# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from offload_lora.lora import inject_lora, freeze_base, get_lora_params
from offload_lora.engine import OffloadTrainer
from offload_lora.pipeline import ElasticPipelineTrainer


class Block(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


class ToyWrapper(nn.Module):
    def __init__(self, vocab=64, hidden=32, n_blocks=6):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([Block(hidden) for _ in range(n_blocks)])
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab)
        from types import SimpleNamespace
        self.config = SimpleNamespace(tie_word_embeddings=False, model_type="toy")


def _lora_state(model):
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def _reference_step(model, input_ids, labels, lr):
    model.zero_grad(set_to_none=True)
    x = model.model.embed_tokens(input_ids)
    for b in model.model.layers:
        x = b(x)
    x = model.model.norm(x)
    logits = model.lm_head(x)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
    )
    loss.backward()
    opt = torch.optim.AdamW(get_lora_params(model), lr=lr)
    opt.step()
    return loss.item()


def test_pipeline_matches_reference_gradients():
    torch.manual_seed(0)
    m = ToyWrapper()
    inject_lora(m, ["fc1", "fc2"], r=4, alpha=8)
    freeze_base(m)
    inp = torch.randint(0, 64, (2, 16))
    labels = inp.clone()
    lr = 1e-3

    ref = copy.deepcopy(m)
    _reference_step(ref, inp, labels, lr)

    pipe = copy.deepcopy(m)
    tr = ElasticPipelineTrainer(
        pipe, lora_targets=["fc1", "fc2"], r=4, alpha=8, lr=lr,
        devices=["cpu", "cpu"], total_steps=1000,
    )
    assert tr.n_stages == 2
    tr.step([(inp, labels)])

    for name in _lora_state(ref):
        a = _lora_state(ref)[name]
        b = _lora_state(pipe)[name]
        assert torch.allclose(a, b, atol=1e-5), f"mismatch {name}"


def test_pipeline_stages1_and_N_consistent():
    # A single device (1 stage) pipeline must equal a 3-device (3 stage) split
    # on the same data and identical LoRA init.
    torch.manual_seed(1)
    m = ToyWrapper(n_blocks=6)
    inject_lora(m, ["fc1", "fc2"], r=4, alpha=8)
    freeze_base(m)
    inp = torch.randint(0, 64, (2, 16))
    labels = inp.clone()
    lr = 1e-3

    one = copy.deepcopy(m)
    tr1 = ElasticPipelineTrainer(one, lora_targets=["fc1", "fc2"], r=4, alpha=8,
                                lr=lr, devices=["cpu"], total_steps=1000)
    tr1.step([(inp, labels)])

    three = copy.deepcopy(m)
    tr3 = ElasticPipelineTrainer(three, lora_targets=["fc1", "fc2"], r=4, alpha=8,
                                 lr=lr, devices=["cpu", "cpu", "cpu"], total_steps=1000)
    assert tr3.n_stages == 3
    tr3.step([(inp, labels)])

    for name in _lora_state(one):
        a = _lora_state(one)[name]
        b = _lora_state(three)[name]
        assert torch.allclose(a, b, atol=1e-5), f"mismatch {name}"


def test_pipeline_loss_decreases():
    torch.manual_seed(2)
    m = ToyWrapper(n_blocks=6)
    inject_lora(m, ["fc1", "fc2"], r=4, alpha=8)
    freeze_base(m)
    tr = ElasticPipelineTrainer(
        m, lora_targets=["fc1", "fc2"], r=4, alpha=8, lr=1e-2,
        devices=["cpu", "cpu"], total_steps=20,
    )
    losses = []
    for _ in range(5):
        inp = torch.randint(0, 64, (2, 16))
        l = tr.step([(inp, inp.clone()), (inp.flip(0), inp.flip(0).clone())])
        losses.append(l)
    assert losses[0] > losses[-1], f"loss did not decrease: {losses}"


def test_pipeline_streaming_tinygpt2_matches_single():
    import os
    base = "/tmp/tinygpt2"
    if not (os.path.isdir(base) and os.path.exists(os.path.join(base, "model.safetensors"))):
        raise pytest.skip("tinygpt2 fixture missing")

    from offload_lora.model import build_skeleton

    # Build once, then deep-copy so both trainers share identical LoRA init.
    torch.manual_seed(3)
    m = build_skeleton(base, "bf16")
    inject_lora(m, "all-linear", r=4, alpha=8)
    freeze_base(m)
    m.eval()  # disable dropout so both trainers are deterministic

    ma = copy.deepcopy(m)
    tr_a = OffloadTrainer(ma, lora_targets="all-linear", r=4, alpha=8, lr=1e-3,
                          device="cpu", stream=True, model_path=base, total_steps=1000)

    mb = copy.deepcopy(m)
    tr_b = ElasticPipelineTrainer(mb, lora_targets="all-linear", r=4, alpha=8, lr=1e-3,
                                  devices=["cpu", "cpu"], stream=True, model_path=base,
                                  total_steps=1000)
    assert tr_b.n_stages == 2

    vocab = m.config.vocab_size
    ids = torch.randint(0, vocab, (1, 10))
    labels = ids.clone()

    tr_a.step(ids, labels)
    tr_b.step([(ids, labels)])

    for (na, pa), (nb, pb) in zip(tr_a.model.named_parameters(), tr_b.model.named_parameters()):
        if pa.requires_grad and ("lora_a" in na or "lora_b" in na):
            assert torch.allclose(pa.detach().cpu(), pb.detach().cpu(), atol=1e-3), \
                f"lora mismatch {na}"
