# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from offload_lora.lora import inject_lora, freeze_base, get_lora_params
from offload_lora.engine import OffloadTrainer


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
        self.config = SimpleNamespace(tie_word_embeddings=False)


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


def _lora_state_dict(model):
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def test_offload_matches_reference_gradients():
    torch.manual_seed(0)
    vocab, hidden, n_blocks = 64, 32, 6
    targets = ["fc1", "fc2"]
    r, alpha, lr = 4, 8.0, 1e-3

    ref = ToyWrapper(vocab, hidden, n_blocks)
    off = ToyWrapper(vocab, hidden, n_blocks)
    off.load_state_dict(ref.state_dict())

    inject_lora(ref, targets, r, alpha, dropout=0.0)
    inject_lora(off, targets, r, alpha, dropout=0.0)
    off.load_state_dict(ref.state_dict())
    freeze_base(ref)
    freeze_base(off)

    input_ids = torch.randint(0, vocab, (2, 16))
    labels = input_ids.clone()

    ref_loss = _reference_step(ref, input_ids, labels, lr)
    trainer = OffloadTrainer(off, lora_targets=targets, r=r, alpha=alpha,
                             lr=lr, device="cpu", offload_activations=True,
                             total_steps=0)
    off_loss = trainer.step(input_ids, labels)

    ref_sd = _lora_state_dict(ref)
    off_sd = _lora_state_dict(off)
    assert set(ref_sd.keys()) == set(off_sd.keys())
    for k in ref_sd:
        assert torch.allclose(ref_sd[k], off_sd[k], atol=1e-5), f"mismatch at {k}"

    assert abs(ref_loss - off_loss) < 1e-4
    print("test_offload_matches_reference_gradients: PASS")


def test_loss_decreases():
    torch.manual_seed(1)
    model = ToyWrapper(64, 32, 6)
    inject_lora(model, ["fc1", "fc2"], 4, 8.0, dropout=0.0)
    freeze_base(model)
    trainer = OffloadTrainer(model, lora_targets=["fc1", "fc2"], r=4, alpha=8.0,
                             lr=1e-2, device="cpu", offload_activations=True,
                             total_steps=0)
    ids = torch.randint(0, 64, (2, 16))
    losses = [trainer.step(ids, ids) for _ in range(20)]
    assert losses[-1] < losses[0], "loss did not decrease"
    print(f"test_loss_decreases: PASS (first={losses[0]:.3f} last={losses[-1]:.3f})")


if __name__ == "__main__":
    test_offload_matches_reference_gradients()
    test_loss_decreases()
    print("ALL TESTS PASSED")
