# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F

from .lora import inject_lora, freeze_base, get_lora_params, save_lora, remap_lora_keys
from .model import get_components, get_prefixes
from .stream import WeightStreamer


def _rotary_pe(rotary, x, device):
    if rotary is None:
        return None
    T = x.size(1)
    B = x.size(0)
    pos = torch.arange(T, device="cpu").unsqueeze(0).expand(B, T)
    cos, sin = rotary(x.to("cpu"), pos)
    return cos.to(device), sin.to(device)


def _block_call(block, x, pe):
    if pe is None:
        return block(x)
    return block(x, position_embeddings=pe)


class Stage:
    """Owns a contiguous slice of transformer blocks resident on one device.

    The slice's base weights are streamed from disk once per training step
    (``begin_step`` -> ``end_step``) and kept resident for every microbatch, so
    the weights are read from disk a single time per step instead of once per
    microbatch. ``window`` caps how many blocks stay materialised at once (LRU).
    """

    def __init__(self, idx, blocks, device, block_prefixes, rotary,
                 is_first, is_last, tied, embed, norm, lm_head,
                 embed_p, norm_p, head_p, stream, streamer,
                 window, offload_activations):
        self.idx = idx
        self.blocks = blocks
        self.device = device
        self.block_prefixes = block_prefixes
        self.rotary = rotary
        self.is_first = is_first
        self.is_last = is_last
        self.tied = tied
        self.embed = embed
        self.norm = norm
        self.lm_head = lm_head
        self.embed_p = embed_p
        self.norm_p = norm_p
        self.head_p = head_p
        self.stream = stream
        self.streamer = streamer
        self.window = max(1, window)
        self.offload_activations = offload_activations
        self.cu_stream = None
        self.resident_idx = {}
        self.lru = []

    def _mat(self, module, prefix):
        if self.stream:
            return self.streamer.materialize(module, prefix, remap=remap_lora_keys)
        return set(n for n, _ in module.named_parameters())

    def _evict(self, module, names):
        if self.stream and names:
            self.streamer.evict(module, names)

    def _ensure_block(self, j):
        block = self.blocks[j]
        if self.stream and j not in self.resident_idx:
            names = self._mat(block, self.block_prefixes[j])
            self.resident_idx[j] = names
            self.lru.append(j)
            while len(self.lru) > self.window:
                old = self.lru.pop(0)
                if old in self.resident_idx:
                    self._evict(self.blocks[old], self.resident_idx[old])
                    del self.resident_idx[old]
        block.to(self.device)

    def _ensure_head(self):
        if self.is_first:
            self._embed_names = self._mat(self.embed, self.embed_p)
            self.embed.to(self.device)
        if self.is_last:
            self._norm_names = self._mat(self.norm, self.norm_p)
            self.norm.to(self.device)
            if self.tied:
                if self.embed is not None:
                    self.lm_head.weight.data = self.embed.weight.data.to(self.device)
                self._head_names = set()
            else:
                self._head_names = self._mat(self.lm_head, self.head_p)
                self.lm_head.to(self.device)

    def _evict_head(self):
        if self.is_first:
            self._evict(self.embed, getattr(self, "_embed_names", set()))
        if self.is_last:
            self._evict(self.norm, getattr(self, "_norm_names", set()))
            self._evict(self.lm_head, getattr(self, "_head_names", set()))

    def begin_step(self):
        self._ensure_head()
        for j in range(len(self.blocks)):
            self._ensure_block(j)

    def end_step(self):
        for j in list(self.resident_idx.keys()):
            self._evict(self.blocks[j], self.resident_idx[j])
        self.resident_idx.clear()
        self.lru.clear()
        self._evict_head()

    def forward(self, x, labels=None):
        pe = _rotary_pe(self.rotary, x, self.device)
        for j, block in enumerate(self.blocks):
            self._ensure_block(j)
            x = _block_call(block, x, pe)
        if self.is_last:
            h = self.norm(x)
            logits = self.lm_head(h)
            if labels is not None:
                return F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                )
            return logits
        return x

    def recompute(self, x):
        pe = _rotary_pe(self.rotary, x, self.device)
        h = x
        for j, block in enumerate(self.blocks):
            self._ensure_block(j)
            h = _block_call(block, h, pe)
        if self.is_last:
            return self.lm_head(self.norm(h))
        return h

    def run_blocks(self, x):
        pe = _rotary_pe(self.rotary, x, self.device)
        h = x
        for j, block in enumerate(self.blocks):
            self._ensure_block(j)
            block.to(self.device)
            h = _block_call(block, h, pe)
        return h


class ElasticPipelineTrainer:
    """Pipeline-parallel LoRA trainer that scales from 1 GPU to N GPUs.

    Blocks are split contiguously across ``devices`` (one stage per device,
    GPipe). Each step runs ``microbatches`` microbatches through a 1F1B
    schedule, accumulating LoRA gradients and applying a single optimizer
    step. Works on CPU for tests; targets CUDA for real training. Floor is
    32 GB host RAM + disk streaming, so the base model never fully loads.
    """

    def __init__(self, model,
                 lora_targets=("q_proj", "v_proj", "k_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"),
                 r: int = 8, alpha: float = 16.0, lr: float = 2e-4,
                 dropout: float = 0.0, weight_decay: float = 0.0,
                 devices=("cuda",), offload_activations: bool = True,
                 grad_clip: float = 0.0, warmup_steps: int = 0,
                 total_steps: int = 0, lr_decay: str = "cosine",
                 stream: bool = False, model_path: str = None, window: int = 8):
        self.devices = list(devices)
        self.offload_activations = offload_activations
        self.dtype = next(model.parameters()).dtype

        self.replaced = inject_lora(model, lora_targets, r, alpha, dropout)
        freeze_base(model)

        self.embed, self.blocks, self.norm, self.lm_head, self.rotary = get_components(model)
        self.model = model
        self.tied = bool(getattr(model.config, "tie_word_embeddings", False))
        if self.tied:
            self.lm_head.weight = torch.nn.Parameter(
                self.lm_head.weight.data.clone().requires_grad_(False)
            )

        self._fix_rotary()

        self.lora_params = get_lora_params(model)
        self.optimizer = torch.optim.AdamW(
            self.lora_params, lr=lr, weight_decay=weight_decay
        )

        self._step = 0
        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_decay = lr_decay
        self.grad_clip = grad_clip

        bf, ep, np_, hp = get_prefixes(model)
        block_prefixes = [bf.format(i) for i in range(len(self.blocks))]
        streamer = None
        if stream and model_path is not None:
            dtype = {torch.bfloat16: "bf16", torch.float16: "fp16",
                     torch.float32: "fp32"}.get(self.dtype, "fp16")
            streamer = WeightStreamer(model_path, dtype)

        n = min(len(self.devices), len(self.blocks))
        bounds = [round(len(self.blocks) * s / n) for s in range(n + 1)]
        self.stages = []
        for i in range(n):
            start, end = bounds[i], bounds[i + 1]
            if end <= start:
                continue
            self.stages.append(Stage(
                idx=i,
                blocks=self.blocks[start:end],
                device=self.devices[i],
                block_prefixes=block_prefixes[start:end],
                rotary=self.rotary,
                is_first=(i == 0),
                is_last=(i == n - 1),
                tied=self.tied,
                embed=self.embed,
                norm=self.norm if i == n - 1 else None,
                lm_head=self.lm_head if i == n - 1 else None,
                embed_p=ep, norm_p=np_, head_p=hp,
                stream=stream, streamer=streamer,
                window=window, offload_activations=offload_activations,
            ))
        self.n_stages = len(self.stages)

        # Per-stage CUDA streams let the GPipe 1F1B schedule overlap compute
        # and host<->device transfers across devices (stage i forward overlaps
        # stage i-1 backward). On CPU this is a no-op and the path is identical
        # to a sequential run.
        self.use_streams = torch.cuda.is_available()
        self.fwd_events = {}
        self.bwd_events = {}
        for st in self.stages:
            if self.use_streams and str(st.device).startswith("cuda"):
                st.cu_stream = torch.cuda.Stream(device=st.device)
            else:
                st.cu_stream = None

    def _fix_rotary(self):
        if self.rotary is None:
            return
        inv = getattr(self.rotary, "inv_freq", None)
        if inv is None or inv.device.type != "meta":
            return
        cfg = self.model.config
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads)
        theta = getattr(cfg, "rope_theta", 10000.0)
        partial = getattr(cfg, "partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial)
        idx = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (idx / rotary_dim))
        self.rotary.inv_freq = inv_freq

    def _lr(self) -> float:
        if self.total_steps <= 1:
            return self.base_lr
        s = self._step
        if s < self.warmup_steps:
            return self.base_lr * (s / max(1, self.warmup_steps))
        if s >= self.total_steps:
            return 0.0
        prog = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        if self.lr_decay == "linear":
            return self.base_lr * (1 - prog)
        if self.lr_decay == "none":
            return self.base_lr
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * prog))

    def _offload(self, t):
        if self.offload_activations:
            return t.detach().cpu()
        return t.detach()

    def _fwd(self, m, mbs, acts, loss_tensors):
        with torch.no_grad():
            input_ids, labels = mbs[m]
            prev_out = None
            for i, stage in enumerate(self.stages):
                if stage.cu_stream is not None:
                    if (i - 1, m) in self.fwd_events:
                        self.fwd_events[(i - 1, m)].wait(stage.cu_stream)
                    ctx = torch.cuda.stream(stage.cu_stream)
                else:
                    ctx = nullcontext()
                with ctx:
                    if stage.is_first:
                        x = stage.embed(input_ids.to("cpu")).to(stage.device)
                        stage_in = input_ids
                    else:
                        x = prev_out.to(stage.device)
                        stage_in = self._offload(prev_out)
                    acts[(i, m)] = stage_in
                    if stage.is_last:
                        out = stage.forward(x, labels.to(stage.device))
                        loss_tensors.append(out)
                        prev_out = None
                    else:
                        prev_out = stage.forward(x)
                if stage.cu_stream is not None:
                    ev = torch.cuda.Event()
                    ev.record(stage.cu_stream)
                    self.fwd_events[(i, m)] = ev

    def _make_input(self, stage, stage_in, input_ids):
        if stage.is_first:
            with torch.no_grad():
                emb = stage.embed(input_ids.to("cpu"))
            return emb.to(stage.device).requires_grad_(True)
        with torch.no_grad():
            inp = stage_in.to(stage.device)
        return inp.requires_grad_(True)

    def _bwd_stage(self, stage, x, grad_from_next, labels):
        device = stage.device
        pe = _rotary_pe(stage.rotary, x, device)
        # Record each block's input (as an independent leaf) by running the
        # stage forward once, then reverse-loop the blocks in isolation
        # (exactly like OffloadTrainer). This avoids fusing many blocks into a
        # single autograd graph, which is unsafe for in-place ops in attention.
        hidden = x
        inputs = []
        for j, block in enumerate(stage.blocks):
            stage._ensure_block(j)
            block.to(device)
            leaf = hidden.detach().requires_grad_(True)
            inputs.append(leaf)
            with torch.enable_grad():
                hidden = _block_call(block, leaf, pe)

        if stage.is_last:
            h = hidden.detach().requires_grad_(True)
            stage.norm.to(device)
            with torch.enable_grad():
                logits = stage.lm_head(stage.norm(h))
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                )
            torch.autograd.backward(loss)
            grad = h.grad
        else:
            grad = grad_from_next

        for j in reversed(range(len(stage.blocks))):
            block = stage.blocks[j]
            inp = inputs[j]
            stage._ensure_block(j)
            block.to(device)
            inp.requires_grad_(True)
            with torch.enable_grad():
                out = _block_call(block, inp, pe)
            torch.autograd.backward(out, grad.to(device))
            grad = inp.grad
        return grad

    def _bwd(self, m, mbs, acts):
        input_ids, labels = mbs[m]
        device = self.stages[-1].device
        labels = labels.to(device)
        grad = None
        for i in reversed(range(self.n_stages)):
            stage = self.stages[i]
            if stage.cu_stream is not None:
                if (i, m) in self.fwd_events:
                    self.fwd_events[(i, m)].wait(stage.cu_stream)
                if grad is not None and (i + 1, m) in self.bwd_events:
                    self.bwd_events[(i + 1, m)].wait(stage.cu_stream)
                ctx = torch.cuda.stream(stage.cu_stream)
            else:
                ctx = nullcontext()
            with ctx:
                x = self._make_input(stage, acts[(i, m)], input_ids)
                grad = self._bwd_stage(stage, x, grad, labels)
            if stage.cu_stream is not None:
                ev = torch.cuda.Event()
                ev.record(stage.cu_stream)
                self.bwd_events[(i, m)] = ev

    def step(self, microbatches):
        self._step += 1
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._lr()
        self.optimizer.zero_grad(set_to_none=True)
        M = len(microbatches)
        n = self.n_stages
        acts = {}
        loss_tensors = []
        self.fwd_events.clear()
        self.bwd_events.clear()
        for stage in self.stages:
            stage.begin_step()
        warm = min(M, n)
        for m in range(warm):
            self._fwd(m, microbatches, acts, loss_tensors)
        start_bwd = 0
        for m in range(warm, M):
            self._fwd(m, microbatches, acts, loss_tensors)
            self._bwd(start_bwd, microbatches, acts)
            start_bwd += 1
        for m in range(start_bwd, M):
            self._bwd(m, microbatches, acts)
        if self.use_streams:
            torch.cuda.synchronize()
        losses = [float(t.detach().float().cpu().item()) for t in loss_tensors]
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.lora_params, self.grad_clip)
        self.optimizer.step()
        for stage in self.stages:
            stage.end_step()
        del acts
        return sum(losses) / max(1, len(losses)) if losses else float("nan")

    @torch.no_grad()
    def evaluate(self, microbatches):
        device = self.stages[-1].device
        total, count = 0.0, 0
        for stage in self.stages:
            stage.begin_step()
        try:
            for input_ids, labels in microbatches:
                labels = labels.to(device)
                x = self.stages[0].embed(input_ids.to("cpu")).to(device)
                for stage in self.stages:
                    x = stage.forward(x.to(stage.device))
                logits = x
                total += F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                ).item()
                count += 1
        finally:
            for stage in self.stages:
                stage.end_step()
        return total / max(1, count)

    def save_lora(self, path):
        save_lora(self.model, path)

    def merge_and_save(self, path):
        for name, module in list(self.model.named_modules()):
            if hasattr(module, "merge"):
                module.merge()
        self.model.save_pretrained(path)
