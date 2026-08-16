# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
import math

import torch
import torch.nn.functional as F

from .lora import inject_lora, freeze_base, get_lora_params, remap_lora_keys
from .model import get_components, get_prefixes


class OffloadTrainer:
    def __init__(
        self,
        model,
        lora_targets=("q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"),
        r: int = 8,
        alpha: float = 16.0,
        lr: float = 2e-4,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        device: str = "cuda",
        offload_activations: bool = True,
        grad_clip: float = 0.0,
        warmup_steps: int = 0,
        total_steps: int = 0,
        lr_decay: str = "cosine",
        stream: bool = False,
        model_path: str = None,
    ):
        self.device = device
        self.offload_activations = offload_activations
        self.stream = stream
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

        self.norm.to(device)
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
        self._pe = None

        if self.stream and model_path is not None:
            from .stream import WeightStreamer
            dtype = {torch.bfloat16: "bf16", torch.float16: "fp16",
                     torch.float32: "fp32"}.get(self.dtype, "fp16")
            self.streamer = WeightStreamer(model_path, dtype)
            bf, ep, np_, hp = get_prefixes(model)
            self.block_prefixes = [bf.format(i) for i in range(len(self.blocks))]
            self.embed_p, self.norm_p, self.head_p = ep, np_, hp
            self.tied = bool(getattr(model.config, "tie_word_embeddings", False))
            self._fix_rotary()
        else:
            self.streamer = None
            self.tied = False

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

    def _mat(self, module, prefix):
        return self.streamer.materialize(module, prefix, remap=remap_lora_keys)

    def _evict(self, module, names):
        if self.stream:
            self.streamer.evict(module, names)

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
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * prog))

    def _act(self, t):
        return t.cpu() if self.offload_activations else t

    def _to_dev(self, t):
        return t.to(self.device)

    def _ensure_pe(self, x):
        if self.rotary is None:
            return None
        T = x.size(1)
        B = x.size(0)
        if self._pe is not None and self._pe[0].size(1) == T:
            return self._pe
        pos = torch.arange(T, device="cpu").unsqueeze(0).expand(B, T)
        cos, sin = self.rotary(x.to("cpu"), pos)
        self._pe = (cos.to(self.device), sin.to(self.device))
        return self._pe

    def _block_call(self, block, x):
        pe = self._ensure_pe(x)
        if pe is None:
            return block(x)
        return block(x, position_embeddings=pe)

    def step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> int:
        self._step += 1
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._lr()
        self.optimizer.zero_grad(set_to_none=True)
        labels = labels.to(self.device)
        self._pe = None

        if self.stream:
            self.norm_names = self._mat(self.norm, self.norm_p)
            self.norm.to(self.device)
            self.embed_names = self._mat(self.embed, self.embed_p)
            if self.tied:
                self.lm_head.weight.data = self.embed.weight.data.to(self.device)
                self.head_names = {"weight"}
            else:
                self.head_names = self._mat(self.lm_head, self.head_p)
            self.lm_head.to(self.device)

        with torch.no_grad():
            x = self.embed(input_ids.to("cpu")).to(self.device)
            if self.stream:
                self._evict(self.embed, self.embed_names)
            acts = [self._act(x)]
            for i, block in enumerate(self.blocks):
                if self.stream:
                    names = self._mat(block, self.block_prefixes[i])
                block.to(self.device)
                x = self._block_call(block, x)
                block.to("cpu")
                if self.stream:
                    self._evict(block, names)
                acts.append(self._act(x))
        x = self.norm(x.to(self.device))
        logits = self.lm_head(x)
        loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        loss_val = loss.item()

        x_dev = self._to_dev(acts[-1])
        x_dev.requires_grad_(True)
        self.norm.to(self.device)
        self.lm_head.to(self.device)
        with torch.enable_grad():
            h = self.norm(x_dev)
            logits2 = self.lm_head(h)
            loss2 = F.cross_entropy(
                logits2[:, :-1].reshape(-1, logits2.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        torch.autograd.backward(loss2)
        grad = x_dev.grad

        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            inp = self._to_dev(acts[i])
            inp.requires_grad_(True)
            if self.stream:
                names = self._mat(block, self.block_prefixes[i])
            block.to(self.device)
            with torch.enable_grad():
                out = self._block_call(block, inp)
            torch.autograd.backward(out, grad.to(self.device))
            g = inp.grad
            block.to("cpu")
            if self.stream:
                self._evict(block, names)
            grad = g

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.lora_params, self.grad_clip)
        self.optimizer.step()

        if self.stream:
            self._evict(self.norm, self.norm_names)
            self._evict(self.lm_head, self.head_names)
        return loss_val

    @torch.no_grad()
    def evaluate(self, input_ids, labels):
        labels = labels.to(self.device)
        self._pe = None
        if self.stream:
            self.norm_names = self._mat(self.norm, self.norm_p)
            self.norm.to(self.device)
            self.embed_names = self._mat(self.embed, self.embed_p)
            if self.tied:
                self.lm_head.weight.data = self.embed.weight.data.to(self.device)
                self.head_names = {"weight"}
            else:
                self.head_names = self._mat(self.lm_head, self.head_p)
            self.lm_head.to(self.device)
        x = self.embed(input_ids.to("cpu")).to(self.device)
        if self.stream:
            self._evict(self.embed, self.embed_names)
        for i, block in enumerate(self.blocks):
            if self.stream:
                names = self._mat(block, self.block_prefixes[i])
            block.to(self.device)
            x = self._block_call(block, x)
            block.to("cpu")
            if self.stream:
                self._evict(block, names)
        x = self.norm(x.to(self.device))
        logits = self.lm_head(x)
        if self.stream:
            self._evict(self.norm, self.norm_names)
            self._evict(self.lm_head, self.head_names)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        ).item()

    def save_lora(self, path):
        from .lora import save_lora
        save_lora(self.model, path)

    def merge_and_save(self, path):
        for name, module in list(self.model.named_modules()):
            if hasattr(module, "merge"):
                module.merge()
        self.model.save_pretrained(path)
