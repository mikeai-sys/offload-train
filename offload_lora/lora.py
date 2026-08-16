# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        if isinstance(base, nn.Linear):
            out_f, in_f = base.weight.shape
        else:
            in_f, out_f = base.weight.shape
        self.lora_a = nn.Parameter(torch.empty(r, in_f))
        self.lora_b = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        xf = x.to(self.lora_a.dtype)
        lora = (xf @ self.lora_a.t() @ self.lora_b.t()).to(base_out.dtype)
        return base_out + self.scaling * lora

    def merge(self):
        with torch.no_grad():
            w = self.base.weight.data + self.scaling * (self.lora_b @ self.lora_a)
            self.base.weight.data.copy_(w)
        return self.base


def _set_module(model, name, new_module):
    *parents, last = name.split(".")
    mod = model
    for p in parents:
        mod = getattr(mod, p)
    setattr(mod, last, new_module)


def remap_lora_keys(module, state_dict):
    out = {}
    for key, tensor in state_dict.items():
        *mods, leaf = key.split(".")
        parent = module
        ok = True
        for m in mods:
            parent = getattr(parent, m, None)
            if parent is None:
                ok = False
                break
        if ok and isinstance(parent, LoRALinear) and leaf in ("weight", "bias"):
            out[".".join(mods) + ".base." + leaf] = tensor
        else:
            out[key] = tensor
    return out


def _module_name_matches(name: str, targets) -> bool:
    if targets == "all-linear" or (isinstance(targets, (list, tuple)) and "all-linear" in targets):
        return name.split(".")[-1] in {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "c_attn", "c_proj", "c_fc", "qkv", "proj",
        }
    leaf = name.split(".")[-1]
    return leaf in set(targets)


def _linear_like():
    try:
        from transformers.pytorch_utils import Conv1D
        return (nn.Linear, Conv1D)
    except Exception:
        return (nn.Linear,)


def inject_lora(model, targets, r: int, alpha: float, dropout: float = 0.0):
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, _linear_like()) and _module_name_matches(name, targets):
            lora = LoRALinear(module, r, alpha, dropout)
            _set_module(model, name, lora)
            replaced.append(name)
    return replaced


def freeze_base(model):
    for name, p in model.named_parameters():
        if "lora_a" not in name and "lora_b" not in name:
            p.requires_grad_(False)


def get_lora_params(model):
    params = []
    for name, p in model.named_parameters():
        if ("lora_a" in name or "lora_b" in name) and p.requires_grad:
            params.append(p)
    return params


def count_params(model):
    total = trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def save_lora(model, path):
    state = {
        n: p
        for n, p in model.named_parameters()
        if ("lora_a" in n or "lora_b" in n) and p.requires_grad
    }
    torch.save(state, path)


def load_lora(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected
