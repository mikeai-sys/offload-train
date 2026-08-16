# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
import json
import os

import torch
from safetensors.torch import safe_open


class WeightStreamer:
    def __init__(self, model_path, dtype):
        if isinstance(dtype, str):
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}.get(dtype, torch.float16)
        self.model_path = model_path
        self.dtype = dtype
        self.shards = {}
        self.handles = {}
        self._discover()

    def _discover(self):
        index = os.path.join(self.model_path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                wm = json.load(f)["weight_map"]
            by_file = {}
            for key, fname in wm.items():
                by_file.setdefault(fname, []).append(key)
            self.shards = by_file
        else:
            fname = "model.safetensors"
            with safe_open(os.path.join(self.model_path, fname), framework="pt", device="cpu") as h:
                self.shards = {fname: list(h.keys())}

    def _handle(self, fname):
        if fname not in self.handles:
            self.handles[fname] = safe_open(
                os.path.join(self.model_path, fname), framework="pt", device="cpu"
            )
        return self.handles[fname]

    def load_prefix(self, prefix):
        sd = {}
        for fname, keys in self.shards.items():
            rel = [k for k in keys if k.startswith(prefix)]
            if not rel:
                continue
            h = self._handle(fname)
            for k in rel:
                name = k[len(prefix):]
                if name.startswith("."):
                    name = name[1:]
                sd[name] = h.get_tensor(k).to(self.dtype)
        return sd

    def materialize(self, module, prefix, remap=None):
        sd = self.load_prefix(prefix)
        if remap is not None:
            sd = remap(module, sd)
        loaded = set(sd.keys())
        for name, p in module.named_parameters():
            if name in sd:
                p.data = sd[name]
        for name, b in module.named_buffers():
            if name in sd:
                b.data = sd[name]
        return loaded

    @staticmethod
    def evict(module, names):
        for name, p in module.named_parameters():
            if name in names and p.numel() > 0:
                p.data = torch.empty(0, device=p.device, dtype=p.dtype)


def stream_prefixes(model_path, prefixes, num_layers, dtype="bf16"):
    streamer = WeightStreamer(model_path, dtype)
    block_fmt, embed_p, norm_p, head_p = prefixes
    block_prefixes = [block_fmt.format(i) for i in range(num_layers)]
    return streamer, block_prefixes, embed_p, norm_p, head_p
