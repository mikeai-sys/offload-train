# offload-lora

> **Powered by Mike Koci from EliseArt** &mdash; https://elise-dev.web.app

Fine-tune **very large models on a single, modest GPU** (e.g. 32 GB) by streaming
**one transformer block at a time** between host RAM and VRAM, while training small
**LoRA adapters**. Only one layer's weights + activations ever occupy the GPU, so a
trillion-parameter model becomes trainable without a cluster of expensive accelerators.

This is the "pipeline/offload" approach you asked for: instead of loading the whole
model onto the GPU, we *flow* the data through the layers, moving each block up to the
GPU for its forward/backward and back down to CPU afterwards.

It is also **elastic**: the transformer blocks are split across **1, 2, 3, or N GPUs**
(pipeline parallelism / GPipe). More GPUs → more speed, and the memory floor is a
single 32 GB device + disk streaming, so it scales from one weak card up to a cluster.

## How it works

The core idea is **weight streaming from disk**: the model is never fully loaded into
RAM. We keep a parameter *skeleton* (meta tensors, no storage) and read one block's
weights from the `safetensors` file only when it is needed, then evict them.

1. **No full load.** At startup we build a skeleton with `from_config` (meta device) —
   the weights occupy essentially no RAM/VRAM. A `WeightStreamer` memory-maps the
   checkpoint so individual tensors can be fetched on demand.
2. **Stream + evict per block.** Each training step:
   - Forward (no graph): for block 0 → N, *materialize* that block's weights from disk
     into RAM, move it to the accelerator, compute, move it back, then **evict** it
     (free its storage). Only one block is resident at a time.
   - Backward: recompute each block under `torch.enable_grad()` (materialize → compute →
     backprop its LoRA slice → evict), so gradients are exact while peak memory stays
     ≈ **one block + one activation + the LM head**.
3. **LoRA adapters** (`lora_A`, `lora_B`) are injected into the target layers and live
   resident (tiny). Only these are trained; the frozen base is streamed.
4. **Tied embeddings** (e.g. Qwen/Llama) are handled by copying `embed_tokens` into
   `lm_head` each step; **rotary position embeddings** are recomputed from the config
   (their `inv_freq` is not stored in the checkpoint).
5. **Optimizer** (AdamW) updates only the LoRA params, so optimizer state is tiny.

Peak GPU memory is bounded by a single layer, which is why a 32 GB GPU can fine-tune
models whose full parameter set would otherwise need hundreds of GB — the weights flow
through the device like a pipeline, and the "last" block is dropped from memory as the
next one is loaded.

## Install

```bash
pip install -e .
# or
pip install .
```

Then `import offload_lora` works, and the `offload-train` CLI is available.

---

# Usage

There are two ways to use offload-lora:

1. **The `offload-train` CLI** — the fastest way to launch a real fine-tune.
2. **The Python library** — for custom training loops, research, or embedding the
   trainers inside a larger program.

---

## 1. CLI: `offload-train`

The CLI auto-detects its mode: if `--model` points at a **local directory containing
`model.safetensors`** (or `model.safetensors.index.json`), it streams weights from disk
(**no full load**). If `--model` is a **Hugging Face repo id**, it loads the model fully
into host memory first (use a local path for the disk-streaming path).

### Quick start commands

Single GPU, local model, streamed from disk:

```bash
offload-train \
  --model /path/to/Qwen2.5-1.5B-Instruct \
  --text corpus.txt \
  --r 8 --alpha 16 \
  --device cuda \
  --seq-len 512 --batch-size 1 \
  --steps 200 --lr 2e-4 \
  --output-lora lora_adapters.pt
```

Two GPUs (pipeline-parallel, blocks split across both):

```bash
offload-train \
  --model /path/to/BigModel \
  --gpus 2 \
  --microbatches 4 --window 16 \
  --device cuda --steps 200
```

Explicit GPU list (equivalent to the above for 2 GPUs):

```bash
offload-train --model /path/to/BigModel --gpus cuda:0,cuda:1 --device cuda --steps 200
```

CPU smoke test (no GPU needed — exercises the same control flow):

```bash
offload-train --model ./Qwen2.5-1.5B-Instruct --device cpu \
  --seq-len 128 --steps 5 --output-lora /tmp/lora.pt
```

From a Hugging Face repo id (full in-RAM load, then LoRA):

```bash
offload-train --model meta-llama/Llama-3-8B --device cuda --steps 100 \
  --tokenizer meta-llama/Llama-3-8B --trust-remote-code
```

### Full command reference

Every flag accepted by `offload-train`:

| flag | default | meaning |
|------|---------|---------|
| `--model` | *(required)* | Hugging Face repo id **or** local path. A local dir with `model.safetensors` triggers disk streaming (no full load). |
| `--text` | `None` | Path to a `.txt` corpus. Defaults to a built-in sample text. |
| `--tokenizer` | `None` | Tokenizer id/path; defaults to `--model`. |
| `--lora-targets` | `None` → `all-linear` | Space-separated module names to adapt (e.g. `q_proj v_proj`), or `all-linear`. |
| `--r` | `8` | LoRA rank. |
| `--alpha` | `16.0` | LoRA scaling (`alpha / r`). |
| `--lora-dropout` | `0.0` | Dropout applied inside the LoRA path. |
| `--dtype` | `bf16` | Weights precision: `bf16` / `fp16` / `fp32`. |
| `--device` | `cuda` | Accelerator for the single-device trainer; use `cpu` for testing. |
| `--gpus` | `auto` | Device split: `auto`, a count like `2`, or a list like `cuda:0,cuda:1`. Blocks are split across these (GPipe). More GPUs = more speed. |
| `--no-offload-activations` | off | If set, keep activations on the accelerator (faster, more VRAM). |
| `--seq-len` | `512` | Token window length. |
| `--batch-size` | `1` | Samples per microbatch. |
| `--microbatches` | `1` | Microbatches per optimizer step (1F1B pipeline schedule). |
| `--window` | `8` | Max transformer blocks kept resident (streamed) per stage at once (LRU eviction). |
| `--grad-accum` | `1` | (Reserved) gradient accumulation steps. |
| `--epochs` | `1` | Training epochs. |
| `--steps` | `0` | Total optimizer steps; `0` means `epochs * 1000`. |
| `--lr` | `2e-4` | Learning rate. |
| `--weight-decay` | `0.0` | AdamW weight decay. |
| `--warmup-steps` | `0` | LR warmup steps. |
| `--grad-clip` | `0.0` | Gradient clipping norm (`0` = off). |
| `--lr-decay` | `cosine` | LR schedule: `cosine` / `linear` / `none`. |
| `--output-lora` | `lora_adapters.pt` | Where to save the trained LoRA adapters. |
| `--output-merged` | `None` | Optional: save a merged full model to this path. |
| `--trust-remote-code` | off | Pass `trust_remote_code=True` to `transformers`. |
| `--log-every` | `10` | Log a loss line every N steps. |

### Notes

- **Elastic multi-GPU scaling.** `--gpus` lets the trainer grow from one device to many.
  Blocks are split contiguously across the listed devices, so two 32 GB GPUs
  (e.g. `--gpus 2`) each hold a slice of a 700B model's layers while the base weights
  stream from disk one block at a time — the model never fully loads into RAM. Layer
  distribution is automatic: `--gpus 2` and `--gpus cuda:0,cuda:1` are equivalent for a
  2-GPU box. Throughput scales roughly with the number of stages, because the 1F1B
  schedule keeps every device busy.
- **Automatic streaming.** Point `--model` at a local checkpoint directory (with
  `model.safetensors`) to get the disk-streaming path. Point it at a HF repo id to load
  fully into host memory instead.
- **Outputs.** `--output-lora` writes the LoRA adapters (`lora_a`/`lora_b`) as a
  `torch.save` state dict. `--output-merged` additionally writes a merged full model
  (base + LoRA baked in).

---

## 2. Library (Python API)

Everything the CLI uses is importable from `offload_lora`:

```python
import offload_lora as ol
# or individually:
from offload_lora import OffloadTrainer, ElasticPipelineTrainer
from offload_lora import (inject_lora, freeze_base, get_lora_params, save_lora,
                          load_lora, count_params, LoRALinear)
from offload_lora import (build_skeleton, load_model_cpu, load_tokenizer,
                          get_components, get_prefixes)
from offload_lora import WeightStreamer
from offload_lora.data import TextWindowDataset, collate_windows, SAMPLE_TEXT
from offload_lora.config import TrainConfig
```

### Trainers

#### `OffloadTrainer` — single device

```python
trainer = ol.OffloadTrainer(
    model,                      # an nn.Module (e.g. from build_skeleton/load_model_cpu)
    lora_targets="all-linear",  # tuple/list of names or "all-linear"
    r=8, alpha=16.0, lr=2e-4,
    dropout=0.0, weight_decay=0.0,
    device="cuda",              # "cuda" or "cpu"
    offload_activations=True,
    grad_clip=0.0,
    warmup_steps=0, total_steps=0, lr_decay="cosine",
    stream=False,               # True + model_path => stream weights from disk
    model_path=None,            # path to the safetensors checkpoint
)
```

Methods:

| method | signature | returns | description |
|--------|-----------|---------|-------------|
| `step` | `step(input_ids, labels)` | `float` loss | One optimizer step on a single (input_ids, labels) pair. Streams/materials blocks, runs forward + recompute backward, applies LR schedule & gradient clip, steps AdamW. |
| `evaluate` | `evaluate(input_ids, labels)` | `float` loss | Forward-only loss (no gradients). |
| `save_lora` | `save_lora(path)` | — | Save `lora_a`/`lora_b` to a file. |
| `merge_and_save` | `merge_and_save(path)` | — | Merge LoRA into base weights and `save_pretrained`. |

Useful attributes: `trainer.replaced` (names of injected layers),
`trainer.lora_params`, `trainer.optimizer`, `trainer.n_stages` (=1),
`trainer._lr()` (current learning rate).

Minimal loop:

```python
tok = ol.load_tokenizer("meta-llama/Llama-3-8B")
ds = TextWindowDataset(tok, open("corpus.txt").read(), seq_len=512)
loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_windows)

model = ol.load_model_cpu("meta-llama/Llama-3-8B", dtype="bf16")
trainer = ol.OffloadTrainer(model, r=8, alpha=16, device="cuda", stream=False)

for input_ids, labels in loader:
    loss = trainer.step(input_ids, labels)
trainer.save_lora("lora_adapters.pt")
```

#### `ElasticPipelineTrainer` — multi-device (pipeline-parallel / GPipe)

```python
trainer = ol.ElasticPipelineTrainer(
    model,
    lora_targets="all-linear",
    r=8, alpha=16.0, lr=2e-4,
    dropout=0.0, weight_decay=0.0,
    devices=("cuda",),           # ("cuda",), ("cuda:0","cuda:1"), or ["cpu","cpu"]
    offload_activations=True,
    grad_clip=0.0,
    warmup_steps=0, total_steps=0, lr_decay="cosine",
    stream=False,                # True + model_path => disk streaming
    model_path=None,
    window=8,                    # max resident blocks per stage (LRU)
)
```

Methods:

| method | signature | returns | description |
|--------|-----------|---------|-------------|
| `step` | `step(microbatches)` | `float` avg loss | One optimizer step over a **list** of `(input_ids, labels)` microbatches, using a 1F1B GPipe schedule with CUDA stream/event overlap across stages. |
| `evaluate` | `evaluate(microbatches)` | `float` loss | Forward-only loss over a list of microbatches. |
| `save_lora` | `save_lora(path)` | — | Save LoRA adapters. |
| `merge_and_save` | `merge_and_save(path)` | — | Merge + `save_pretrained`. |

Useful attributes: `trainer.n_stages` (number of pipeline stages = min(devices, blocks)),
`trainer.replaced`, `trainer.use_streams` (True if CUDA available),
`trainer.fwd_events` / `trainer.bwd_events` (overlap bookkeeping), `trainer._lr()`,
`trainer.stages` (the `Stage` objects).

Minimal loop:

```python
model = ol.build_skeleton("/path/to/BigModel", dtype="bf16")
trainer = ol.ElasticPipelineTrainer(
    model, lora_targets="all-linear", r=8, alpha=16,
    devices=["cuda:0", "cuda:1"], stream=True, model_path="/path/to/BigModel",
    window=8,
)
microbatches = [(ids1, lab1), (ids2, lab2)]
loss = trainer.step(microbatches)
trainer.save_lora("lora_adapters.pt")
```

> On a CPU-only machine `devices=["cpu","cpu"]` still runs correctly (the CUDA stream
> overlap becomes a no-op, so it is simply a sequential run), which makes it ideal for
> tests.

---

### LoRA tools

| function | signature | returns / effect |
|----------|-----------|------------------|
| `inject_lora` | `inject_lora(model, targets, r, alpha, dropout=0.0)` | Replaces target `Linear`/`Conv1D` layers with `LoRALinear`. Returns a list of replaced module names. |
| `freeze_base` | `freeze_base(model)` | Sets `requires_grad=False` on every non-LoRA parameter (the frozen base). |
| `get_lora_params` | `get_lora_params(model)` | Returns the list of trainable LoRA `Parameter`s. |
| `count_params` | `count_params(model)` | Returns `(total_params, trainable_params)`. |
| `save_lora` | `save_lora(model, path)` | Saves the LoRA state dict to `path`. |
| `load_lora` | `load_lora(model, path)` | Loads LoRA weights into `model`; returns `(missing, unexpected)` keys. |
| `LoRALinear` | `LoRALinear(base_linear, r, alpha, dropout=0.0)` | Drop-in LoRA layer wrapping a `nn.Linear`/`Conv1D`. Has `.merge()` to bake LoRA into the base. |

`LoRALinear.forward(x)` = `base(x) + scaling * (x @ lora_a.T @ lora_b.T)`, computed in
fp32 and cast back to the base dtype.

---

### Model tools

| function | signature | returns / effect |
|----------|-----------|------------------|
| `build_skeleton` | `build_skeleton(model_name_or_path, dtype="bf16", trust_remote_code=False)` | Builds a model from its **config only** (`from_config`) — weights are random/`meta`, giving the disk-streaming skeleton. Use with `stream=True`. |
| `load_model_cpu` | `load_model_cpu(model_name_or_path, dtype="bf16", trust_remote_code=False)` | Loads the full model into host RAM (no streaming). |
| `load_tokenizer` | `load_tokenizer(model_name_or_path, trust_remote_code=False)` | Loads the tokenizer (sets `pad_token = eos_token` if missing). |
| `get_components` | `get_components(model)` | Returns `(embed, blocks, norm, lm_head, rotary)` for supported architectures. |
| `get_prefixes` | `get_prefixes(model)` | Returns `(block_prefix_fmt, embed_prefix, norm_prefix, head_prefix)` used by the streamer. |

Supported architectures (in `get_components` / `get_prefixes`): Llama, Qwen2/Qwen3,
Mistral, Gemma, Phi, Cohere, DeepSeek, BLOOM, GPT-NeoX, Llama4, GPT-2 (incl. `Conv1D`),
OPT, T5. Extend these two functions for more.

---

### Streaming tool

`WeightStreamer` memory-maps a checkpoint's `safetensors` so individual tensors can be
fetched on demand (this is what powers the no-full-load path):

```python
streamer = WeightStreamer(model_path, dtype="bf16")   # dtype: "bf16"/"fp16"/"fp32" or torch.dtype
sd = streamer.load_prefix("transformer.h.0.")          # {name: tensor} for that prefix
names = streamer.materialize(module, "transformer.h.0.", remap=ol.lora.remap_lora_keys)
streamer.evict(module, names)                          # free the materialized storage
```

| method | signature | description |
|--------|-----------|-------------|
| `load_prefix` | `load_prefix(prefix)` | Returns `{stripped_name: tensor}` for all keys under `prefix`. |
| `materialize` | `materialize(module, prefix, remap=None)` | Loads the prefix's tensors into `module`'s params/buffers and returns the set of loaded param names. |
| `evict` | `evict(module, names)` | Frees the listed parameters' storage. |

`remap_lora_keys(module, state_dict)` rewrites streamed base weights so they land on the
`LoRALinear.base.*` parameter (e.g. `attn.c_attn.weight` → `attn.c_attn.base.weight`).

---

### Data tools

| tool | signature | description |
|------|-----------|-------------|
| `TextWindowDataset` | `TextWindowDataset(tokenizer, text, seq_len)` | A `torch.utils.data.Dataset` of `(input_ids, labels)` token windows (labels are input shifted by one). |
| `collate_windows` | `collate_windows(batch)` | Collates a list of `(input_ids, labels)` into batched tensors. |
| `SAMPLE_TEXT` | `str` | Built-in corpus used when `--text` is not given. |

---

### Config: `TrainConfig`

The CLI builds a `TrainConfig` dataclass internally; you can construct one directly and
reuse the same fields the CLI exposes:

```python
from offload_lora.config import TrainConfig

cfg = TrainConfig(
    model_name_or_path="/path/to/BigModel",
    lora_targets="all-linear", r=8, alpha=16.0, dtype="bf16",
    device="cuda", gpus="auto", seq_len=512, microbatches=4, window=8,
    steps=200, lr=2e-4, output_lora="lora_adapters.pt",
)
```

All fields (with defaults): `model_name_or_path="sshleifer/tiny-gpt2"`,
`text_path=None`, `tokenizer_name=None`, `lora_targets` (default q/k/v/o + gate/up/down
proj), `r=8`, `alpha=16.0`, `lora_dropout=0.0`, `dtype="bf16"`, `device="cuda"`,
`offload_activations=True`, `seq_len=512`, `batch_size=1`, `grad_accum=1`,
`microbatches=1`, `window=8`, `epochs=1`, `steps=0`, `lr=2e-4`, `weight_decay=0.0`,
`warmup_steps=0`, `grad_clip=0.0`, `lr_decay="cosine"`, `output_lora="lora_adapters.pt"`,
`output_merged=None`, `trust_remote_code=False`, `log_every=10`.

---

## Validate on CPU (no GPU needed)

The tool runs on `--device cpu` too (offload becomes a host↔host move, still exercising
the exact same control flow). Use the local Qwen model as a smoke test:

```bash
offload-train --model ./Qwen2.5-1.5B-Instruct --device cpu \
  --seq-len 128 --steps 5 --output-lora /tmp/lora.pt
```

## Limits & notes

- This trains **LoRA adapters**, not full weights — the realistic path for a 32 GB GPU.
- Throughput trades memory for extra disk↔host↔device transfers; fast storage and a fast
  PCIe/NVLink link help. The recompute (gradient-checkpoint style) doubles the forward
  compute versus a resident model.
- Full *pretraining* of a trillion-parameter model is not what this does — that needs
  optimizer state for every parameter. This makes *fine-tuning* huge frozen models
  feasible on commodity hardware.
- To train on a GPU, pass `--device cuda`. On CPU it still runs (useful for smoke
  tests) but is slow; the 1.5B Qwen example above takes minutes per step on CPU.
- Supported architectures: Llama/Qwen/Mistral/Gemma/Phi/Cohere/DeepSeek families,
  GPT-2 (incl. `Conv1D`), OPT, BLOOM, GPT-NeoX. Extend `get_components()` /
  `get_prefixes()` in `offload_lora/model.py` for more.
