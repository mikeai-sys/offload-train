# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
import argparse
import sys

import torch

from .config import TrainConfig
from .data import TextWindowDataset, collate_windows, SAMPLE_TEXT
from .lora import count_params
from .model import load_model_cpu, load_tokenizer
from .pipeline import ElasticPipelineTrainer


def _parse_devices(spec):
    if spec in (None, "", "auto"):
        return ["cuda" if torch.cuda.is_available() else "cpu"]
    if "," in spec:
        return [d.strip() for d in spec.split(",") if d.strip()]
    try:
        n = int(spec)
        return [f"cuda:{i}" for i in range(n)] if torch.cuda.is_available() else ["cpu"]
    except ValueError:
        return [spec]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="offload-train",
        description="Fine-tune huge models on weak GPUs via pipeline-parallel "
                    "CPU/GPU offload + LoRA (stream one layer at a time).",
    )
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--text", default=None, help="Path to a .txt corpus (defaults to built-in sample)")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--lora-targets", nargs="*", default=None,
                   help="Module names to adapt, or 'all-linear'")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpus", default="auto",
                   help="Devices: 'auto', a count like '2', or a comma list "
                        "like 'cuda:0,cuda:1'. Layers are split across them.")
    p.add_argument("--no-offload-activations", action="store_true")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--microbatches", type=int, default=1,
                   help="Microbatches per optimizer step (pipeline GPipe).")
    p.add_argument("--window", type=int, default=8,
                   help="Max resident blocks kept streamed per stage (LRU).")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--lr-decay", default="cosine", choices=["cosine", "linear", "none"])
    p.add_argument("--output-lora", default="lora_adapters.pt")
    p.add_argument("--output-merged", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(
        model_name_or_path=args.model,
        text_path=args.text,
        tokenizer_name=args.tokenizer,
        lora_targets=args.lora_targets,
        r=args.r,
        alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        dtype=args.dtype,
        device=args.device,
        offload_activations=not args.no_offload_activations,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        microbatches=args.microbatches,
        window=args.window,
        epochs=args.epochs,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        lr_decay=args.lr_decay,
        output_lora=args.output_lora,
        output_merged=args.output_merged,
        trust_remote_code=args.trust_remote_code,
        log_every=args.log_every,
    )
    if isinstance(cfg.lora_targets, str):
        if cfg.lora_targets != "all-linear":
            cfg.lora_targets = cfg.lora_targets.split(",")
    elif cfg.lora_targets is None:
        cfg.lora_targets = "all-linear"

    print(f"[offload-train] loading model from {cfg.model_name_or_path} ...", flush=True)
    import os
    base = cfg.model_name_or_path
    has_local = os.path.isdir(base) and (
        os.path.exists(os.path.join(base, "model.safetensors"))
        or os.path.exists(os.path.join(base, "model.safetensors.index.json"))
    )
    if has_local:
        from .model import build_skeleton
        model = build_skeleton(base, cfg.dtype, cfg.trust_remote_code)
        stream = True
        print(f"[offload-train] streaming weights from disk (no full load)", flush=True)
    else:
        from .model import load_model_cpu
        model = load_model_cpu(base, cfg.dtype, cfg.trust_remote_code)
        stream = False
    tok = load_tokenizer(cfg.tokenizer_name or base, cfg.trust_remote_code)

    total, trainable = count_params(model)
    print(f"[offload-train] base params={total/1e9:.2f}B  trainable(LoRA)={trainable/1e6:.2f}M", flush=True)

    devices = _parse_devices(args.gpus)
    microbatches = max(1, cfg.microbatches)
    print(f"[offload-train] pipeline across devices={devices} "
          f"microbatches/step={microbatches}", flush=True)

    trainer = ElasticPipelineTrainer(
        model,
        lora_targets=cfg.lora_targets,
        r=cfg.r,
        alpha=cfg.alpha,
        lr=cfg.lr,
        dropout=cfg.lora_dropout,
        weight_decay=cfg.weight_decay,
        devices=devices,
        offload_activations=cfg.offload_activations,
        grad_clip=cfg.grad_clip,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.steps or (cfg.epochs * 1000),
        lr_decay=cfg.lr_decay,
        stream=stream,
        model_path=base if stream else None,
        window=cfg.window,
    )
    print(f"[offload-train] LoRA injected into {len(trainer.replaced or [])} linear layers "
          f"across {trainer.n_stages} stage(s)", flush=True)

    text = SAMPLE_TEXT
    if cfg.text_path:
        with open(cfg.text_path, "r", encoding="utf-8") as f:
            text = f.read()
    ds = TextWindowDataset(tok, text, cfg.seq_len)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_windows
    )

    global_step = 0
    pending = []
    for epoch in range(cfg.epochs):
        for batch in loader:
            pending.append(batch)
            if len(pending) < microbatches:
                continue
            loss = trainer.step(pending)
            pending = []
            global_step += 1
            if global_step % cfg.log_every == 0:
                print(f"[offload-train] step={global_step} loss={loss:.4f} lr={trainer._lr():.2e}",
                      flush=True)
            if cfg.steps and global_step >= cfg.steps:
                break
        if pending:
            loss = trainer.step(pending)
            pending = []
            global_step += 1
            if cfg.steps and global_step >= cfg.steps:
                break
        if cfg.steps and global_step >= cfg.steps:
            break

    trainer.save_lora(cfg.output_lora)
    print(f"[offload-train] saved LoRA adapters -> {cfg.output_lora}", flush=True)
    if cfg.output_merged:
        trainer.merge_and_save(cfg.output_merged)
        print(f"[offload-train] saved merged model -> {cfg.output_merged}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
