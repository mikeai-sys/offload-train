# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
import torch
import torch.nn as nn


_LLAMA_STYLE = {
    "llama", "mistral", "qwen2", "qwen2_moe", "qwen3",
    "gemma", "gemma2", "phi", "phi3", "cohere", "falcon",
    "baichuan", "internlm2", "deepseek", "deepseek_v3", "stablelm",
}


def _rotary_of(model):
    m = getattr(model, "model", None)
    if m is not None and hasattr(m, "rotary_emb"):
        return m.rotary_emb
    return None


def get_components(model):
    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", "").lower() if cfg is not None else ""

    m = getattr(model, "model", None)
    if m is not None and hasattr(m, "embed_tokens") and hasattr(m, "layers"):
        norm = getattr(m, "norm", None)
        if hasattr(model, "lm_head"):
            return m.embed_tokens, list(m.layers), norm, model.lm_head, _rotary_of(model)

    if model_type in _LLAMA_STYLE:
        m = model.model
        return m.embed_tokens, list(m.layers), m.norm, model.lm_head, _rotary_of(model)

    if model_type == "gpt2":
        tr = model.transformer
        return tr.wte, list(tr.h), tr.ln_f, model.lm_head, None

    if model_type in ("bloom", "gpt_neox", "llama4"):
        m = model.model
        return m.embed_tokens, list(m.layers), m.norm, model.lm_head, _rotary_of(model)

    if model_type == "t5":
        return model.shared, list(model.encoder.block), None, model.lm_head, None

    if model_type == "opt":
        return model.model.decoder.embed_tokens, list(model.model.decoder.layers), \
            model.model.decoder.final_layer_norm, model.lm_head, None

    for attempt in ("model.embed_tokens", "model.embeddings", "transformer.wte",
                    "model.decoder.embed_tokens"):
        try:
            embed = eval("model." + attempt)
        except Exception:
            embed = None
        if embed is not None:
            break
    raise ValueError(
        f"Unsupported model_type={model_type!r}. Please open an issue or extend "
        "get_components() in offload_lora/model.py for this architecture."
    )


def load_model_cpu(model_name_or_path, dtype="bf16", trust_remote_code=False):
    from transformers import AutoModelForCausalLM, AutoConfig

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def build_skeleton(model_name_or_path, dtype="bf16", trust_remote_code=False):
    from transformers import AutoModelForCausalLM, AutoConfig

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model


_PREFIXES = {
    "llama": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "mistral": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "qwen2": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "qwen2_moe": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "qwen3": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "gemma": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "gemma2": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "phi": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "phi3": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "cohere": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "falcon": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "baichuan": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "internlm2": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "deepseek": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "stablelm": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "bloom": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "gpt_neox": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "llama4": ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head."),
    "gpt2": ("transformer.h.{}.", "transformer.wte.", "transformer.ln_f.", "lm_head."),
    "opt": ("model.model.decoder.layers.{}.", "model.model.decoder.embed_tokens.",
            "model.model.decoder.final_layer_norm.", "lm_head."),
}


def get_prefixes(model):
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", "").lower() if cfg is not None else ""
    if mt in _PREFIXES:
        return _PREFIXES[mt]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return ("model.layers.{}.", "model.embed_tokens.", "model.norm.", "lm_head.")
    raise ValueError(f"Unsupported model_type={mt!r} for streaming.")


def load_tokenizer(model_name_or_path, trust_remote_code=False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
