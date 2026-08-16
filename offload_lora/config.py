# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class TrainConfig:
    model_name_or_path: str = "sshleifer/tiny-gpt2"
    text_path: Optional[str] = None
    tokenizer_name: Optional[str] = None

    lora_targets: Union[str, List[str]] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    r: int = 8
    alpha: float = 16.0
    lora_dropout: float = 0.0

    dtype: str = "bf16"
    device: str = "cuda"
    offload_activations: bool = True

    seq_len: int = 512
    batch_size: int = 1
    grad_accum: int = 1
    microbatches: int = 1
    window: int = 8
    epochs: int = 1
    steps: int = 0

    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_clip: float = 0.0
    lr_decay: str = "cosine"

    output_lora: str = "lora_adapters.pt"
    output_merged: Optional[str] = None
    trust_remote_code: bool = False
    log_every: int = 10
