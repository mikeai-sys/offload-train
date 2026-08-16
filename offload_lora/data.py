# Powered by Mike Koci from EliseArt - https://elise-dev.web.app
import torch
from torch.utils.data import Dataset


class TextWindowDataset(Dataset):
    def __init__(self, tokenizer, text, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        self.ids = ids

    def __len__(self):
        return max(1, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.ids[idx: idx + self.seq_len + 1]
        if chunk.size(0) < self.seq_len + 1:
            pad = self.seq_len + 1 - chunk.size(0)
            chunk = torch.cat([chunk, torch.full((pad,), self.tokenizer.pad_token_id, dtype=torch.long)])
        input_ids = chunk[:-1]
        labels = chunk[1:]
        return input_ids, labels


def collate_windows(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return input_ids, labels


SAMPLE_TEXT = (
    "The offload trainer streams one transformer block onto the accelerator at a "
    "time, so only a single layer's weights and activations ever occupy VRAM. "
    "The frozen base model lives in host memory while small trainable LoRA "
    "adapters are updated with gradients. This makes it possible to fine tune "
    "very large language models on hardware that could never hold the full "
    "parameter count at once. Pipeline style computation keeps memory bounded "
    "and lets commodity GPUs participate in training trillion parameter scale "
    "networks by trading memory for additional data transfers. Streaming, "
    "checkpointing and low rank adaptation together unlock affordable training."
)
