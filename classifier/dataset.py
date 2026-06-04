"""
dataset.py — Dataset loader for KyrgyzBERT intent classification fine-tuning.
"""

import json
import os
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class IntentDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        with open(data_path, encoding="utf-8") as f:
            all_items = json.load(f)

        # Must shuffle before split — data is ordered by class,
        # so without this the val set only sees the last class.
        random.seed(seed)
        random.shuffle(all_items)

        split_idx = int(len(all_items) * (1 - val_ratio))
        if split == "train":
            items = all_items[:split_idx]
        else:
            items = all_items[split_idx:]

        self.texts = [item["text"] for item in items]
        self.labels = [item["label_id"] for item in items]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
