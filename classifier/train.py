"""
train.py — Fine-tune KyrgyzBERT for 4-class intent classification.

Classes:
    0: ort_validator       — ORT score / eligibility / discount questions
    1: orientation_engine  — career guidance / major selection uncertainty
    2: program_comparator  — program comparison requests
    3: human_handoff       — requests for human officer

Expected training time on RTX 4080: ~40 minutes.
Target: accuracy > 85%, f1_macro > 0.83

Usage:
    python classifier/train.py
    python classifier/train.py --data classifier/training_data.json --epochs 5
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)


def train(data_path: str, epochs: int = 10, batch_size: int = 16):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score

    from classifier.dataset import IntentDataset

    model_name = _cfg["classifier"]["model_name"]
    output_dir = _cfg["classifier"]["finetuned_path"]
    labels = _cfg["classifier"]["labels"]
    num_labels = len(labels)

    print(f"[train] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[train] Loading model: {model_name} ({num_labels} classes)")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    model.to(device)

    train_ds = IntentDataset(data_path, tokenizer, split="train")
    val_ds = IntentDataset(data_path, tokenizer, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )

    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(batch["labels"].tolist())

        acc = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        per_class = f1_score(all_true, all_preds, average=None, zero_division=0)
        label_names = list(_cfg["classifier"]["labels"].keys())
        class_str = " | ".join(f"{label_names[i]}={per_class[i]:.2f}" for i in range(len(per_class)))
        print(f"[train] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | acc={acc:.3f} | f1_macro={f1:.3f}")
        print(f"         {class_str}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[train] ✓ Saved best model (f1={f1:.3f}) → {output_dir}")

    print(f"[train] Training complete. Best f1_macro: {best_f1:.3f}")
    if best_f1 < 0.83:
        print("[train] ⚠️  f1_macro below target (0.83). Consider generating more training data.")
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "training_data.json"),
        help="Path to training_data.json",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch_size)
