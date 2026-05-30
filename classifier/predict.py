"""
predict.py — KyrgyzBERT intent inference.

Returns (intent_label, confidence) for a given text.
Model is loaded once and cached — safe to call per-request.
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_model = None
_tokenizer = None
_id_to_label: dict[int, str] = {
    v: k for k, v in _cfg["classifier"]["labels"].items()
}


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    finetuned_path = _cfg["classifier"]["finetuned_path"]
    base_model = _cfg["classifier"]["model_name"]

    if os.path.isdir(finetuned_path) and os.listdir(finetuned_path):
        load_path = finetuned_path
        print(f"[predict] Loading fine-tuned classifier from {finetuned_path}")
    else:
        load_path = base_model
        print(f"[predict] Fine-tuned model not found — loading base {base_model} (untrained)")

    _tokenizer = AutoTokenizer.from_pretrained(load_path)
    _model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=len(_cfg["classifier"]["labels"]),
    )
    _model.eval()
    print("[predict] Classifier loaded.")


def predict_intent(text: str) -> tuple[str, float]:
    """
    Run intent classification on text.

    Returns:
        (intent_label, confidence) where intent_label is one of:
        'ort_validator', 'orientation_engine', 'program_comparator', 'human_handoff'
    """
    _load_model()

    import torch
    import torch.nn.functional as F

    encoding = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )

    with torch.no_grad():
        outputs = _model(**encoding)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()

    confidence = float(probs.max().item())
    label_id = int(probs.argmax().item())
    intent = _id_to_label.get(label_id, "ort_validator")

    return intent, confidence
