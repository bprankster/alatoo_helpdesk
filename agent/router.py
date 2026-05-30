"""
agent/router.py — KyrgyzBERT-based fast-path intent router.

Two modes controlled by config.yaml agent.use_classifier:
  true  → KyrgyzBERT classifies intent; if confidence ≥ threshold → direct tool dispatch
  false → always return 'react_agent' (pure ReAct, no classifier) — A/B baseline

This design lets the ablation study compare:
  Condition A: use_classifier=true  (KyrgyzBERT fast path + ReAct fallback)
  Condition B: use_classifier=false (pure ReAct for every query)
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_USE_CLASSIFIER: bool = _cfg["agent"].get("use_classifier", False)
_THRESHOLD: float = _cfg["classifier"]["confidence_threshold"]


def route_query(text: str) -> str:
    """
    Route a query to either a named tool or 'react_agent'.

    Returns one of:
        'ort_validator'       — ORT eligibility / discount check
        'orientation_engine'  — RIASEC career guidance survey
        'program_comparator'  — program comparison
        'human_handoff'       — escalate to human officer
        'react_agent'         — fall through to Qwen3 ReAct (default when classifier disabled)

    When use_classifier=false: always returns 'react_agent' (ablation baseline B).
    When use_classifier=true: returns tool name if confidence ≥ threshold, else 'react_agent'.
    """
    if not _USE_CLASSIFIER:
        return "react_agent"

    from classifier.predict import predict_intent
    intent, confidence = predict_intent(text)

    if confidence >= _THRESHOLD:
        return intent
    return "react_agent"
