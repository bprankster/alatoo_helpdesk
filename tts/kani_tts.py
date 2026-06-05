"""
tts/kani_tts.py — Kyrgyz TTS wrapper for nineninesix/kani-tts-400m-ky.

Only called when the platform is in config.yaml tts.enabled_platforms.
Telegram is excluded (text-only). Web kiosk gets audio responses.
Loads on demand per request; unloads after each call to free VRAM for Qwen3/BGE-m3.
"""

import os
import re
import sys
import uuid
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_MAX_TTS_CHARS = 1200  # ~20-30s of speech; model supports up to 3000 tokens

def _clean_for_tts(text: str) -> str:
    """Normalize LLM markdown output into plain speakable text."""
    # Numbered / bulleted list items → inline with comma separator
    text = re.sub(r"\n\s*[\-\*\•]\s*", ", ", text)
    text = re.sub(r"\n\s*\d+[\.\)]\s*", ", ", text)
    # Remove remaining markdown symbols
    text = re.sub(r"[*_`#\[\]()>]", "", text)
    # Collapse runs of whitespace / newlines into single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"  +", " ", text)
    # Remove artefact comma after colon (": , item" → ": item")
    text = re.sub(r":\s*,\s*", ": ", text)
    # Strip leading comma artefact
    text = re.sub(r"^[\s,]+", "", text)
    return text.strip()


def _truncate(text: str, limit: int = _MAX_TTS_CHARS) -> str:
    """Clean markdown and truncate to the nearest sentence end."""
    text = _clean_for_tts(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (".", "!", "?"):
        idx = cut.rfind(sep)
        if idx > limit // 2:
            return cut[: idx + 1].strip()
    return cut.strip()

_model = None


def is_enabled(platform: str) -> bool:
    """Return True if TTS should run for the given platform."""
    return platform in _cfg["tts"]["enabled_platforms"]


def load() -> None:
    """Load the TTS model into memory."""
    global _model
    if _model is not None:
        return
    model_name = _cfg["tts"]["model"]
    print(f"[tts] Loading {model_name}…")
    from kani_tts import KaniTTS
    _model = KaniTTS(model_name)
    print("[tts] Model loaded.")


def speak(text: str, filename: str | None = None) -> str:
    """
    Convert text to Kyrgyz speech. Returns absolute path to the generated wav file.
    Loads model on demand, unloads after each call to free VRAM.
    Only call after checking is_enabled(platform).
    """
    global _model
    try:
        if _model is None:
            load()

        output_dir = _PROJECT_ROOT / _cfg["tts"]["output_dir"].lstrip("./")
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = filename if filename else f"{uuid.uuid4().hex[:12]}.wav"
        output_path = str(output_dir / fname)

        clean = _truncate(text)
        print(f"[tts] synthesizing: {repr(clean)}")
        audio, _ = _model(clean)
        _model.save_audio(audio, output_path)
        print(f"[tts] Saved: {output_path}")
        return output_path
    finally:
        if _model is not None:
            del _model
            _model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[tts] Model unloaded.")
