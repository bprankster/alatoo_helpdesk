"""
tts/kani_tts.py — Kyrgyz TTS wrapper for nineninesix/kani-tts-400m-ky.

Only loads if the platform is in config.yaml tts.enabled_platforms.
Telegram is excluded (text-only). Web kiosk gets audio responses.
"""

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_model = None


def is_enabled(platform: str) -> bool:
    """Return True if TTS should run for the given platform."""
    return platform in _cfg["tts"]["enabled_platforms"]


def load() -> None:
    """Load the TTS model into memory. Call once at startup."""
    global _model
    model_name = _cfg["tts"]["model"]
    print(f"[tts] Loading {model_name}…")
    from kani_tts import KaniTTS
    _model = KaniTTS(model_name)
    print("[tts] Model loaded.")


def speak(text: str, filename: str = "response.wav") -> str:
    """
    Convert text to Kyrgyz speech. Returns path to the generated wav file.

    Only call after checking is_enabled(platform).
    Raises RuntimeError if model not loaded.

    Args:
        text: text to speak (Kyrgyz or Russian)
        filename: output filename (relative to tts.output_dir)

    Returns:
        Absolute path to the generated wav file.
    """
    if _model is None:
        raise RuntimeError("[tts] Model not loaded — call tts.load() at startup")

    output_dir = Path(_cfg["tts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / filename)

    audio, _ = _model(text)
    _model.save_audio(audio, output_path)
    return output_path
