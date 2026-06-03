"""
voice/stt.py — Speech-to-text using kyrgyz-whisper-medium.

Single model handles Kyrgyz + Russian + English natively.
No language routing needed — auto-detection via Whisper.
"""

import os
import sys
import tempfile
from pathlib import Path

import yaml
from faster_whisper import WhisperModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

# Hardcoded path to the downloaded model — bypasses HF cache lookup
MODEL_PATH = "/home/cs/.cache/huggingface/kyrgyz-whisper-ct2"

_model = None


def load():
    """Load the model at startup. Called from api/main.py lifespan."""
    global _model
    if _model is not None:
        return
    device = cfg["stt"]["device"]
    compute_type = cfg["stt"]["compute_type"]
    print(f"[stt] Loading nineninesix/kyrgyz-whisper-medium on {device}…")
    _model = WhisperModel(
        MODEL_PATH,
        device=device,
        compute_type=compute_type,
    )
    print("[stt] Model loaded.")


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        load()
    return _model


def transcribe(audio_path: str) -> str:
    """
    Transcribe audio file. Auto-detects Kyrgyz, Russian, or English.

    Args:
        audio_path: path to audio file (wav, ogg, mp3, m4a, etc.)

    Returns:
        Transcribed text, or empty string on failure.
    """
    model = _get_model()
    try:
        segments, info = model.transcribe(
            str(audio_path),
            initial_prompt=cfg["stt"]["contextual_bias"],
            language=None,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        print(
            f"[stt] Detected: {info.language} "
            f"(prob={info.language_probability:.2f}) | {text[:80]}"
        )
        return text
    except Exception as e:
        print(f"[stt] Transcription error: {e}")
        return ""


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".ogg") -> str:
    """Convenience wrapper: write bytes to temp file, transcribe, clean up."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)