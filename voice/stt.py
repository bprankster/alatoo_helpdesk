"""
stt.py — Speech-to-Text using faster-whisper with contextual biasing.

Handles .ogg (Telegram), .wav, .mp3, .m4a input.
Language detection is automatic so Russian and Kyrgyz both work.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CONTEXTUAL_PROMPT,
)

_model = None   # loaded lazily on first call


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"[stt] Loading faster-whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}…")
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _model


SUPPORTED_EXTENSIONS = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".webm"}


def transcribe(audio_path: str, language: Optional[str] = None) -> str:
    """
    Transcribe an audio file to text.

    Args:
        audio_path: path to audio file
        language:   ISO 639-1 code ('ru', 'ky') or None for auto-detection

    Returns:
        Transcribed text string, or empty string on failure.
    """
    path = Path(audio_path)
    if not path.exists():
        print(f"[stt] File not found: {audio_path}")
        return ""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[stt] Unsupported format: {path.suffix}")
        return ""

    model = _get_model()
    try:
        segments, info = model.transcribe(
            str(path),
            initial_prompt=WHISPER_CONTEXTUAL_PROMPT,
            language=language,          # None = auto-detect
            beam_size=5,
            vad_filter=True,            # skip silent segments
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        detected_lang = info.language
        print(f"[stt] Detected language: {detected_lang} | Transcribed: {text[:80]}")
        return text
    except Exception as e:
        print(f"[stt] Transcription failed: {e}")
        return ""


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".ogg") -> str:
    """
    Transcribe raw audio bytes (e.g. from a Telegram voice message).

    Writes to a temp file, transcribes, then cleans up.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python voice/stt.py <audio_file>")
        sys.exit(1)
    result = transcribe(sys.argv[1])
    print(f"\nTranscription:\n{result}")
