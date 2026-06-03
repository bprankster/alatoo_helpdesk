"""
voice/stt.py — Speech-to-text using kyrgyz-whisper-medium via transformers pipeline.

Single model handles Kyrgyz + Russian + English natively.
No language routing needed — Whisper auto-detects language.
Loads on demand per request; unloads after each call to keep VRAM free for Qwen3/BGE-m3.
"""

import os
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

MODEL_PATH = (
    "/home/cs/.cache/huggingface/hub/"
    "models--nineninesix--kyrgyz-whisper-medium/snapshots/"
    "bb00894d615500bc76aeb6b042d135555dfec125"
)

_pipeline = None


def _load_pipeline(device: int = 0):
    """Load transformers ASR pipeline. device=0 for GPU, device=-1 for CPU."""
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition",
        model=MODEL_PATH,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )


def load() -> None:
    """Load STT pipeline on GPU. Falls back to CPU on CUDA OOM."""
    global _pipeline
    if _pipeline is not None:
        return
    print("[stt] Loading kyrgyz-whisper-medium…")
    try:
        _pipeline = _load_pipeline(device=0)
        print("[stt] Loaded on GPU.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[stt] CUDA OOM at load — using CPU: {e}")
            _pipeline = _load_pipeline(device=-1)
            print("[stt] Loaded on CPU.")
        else:
            raise


def _unload() -> None:
    global _pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[stt] Unloaded.")


def _run_inference(pipe, audio_path: str) -> str:
    result = pipe(
        str(audio_path),
        generate_kwargs={"task": "transcribe"},
        return_timestamps=False,
    )
    text = result.get("text", "").strip()
    print(f"[stt] Transcribed: {text[:80]}")
    return text


def transcribe(audio_path: str) -> str:
    """
    Transcribe audio file. Auto-detects Kyrgyz, Russian, or English.
    Loads model on demand, unloads after each call to free VRAM.

    Args:
        audio_path: path to audio file (wav, ogg, mp3, m4a, etc.)

    Returns:
        Transcribed text, or empty string on failure.
    """
    try:
        load()
        try:
            return _run_inference(_pipeline, audio_path)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[stt] CUDA OOM during inference — retrying on CPU")
                _unload()
                cpu_pipe = _load_pipeline(device=-1)
                try:
                    return _run_inference(cpu_pipe, audio_path)
                finally:
                    del cpu_pipe
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            raise
    except Exception as e:
        print(f"[stt] Transcription error: {e}")
        return ""
    finally:
        _unload()


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".ogg") -> str:
    """Convenience wrapper: write bytes to temp file, transcribe, clean up."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
