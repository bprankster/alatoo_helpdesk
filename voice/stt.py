"""
voice/stt.py — kyrgyz-whisper-medium with official custom tokenizer.
"""

import os, sys, tempfile, yaml, torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = "/home/cs/.cache/huggingface/hub/models--nineninesix--kyrgyz-whisper-medium/snapshots/bb00894d615500bc76aeb6b042d135555dfec125"

_pipe = None

def load():
    global _pipe
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline,
        WhisperFeatureExtractor,
        AutoTokenizer,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[stt] Loading kyrgyz-whisper-medium on {device}…")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH)

    # Critical: trust_remote_code=True loads the custom <|ky|> Kyrgyz token
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        language="kyrgyz",
        task="transcribe",
    )

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("[stt] kyrgyz-whisper-medium loaded with Kyrgyz tokenizer.")


def _get_pipe():
    global _pipe
    if _pipe is None:
        load()
    return _pipe


def _is_valid_audio(audio_path: str) -> bool:
    """Reject silent or too-short audio to prevent Whisper hallucination."""
    try:
        import wave, struct
        with wave.open(str(audio_path), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            if len(frames) < 8000:  # less than ~0.5s at 8kHz
                print("[stt] Audio too short, skipping")
                return False
            # Check if audio is silence (all near-zero samples)
            samples = struct.unpack(f'{len(frames)//2}h', frames)
            avg_amplitude = sum(abs(s) for s in samples) / len(samples)
            if avg_amplitude < 50:
                print(f"[stt] Audio too quiet (amplitude={avg_amplitude:.1f}), skipping")
                return False
        return True
    except Exception:
        return True  # if we can't check, proceed anyway


def transcribe(audio_path: str) -> str:
    if not _is_valid_audio(audio_path):
        return ""
    pipe = _get_pipe()
    try:
        result = pipe(
            str(audio_path),
            generate_kwargs={
                "task": "transcribe",
                "language": "ky",          # force Kyrgyz — auto-detect mis-fires on KG/RU mix
                "temperature": 0.0,        # deterministic decoding, suppresses hallucinations
                "condition_on_prev_tokens": False,  # prevents hallucination loops between segments
            },
        )
        text = result["text"].strip()
        print(f"[stt] Transcribed: {text[:80]}")
        return text
    except torch.cuda.OutOfMemoryError:
        print("[stt] CUDA OOM — clearing cache and retrying on CPU")
        global _pipe
        del _pipe
        _pipe = None
        torch.cuda.empty_cache()
        # Reload on CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        load()
        return transcribe(audio_path)
    except Exception as e:
        print(f"[stt] Error: {e}")
        return ""


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".ogg") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
