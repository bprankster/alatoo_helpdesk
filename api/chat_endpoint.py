"""
chat_endpoint.py — POST /chat and POST /voice routes.

Web platform gets TTS audio in response when tts.enabled_platforms includes 'web'.
Telegram gets text only (no TTS).
"""

import os
import sys
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import guardrails
from agent.core import run_agent
from agent.session import clear_session, get_session

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "web_anonymous"
    platform: str = "web"


class ChatResponse(BaseModel):
    reply: str
    audio_path: str | None = None
    session_cleared: bool = False


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Handle a plain-text message. Returns TTS audio path for web platform."""
    guard = guardrails.check(req.message)
    if guard.blocked:
        return ChatResponse(reply=guard.reply)

    session = get_session(req.user_id)
    if guard.off_topic:
        return ChatResponse(reply=guard.reply)

    reply = run_agent(req.message, session)

    audio_path: str | None = None
    if req.platform == "web":
        try:
            import tts.kani_tts as tts_mod
            if tts_mod.is_enabled("web"):
                import hashlib
                fname = hashlib.md5(reply.encode()).hexdigest()[:12] + ".wav"
                audio_path = tts_mod.speak(reply, filename=fname)
        except Exception as e:
            print(f"[chat] TTS failed (non-fatal): {e}")

    return ChatResponse(reply=reply, audio_path=audio_path)


@router.post("/voice", response_model=ChatResponse)
async def voice(
    audio: UploadFile = File(...),
    user_id: str = Form(default="web_anonymous"),
    platform: str = Form(default="web"),
) -> ChatResponse:
    """Handle a voice file upload — transcribe then route to agent."""
    from voice.stt import transcribe

    suffix = os.path.splitext(audio.filename or "audio.ogg")[1] or ".ogg"
    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        text = transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not text:
        return ChatResponse(
            reply="Не удалось распознать речь. Пожалуйста, попробуйте ещё раз или напишите текстом."
        )

    guard = guardrails.check(text)
    if guard.blocked:
        return ChatResponse(reply=guard.reply)

    session = get_session(user_id)
    if guard.off_topic:
        return ChatResponse(reply=guard.reply)

    reply = run_agent(text, session)

    audio_path: str | None = None
    if platform == "web":
        try:
            import tts.kani_tts as tts_mod
            if tts_mod.is_enabled("web"):
                import hashlib
                fname = hashlib.md5(reply.encode()).hexdigest()[:12] + ".wav"
                audio_path = tts_mod.speak(reply, filename=fname)
        except Exception as e:
            print(f"[voice] TTS failed (non-fatal): {e}")

    return ChatResponse(
        reply=f"🎙️ _Распознано:_ «{text}»\n\n{reply}",
        audio_path=audio_path,
    )


@router.post("/clear_session")
async def clear(user_id: str = "web_anonymous") -> dict:
    clear_session(user_id)
    return {"status": "ok", "message": "Сессия сброшена."}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
