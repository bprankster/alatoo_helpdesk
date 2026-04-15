"""
chat_endpoint.py — POST /chat and POST /voice routes for the Gradio kiosk.
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


class ChatResponse(BaseModel):
    reply: str
    session_cleared: bool = False


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Handle a plain-text message from the Gradio kiosk."""
    guard = guardrails.check(req.message)
    if guard.blocked:
        return ChatResponse(reply=guard.reply)

    session = get_session(req.user_id)
    if guard.off_topic:
        return ChatResponse(reply=guard.reply)

    reply = run_agent(req.message, session)
    return ChatResponse(reply=reply)


@router.post("/voice", response_model=ChatResponse)
async def voice(
    audio: UploadFile = File(...),
    user_id: str = Form(default="web_anonymous"),
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
    return ChatResponse(reply=f"🎙️ _Распознано:_ «{text}»\n\n{reply}")


@router.post("/clear_session")
async def clear(user_id: str = "web_anonymous") -> dict:
    clear_session(user_id)
    return {"status": "ok", "message": "Сессия сброшена."}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
