"""
main.py — FastAPI application entry point.

Routes:
  POST /chat          — text chat (Gradio kiosk)
  POST /voice         — voice upload (Gradio kiosk)
  POST /clear_session — reset session
  GET  /health        — health check
  POST /telegram      — Telegram webhook
  GET  /              — Gradio kiosk (mounted as ASGI sub-app)
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat_endpoint import router as chat_router
from api.telegram_bot import router as telegram_router, set_webhook

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load heavyweight models once at startup, not per-request
    print("[startup] Loading embedding model…")
    from data_ingestion.embedder import get_embedding_model
    get_embedding_model()

    print("[startup] Warming up LLM connection…")
    from agent.core import get_llm
    get_llm()

    print("[startup] Checking STT availability…")
    try:
        import voice.stt  # noqa: F401 — load-on-demand; just verify import
        print("[startup] STT ready (loads per request).")
    except Exception as e:
        print(f"[startup] STT not available: {e}")

    print("[startup] Checking TTS availability…")
    try:
        import tts.kani_tts  # noqa: F401 — load-on-demand; just verify import
        print("[startup] TTS ready (loads per request).")
    except Exception as e:
        print(f"[startup] TTS not available: {e}")

    print("[startup] Registering Telegram webhook…")
    await set_webhook()

    print("[startup] Ready.")
    yield


app = FastAPI(
    title="Ala-Too University Admissions Agent",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(telegram_router)

from ui.kiosk import build_demo
gradio_app = gr.mount_gradio_app(app, build_demo(), path="/kiosk")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=_cfg["api"]["host"],
        port=_cfg["api"]["port"],
        reload=False,
    )
