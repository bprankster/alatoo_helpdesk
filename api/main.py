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

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat_endpoint import router as chat_router
from api.telegram_bot import router as telegram_router, set_webhook
from config import API_HOST, API_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register Telegram webhook on startup
    await set_webhook()
    yield


app = FastAPI(
    title="Ala-Too University Admissions Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(chat_router)
app.include_router(telegram_router)

# Mount Gradio kiosk UI at /kiosk
from ui.kiosk import build_demo
gradio_app = gr.mount_gradio_app(app, build_demo(), path="/kiosk")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
