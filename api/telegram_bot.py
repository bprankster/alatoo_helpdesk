"""
telegram_bot.py — Telegram webhook handler via python-telegram-bot.

Each Telegram user gets an isolated session keyed by their Telegram user ID.
Handles both text messages and voice notes.
"""

import os
import sys

from fastapi import APIRouter, Request, Response
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, CommandHandler, filters

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_URL
from agent import guardrails
from agent.core import run_agent
from agent.session import clear_session, get_session

router = APIRouter()

# Build the application once (not started with polling — webhook mode)
_app: Application | None = None


def _get_app() -> Application:
    global _app
    if _app is None:
        _app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        _app.add_handler(CommandHandler("start", _handle_start))
        _app.add_handler(CommandHandler("reset", _handle_reset))
        _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
        _app.add_handler(MessageHandler(filters.VOICE, _handle_voice))
    return _app


# ── Handlers ───────────────────────────────────────────────────────────────────

async def _handle_start(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    clear_session(user_id)
    await update.message.reply_text(
        "Саламатсызбы! Добро пожаловать в приёмную комиссию Ала-Тоо Университета.\n\n"
        "Я могу помочь вам:\n"
        "• Проверить ваш балл ОРТ для поступления\n"
        "• Определить подходящую специальность (опрос RIASEC)\n"
        "• Сравнить программы\n"
        "• Связать вас с сотрудником\n\n"
        "Задайте ваш вопрос текстом или голосовым сообщением.\n\n"
        "Жардам алуу үчүн суроонузду жазыңыз же үн жазуу жиберіңіз."
    )


async def _handle_reset(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    clear_session(user_id)
    await update.message.reply_text("Сессия сброшена. Начнём сначала!")


async def _handle_text(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    text = update.message.text or ""

    guard = guardrails.check(text)
    if guard.blocked or guard.off_topic:
        await update.message.reply_text(guard.reply)
        return

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    session = get_session(user_id)
    reply = run_agent(text, session)
    await update.message.reply_text(reply, parse_mode="Markdown")


async def _handle_voice(update: Update, context) -> None:
    from voice.stt import transcribe_bytes

    user_id = str(update.effective_user.id)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    # Download voice note (.ogg format from Telegram)
    voice_file = await update.message.voice.get_file()
    audio_bytes = bytes(await voice_file.download_as_bytearray())

    text = transcribe_bytes(audio_bytes, suffix=".ogg")
    if not text:
        await update.message.reply_text(
            "Не удалось распознать речь. Пожалуйста, попробуйте ещё раз или напишите текстом."
        )
        return

    guard = guardrails.check(text)
    if guard.blocked or guard.off_topic:
        await update.message.reply_text(guard.reply)
        return

    session = get_session(user_id)
    reply = run_agent(text, session)
    await update.message.reply_text(
        f"🎙️ _Распознано:_ «{text}»\n\n{reply}", parse_mode="Markdown"
    )


# ── Webhook route ──────────────────────────────────────────────────────────────

@router.post("/telegram")
async def telegram_webhook(request: Request) -> Response:
    """Receive Telegram updates and dispatch them to handlers."""
    app = _get_app()
    data = await request.json()
    update = Update.de_json(data, app.bot)
    async with app:
        await app.process_update(update)
    return Response(status_code=200)


async def set_webhook() -> None:
    """Register the webhook URL with Telegram on startup."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_WEBHOOK_URL:
        print("[telegram] Webhook not configured — skipping registration.")
        return
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    webhook_url = f"{TELEGRAM_WEBHOOK_URL.rstrip('/')}/telegram"
    await bot.set_webhook(url=webhook_url)
    print(f"[telegram] Webhook registered: {webhook_url}")
