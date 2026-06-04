"""
telegram_bot.py — Telegram bot handler.

Supports both webhook mode (when TELEGRAM_WEBHOOK_URL is set) and
polling mode (run standalone: python -m api.telegram_bot).

Features:
  - /start — welcome + inline quick-action buttons
  - /reset — clear session
  - Inline buttons: quick questions, proforientation, human handoff
  - Text and voice message handling
"""

import os
import sys

import yaml
from fastapi import APIRouter, Request, Response
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

TELEGRAM_BOT_TOKEN: str = (
    os.getenv("TELEGRAM_BOT_TOKEN") or _cfg["telegram"].get("bot_token", "")
)
TELEGRAM_WEBHOOK_URL: str = (
    os.getenv("TELEGRAM_WEBHOOK_URL") or _cfg["telegram"].get("webhook_url", "")
)

from agent import guardrails
from agent.core import run_agent
from agent.session import clear_session, get_session

router = APIRouter()

_app: Application | None = None

# ── Inline keyboard definitions ────────────────────────────────────────────────

# Callback data constants
CB_ORIENT   = "cb:orient"
CB_HANDOFF  = "cb:handoff"
CB_ORT      = "cb:ort"
CB_DOCS     = "cb:docs"
CB_DISCOUNT = "cb:discount"
CB_IT       = "cb:it"
CB_MED      = "cb:med"
CB_ECO      = "cb:eco"
CB_PSY      = "cb:psy"
CB_MENU     = "cb:menu"

# Map callback → question text sent to agent
_CB_QUESTIONS: dict[str, str] = {
    CB_ORT:      "Какой минимальный балл ОРТ нужен для поступления в МУА?",
    CB_DOCS:     "Какие документы нужны для поступления в МУА?",
    CB_DISCOUNT: "Расскажи про скидки по ОРТ в МУА",
    CB_IT:       "Расскажи про факультет инженерии и информатики МУА",
    CB_MED:      "Расскажи про медицинский факультет МУА",
    CB_ECO:      "Расскажи про факультет экономики и управления МУА",
    CB_PSY:      "Расскажи про психологию и социальные науки в МУА",
}

_ORIENT_TRIGGER = (
    "Я хочу пройти тест на профориентацию. Помоги определить подходящую специальность."
)
_HANDOFF_TRIGGER = "Соедините меня с сотрудником приёмной комиссии."


def _main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🎯 Профориентация", callback_data=CB_ORIENT),
            InlineKeyboardButton("👤 Позвать сотрудника", callback_data=CB_HANDOFF),
        ],
        [
            InlineKeyboardButton("📊 Порог ОРТ", callback_data=CB_ORT),
            InlineKeyboardButton("💰 Скидки по ОРТ", callback_data=CB_DISCOUNT),
        ],
        [
            InlineKeyboardButton("📋 Документы", callback_data=CB_DOCS),
            InlineKeyboardButton("💻 IT факультет", callback_data=CB_IT),
        ],
        [
            InlineKeyboardButton("🏥 Медицина", callback_data=CB_MED),
            InlineKeyboardButton("📈 Экономика", callback_data=CB_ECO),
        ],
        [
            InlineKeyboardButton("🧠 Психология / Соц. науки", callback_data=CB_PSY),
        ],
    ])


def _back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("◀️ Главное меню", callback_data=CB_MENU),
    ]])


WELCOME_TEXT = (
    "Саламатсызбы! 👋 Добро пожаловать в приёмную комиссию *Ала-Тоо Университета*.\n\n"
    "Я могу помочь:\n"
    "• Проверить балл ОРТ и рассчитать скидку\n"
    "• Подобрать специальность (тест RIASEC)\n"
    "• Ответить на вопросы о программах и документах\n"
    "• Связать вас с сотрудником\n\n"
    "Выберите действие или напишите вопрос текстом / голосом 👇"
)


# ── App builder ────────────────────────────────────────────────────────────────

def _get_app() -> Application:
    global _app
    if _app is None:
        _app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        _app.add_handler(CommandHandler("start", _handle_start))
        _app.add_handler(CommandHandler("reset", _handle_reset))
        _app.add_handler(CommandHandler("menu", _handle_menu))
        _app.add_handler(CallbackQueryHandler(_handle_callback))
        _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
        _app.add_handler(MessageHandler(filters.VOICE, _handle_voice))
    return _app


# ── Handlers ───────────────────────────────────────────────────────────────────

async def _handle_start(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    clear_session(user_id)
    await update.message.reply_text(
        WELCOME_TEXT,
        parse_mode="Markdown",
        reply_markup=_main_keyboard(),
    )


async def _handle_menu(update: Update, context) -> None:
    await update.message.reply_text(
        "Выберите действие 👇",
        reply_markup=_main_keyboard(),
    )


async def _handle_reset(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    clear_session(user_id)
    await update.message.reply_text(
        "Сессия сброшена. Начнём сначала!",
        reply_markup=_main_keyboard(),
    )


async def _handle_callback(update: Update, context) -> None:
    query = update.callback_query
    await query.answer()  # removes the loading spinner

    user_id = str(update.effective_user.id)
    data = query.data

    # Back to menu
    if data == CB_MENU:
        await query.message.reply_text(
            "Выберите действие 👇",
            reply_markup=_main_keyboard(),
        )
        return

    # Proforientation
    if data == CB_ORIENT:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
        clear_session(user_id + ":orient")
        session = get_session(user_id + ":orient")
        reply = run_agent(_ORIENT_TRIGGER, session)
        await query.message.reply_text(reply, reply_markup=_back_keyboard())
        return

    # Human handoff
    if data == CB_HANDOFF:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
        session = get_session(user_id)
        reply = run_agent(_HANDOFF_TRIGGER, session)
        await query.message.reply_text(reply, reply_markup=_back_keyboard())
        return

    # Quick questions
    question = _CB_QUESTIONS.get(data)
    if question:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
        session = get_session(user_id)
        reply = run_agent(question, session)
        await query.message.reply_text(
            reply,
            parse_mode="Markdown",
            reply_markup=_back_keyboard(),
        )


async def _handle_text(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    text = update.message.text or ""

    guard = guardrails.check(text)
    if guard.blocked or guard.off_topic:
        await update.message.reply_text(guard.reply, reply_markup=_main_keyboard())
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    session = get_session(user_id)
    reply = run_agent(text, session)
    await update.message.reply_text(
        reply,
        parse_mode="Markdown",
        reply_markup=_back_keyboard(),
    )


async def _handle_voice(update: Update, context) -> None:
    from voice.stt import transcribe_bytes

    user_id = str(update.effective_user.id)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    voice_file = await update.message.voice.get_file()
    audio_bytes = bytes(await voice_file.download_as_bytearray())

    text = transcribe_bytes(audio_bytes, suffix=".ogg")
    if not text:
        await update.message.reply_text(
            "Не удалось распознать речь. Попробуйте ещё раз или напишите текстом.",
            reply_markup=_main_keyboard(),
        )
        return

    guard = guardrails.check(text)
    if guard.blocked or guard.off_topic:
        await update.message.reply_text(guard.reply, reply_markup=_main_keyboard())
        return

    session = get_session(user_id)
    reply = run_agent(text, session)
    await update.message.reply_text(
        f"🎙 _Распознано:_ «{text}»\n\n{reply}",
        parse_mode="Markdown",
        reply_markup=_back_keyboard(),
    )


# ── Webhook route (used when webhook_url is configured) ────────────────────────

@router.post("/telegram")
async def telegram_webhook(request: Request) -> Response:
    app = _get_app()
    data = await request.json()
    update = Update.de_json(data, app.bot)
    async with app:
        await app.process_update(update)
    return Response(status_code=200)


async def set_webhook() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_WEBHOOK_URL:
        print("[telegram] Webhook not configured — bot will not receive messages via webhook.")
        print("[telegram] To receive messages, run polling mode: python -m api.telegram_bot")
        return
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        webhook_url = f"{TELEGRAM_WEBHOOK_URL.rstrip('/')}/telegram"
        await bot.set_webhook(url=webhook_url)
        print(f"[telegram] Webhook registered: {webhook_url}")
    except Exception as e:
        print(f"[telegram] Webhook registration failed: {e} — continuing without it.")


# ── Polling mode (run standalone when no public domain) ────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _delete_webhook_and_poll():
        """Remove any stale webhook so polling can take over."""
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.delete_webhook(drop_pending_updates=True)
        print("[telegram] Webhook cleared. Starting polling…")

    asyncio.run(_delete_webhook_and_poll())

    app = _get_app()
    print("[telegram] Polling started. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)
