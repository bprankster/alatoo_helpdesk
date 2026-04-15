"""
human_handoff.py — Tool D: Human_Handoff_Trigger

Sends a structured summary to the admissions officer's Telegram channel.
Summary is built ONLY from session state variables — never from LLM text —
to prevent hallucinated information reaching the officer.
"""

import os
import sys
from datetime import datetime

from langchain.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config import TELEGRAM_BOT_TOKEN, OFFICER_CHAT_ID


def _send_telegram_message(text: str) -> bool:
    """Send a message to the officer's Telegram chat. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not OFFICER_CHAT_ID:
        print("[handoff] Telegram credentials not configured — skipping send.")
        return False
    try:
        import httpx
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": OFFICER_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }
        resp = httpx.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[handoff] Failed to send Telegram message: {e}")
        return False


def human_handoff(input_text: str) -> str:
    """
    Notify the admissions officer with a structured session summary.

    The summary is built from session state variables only.
    Input: reason the student needs a human (used as context note only).
    """
    from agent.core import get_active_session

    session = get_active_session()
    if session is None:
        return (
            "Не удалось передать данные: сессия не найдена. "
            "Пожалуйста, свяжитесь с приёмной комиссией напрямую: "
            "+996 (312) 123-456 или admissions@alatoo.edu.kg"
        )

    summary = session.to_summary_dict()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build the officer notification (structured, no LLM summary)
    officer_msg = (
        f"🔔 *Запрос на помощь сотрудника*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 Время: {now}\n"
        f"👤 ID сессии: `{summary['user_id']}`\n"
        f"⏱ Длительность сессии: {summary['session_duration_min']} мин\n"
        f"💬 Сообщений обменялось: {summary['messages_exchanged']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Балл ОРТ: {summary['ort_score'] or 'не указан'}\n"
        f"🎓 Интерес к программе: {summary['ort_program'] or 'не указана'}\n"
        f"🧠 Профиль RIASEC: {summary['riasec_result'] or 'не пройден'} "
        f"(шаг {summary['riasec_step']}/{5})\n"
        f"📌 Текущая тема: {summary['current_topic'] or 'не определена'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📝 Причина обращения: {input_text[:300]}\n"
    )

    sent = _send_telegram_message(officer_msg)

    if sent:
        return (
            "✅ Ваш запрос передан сотруднику приёмной комиссии. "
            "Они свяжутся с вами в ближайшее время.\n\n"
            "Если вопрос срочный, вы также можете обратиться напрямую:\n"
            "📞 +996 (312) 123-456\n"
            "📧 admissions@alatoo.edu.kg\n"
            "🏢 Корпус A, каб. 101, пн–пт 9:00–17:00"
        )
    else:
        return (
            "⚠️ Не удалось автоматически уведомить сотрудника. "
            "Пожалуйста, обратитесь в приёмную комиссию напрямую:\n"
            "📞 +996 (312) 123-456\n"
            "📧 admissions@alatoo.edu.kg\n"
            "🏢 Корпус A, каб. 101, пн–пт 9:00–17:00"
        )


human_handoff_tool = Tool(
    name="Human_Handoff_Trigger",
    func=human_handoff,
    description=(
        "Use this tool when: the student explicitly asks to speak to a human officer, "
        "the student's question cannot be answered from the database, "
        "or the situation requires human judgement (e.g. special circumstances, complaints). "
        "Input: a brief description of why the student needs human assistance."
    ),
)
