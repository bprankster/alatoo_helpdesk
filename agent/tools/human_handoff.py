"""
human_handoff.py — Tool D: Human_Handoff_Trigger

Sends a structured summary to the admissions officer's Telegram channel.
Summary is built ONLY from session state variables — never from LLM text —
to prevent hallucinated information reaching the officer.
"""

import os
import sys
from datetime import datetime

import yaml
from langchain_core.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

# Telegram credentials: prefer env vars (secrets), fall back to config.yaml
_TELEGRAM_BOT_TOKEN: str = (
    os.getenv("TELEGRAM_BOT_TOKEN") or _cfg["telegram"].get("bot_token", "")
)
_OFFICER_CHAT_ID: str = (
    os.getenv("OFFICER_CHAT_ID") or _cfg["telegram"].get("officer_chat_id", "")
)

# Real MUA admissions contact info
CONTACT_INFO = (
    "📞 +996 555 820 000 (WhatsApp)\n"
    "📧 admission@alatoo.edu.kg\n"
    "🏢 ул. Анкара (Горький) 1/10, мкр. «Тунгуч», г. Бишкек (D-блок, 1 этаж)\n"
    "🌐 https://2020.edu.gov.kg/vuz"
)


def _is_real_token(token: str) -> bool:
    """A real Telegram bot token always contains a colon, e.g. '123456:ABC-...'"""
    return bool(token) and ":" in token and "placeholder" not in token.lower() and "your_" not in token.lower()

def _send_telegram_message(text: str) -> bool:
    if not _is_real_token(_TELEGRAM_BOT_TOKEN) or not _OFFICER_CHAT_ID:
        print("[handoff] Telegram credentials not configured — skipping send.")
        return False
    try:
        import httpx
        url = f"https://api.telegram.org/bot{_TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": _OFFICER_CHAT_ID,
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

    Use when: student asks for a human, question is unanswerable, or special circumstances.
    Do NOT use for standard ORT/program queries that tools can handle.
    Input: brief description of why student needs human assistance.
    """
    from agent.core import get_active_session

    session = get_active_session()
    if session is None:
        return (
            "Не удалось передать данные: сессия не найдена. "
            "Пожалуйста, свяжитесь с приёмной комиссией напрямую:\n"
            + CONTACT_INFO
        )

    summary = session.to_summary_dict()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

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
            "✅ Ваш запрос передан сотруднику приёмной комиссии МУА. "
            "Они свяжутся с вами в ближайшее время.\n\n"
            "Если вопрос срочный, вы также можете обратиться напрямую:\n"
            + CONTACT_INFO
        )
    else:
        return (
            "⚠️ Не удалось автоматически уведомить сотрудника. "
            "Пожалуйста, обратитесь в приёмную комиссию напрямую:\n"
            + CONTACT_INFO
        )


human_handoff_tool = Tool(
    name="Human_Handoff_Trigger",
    func=human_handoff,
    description=(
        "Use this tool when: the student explicitly asks to speak to a human officer, "
        "the student's question cannot be answered from the database, "
        "or the situation requires human judgement (e.g. special circumstances, complaints, "
        "questions about enrollment status). "
        "Input: a brief description of why the student needs human assistance."
    ),
)
