"""
guardrails.py — Input filter applied BEFORE any text reaches the Grok API.

Two checks:
  1. Prompt injection detection → block immediately
  2. Domain bounding → warn agent if query is off-topic
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import INJECTION_KEYWORDS, DOMAIN_KEYWORDS


INJECTION_BLOCK_REPLY = (
    "Извините, я не могу обработать этот запрос. "
    "Я здесь, чтобы помочь с вопросами поступления в Ала-Тоо Университет."
    "\n\nКечиресиз, мен бул суранычты аткара албайм. "
    "Мен Ала-Тоо Университетине кабылуу боюнча суроолорго жардам берем."
)

OFF_TOPIC_REPLY = (
    "Я специализируюсь только на вопросах поступления, учебных программах "
    "и профессиональной ориентации в Ала-Тоо Университете. "
    "Пожалуйста, задайте вопрос по этим темам.\n\n"
    "Мен Ала-Тоо Университетинин кабылуу, окуу программалары жана "
    "кесиптик багыттоо суроолору боюнча гана жардам бере алам."
)


def is_injection(text: str) -> bool:
    """Return True if the input contains prompt-injection patterns."""
    lower = text.lower()
    return any(kw in lower for kw in INJECTION_KEYWORDS)


def is_on_topic(text: str) -> bool:
    """Return True if the input is plausibly related to admissions/orientation."""
    lower = text.lower()
    return any(kw in lower for kw in DOMAIN_KEYWORDS)


class GuardrailResult:
    __slots__ = ("blocked", "off_topic", "reply")

    def __init__(self, blocked: bool, off_topic: bool, reply: str | None):
        self.blocked = blocked
        self.off_topic = off_topic
        self.reply = reply


def check(text: str) -> GuardrailResult:
    """
    Run both checks and return a GuardrailResult.

    If blocked=True, reply must be sent to the user immediately without calling the agent.
    If off_topic=True, the agent is still called but with a domain-bounding reminder.
    """
    if is_injection(text):
        return GuardrailResult(blocked=True, off_topic=False, reply=INJECTION_BLOCK_REPLY)
    if not is_on_topic(text):
        return GuardrailResult(blocked=False, off_topic=True, reply=OFF_TOPIC_REPLY)
    return GuardrailResult(blocked=False, off_topic=False, reply=None)
