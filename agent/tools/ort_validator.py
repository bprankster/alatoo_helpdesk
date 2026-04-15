"""
ort_validator.py — Tool A: ORT_Validator

Extracts a student's ORT score from the input string and compares it against
the minimum thresholds stored in ort_thresholds.json.
Pure Python math — no LLM hallucination risk.
"""

import json
import re
import os
import sys

from langchain.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config import ORT_THRESHOLDS_FILE


def _load_thresholds() -> dict:
    if not os.path.exists(ORT_THRESHOLDS_FILE):
        return {}
    with open(ORT_THRESHOLDS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("programs", {})


SCORE_PATTERN = re.compile(r"\b(\d{2,3})\s*(?:баллов|балл|points|очков|очко)?\b")
PROGRAM_ALIASES: dict[str, str] = {
    "cs": "Computer Science",
    "информатика": "Computer Science",
    "computer science": "Computer Science",
    "программирование": "Software Engineering",
    "software": "Software Engineering",
    "it": "Information Technology",
    "информационные технологии": "Information Technology",
    "экономика": "Economics",
    "economics": "Economics",
    "финансы": "Finance",
    "finance": "Finance",
    "бухучет": "Accounting",
    "accounting": "Accounting",
    "бизнес": "Business Administration",
    "business": "Business Administration",
    "менеджмент": "Management",
    "management": "Management",
    "право": "Law",
    "юриспруденция": "Law",
    "law": "Law",
    "международные отношения": "International Relations",
    "педагогика": "Education (Primary)",
    "education": "Education (Primary)",
    "психология": "Psychology",
    "psychology": "Psychology",
    "дизайн": "Design",
    "design": "Design",
    "архитектура": "Architecture",
    "architecture": "Architecture",
    "строительство": "Civil Engineering",
    "civil engineering": "Civil Engineering",
}


def _extract_score(text: str) -> int | None:
    matches = SCORE_PATTERN.findall(text)
    candidates = [int(m) for m in matches if 80 <= int(m) <= 260]
    return candidates[0] if candidates else None


def _extract_program(text: str) -> str | None:
    lower = text.lower()
    for alias, canonical in PROGRAM_ALIASES.items():
        if alias in lower:
            return canonical
    return None


def ort_validator(input_text: str) -> str:
    """
    Check ORT eligibility.

    Input examples:
      "Мой ОРТ 145 балл. Могу ли я поступить на Computer Science?"
      "У меня 132 балла, хочу на юридический"
    """
    thresholds = _load_thresholds()
    score = _extract_score(input_text)
    program = _extract_program(input_text)

    if score is None:
        return (
            "Пожалуйста, укажите ваш балл ОРТ (например: «у меня 145 баллов»). "
            "Тогда я смогу проверить ваши шансы на поступление."
        )

    if program is None:
        # Show all thresholds
        lines = [f"  • {prog}: от {info['min_ort_score']} (платно от {info['paid_min_score']})"
                 for prog, info in thresholds.items()]
        return (
            f"Ваш балл ОРТ: **{score}**.\n\n"
            "Укажите, на какую специальность вы хотите поступить, "
            "и я сравню ваш балл с проходным.\n\n"
            "Минимальные баллы по специальностям:\n" + "\n".join(lines)
        )

    info = thresholds.get(program)
    if info is None:
        return (
            f"Специальность «{program}» не найдена в базе данных. "
            "Пожалуйста, уточните название или свяжитесь с приёмной комиссией."
        )

    budget_min = info["min_ort_score"]
    paid_min = info["paid_min_score"]
    seats = info.get("budget_seats", "?")

    if score >= budget_min:
        status = (
            f"✅ Ваш балл ОРТ ({score}) **превышает** бюджетный порог "
            f"для специальности «{program}» ({budget_min} баллов). "
            f"Количество бюджетных мест: {seats}. "
            "Вы можете подать документы на бюджет. "
            "Окончательное решение принимает приёмная комиссия."
        )
    elif score >= paid_min:
        status = (
            f"⚠️ Ваш балл ОРТ ({score}) ниже бюджетного порога ({budget_min}), "
            f"но выше минимума для платного обучения ({paid_min}) "
            f"по специальности «{program}». "
            "Вы можете поступить на платной основе. "
            "Окончательное решение принимает приёмная комиссия."
        )
    else:
        status = (
            f"❌ К сожалению, ваш балл ОРТ ({score}) ниже минимума "
            f"для платного обучения ({paid_min}) по специальности «{program}». "
            "Рекомендуем рассмотреть другие специальности или пересдачу ОРТ. "
            "Вы можете также обратиться в приёмную комиссию за индивидуальной консультацией."
        )

    return status


ort_validator_tool = Tool(
    name="ORT_Validator",
    func=ort_validator,
    description=(
        "Use this tool when a student mentions their ORT exam score and wants to know "
        "if they are eligible for a specific program. "
        "Input should include the student's score and desired program name. "
        "Example: 'у меня 145 баллов, хочу на Computer Science'"
    ),
)
