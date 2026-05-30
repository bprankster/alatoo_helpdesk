"""
ort_validator.py — Tool A: ORT_Validator

Checks a student's ORT score against real MUA 2024-2025 admissions rules.
No budget places — private university. Discounts based on ORT score.
Pure Python logic — no LLM, no hallucination risk.
"""

import json
import os
import re
import sys

import yaml
from langchain_core.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_ORT_FILE = _cfg["data"]["ort_thresholds"]

SCORE_PATTERN = re.compile(r"\b(\d{2,3})\s*(?:баллов|балл|points|упай|очков|очко)?\b")

FACULTY_ALIASES: dict[str, str] = {
    "инженер": "Факультет инженерии и информатики",
    "информатик": "Факультет инженерии и информатики",
    "it": "Факультет инженерии и информатики",
    "cs": "Факультет инженерии и информатики",
    "computer": "Факультет инженерии и информатики",
    "программирован": "Факультет инженерии и информатики",
    "data science": "Факультет инженерии и информатики",
    "кибербезопасн": "Факультет инженерии и информатики",
    "искусственный интеллект": "Факультет инженерии и информатики",
    "робот": "Факультет инженерии и информатики",
    "медицин": "Медицинский факультет",
    "лечебн": "Медицинский факультет",
    "врач": "Медицинский факультет",
    "экономик": "Факультет экономики и управления",
    "менеджмент": "Факультет экономики и управления",
    "юрид": "Факультет экономики и управления",
    "право": "Факультет экономики и управления",
    "финанс": "Факультет экономики и управления",
    "туризм": "Факультет экономики и управления",
    "гостеприимств": "Факультет экономики и управления",
    "лингвистик": "Факультет гуманитарных наук",
    "перевод": "Факультет гуманитарных наук",
    "педагогик": "Факультет гуманитарных наук",
    "филолог": "Факультет гуманитарных наук",
    "психолог": "Факультет социальных наук",
    "журналист": "Факультет социальных наук",
    "международн отнош": "Факультет социальных наук",
    "дипломат": "Факультет социальных наук",
}


def _load_thresholds() -> dict:
    if not os.path.exists(_ORT_FILE):
        return {}
    with open(_ORT_FILE, encoding="utf-8") as f:
        return json.load(f)


def _extract_score(text: str) -> int | None:
    matches = SCORE_PATTERN.findall(text)
    candidates = [int(m) for m in matches if 80 <= int(m) <= 260]
    return candidates[0] if candidates else None


def _extract_faculty(text: str) -> str | None:
    lower = text.lower()
    for alias, faculty in FACULTY_ALIASES.items():
        if alias in lower:
            return faculty
    return None


def _get_discount(score: int, data: dict) -> int:
    """Return the ORT-based discount percentage for a given score."""
    for row in data.get("discount_table", []):
        score_range = row["score_range"]
        if score_range == "gold_certificate":
            continue
        try:
            low, high = score_range.split("-")
            if int(low) <= score <= int(high):
                return row["discount_percent"]
        except ValueError:
            continue
    return 0


def ort_validator(input_text: str) -> str:
    """
    Check ORT eligibility and calculate discount for MUA admissions.

    Use when student mentions their ORT score and asks about admission eligibility.
    Do NOT use for career guidance or program comparison.
    Input: student's message containing their ORT score.
    """
    data = _load_thresholds()
    min_threshold = data.get("min_ort_threshold", 110)
    score = _extract_score(input_text)
    faculty = _extract_faculty(input_text)

    if score is None:
        return (
            "Пожалуйста, укажите ваш балл ОРТ (например: «у меня 145 баллов»). "
            "Тогда я смогу проверить ваши шансы на поступление в МУА."
        )

    if score < min_threshold:
        return (
            f"❌ Ваш балл ОРТ ({score}) ниже минимального порога {min_threshold} баллов "
            f"для поступления в МУА.\n\n"
            f"💡 Альтернатива: IT&Business колледж МУА — поступление без ОРТ, "
            f"срок 1 год 10 месяцев, затем можно перейти на 2-й курс университета.\n\n"
            f"📞 Приёмная комиссия: +996 555 820 000 (WhatsApp)"
        )

    discount = _get_discount(score, data)
    if discount > 0:
        discount_msg = f"🎓 Скидка на обучение: **{discount}%**"
    else:
        discount_msg = "📋 Скидка по ОРТ не предусмотрена (балл ниже 171)"

    # Faculty-specific additional requirements
    additional = ""
    add_reqs = data.get("additional_subject_requirements", {})
    if faculty:
        req = add_reqs.get(faculty)
        if req:
            req_parts = [f"{subj.capitalize()} ≥{pts}" for subj, pts in req.items()]
            additional = (
                f"\n\n⚠️ Для поступления на **{faculty}** также требуются "
                f"дополнительные предметы ОРТ: {' и '.join(req_parts)}."
            )
    elif add_reqs:
        # No faculty specified — mention that some faculties have extra requirements
        additional = (
            "\n\n⚠️ Уточните факультет: для Инженерии нужны Математика+Физика ≥60, "
            "для Медицины — Биология+Химия ≥60."
        )

    return (
        f"✅ Ваш балл ОРТ ({score}) превышает минимальный порог ({min_threshold}). "
        f"Вы можете подать документы в МУА.\n\n"
        f"{discount_msg}{additional}\n\n"
        f"⚠️ Окончательное решение принимает приёмная комиссия. "
        f"Подача через портал: 2020.edu.gov.kg/vuz\n"
        f"📞 +996 555 820 000 (WhatsApp) | 📧 admission@alatoo.edu.kg"
    )


ort_validator_tool = Tool(
    name="ORT_Validator",
    func=ort_validator,
    description=(
        "Use this tool when a student mentions their ORT exam score and wants to know "
        "if they are eligible to apply to MUA (Ala-Too University). "
        "Checks eligibility against the 110-point minimum and calculates the ORT-based discount. "
        "Input: the student's message containing their ORT score. "
        "Examples: 'у меня 145 баллов, могу поступить?', 'мой ОРТ 183, есть ли скидка?', "
        "'Менин ОРТ балым 138. CS факультетине кире аламбы?'"
    ),
)
