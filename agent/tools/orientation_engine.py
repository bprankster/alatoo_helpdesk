"""
orientation_engine.py — Tool B: Professional_Orientation_Engine

Adaptive 5-question RIASEC survey using LLM-generated questions.
Each question is generated dynamically based on previous answers,
targeting the most uncertain RIASEC dimensions (Qwen3 with /think mode).
Maps final result to real MUA faculty names from riasec_mapping.json.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml
from langchain_core.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_RIASEC_FILE = str(_PROJECT_ROOT / _cfg["data"]["riasec_mapping"].lstrip("./"))

MAX_QUESTIONS: int = _cfg["orientation"]["max_questions"]
VALID_ANSWERS = {"R", "I", "A", "S", "E", "C"}

_RIASEC_NAMES = {
    "R": "Реалистичный",
    "I": "Исследовательский",
    "A": "Артистический",
    "S": "Социальный",
    "E": "Предприимчивый",
    "C": "Конвенциональный",
}


def _load_mapping() -> dict:
    with open(_RIASEC_FILE, encoding="utf-8") as f:
        return json.load(f)


def _extract_answer(text: str) -> str | None:
    """Pull a RIASEC letter from user response, or detect option number (1-4)."""
    upper = text.strip().upper()
    # Direct letter answer
    for char in upper:
        if char in VALID_ANSWERS:
            return char
    # Numeric answer "1", "2", "3", "4" — will be resolved by the caller
    # using the current question's option list
    return None


def _compute_result(answers: list[str]) -> tuple[str, list[dict]]:
    """Return top-2 RIASEC types and faculty recommendations scored by profile match."""
    mapping = _load_mapping()
    counts = Counter(answers)
    top2 = [t for t, _ in counts.most_common(2)]
    riasec_type = "".join(top2)
    student_set = set(top2)

    # Score each faculty by how many of its RIASEC types overlap with the student's top types
    scored: list[tuple[int, str, dict]] = []
    for name, info in mapping.get("faculties", {}).items():
        overlap = len(set(info.get("riasec_types", [])) & student_set)
        if overlap > 0:
            scored.append((overlap, name, info))

    scored.sort(key=lambda x: x[0], reverse=True)

    recs = [
        {
            "faculty": name,
            "programs": info.get("programs", []),
            "ort_req": info.get("additional_ort", ""),
        }
        for _, name, info in scored[:2]
    ]
    return riasec_type, recs


def _generate_next_question(history: list[dict]) -> dict:
    """
    Use Qwen3 to generate the next adaptive RIASEC question.

    Returns a dict: {question: str, options: [{text: str, riasec: str}]}
    Falls back to a safe static question if LLM call fails.
    """
    import re as _re
    from agent.core import get_llm

    is_first = len(history) == 0
    history_text = "\n".join(
        f"Q{i+1}: {h['question']}\nОтвет: {h['answer']} ({h.get('riasec', '?')})"
        for i, h in enumerate(history)
    )

    if is_first:
        context = "Это первый вопрос — широкий, ситуационный, о предпочтениях студента."
    else:
        context = (
            f"Предыдущие ответы:\n{history_text}\n\n"
            "Определи RIASEC-типы, которые ещё не подтверждены. "
            "Задай вопрос, максимально различающий оставшиеся типы."
        )

    prompt = (
        "/no_think "
        "Ты консультант по профориентации (Holland RIASEC: "
        "R=Реалистичный I=Исследовательский A=Артистический "
        "S=Социальный E=Предприимчивый C=Конвенциональный).\n"
        f"{context}\n"
        f"Придумай вопрос №{len(history)+1} из {MAX_QUESTIONS} на русском языке. "
        "Ситуационный сценарий. Ровно 4 варианта ответа, каждый — отдельный RIASEC-тип (R/I/A/S/E/C).\n"
        "Ответь СТРОГО только JSON-объектом без markdown и пояснений:\n"
        '{"question":"...","options":['
        '{"text":"...","riasec":"R"},{"text":"...","riasec":"I"},'
        '{"text":"...","riasec":"A"},{"text":"...","riasec":"S"}]}'
    )

    raw = ""
    llm = get_llm()
    try:
        raw = llm.invoke(prompt).content
        # Strip any <think>...</think> block
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[-1].strip()
        # Strip markdown code fences
        raw = _re.sub(r"```\w*\n?", "", raw).strip()
        # Find JSON object (robust: handles leading/trailing text)
        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if m:
            return json.loads(m.group())
        return json.loads(raw)
    except Exception as e:
        print(f"[orientation] LLM generation failed: {e} | raw={repr(raw[:200])}")
        return _fallback_question(len(history))


def _fallback_question(step: int) -> dict:
    """Static fallback questions if LLM call fails."""
    fallbacks = [
        {
            "question": "Представьте свободный вечер. Что вы предпочтёте делать?",
            "options": [
                {"text": "Починить что-то или собрать своими руками", "riasec": "R"},
                {"text": "Читать научную статью или смотреть документальный фильм", "riasec": "I"},
                {"text": "Нарисовать, написать или сыграть музыку", "riasec": "A"},
                {"text": "Встретиться с друзьями и помочь кому-то с проблемой", "riasec": "S"},
            ],
        },
        {
            "question": "Какая рабочая среда вас привлекает больше всего?",
            "options": [
                {"text": "Лаборатория или мастерская — работа руками", "riasec": "R"},
                {"text": "Исследовательский центр или библиотека", "riasec": "I"},
                {"text": "Творческая студия или медиаагентство", "riasec": "A"},
                {"text": "Школа, больница или НКО", "riasec": "S"},
            ],
        },
        {
            "question": "Какое задание вас больше воодушевит на работе?",
            "options": [
                {"text": "Настроить и оптимизировать техническую систему", "riasec": "R"},
                {"text": "Провести анализ данных и найти закономерности", "riasec": "I"},
                {"text": "Создать дизайн или придумать концепцию кампании", "riasec": "A"},
                {"text": "Провести мастер-класс или помочь группе людей", "riasec": "S"},
            ],
        },
        {
            "question": "Какой из этих предметов в школе нравился вам больше всего?",
            "options": [
                {"text": "Физика или технология", "riasec": "R"},
                {"text": "Математика или биология", "riasec": "I"},
                {"text": "Литература или искусство", "riasec": "A"},
                {"text": "История или обществознание", "riasec": "S"},
            ],
        },
        {
            "question": "Какой тип проекта вы бы выбрали для дипломной работы?",
            "options": [
                {"text": "Спроектировать и построить устройство или систему", "riasec": "R"},
                {"text": "Провести научное исследование с экспериментами", "riasec": "I"},
                {"text": "Создать художественное произведение или медиапроект", "riasec": "A"},
                {"text": "Изучить социальную проблему и предложить её решение", "riasec": "S"},
            ],
        },
    ]
    return fallbacks[step % len(fallbacks)]


_CYR_LABELS = ["А", "Б", "В", "Г"]


def _format_question(q_data: dict, step: int) -> str:
    """Format a question dict into a readable string for the student."""
    options_text = "\n".join(
        f"  {_CYR_LABELS[i]}. {opt['text']}"
        for i, opt in enumerate(q_data["options"])
    )
    return (
        f"Вопрос ({step}/{MAX_QUESTIONS}):\n\n"
        f"{q_data['question']}\n\n"
        f"{options_text}\n\n"
        "Ответьте буквой (А/Б/В/Г) или напишите своими словами."
    )


def _resolve_option_answer(text: str, options: list[dict]) -> str | None:
    """Resolve A/B/C/D or А/Б/В/Г letter to RIASEC type from option list."""
    upper = text.strip().upper()
    cyrillic_map = {"А": 0, "Б": 1, "В": 2, "Г": 3}
    latin_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    idx = cyrillic_map.get(upper[0]) if upper else None
    if idx is None:
        idx = latin_map.get(upper[0]) if upper else None
    if idx is not None and idx < len(options):
        return options[idx]["riasec"]
    # Try numeric
    try:
        idx = int(upper[0]) - 1
        if 0 <= idx < len(options):
            return options[idx]["riasec"]
    except (ValueError, IndexError):
        pass
    return None


def _format_result(riasec_type: str, recs: list[dict]) -> str:
    """Format the final RIASEC survey result with faculty + program recommendations."""
    type_desc = " + ".join(_RIASEC_NAMES.get(t, t) for t in riasec_type)
    lines = [
        f"Спасибо! Тест завершён.",
        f"\n🧠 Ваш профиль RIASEC: **{riasec_type}** ({type_desc})",
    ]

    for i, rec in enumerate(recs):
        label = "🎓 Основная рекомендация" if i == 0 else "🔄 Альтернатива"
        lines.append(f"\n{label}: **{rec['faculty']}**")

        programs = rec["programs"][:4]
        if programs:
            lines.append("📚 Программы:")
            for prog in programs:
                careers = ", ".join(prog["careers"][:3])
                lines.append(f"  • **{prog['name']}** ({prog['duration']}) — {careers}")

        ort = rec.get("ort_req", "")
        if ort and ort != "не требуется":
            lines.append(f"  ⚠️ Требования ОРТ: {ort}")

    lines.append(
        "\n💬 Чтобы сравнить программы или проверить проходной балл ОРТ — просто спросите!"
    )
    return "\n".join(lines)


def orientation_engine(input_text: str) -> str:
    """
    Adaptive RIASEC survey with LLM-generated questions.

    Use when student is undecided about major/career, or survey is already in progress.
    Do NOT use for ORT score checks or direct program comparisons.
    Input: student's latest message (may contain their survey answer).
    """
    from agent.core import get_active_session

    session = get_active_session()
    if session is None:
        return "Ошибка: сессия не найдена. Пожалуйста, начните разговор заново."

    # Ensure survey state exists on session
    if not hasattr(session, "riasec_history"):
        session.riasec_history = []
    if not hasattr(session, "riasec_current_question"):
        session.riasec_current_question = None

    # ── Survey already complete ────────────────────────────────────────────────
    if session.riasec_complete() and session.riasec_result:
        riasec_type = session.riasec_result
        _, recs = _compute_result(session.riasec_answers)
        return _format_result(riasec_type, recs)

    # ── Process answer if survey is in progress ────────────────────────────────
    if session.riasec_in_progress():
        current_q = session.riasec_current_question
        riasec_answer = None

        if current_q:
            riasec_answer = _resolve_option_answer(input_text, current_q["options"])

        if riasec_answer is None:
            riasec_answer = _extract_answer(input_text)

        if riasec_answer is None:
            q_text = _format_question(current_q, session.riasec_step) if current_q else ""
            return (
                "Пожалуйста, ответьте буквой варианта (А, Б, В или Г).\n\n"
                + q_text
            )

        # Record answer with question context
        if current_q:
            session.riasec_history.append({
                "question": current_q["question"],
                "answer": input_text[:100],
                "riasec": riasec_answer,
            })
        session.riasec_answers.append(riasec_answer)

        # Survey complete?
        if len(session.riasec_answers) >= MAX_QUESTIONS:
            riasec_type, recs = _compute_result(session.riasec_answers)
            session.riasec_result = riasec_type
            session.riasec_step = MAX_QUESTIONS
            return _format_result(riasec_type, recs)

        # Generate next question
        session.riasec_step += 1
        next_q = _generate_next_question(session.riasec_history)
        session.riasec_current_question = next_q
        return _format_question(next_q, session.riasec_step)

    # ── Start the survey ───────────────────────────────────────────────────────
    session.riasec_step = 1
    session.riasec_answers = []
    session.riasec_history = []

    first_q = _generate_next_question([])
    session.riasec_current_question = first_q

    return (
        "Отлично! Давайте определим, какая специальность вам подходит.\n"
        f"Я задам {MAX_QUESTIONS} ситуационных вопроса (метод Голланда — RIASEC).\n\n"
        + _format_question(first_q, 1)
    )


orientation_engine_tool = Tool(
    name="Professional_Orientation_Engine",
    func=orientation_engine,
    description=(
        "Use this tool when a student is undecided about their major or career path, "
        "wants help choosing a faculty, or when a RIASEC survey is already in progress "
        "and the student has answered a question. "
        "This tool runs an adaptive 5-question Holland RIASEC survey with LLM-generated "
        "situational questions and maps the result to real Ala-Too University faculties. "
        "Input: the student's latest message (question or survey answer). "
        "Do NOT use for ORT score checks or direct program comparisons."
    ),
)
