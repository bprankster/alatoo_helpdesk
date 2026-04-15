"""
orientation_engine.py — Tool B: Professional_Orientation_Engine

Stateful RIASEC 5-question survey that maps the student's answers to
Ala-Too University faculties. Reads/writes from the active SessionState.
Hard-capped at exactly 5 questions, then forces a faculty recommendation.
"""

import json
import os
import sys
from collections import Counter

from langchain.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config import RIASEC_MAPPING_FILE, RIASEC_MAX_QUESTIONS


def _load_mapping() -> dict:
    with open(RIASEC_MAPPING_FILE, encoding="utf-8") as f:
        return json.load(f)


VALID_ANSWERS = {"R", "I", "A", "S", "E", "C"}


def _extract_answer(text: str) -> str | None:
    """Pull a single RIASEC letter from the user's response."""
    upper = text.strip().upper()
    for char in upper:
        if char in VALID_ANSWERS:
            return char
    return None


def _compute_result(answers: list[str]) -> tuple[str, list[str]]:
    """Return the top-2 RIASEC types and the corresponding faculty list."""
    mapping = _load_mapping()
    counts = Counter(answers)
    top2 = [t for t, _ in counts.most_common(2)]
    riasec_type = "".join(top2)

    faculties: list[str] = []
    for t in top2:
        faculties.extend(mapping["mapping"].get(t, {}).get("faculties", []))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_faculties = [f for f in faculties if not (f in seen or seen.add(f))]
    return riasec_type, unique_faculties


def orientation_engine(input_text: str) -> str:
    """
    Run or advance the RIASEC survey for the active session.

    The tool is called by the agent whenever the student expresses uncertainty
    about their major, OR when the agent is advancing an in-progress survey.
    """
    # Import here to avoid circular import at module load time
    from agent.core import get_active_session

    session = get_active_session()
    if session is None:
        return "Ошибка: сессия не найдена. Пожалуйста, начните разговор заново."

    mapping = _load_mapping()
    questions = mapping["survey_questions"]

    # ── Survey already complete ────────────────────────────────────────────────
    if session.riasec_complete() and session.riasec_result:
        riasec_type = session.riasec_result
        _, faculties = _compute_result(session.riasec_answers)
        faculty_list = "\n".join(f"  • {f}" for f in faculties)
        return (
            f"Ваш профиль RIASEC: **{riasec_type}**.\n\n"
            f"Рекомендуемые специальности Ала-Тоо Университета:\n{faculty_list}\n\n"
            "Хотите узнать подробнее об одной из этих программ? "
            "Я могу сравнить их или проверить ваш балл ОРТ для поступления."
        )

    # ── Process answer if survey is in progress ────────────────────────────────
    if session.riasec_in_progress():
        answer = _extract_answer(input_text)
        if answer is None:
            # Re-ask the current question
            q = questions[session.riasec_step - 1]
            return (
                "Пожалуйста, ответьте одной буквой: R, I, A, S, E или C.\n\n"
                + q["question_ru"]
            )
        session.riasec_answers.append(answer)

        # Check if all questions answered
        if len(session.riasec_answers) >= RIASEC_MAX_QUESTIONS:
            riasec_type, faculties = _compute_result(session.riasec_answers)
            session.riasec_result = riasec_type
            session.riasec_step = RIASEC_MAX_QUESTIONS
            faculty_list = "\n".join(f"  • {f}" for f in faculties)
            return (
                f"Спасибо! Вы ответили на все {RIASEC_MAX_QUESTIONS} вопроса.\n\n"
                f"Ваш профиль RIASEC: **{riasec_type}**.\n\n"
                f"Рекомендуемые специальности Ала-Тоо Университета:\n{faculty_list}\n\n"
                "Хотите узнать подробнее о любой из этих программ?"
            )

        # Ask next question
        session.riasec_step += 1
        next_q = questions[session.riasec_step - 1]
        progress = f"({session.riasec_step}/{RIASEC_MAX_QUESTIONS})"
        return f"Вопрос {progress}:\n\n{next_q['question_ru']}"

    # ── Start the survey ───────────────────────────────────────────────────────
    session.riasec_step = 1
    session.riasec_answers = []
    first_q = questions[0]
    return (
        "Отлично! Давайте определим, какая специальность вам подходит, "
        f"с помощью короткого опроса из {RIASEC_MAX_QUESTIONS} вопросов "
        "(метод Голланда — RIASEC).\n\n"
        f"Вопрос (1/{RIASEC_MAX_QUESTIONS}):\n\n"
        + first_q["question_ru"]
    )


orientation_engine_tool = Tool(
    name="Professional_Orientation_Engine",
    func=orientation_engine,
    description=(
        "Use this tool when a student is undecided about their major or career path, "
        "or when a RIASEC survey is already in progress and the student has provided "
        "an answer (R/I/A/S/E/C) to the current question. "
        "This tool administers a 5-question Holland RIASEC survey and maps the result "
        "to Ala-Too University faculties. "
        "Input: the student's latest message (may contain their survey answer)."
    ),
)
