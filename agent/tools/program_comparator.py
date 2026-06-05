"""
program_comparator.py — Tool C: Program_Comparator_RAG

Searches ChromaDB to compare two or more university programs side by side.
Response is built strictly from retrieved documents — no LLM fabrication.
"""

import os
import re
import sys

import yaml
from langchain_core.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

# Updated aliases matching real MUA faculty names
FACULTY_ALIASES: dict[str, str] = {
    "инженер": "Факультет инженерии и информатики",
    "информатик": "Факультет инженерии и информатики",
    "it": "Факультет инженерии и информатики",
    "cs": "Факультет инженерии и информатики",
    "computer": "Факультет инженерии и информатики",
    "программирован": "Факультет инженерии и информатики",
    "data science": "Факультет инженерии и информатики",
    "ии": "Факультет инженерии и информатики",
    "кибербезопасност": "Факультет инженерии и информатики",
    "робот": "Факультет инженерии и информатики",
    "экономик": "Факультет экономики и управления",
    "менеджмент": "Факультет экономики и управления",
    "management": "Факультет экономики и управления",
    "финанс": "Факультет экономики и управления",
    "юридическ": "Факультет экономики и управления",
    "право": "Факультет экономики и управления",
    "law": "Факультет экономики и управления",
    "туризм": "Факультет экономики и управления",
    "гостеприимств": "Факультет экономики и управления",
    "экология": "Факультет экономики и управления",
    "лингвистик": "Факультет гуманитарных наук",
    "перевод": "Факультет гуманитарных наук",
    "филолог": "Факультет гуманитарных наук",
    "педагогик": "Факультет гуманитарных наук",
    "stem": "Факультет гуманитарных наук",
    "медицин": "Медицинский факультет",
    "лечебн": "Медицинский факультет",
    "medicine": "Медицинский факультет",
    "психолог": "Факультет социальных наук",
    "журналистик": "Факультет социальных наук",
    "международн": "Факультет социальных наук",
    "social": "Факультет социальных наук",
}

COMPARE_PATTERN = re.compile(
    r"(?:сравни|сравните|compare|versus|vs\.?|и\s|and\s|,)\s*",
    re.IGNORECASE,
)


def _extract_programs(text: str) -> list[str]:
    lower = text.lower()
    for prefix in ["сравни программы", "сравни факультеты", "сравните",
                   "compare programs", "compare faculties", "compare", "сравни"]:
        lower = lower.replace(prefix, "")
    parts = COMPARE_PATTERN.split(lower.strip())
    programs = [p.strip() for p in parts if p.strip()]
    return programs[:4]


def _resolve_faculty(program_text: str) -> str | None:
    lower = program_text.lower()
    for alias, faculty in FACULTY_ALIASES.items():
        if alias in lower:
            return faculty
    return None


def program_comparator(input_text: str) -> str:
    """
    Compare two or more university programs using ChromaDB retrieval.

    Use when student asks to compare programs.
    Do NOT use for ORT score checks or career guidance.
    Input: comparison request as full sentence.
    """
    from data_ingestion.embedder import query_collection

    programs = _extract_programs(input_text)

    if len(programs) < 2:
        return (
            "Пожалуйста, укажите минимум две специальности для сравнения. "
            "Например: «Сравни IT и экономику» или «Compare психологию и журналистику»."
        )

    top_k = _cfg["retrieval"]["top_k"]
    sections: list[str] = []

    for prog in programs:
        faculty = _resolve_faculty(prog)
        where = {"faculty": faculty} if faculty else None

        results = query_collection(
            query_text=f"программа учебный план карьера специальность {prog}",
            n_results=top_k,
            where=where,
            similarity_threshold=0.65,
        )

        if not results:
            sections.append(f"### {prog.title()}\n_Информация не найдена в базе данных._")
            continue

        combined = "\n".join(r["text"] for r in results)
        if len(combined) > 800:
            combined = combined[:800] + "…"

        sections.append(f"### {prog.title()}\n{combined}")

    header = "## Сравнение программ\n\n"
    body = "\n\n---\n\n".join(sections)
    footer = (
        "\n\n---\n_Информация получена из базы данных Ала-Тоо Университета. "
        "Для точных деталей обратитесь в приёмную комиссию: +996 555 820 000_"
    )
    return header + body + footer


program_comparator_tool = Tool(
    name="Program_Comparator_RAG",
    func=program_comparator,
    description=(
        "Use this tool ONLY when a student explicitly asks to COMPARE two or more programs "
        "side by side using words like 'сравни', 'compare', 'vs', 'versus', 'или', 'or'. "
        "Examples: 'Сравни IT и экономику', 'Compare психологию и журналистику'. "
        "Do NOT use for: ORT score checks, 'which faculty should I choose', "
        "'кайсы факультетке барсам', 'what faculties exist', general faculty questions, "
        "or when student is undecided about career path — use University_KB_Search instead."
    ),
)
