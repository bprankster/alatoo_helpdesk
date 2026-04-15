"""
program_comparator.py — Tool C: Program_Comparator_RAG

Searches ChromaDB to compare two or more university programs side by side.
Response is built strictly from retrieved documents — no LLM fabrication.
"""

import os
import re
import sys

from langchain.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config import RETRIEVAL_TOP_K

# Faculty aliases for metadata filtering
FACULTY_ALIASES: dict[str, str] = {
    "cs": "CS", "computer science": "CS", "информатика": "CS",
    "software": "CS", "it": "CS",
    "economics": "Economics", "экономика": "Economics",
    "finance": "Economics", "финансы": "Economics",
    "business": "Business", "бизнес": "Business",
    "management": "Business", "менеджмент": "Business",
    "law": "Law", "право": "Law", "юридический": "Law",
    "education": "Education", "педагогика": "Education",
    "psychology": "Education", "психология": "Education",
    "design": "Design", "дизайн": "Design",
    "engineering": "Engineering", "инженерия": "Engineering",
    "architecture": "Engineering", "архитектура": "Engineering",
}

# Patterns to split "compare X and Y" or "X vs Y"
COMPARE_PATTERN = re.compile(
    r"(?:сравни|сравните|compare|versus|vs\.?|и|and|,)\s+",
    re.IGNORECASE,
)


def _extract_programs(text: str) -> list[str]:
    """Pull program names from a comparison query."""
    lower = text.lower()
    # Remove common preamble
    for prefix in ["сравни программы", "сравните", "compare programs", "compare", "сравни"]:
        lower = lower.replace(prefix, "")
    parts = COMPARE_PATTERN.split(lower.strip())
    programs = [p.strip() for p in parts if p.strip()]
    return programs[:4]   # cap at 4 programs


def _resolve_faculty(program_text: str) -> str | None:
    lower = program_text.lower()
    for alias, faculty in FACULTY_ALIASES.items():
        if alias in lower:
            return faculty
    return None


def program_comparator(input_text: str) -> str:
    """
    Compare two or more university programs using ChromaDB retrieval.

    Input example: "Сравни CS и Economics"
    """
    # Lazy import to avoid loading ChromaDB at module level
    from data_ingestion.embedder import query_collection

    programs = _extract_programs(input_text)

    if len(programs) < 2:
        return (
            "Пожалуйста, укажите минимум две специальности для сравнения. "
            "Например: «Сравни CS и Economics» или «Compare Law and Business»."
        )

    sections: list[str] = []
    for prog in programs:
        faculty = _resolve_faculty(prog)
        where = {"faculty": faculty} if faculty else None

        results = query_collection(
            query_text=f"program description curriculum tuition {prog}",
            n_results=RETRIEVAL_TOP_K,
            where=where,
        )

        if not results:
            sections.append(f"### {prog.title()}\n_Информация не найдена в базе данных._")
            continue

        # Stitch retrieved chunks into a summary
        combined = "\n".join(r["text"] for r in results)
        # Trim to avoid overly long output
        if len(combined) > 800:
            combined = combined[:800] + "…"

        sections.append(f"### {prog.title()}\n{combined}")

    header = "## Сравнение программ\n\n"
    body = "\n\n---\n\n".join(sections)
    footer = (
        "\n\n---\n_Информация получена из базы данных Ала-Тоо Университета. "
        "Для точных деталей обратитесь в приёмную комиссию._"
    )
    return header + body + footer


program_comparator_tool = Tool(
    name="Program_Comparator_RAG",
    func=program_comparator,
    description=(
        "Use this tool when a student wants to compare two or more university programs "
        "or faculties side by side (e.g. 'Compare CS and Economics', "
        "'Сравни юридический и бизнес'). "
        "Input: the student's comparison request as a full sentence."
    ),
)
