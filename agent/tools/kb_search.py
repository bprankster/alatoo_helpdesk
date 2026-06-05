"""
kb_search.py — Tool: University_KB_Search

General-purpose RAG search over the ChromaDB knowledge base.
Handles any factual question about the university not covered by
the specialised tools (ORT, orientation, comparator).
"""

import os
import sys

import yaml
from langchain_core.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

# Lang prefixes that run_agent prepends — strip before embedding so they don't
# pollute the vector query with instruction text.
_LANG_PREFIXES = [
    "МААНИЛҮҮ: Кыргыз тилинде гана жооп бер. ",
    "IMPORTANT: Reply in English only. ",
]

# Use a more lenient threshold than the default (0.40) — general questions
# have broader phrasing and need a wider similarity window.
_KB_SEARCH_THRESHOLD = 0.65


def _strip_prefix(query: str) -> str:
    for prefix in _LANG_PREFIXES:
        if query.startswith(prefix):
            return query[len(prefix):]
    return query


def university_kb_search(query: str) -> str:
    """
    Search the university knowledge base for any factual question.

    Use for general questions about faculties, programs, fees, documents,
    contacts, admissions process, campus, etc.
    Do NOT use for ORT score checks, career guidance surveys, or program comparisons.
    Input: the student's question as a full sentence.
    """
    from data_ingestion.embedder import query_collection
    from agent.core import get_llm, detect_language

    clean_query = _strip_prefix(query)

    results = query_collection(
        query_text=clean_query,
        n_results=_cfg["retrieval"]["top_k"],
        similarity_threshold=_KB_SEARCH_THRESHOLD,
    )

    if not results:
        # Signal the ReAct agent to escalate to Human_Handoff_Trigger
        return (
            "[NO_INFORMATION] Информация по этому вопросу не найдена в базе данных. "
            "Студента необходимо перенаправить к сотруднику приёмной комиссии."
        )

    context = "\n\n".join(r["text"] for r in results)
    if len(context) > 2000:
        context = context[:2000] + "…"

    lang = detect_language(clean_query)
    lang_instruction = {
        "ky": "Кыргыз тилинде гана жооп бер. Бардык сандарды жана форматтоону сакта.",
        "ru": "Отвечай только на русском языке.",
        "en": "Reply in English only.",
    }.get(lang, "Отвечай только на русском языке.")

    prompt = (
        f"/no_think {lang_instruction}\n\n"
        "Ты — ассистент приёмной комиссии Ала-Тоо Университета. "
        "Используй ТОЛЬКО информацию из контекста ниже. "
        "Не придумывай факты. Отвечай кратко и по делу.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {clean_query}\n\n"
        "Ответ:"
    )

    try:
        llm = get_llm(thinking=False)
        answer = llm.invoke(prompt).content.strip()
        return answer
    except Exception as e:
        print(f"[kb_search] LLM synthesis failed: {e}")
        return context[:1200]


university_kb_search_tool = Tool(
    name="University_KB_Search",
    func=university_kb_search,
    description=(
        "Use this tool to answer ANY factual question about Ala-Too University "
        "that is not covered by the other tools. This includes: list of faculties, "
        "admission documents required, tuition fees, contacts, campus location, "
        "program descriptions, academic calendar, scholarships, and general university info. "
        "Examples: 'Какие факультеты есть в МУА?', 'Какие документы нужны для поступления?', "
        "'кайсы факультеттер бар?', 'What is the tuition fee?', 'Где находится университет?'. "
        "Input: the student's question as a full sentence. "
        "If the tool returns [NO_INFORMATION], call Human_Handoff_Trigger immediately."
    ),
)
