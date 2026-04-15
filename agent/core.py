"""
core.py — ReAct agent wired to Grok (xAI) via LangChain.

Entry point: run_agent(user_input, session) → str
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from config import XAI_API_KEY, GROK_BASE_URL, GROK_MODEL, LLM_TEMPERATURE
from agent.session import SessionState
from agent.tools.ort_validator import ort_validator_tool
from agent.tools.orientation_engine import orientation_engine_tool
from agent.tools.program_comparator import program_comparator_tool
from agent.tools.human_handoff import human_handoff_tool


# ── System / prompt template ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an official admissions assistant for Ala-Too International University in Bishkek, Kyrgyzstan.
Your role is strictly limited to:
  1. Answering questions about admissions, ORT scores, tuition fees, programs, and syllabi.
  2. Helping students choose a faculty using professional orientation (RIASEC survey).
  3. Comparing university programs side by side.
  4. Handing off to a human officer when needed.

HARD RULES — never violate these:
  - NEVER guarantee admission, scholarships, or specific outcomes.
  - If you cannot find the answer in your tools or knowledge base, say:
    "У меня нет этой информации. Пожалуйста, обратитесь в приёмную комиссию."
  - NEVER discuss topics unrelated to Ala-Too University admissions or career orientation.
  - Answer in the same language the student used (Russian or Kyrgyz or both).
  - Do not fabricate ORT scores, tuition fees, or program details.

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: reason about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# ── LLM singleton ──────────────────────────────────────────────────────────────

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=GROK_MODEL,
            base_url=GROK_BASE_URL,
            api_key=XAI_API_KEY,
            temperature=LLM_TEMPERATURE,
        )
    return _llm


# ── Agent executor singleton ───────────────────────────────────────────────────

_agent_executor: AgentExecutor | None = None


def get_agent_executor() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        tools: list[Tool] = [
            ort_validator_tool,
            orientation_engine_tool,
            program_comparator_tool,
            human_handoff_tool,
        ]
        prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
        agent = create_react_agent(get_llm(), tools, prompt)
        _agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=8,          # prevent runaway loops
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )
    return _agent_executor


# ── Public entry point ─────────────────────────────────────────────────────────

def run_agent(user_input: str, session: SessionState) -> str:
    """
    Run the ReAct agent for a single user turn.

    Args:
        user_input: the (already guardrail-checked) user message
        session: the caller's SessionState (passed to tools via a thread-local)

    Returns:
        Agent's final answer string.
    """
    # Inject session into a context var so tools can read/write it
    _set_active_session(session)
    session.add_message("user", user_input)

    try:
        result = get_agent_executor().invoke({"input": user_input})
        answer: str = result.get("output", "").strip()
    except Exception as e:
        print(f"[agent] Error: {e}")
        answer = (
            "Произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте ещё раз или обратитесь в приёмную комиссию."
        )

    session.add_message("assistant", answer)
    return answer


# ── Session context (thread-local, avoids global state between users) ──────────

import threading
_local = threading.local()


def _set_active_session(session: SessionState) -> None:
    _local.session = session


def get_active_session() -> SessionState | None:
    return getattr(_local, "session", None)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agent.session import get_session
    from agent import guardrails

    test_queries = [
        "Какой минимальный балл ОРТ для поступления на Computer Science?",
        "Я не знаю, какую специальность выбрать. Помоги мне.",
        "Сравни программы CS и Economics.",
    ]

    session = get_session("cli_test_user")
    for q in test_queries:
        print(f"\n{'='*60}\nUSER: {q}")
        guard = guardrails.check(q)
        if guard.blocked:
            print(f"BLOCKED: {guard.reply}")
        else:
            reply = run_agent(q, session)
            print(f"AGENT: {reply}")
