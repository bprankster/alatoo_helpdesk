"""
core.py — ReAct agent wired to Qwen3-14B via Ollama (local, no cloud LLMs).

Entry point: run_agent(user_input, session) → str
"""

import os
import sys
import threading

import yaml
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.session import SessionState
from agent.tools.ort_validator import ort_validator_tool
from agent.tools.orientation_engine import orientation_engine_tool
from agent.tools.program_comparator import program_comparator_tool
from agent.tools.human_handoff import human_handoff_tool

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as _f:
    _cfg = yaml.safe_load(_f)


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


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_llm(thinking: bool = False) -> ChatOllama:
    """
    thinking=True  → Qwen3 extended reasoning (for RIASEC diagnosis)
    thinking=False → direct fast response (ORT checks, comparisons)
    """
    return ChatOllama(
        model=_cfg["llm"]["model"],
        base_url=_cfg["llm"]["base_url"],
        temperature=_cfg["llm"]["temperature"],
        num_predict=_cfg["llm"]["num_predict"],
        num_ctx=_cfg["llm"]["num_ctx"],
    )


def format_prompt(text: str, thinking: bool = False) -> str:
    """Prepend Qwen3 thinking mode prefix."""
    return ("/think " if thinking else "/no_think ") + text


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
            max_iterations=_cfg["agent"]["max_iterations"],
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )
    return _agent_executor


# ── Public entry point ─────────────────────────────────────────────────────────

def run_agent(user_input: str, session: SessionState) -> str:
    """
    Run the agent for a single user turn.

    Fast path (use_classifier=true): KyrgyzBERT classifies intent → direct tool call
    if confidence ≥ threshold. Falls back to ReAct on low confidence or errors.

    Slow path (use_classifier=false): always uses Qwen3 ReAct (ablation baseline).

    Args:
        user_input: the (already guardrail-checked) user message
        session: the caller's SessionState (passed to tools via a thread-local)

    Returns:
        Agent's final answer string.
    """
    _set_active_session(session)
    session.add_message("user", user_input)

    # Try classifier fast path
    from agent.router import route_query
    route = route_query(user_input)

    if route != "react_agent":
        _tool_map = {
            "ort_validator": ort_validator_tool,
            "orientation_engine": orientation_engine_tool,
            "program_comparator": program_comparator_tool,
            "human_handoff": human_handoff_tool,
        }
        tool = _tool_map.get(route)
        if tool:
            try:
                print(f"[router] Fast path → {route}")
                answer = tool.func(user_input)
                session.add_message("assistant", answer)
                return answer
            except Exception as e:
                print(f"[router] Fast path failed ({e}), falling back to ReAct")

    # ReAct path (Qwen3)
    try:
        result = get_agent_executor().invoke({"input": user_input})
        answer: str = result.get("output", "").strip()
    except Exception as e:
        print(f"[agent] Error: {e}")
        answer = _cfg["agent"]["fallback_message"]

    session.add_message("assistant", answer)
    return answer


# ── Session context (thread-local, avoids global state between users) ──────────

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
        "Какой минимальный балл ОРТ для поступления?",
        "Я не знаю, какую специальность выбрать. Помоги мне.",
        "Сравни программы IT и экономики.",
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
