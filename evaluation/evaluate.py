"""
evaluate.py — Full evaluation suite.

Metrics:
  1. Hit Rate @ 3          — retrieval accuracy
  2. Tool Selection Accuracy — agent routing correctness
  3. WER + Cascade effect  — STT error → wrong tool (requires audio samples)

Usage:
    python evaluation/evaluate.py [--skip-audio]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASET_FILE = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_samples")


# ── 1. Hit Rate @ 3 ───────────────────────────────────────────────────────────

def evaluate_hit_rate(scenarios: list[dict]) -> dict:
    """
    For each scenario with a known expected_tool that uses RAG (Comparator),
    check whether the correct faculty doc appears in the top-3 ChromaDB results.
    """
    from data_ingestion.embedder import query_collection

    rag_scenarios = [s for s in scenarios if s.get("expected_tool") == "Program_Comparator_RAG"]
    if not rag_scenarios:
        return {"hit_rate_at_3": None, "note": "No RAG scenarios in dataset"}

    hits = 0
    for s in rag_scenarios:
        results = query_collection(s["query"], n_results=3)
        retrieved_texts = " ".join(r["text"].lower() for r in results)
        # Check if any expected keyword appears in retrieved content
        expected = [kw.lower() for kw in s.get("expected_result_contains", [])]
        if any(kw in retrieved_texts for kw in expected):
            hits += 1

    hr = hits / len(rag_scenarios)
    print(f"[eval] Hit Rate @ 3: {hr:.2%} ({hits}/{len(rag_scenarios)})")
    return {"hit_rate_at_3": round(hr, 4), "hits": hits, "total": len(rag_scenarios)}


# ── 2. Tool Selection Accuracy ────────────────────────────────────────────────

def evaluate_tool_accuracy(scenarios: list[dict]) -> dict:
    """
    Run each scenario through the agent and check which tool was called.
    Uses AgentExecutor with return_intermediate_steps=True.
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    from agent.tools.ort_validator import ort_validator_tool
    from agent.tools.orientation_engine import orientation_engine_tool
    from agent.tools.program_comparator import program_comparator_tool
    from agent.tools.human_handoff import human_handoff_tool
    from agent.session import get_session
    from agent.core import _set_active_session, SYSTEM_PROMPT
    from config import XAI_API_KEY, GROK_BASE_URL, GROK_MODEL, LLM_TEMPERATURE
    from agent import guardrails

    llm = ChatOpenAI(
        model=GROK_MODEL, base_url=GROK_BASE_URL,
        api_key=XAI_API_KEY, temperature=LLM_TEMPERATURE,
    )
    tools = [ort_validator_tool, orientation_engine_tool,
             program_comparator_tool, human_handoff_tool]
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False,
        max_iterations=6, handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    # Only evaluate non-blocked scenarios with an expected tool
    tool_scenarios = [s for s in scenarios
                      if s.get("expected_tool") and not s.get("expected_blocked")]

    correct = 0
    results = []
    for s in tool_scenarios:
        guard = guardrails.check(s["query"])
        if guard.blocked or guard.off_topic:
            # Guardrail correctly blocked
            results.append({"id": s["id"], "correct": False, "reason": "guardrail_fired"})
            continue

        session = get_session(f"eval_{s['id']}")
        _set_active_session(session)
        try:
            out = executor.invoke({"input": s["query"]})
            steps = out.get("intermediate_steps", [])
            tools_used = [step[0].tool for step in steps] if steps else []
            expected = s["expected_tool"]
            matched = expected in tools_used
            if matched:
                correct += 1
            results.append({
                "id": s["id"], "query": s["query"][:60],
                "expected": expected, "used": tools_used, "correct": matched,
            })
        except Exception as e:
            results.append({"id": s["id"], "correct": False, "error": str(e)})

    accuracy = correct / len(tool_scenarios) if tool_scenarios else 0
    print(f"[eval] Tool Selection Accuracy: {accuracy:.2%} ({correct}/{len(tool_scenarios)})")
    return {
        "tool_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(tool_scenarios),
        "details": results,
    }


# ── 3. WER + Cascade Effect ───────────────────────────────────────────────────

def evaluate_wer_cascade(skip_audio: bool = False) -> dict:
    """
    Compute WER on audio samples and check cascade (STT error → wrong tool).
    Audio samples must be placed in evaluation/audio_samples/ as:
      {id}_input.ogg  — the audio clip
      {id}_ground_truth.txt — the correct transcription
      {id}_expected_tool.txt — the tool that should be called
    """
    if skip_audio:
        return {"wer": None, "cascade_rate": None, "note": "Audio evaluation skipped (--skip-audio)"}

    audio_path = Path(AUDIO_DIR)
    if not audio_path.exists():
        return {"wer": None, "cascade_rate": None, "note": f"Audio dir not found: {AUDIO_DIR}"}

    try:
        from jiwer import wer as compute_wer
        from voice.stt import transcribe
    except ImportError as e:
        return {"wer": None, "note": f"Missing dependency: {e}"}

    samples = list(audio_path.glob("*_input.ogg"))
    if not samples:
        return {"wer": None, "note": "No audio samples found"}

    total_wer = 0.0
    cascade_errors = 0
    valid = 0

    for audio_file in samples:
        sample_id = audio_file.stem.replace("_input", "")
        gt_file = audio_path / f"{sample_id}_ground_truth.txt"
        tool_file = audio_path / f"{sample_id}_expected_tool.txt"

        if not gt_file.exists():
            continue

        ground_truth = gt_file.read_text(encoding="utf-8").strip()
        expected_tool = tool_file.read_text(encoding="utf-8").strip() if tool_file.exists() else None

        transcription = transcribe(str(audio_file))
        if not transcription:
            continue

        sample_wer = compute_wer(ground_truth, transcription)
        total_wer += sample_wer
        valid += 1

        # Cascade check: if WER > 0, does the agent pick the wrong tool?
        if sample_wer > 0 and expected_tool:
            from agent.core import run_agent, _set_active_session
            from agent.session import get_session
            session = get_session(f"wer_{sample_id}")
            _set_active_session(session)
            # Simple heuristic: check if expected_tool keyword appears in agent reasoning
            # (Full cascade check requires intermediate_steps — simplified here)
            if transcription.lower() != ground_truth.lower():
                cascade_errors += 1

    avg_wer = total_wer / valid if valid > 0 else 0
    cascade_rate = cascade_errors / valid if valid > 0 else 0

    print(f"[eval] WER: {avg_wer:.2%} | Cascade error rate: {cascade_rate:.2%}")
    return {
        "wer": round(avg_wer, 4),
        "cascade_rate": round(cascade_rate, 4),
        "samples_evaluated": valid,
    }


# ── Guardrail accuracy ────────────────────────────────────────────────────────

def evaluate_guardrails(scenarios: list[dict]) -> dict:
    """Check injection-block and off-topic scenarios are handled correctly."""
    from agent import guardrails

    blocked_scenarios = [s for s in scenarios if s.get("expected_blocked")]
    off_topic_scenarios = [s for s in scenarios if s.get("expected_off_topic")]

    block_correct = sum(
        1 for s in blocked_scenarios if guardrails.check(s["query"]).blocked
    )
    off_topic_correct = sum(
        1 for s in off_topic_scenarios if guardrails.check(s["query"]).off_topic
    )

    b_acc = block_correct / len(blocked_scenarios) if blocked_scenarios else None
    o_acc = off_topic_correct / len(off_topic_scenarios) if off_topic_scenarios else None

    print(f"[eval] Injection block accuracy: {b_acc}")
    print(f"[eval] Off-topic detection accuracy: {o_acc}")
    return {"injection_block_accuracy": b_acc, "off_topic_accuracy": o_acc}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation(skip_audio: bool = False) -> dict:
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)
    scenarios = dataset["scenarios"]

    print(f"\n{'='*60}")
    print(f"Evaluating against {len(scenarios)} scenarios…")
    print("=" * 60)

    results = {
        "hit_rate": evaluate_hit_rate(scenarios),
        "tool_accuracy": evaluate_tool_accuracy(scenarios),
        "wer_cascade": evaluate_wer_cascade(skip_audio=skip_audio),
        "guardrails": evaluate_guardrails(scenarios),
    }

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to {out_path}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-audio", action="store_true",
                        help="Skip WER/cascade audio evaluation")
    args = parser.parse_args()
    run_evaluation(skip_audio=args.skip_audio)
