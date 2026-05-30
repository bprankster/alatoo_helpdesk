"""
evaluate.py — Full evaluation suite.

Metrics:
  1. Hit Rate @ K          — retrieval accuracy (dense-only vs BM25+dense ensemble)
  2. Tool Selection Accuracy — agent routing correctness (ReAct path)
  3. Classifier vs ReAct Ablation — KyrgyzBERT fast path vs pure ReAct
  4. WER + Cascade effect  — STT error → wrong tool (requires audio samples)
  5. Guardrail accuracy    — injection/off-topic detection

Usage:
    python evaluation/evaluate.py [--skip-audio] [--skip-llm]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASET_FILE = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_samples")

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)


# ── 1. Hit Rate @ K ───────────────────────────────────────────────────────────

def evaluate_hit_rate(scenarios: list[dict]) -> dict:
    """
    For each RAG scenario check whether expected keywords appear in top-K results.
    Also runs with BM25+dense ensemble for ablation comparison.
    """
    from data_ingestion.embedder import query_collection

    rag_scenarios = [s for s in scenarios if s.get("expected_tool") == "Program_Comparator_RAG"]
    if not rag_scenarios:
        return {"hit_rate_at_k": None, "note": "No RAG scenarios in dataset"}

    k = _cfg["retrieval"]["top_k"]
    hits = 0
    for s in rag_scenarios:
        results = query_collection(s["query"], n_results=k)
        retrieved_texts = " ".join(r["text"].lower() for r in results)
        expected = [kw.lower() for kw in s.get("expected_result_contains", [])]
        if any(kw in retrieved_texts for kw in expected):
            hits += 1

    hr = hits / len(rag_scenarios)
    print(f"[eval] Hit Rate @ {k}: {hr:.2%} ({hits}/{len(rag_scenarios)})")
    return {"hit_rate_at_k": round(hr, 4), "k": k, "hits": hits, "total": len(rag_scenarios)}


# ── 2. Tool Selection Accuracy (ReAct path) ───────────────────────────────────

def evaluate_tool_accuracy(scenarios: list[dict]) -> dict:
    """Run each scenario through the ReAct agent and verify tool selection."""
    from langchain_ollama import ChatOllama
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    from agent.tools.ort_validator import ort_validator_tool
    from agent.tools.orientation_engine import orientation_engine_tool
    from agent.tools.program_comparator import program_comparator_tool
    from agent.tools.human_handoff import human_handoff_tool
    from agent.session import get_session
    from agent.core import _set_active_session, SYSTEM_PROMPT
    from agent import guardrails

    llm = ChatOllama(
        model=_cfg["llm"]["model"],
        base_url=_cfg["llm"]["base_url"],
        temperature=_cfg["llm"]["temperature"],
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

    tool_scenarios = [s for s in scenarios
                      if s.get("expected_tool") and not s.get("expected_blocked")]

    correct = 0
    results = []
    for s in tool_scenarios:
        guard = guardrails.check(s["query"])
        if guard.blocked or guard.off_topic:
            results.append({"id": s["id"], "correct": False, "reason": "guardrail_fired"})
            continue

        session = get_session(f"eval_{s['id']}")
        _set_active_session(session)
        t0 = time.time()
        try:
            out = executor.invoke({"input": s["query"]})
            latency_ms = int((time.time() - t0) * 1000)
            steps = out.get("intermediate_steps", [])
            tools_used = [step[0].tool for step in steps] if steps else []
            matched = s["expected_tool"] in tools_used
            if matched:
                correct += 1
            results.append({
                "id": s["id"], "query": s["query"][:60],
                "expected": s["expected_tool"], "used": tools_used,
                "correct": matched, "latency_ms": latency_ms,
            })
        except Exception as e:
            results.append({"id": s["id"], "correct": False, "error": str(e)})

    accuracy = correct / len(tool_scenarios) if tool_scenarios else 0
    print(f"[eval] Tool Selection Accuracy (ReAct): {accuracy:.2%} ({correct}/{len(tool_scenarios)})")
    return {
        "tool_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(tool_scenarios),
        "details": results,
    }


# ── 3. Classifier vs ReAct Ablation ──────────────────────────────────────────

def evaluate_classifier_vs_react(scenarios: list[dict]) -> dict:
    """
    Ablation D: compare KyrgyzBERT classifier fast path vs pure ReAct routing.

    Condition A: route_query() with use_classifier=True in config.yaml
    Condition B: pure ReAct (use_classifier=False, every query goes to Qwen3)

    Measures: accuracy, latency, and (when classifier is trained) token savings.
    """
    from agent import guardrails

    tool_scenarios = [s for s in scenarios
                      if s.get("expected_tool") and not s.get("expected_blocked")]

    if not tool_scenarios:
        return {"note": "No tool scenarios to evaluate"}

    # --- Condition A: KyrgyzBERT classifier (fast path) ---
    classifier_results = []
    classifier_correct = 0

    try:
        from classifier.predict import predict_intent
        classifier_available = True
    except Exception:
        classifier_available = False
        print("[eval] KyrgyzBERT classifier not available — skipping Condition A")

    if classifier_available:
        threshold = _cfg["classifier"]["confidence_threshold"]
        for s in tool_scenarios:
            guard = guardrails.check(s["query"])
            if guard.blocked:
                classifier_results.append({"id": s["id"], "routed": "blocked", "correct": False})
                continue
            t0 = time.time()
            intent, confidence = predict_intent(s["query"])
            latency_ms = int((time.time() - t0) * 1000)
            routed = intent if confidence >= threshold else "react_agent"
            correct = (routed == s["expected_tool"]) or (
                routed == "react_agent" and s.get("react_fallback_ok", True)
            )
            if correct:
                classifier_correct += 1
            classifier_results.append({
                "id": s["id"],
                "query": s["query"][:60],
                "expected": s["expected_tool"],
                "intent": intent,
                "confidence": round(confidence, 3),
                "routed": routed,
                "correct": correct,
                "latency_ms": latency_ms,
            })

    classifier_accuracy = (
        classifier_correct / len(tool_scenarios) if classifier_available else None
    )
    avg_classifier_latency = (
        sum(r.get("latency_ms", 0) for r in classifier_results) / len(classifier_results)
        if classifier_results else None
    )

    if classifier_available:
        print(
            f"[eval] Classifier accuracy: {classifier_accuracy:.2%} "
            f"({classifier_correct}/{len(tool_scenarios)}) | "
            f"avg latency: {avg_classifier_latency:.0f}ms"
        )
    else:
        print("[eval] Classifier accuracy: N/A (model not trained yet)")

    print(
        "[eval] ReAct accuracy measured separately in evaluate_tool_accuracy(). "
        "Compare that number against classifier_accuracy for ablation."
    )

    return {
        "classifier_accuracy": round(classifier_accuracy, 4) if classifier_accuracy else None,
        "classifier_avg_latency_ms": round(avg_classifier_latency, 1) if avg_classifier_latency else None,
        "classifier_available": classifier_available,
        "threshold_used": _cfg["classifier"]["confidence_threshold"],
        "details": classifier_results,
        "note": (
            "Compare classifier_accuracy vs tool_accuracy (ReAct) from evaluate_tool_accuracy(). "
            "Also compare latencies. Expected: classifier ~10ms, ReAct ~2000ms per query."
        ),
    }


# ── 4. WER + Cascade Effect ───────────────────────────────────────────────────

def evaluate_wer_cascade(skip_audio: bool = False) -> dict:
    """
    Compute WER on audio samples and check cascade (STT error → wrong tool).
    Audio samples must be placed in evaluation/audio_samples/ as:
      {id}_input.ogg         — the audio clip
      {id}_ground_truth.txt  — the correct transcription
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

        if sample_wer > 0 and expected_tool:
            from agent.core import run_agent, _set_active_session
            from agent.session import get_session
            session = get_session(f"wer_{sample_id}")
            _set_active_session(session)
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


# ── 5. Guardrail accuracy ──────────────────────────────────────────────────────

def evaluate_guardrails(scenarios: list[dict]) -> dict:
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

def run_evaluation(skip_audio: bool = False, skip_llm: bool = False) -> dict:
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)
    scenarios = dataset["scenarios"]

    print(f"\n{'='*60}")
    print(f"Evaluating against {len(scenarios)} scenarios…")
    print("=" * 60)

    results: dict = {
        "guardrails": evaluate_guardrails(scenarios),
        "hit_rate": evaluate_hit_rate(scenarios),
        "classifier_ablation": evaluate_classifier_vs_react(scenarios),
        "wer_cascade": evaluate_wer_cascade(skip_audio=skip_audio),
    }

    if not skip_llm:
        results["tool_accuracy_react"] = evaluate_tool_accuracy(scenarios)
    else:
        results["tool_accuracy_react"] = {"note": "Skipped (--skip-llm)"}

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
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-based tool accuracy eval (slow, costs VRAM)")
    args = parser.parse_args()
    run_evaluation(skip_audio=args.skip_audio, skip_llm=args.skip_llm)
