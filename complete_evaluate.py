"""
complete_evaluate.py
Complete evaluation pipeline for the multi-agent system.

Combines:
  1. Tool Accuracy        — binary: did the agent call the right tool?
  2. Tool Precision/Recall/F1 — nuanced: missing vs over-calling
  3. Answer Accuracy      — keyword match (baseline)
  4. LLM Judge Score      — semantic correctness (Ollama-backed)
  5. Latency              — response time per query

Usage
-----
    # Terminal 1 — start the agent API
    uv run uvicorn main:app --reload --port 8000

    # Terminal 2 — run the evaluation
    python complete_evaluate.py

Environment
-----------
    AGENT_URL   — agent endpoint (default: http://localhost:8000/invoke)
    JUDGE_KIND  — "rule" or "llm" (default: "llm")
    JUDGE_MODEL — Ollama model for LLM judge (default: "llama3.2")
"""

import json
import os
import time
from typing import Dict, List

import requests

from tool_call_f1 import AggregateF1, score_tool_calls
from answer_judge import AggregateJudge, OllamaClient, get_judge


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8000/invoke")
JUDGE_KIND = os.getenv("JUDGE_KIND", "llm")
#JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.2")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.2:3b")
DATASET_PATH = os.getenv("DATASET_PATH", "dataset.json")


# ---------------------------------------------------------------------------
# Agent call
# ---------------------------------------------------------------------------
def call_real_agent(query: str) -> Dict[str, str]:
    """POST a query to the agent API and return used_tool + final_answer."""
    try:
        response = requests.post(
            AGENT_URL,
            json={"query": query},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "used_tool": data.get("used_tool", "None"),
            "final_answer": data.get("final_answer", ""),
        }
    except Exception as e:
        return {
            "used_tool": "Error",
            "final_answer": f"API Error: {e}",
        }


# ---------------------------------------------------------------------------
# Judge factory
# ---------------------------------------------------------------------------
def build_judge():
    """Build the answer judge based on environment configuration."""
    if JUDGE_KIND == "rule":
        return get_judge("rule")
    try:
        client = OllamaClient(model=JUDGE_MODEL)
        return get_judge("llm", client=client)
    except Exception as e:
        print(f"⚠️  Falling back to rule-based judge: {e}")
        return get_judge("rule")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_agent(dataset_path: str) -> None:
    print(f"📥 Loading dataset from: {dataset_path}")
    print(f"🔗 Agent endpoint:       {AGENT_URL}")
    print(f"⚖️  Judge:                {JUDGE_KIND} (model={JUDGE_MODEL})\n")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    judge = build_judge()
    agg_f1 = AggregateF1()
    agg_judge = AggregateJudge()

    total_tests = len(dataset)
    correct_tool_count = 0
    keyword_match_count = 0
    total_latency = 0.0

    for test in dataset:
        query = test["query"]
        expected_tool = test["expected_tool"]
        expected_keywords = test.get("expected_keywords", [])

        print(f"🧪 Testing: {query}")

        # --- Call agent and measure latency ---
        start = time.time()
        response = call_real_agent(query)
        latency = time.time() - start
        total_latency += latency

        used_tool = response.get("used_tool", "None")
        final_answer = response.get("final_answer", "")

        # --- 1. Tool accuracy (binary) ---
        tool_correct = used_tool == expected_tool
        if tool_correct:
            correct_tool_count += 1
            tool_status = "✅"
        else:
            tool_status = f"❌ (Used: {used_tool}, Expected: {expected_tool})"

        # --- 2. Tool F1 (nuanced) ---
        f1_result = score_tool_calls([expected_tool], [used_tool])
        agg_f1.add(f1_result)

        # --- 3. Answer keyword accuracy (baseline) ---
        answer_lower = final_answer.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        keyword_correct = found == len(expected_keywords) if expected_keywords else True
        if keyword_correct:
            keyword_match_count += 1
            answer_status = "✅"
        else:
            answer_status = f"❌ ({found}/{len(expected_keywords)} keywords)"

        # --- 4. LLM judge (semantic) ---
        judge_result = judge.evaluate(
            query=query,
            answer=final_answer,
            expected_keywords=expected_keywords,
        )
        agg_judge.add(judge_result)

        # --- Print per-case result ---
        print(f"   🔧 Tool:        {tool_status}")
        print(f"   📐 Tool F1:     {f1_result.f1:.2f} "
              f"(P={f1_result.precision:.2f}, R={f1_result.recall:.2f})")
        print(f"   💬 Keywords:    {answer_status}")
        print(f"   ⚖️  Judge:       {judge_result.score:.2f} "
              f"[{judge_result.judge_type}] — {judge_result.reason}")
        print(f"   ⏱️  Latency:     {latency:.2f}s\n")

    # --- Summary ---
    print("=" * 50)
    print("📊 Evaluation Summary")
    print("=" * 50)
    print(f"Total Tests:          {total_tests}")
    print(f"🔧 Tool Accuracy:     {correct_tool_count/total_tests*100:5.1f}%  "
          f"({correct_tool_count}/{total_tests})")
    print(f"📐 Tool Precision:    {agg_f1.avg_precision:5.2f}")
    print(f"📐 Tool Recall:       {agg_f1.avg_recall:5.2f}")
    print(f"📐 Tool F1:           {agg_f1.avg_f1:5.2f}")
    print(f"📐 Exact Match Rate:  {agg_f1.exact_match_rate:5.1%}")
    print(f"💬 Keyword Accuracy:  {keyword_match_count/total_tests*100:5.1f}%  "
          f"({keyword_match_count}/{total_tests})")
    print(f"⚖️  Avg Judge Score:   {agg_judge.avg_score:5.2f}")
    print(f"⚖️  Judge Pass Rate:   {agg_judge.pass_rate:5.1%}")
    print(f"⏱️  Avg Latency:       {total_latency/total_tests:5.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_agent(DATASET_PATH)