# AI Agent Evaluation Framework

An automated evaluation pipeline for testing LangGraph-based AI agents.
This framework goes beyond binary pass/fail by measuring tool-call quality,
answer correctness (rule + LLM), and latency against a Golden Dataset.

# Why this matters

Most agent demos stop at "it works". Production agents need regression testing:
when you change a prompt, swap a model, or add a tool, you need to know whether
the agent got better or worse, and why.

This framework measures four things per test case:

```
Metric                  What it answers
----------------------  --------------------------------------------------
Tool Accuracy           Did the agent call the right tool? (binary)
Tool F1                 Is the agent missing tools or over-calling them?
Answer Quality          Keyword match AND LLM-as-judge semantic score
Latency                 How long did the agent take?
```

# Features

```
- Tool Accuracy: binary, correct tool called or not
- Tool-Call F1: separates "missing a tool" (low recall) from "over-calling" (low precision)
- Rule-Based Answer Judge: deterministic, free, runs in CI (no API key)
- LLM-as-Judge: semantic scoring via Ollama (local, free) or any LLM client
- Latency Tracking: per-query and average
- Golden Dataset: JSON-based test cases
- Regression Testing: run after every prompt or model change
- Pluggable Judges: swap rule and LLM without changing evaluation code
```

# Architecture

```
  dataset.json
       |
       v
  complete_evaluate.py  -->  HTTP POST  -->  FastAPI /invoke
       |                                            |
       |                                            v
       |                                     LangGraph Agent
       |                                            |
       |                                            v
       +--> tool_call_f1.py  +  answer_judge.py  -->  Evaluation Summary
```

# Tech Stack

```
- Python: evaluation pipeline
- FastAPI: agent API wrapper
- LangGraph: multi-agent system under test
- Ollama: local LLM for LLM-as-judge (optional)
```

# Installation

```
pip install requests ollama

# Start the Agent API (Terminal 1)
cd ../agent
uv run uvicorn main:app --reload --port 8000

# Run evaluation (Terminal 2)
cd ../agent-evaluator
python complete_evaluate.py
```

# Configuration

All configuration is via environment variables. No code changes needed to
switch judges or models.

```
Variable       Default                        Description
-------------  -----------------------------  ---------------------------
AGENT_URL      http://localhost:8000/invoke   Agent endpoint
JUDGE_KIND     llm                            "rule" or "llm"
JUDGE_MODEL    llama3.2:3b                    Ollama model for LLM judge
DATASET_PATH   dataset.json                   Path to Golden Dataset
```

Example: run with the rule-based judge only (no Ollama required):

```
JUDGE_KIND=rule python complete_evaluate.py
```

# Dataset Format

Each test case in dataset.json:

```
{
  "id": "test_001",
  "query": "What is the weather in Auckland today?",
  "expected_tool": "get_weather",
  "expected_keywords": ["Auckland", "temperature"]
}
```

# Project Structure

```
agent-evaluator/
  complete_evaluate.py   Full evaluation pipeline
  evaluate.py            v1 (binary metrics only, kept for reference)
  tool_call_f1.py        Tool-call precision / recall / F1
  answer_judge.py        Rule-based + LLM-as-judge implementations
  dataset.json           Golden Dataset
  README.md
```

# Sample Output

```
Testing: What is the weather in Auckland today?
   Tool:        OK
   Tool F1:     1.00 (P=1.00, R=1.00)
   Keywords:    OK
   Judge:       0.70 [llm] - Answer contains the city name and
                            temperature, but lacks the expected keyword.
   Latency:     10.89s

==================================================
Evaluation Summary
==================================================
Total Tests:          3
Tool Accuracy:        66.7%  (2/3)
Tool Precision:       0.67
Tool Recall:          0.67
Tool F1:              0.67
Exact Match Rate:     66.7%
Keyword Accuracy:     66.7%  (2/3)
Avg Judge Score:      0.57
Judge Pass Rate:      66.7%
Avg Latency:          11.90s
==================================================
```

# Findings

```
- Weather and Word Count queries route to the correct tool and produce
  relevant answers.

- The World Cup query falls through to a direct answer instead of calling
  search_wikipedia. The supervisor's routing needs tightening for
  factual and long-tail queries.

- Latency is high (about 12s) because the local LLM runs on CPU. Acceptable
  for offline evaluation; would need a GPU or hosted model for CI.

- LLM-as-judge produces more nuanced scores than keyword matching. For
  example, it correctly penalised an answer that was semantically fine but
  missed a required keyword, while keyword matching alone could not explain
  why.
```

# Next Steps

```
- Add more test cases (edge cases, adversarial inputs, multi-tool queries)
- Add a judge calibration step: compare LLM judge vs rule judge to detect
  judge drift
- Export results to Markdown or JSON for CI regression baselines
- Add GitHub Actions workflow (rule-based judge only, no Ollama needed)
- Add a Streamlit dashboard for visualisation
```

# License

```
MIT
```