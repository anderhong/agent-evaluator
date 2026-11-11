"""
answer_judge.py
Answer Quality Judge — evaluates whether an agent's final answer is
actually correct, not just whether it contains the right keywords.

Why this matters
----------------
- Keyword matching is brittle: "temperature" != "temp", "Auckland" != "Tāmaki Makaurau".
- A real LLM judge can assess *semantic* correctness, not just string overlap.
- But an LLM judge costs money and needs an API key, which breaks CI.
  So we provide two implementations behind one interface:
    1. RuleBasedJudge  — deterministic, free, runs in CI (default)
    2. LLMJudge        — plug in OpenAI/Anthropic/Ollama when you want semantic scoring

Usage
-----
    from answer_judge import get_judge

    judge = get_judge("rule")          # default, no API key needed
    result = judge.evaluate(
        query="What is the weather in Auckland?",
        answer="It is 18°C and cloudy in Auckland.",
        expected_keywords=["Auckland", "°C"],
    )
    print(result.score, result.reason)
"""

from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Result object — shared by every judge implementation
# ---------------------------------------------------------------------------
@dataclass
class JudgeResult:
    """Outcome of judging a single answer."""

    score: float          # 0.0 – 1.0 (1.0 = fully correct)
    passed: bool          # True if score >= threshold
    reason: str           # Human-readable explanation
    judge_type: str       # "rule" | "llm"


# ---------------------------------------------------------------------------
# Base interface — every judge must implement evaluate()
# ---------------------------------------------------------------------------
class BaseJudge:
    """Abstract base class for answer judges."""

    name: str = "base"

    def evaluate(
        self,
        query: str,
        answer: str,
        expected_keywords: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> JudgeResult:
        """
        Judge a single answer.

        Args:
            query:             The original user query.
            answer:            The agent's final answer.
            expected_keywords: Keywords that should appear in a correct answer.
            reference_answer:  Optional gold answer for semantic comparison.

        Returns:
            JudgeResult with score in [0.0, 1.0].
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Implementation 1: Rule-based judge (free, deterministic, CI-friendly)
# ---------------------------------------------------------------------------
class RuleBasedJudge(BaseJudge):
    """
    Scores an answer by keyword recall.

    score = (# expected keywords found) / (# expected keywords)

    This is intentionally simple. It is a *baseline* — good enough for
    regression testing, but blind to synonyms and paraphrases.
    """

    name = "rule"

    def __init__(self, pass_threshold: float = 0.8):
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        query: str,
        answer: str,
        expected_keywords: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> JudgeResult:
        if not expected_keywords:
            return JudgeResult(
                score=1.0,
                passed=True,
                reason="No expected keywords provided; defaulting to pass.",
                judge_type=self.name,
            )

        answer_lower = answer.lower()
        found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

        score = len(found) / len(expected_keywords)
        passed = score >= self.pass_threshold

        if passed:
            reason = f"Found {len(found)}/{len(expected_keywords)} keywords."
        else:
            reason = (
                f"Found {len(found)}/{len(expected_keywords)} keywords. "
                f"Missing: {missing}"
            )

        return JudgeResult(
            score=score,
            passed=passed,
            reason=reason,
            judge_type=self.name,
        )


# ---------------------------------------------------------------------------
# Implementation 2: LLM-as-judge (semantic, needs an LLM client)
# ---------------------------------------------------------------------------
class LLMJudge(BaseJudge):
    """
    Uses an LLM to score answer correctness on a 0-10 scale, normalised to [0, 1].

    This implementation is deliberately kept thin — the goal of this module
    is to define the *interface*, not to couple the evaluator to one vendor.
    Plug in OpenAI, Anthropic, Azure OpenAI, or a local Ollama model as needed.
    """

    name = "llm"

    JUDGE_PROMPT = """You are an evaluation judge. Score the answer on a scale of 0-10.

Query: {query}
Expected keywords: {keywords}
Reference answer (if any): {reference}
Agent answer: {answer}

Return ONLY a JSON object:
{{"score": <0-10 integer>, "reason": "<one sentence>"}}
"""

    def __init__(self, client=None, pass_threshold: float = 0.7):
        """
        Args:
            client: A pre-configured LLM client with a .complete(prompt) -> str method.
                    If None, this judge cannot be used (caller should fall back).
            pass_threshold: Minimum normalised score to mark as passed.
        """
        if client is None:
            raise RuntimeError(
                "LLMJudge requires an LLM client. "
                "Use get_judge('rule') for a key-free fallback."
            )
        self.client = client
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        query: str,
        answer: str,
        expected_keywords: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> JudgeResult:
        import json

        prompt = self.JUDGE_PROMPT.format(
            query=query,
            keywords=expected_keywords or [],
            reference=reference_answer or "N/A",
            answer=answer,
        )
        raw = self.client.complete(prompt)

        try:
            parsed = json.loads(raw)
            raw_score = float(parsed.get("score", 0))
            reason = parsed.get("reason", "")
        except (ValueError, TypeError):
            return JudgeResult(
                score=0.0,
                passed=False,
                reason=f"Failed to parse judge output: {raw[:100]}",
                judge_type=self.name,
            )

        score = max(0.0, min(1.0, raw_score / 10.0))
        return JudgeResult(
            score=score,
            passed=score >= self.pass_threshold,
            reason=reason,
            judge_type=self.name,
        )


# ---------------------------------------------------------------------------
# Factory — pick a judge by name, with safe fallback
# ---------------------------------------------------------------------------
def get_judge(kind: str = "rule", client=None) -> BaseJudge:
    """
    Return a judge instance.

    Args:
        kind:   "rule" (default, no API key) or "llm" (requires client).
        client: LLM client for the "llm" judge.

    Returns:
        A BaseJudge implementation.
    """
    if kind == "rule":
        return RuleBasedJudge()
    if kind == "llm":
        return LLMJudge(client=client)
    raise ValueError(f"Unknown judge kind: {kind!r}. Use 'rule' or 'llm'.")


# ---------------------------------------------------------------------------
# Aggregate helper — mirrors AggregateF1 from tool_call_f1.py
# ---------------------------------------------------------------------------
@dataclass
class AggregateJudge:
    """Averages judge scores across many test cases."""

    results: List[JudgeResult] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []

    def add(self, r: JudgeResult) -> None:
        self.results.append(r)

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)


# ---------------------------------------------------------------------------
# Ollama client adapter (optional — only needed if you want LLMJudge)
# ---------------------------------------------------------------------------
class OllamaClient:
    """
    Minimal adapter so Ollama can be used as the LLM backend for LLMJudge.

    Requires: `uv add ollama` and a running Ollama server (`ollama serve`).

    Usage:
        from answer_judge import get_judge, OllamaClient
        client = OllamaClient(model="llama3.2")
        judge = get_judge("llm", client=client)
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def complete(self, prompt: str) -> str:
        import ollama  # imported lazily so the module works without ollama installed
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


# --- Quick self-test (no API key required) ---
if __name__ == "__main__":
    judge = get_judge("rule")

    # Case 1: all keywords present -> pass
    r1 = judge.evaluate(
        query="What is the weather in Auckland?",
        answer="It is 18°C and cloudy in Auckland.",
        expected_keywords=["Auckland", "°C"],
    )
    assert r1.passed and r1.score == 1.0

    # Case 2: partial keywords -> fail (below 0.8 threshold)
    r2 = judge.evaluate(
        query="What is the weather in Auckland?",
        answer="It is cloudy.",
        expected_keywords=["Auckland", "°C"],
    )
    assert not r2.passed and r2.score == 0.0

    # Case 3: half keywords -> fail (0.5 < 0.8)
    r3 = judge.evaluate(
        query="What is the weather in Auckland?",
        answer="It is 18°C.",
        expected_keywords=["Auckland", "°C"],
    )
    assert not r3.passed and r3.score == 0.5

    # Aggregate
    agg = AggregateJudge()
    for r in (r1, r2, r3):
        agg.add(r)
    print(f"Avg Score: {agg.avg_score:.2f}")
    print(f"Pass Rate: {agg.pass_rate:.2%}")
    print("✅ All rule-based self-tests passed.")

    # Demonstrate the LLM judge safely falls back when no client is given
    try:
        get_judge("llm")
    except RuntimeError as e:
        print(f"✅ LLMJudge correctly refused to start without a client.")

    # Optional: try Ollama if it is installed and running
    try:
        #ollama_judge = get_judge("llm", client=OllamaClient(model="llama3.2"))
        ollama_judge = get_judge("llm", client=OllamaClient(model="qwen2.5-coder:7b"))
        demo = ollama_judge.evaluate(
            query="What is the weather in Auckland?",
            answer="It is 18°C and cloudy in Auckland.",
            expected_keywords=["Auckland", "°C"],
        )
        print(f"✅ Ollama judge works — score={demo.score:.2f}, reason={demo.reason!r}")
    except Exception as e:
        print(f"⚠️  Ollama judge skipped: {type(e).__name__}: {e}")