"""
tool_call_f1.py
Tool Call F1 — goes beyond "did it call the right tool?" by measuring
precision / recall / F1 per test case.

Why this matters
----------------
- Binary accuracy hides problems: if an agent calls 3 tools and only 1 is
  correct, binary accuracy marks the whole case as wrong with no nuance.
- F1 separates "missing a tool" (low recall) from "over-calling" (low precision).
- Being able to explain precision vs recall is what separates a senior
  engineer from a junior one in an interview.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ToolCallResult:
    """Evaluation result for a single test case's tool calls."""

    expected: List[str]          # e.g. ["get_weather"]
    actual: List[str]            # e.g. ["get_weather", "calculate"]
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def is_exact_match(self) -> bool:
        """True if expected and actual tool sets are identical (order-agnostic)."""
        return set(self.expected) == set(self.actual)


def score_tool_calls(expected: List[str], actual: List[str]) -> ToolCallResult:
    """
    Compute tool-call F1 for a single test case.

    Args:
        expected: Expected tool name(s) — usually one, but may be multiple.
        actual:   Tool name(s) the agent actually called.

    Returns:
        ToolCallResult containing precision / recall / f1.
    """
    expected_set = set(expected)
    actual_set = set(actual)

    tp = len(expected_set & actual_set)
    fp = len(actual_set - expected_set)
    fn = len(expected_set - actual_set)

    return ToolCallResult(
        expected=list(expected_set),
        actual=list(actual_set),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )


@dataclass
class AggregateF1:
    """Aggregated tool-call F1 across an entire dataset."""

    results: List[ToolCallResult] = field(default_factory=list)

    def add(self, r: ToolCallResult) -> None:
        self.results.append(r)

    @property
    def avg_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.precision for r in self.results) / len(self.results)

    @property
    def avg_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall for r in self.results) / len(self.results)

    @property
    def avg_f1(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.f1 for r in self.results) / len(self.results)

    @property
    def exact_match_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.is_exact_match) / len(self.results)


# --- Quick self-test (no Ollama, no API key required) ---
if __name__ == "__main__":
    # Case 1: perfect match
    r1 = score_tool_calls(["get_weather"], ["get_weather"])
    assert r1.f1 == 1.0 and r1.is_exact_match

    # Case 2: missing tool (low recall)
    r2 = score_tool_calls(["search_wikipedia"], [])
    assert r2.recall == 0.0

    # Case 3: over-calling (low precision)
    r3 = score_tool_calls(["calculate"], ["calculate", "get_weather"])
    assert r3.precision == 0.5 and r3.recall == 1.0

    # Case 4: multiple tools, all correct (order-agnostic)
    r4 = score_tool_calls(
        ["get_weather", "get_stock_price"],
        ["get_stock_price", "get_weather"],
    )
    assert r4.f1 == 1.0 and r4.is_exact_match

    agg = AggregateF1()
    for r in (r1, r2, r3, r4):
        agg.add(r)

    print(f"Avg Precision: {agg.avg_precision:.2f}")
    print(f"Avg Recall:    {agg.avg_recall:.2f}")
    print(f"Avg F1:        {agg.avg_f1:.2f}")
    print(f"Exact Match:   {agg.exact_match_rate:.2%}")
    print("✅ All self-tests passed.")