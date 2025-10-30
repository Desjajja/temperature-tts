"""
Math-Verify integration verifier.

Requires: pip install 'math-verify[antlr4_13_2]'

API contract:
- verify(prediction_text: str, ref_answer: Any) -> bool
  Returns True if prediction_text contains a mathematically equivalent answer to ref_answer.
"""
from typing import Any


def verify(prediction_text: str, ref_answer: Any) -> bool:
    try:
        # Lazy import so that project remains usable without the dependency
        from math_verify import parse as mv_parse, verify as mv_verify  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Math-Verify not available. Install with: pip install 'math-verify[antlr4_13_2]'"
        ) from e

    try:
        gold = mv_parse(str(ref_answer) if ref_answer is not None else "")
        pred = mv_parse(prediction_text or "")
        return bool(mv_verify(gold, pred))
    except Exception:
        # Best-effort fallback: treat failures as incorrect
        return False
