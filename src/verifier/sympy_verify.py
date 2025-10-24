
# Optional verifier for expression equality (requires sympy).
import re

def _extract_boxed_expr(text: str):
    m = re.search(r"\\boxed\{(.+?)\}", text or "", flags=re.DOTALL)
    return m.group(1).strip() if m else None

def verify(prediction_text: str, ref_expr: str) -> bool:
    try:
        import sympy as sp
    except Exception:
        pred = _extract_boxed_expr(prediction_text)
        return (pred is not None) and (pred == ref_expr)

    pred = _extract_boxed_expr(prediction_text)
    if pred is None:
        return False
    try:
        x = sp.symbols("x")
        return sp.simplify(sp.sympify(pred) - sp.sympify(ref_expr)) == 0
    except Exception:
        return False
