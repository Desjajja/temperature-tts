
import re
BOXED_INT = re.compile(r"\\boxed\{(-?\d+)\}")

def extract_boxed_int(text: str):
    m = BOXED_INT.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _coerce_ref_to_int(ref_answer):
    """Attempt to coerce the gold answer to int.

    Accepts int or numeric string (e.g., "5"). Returns None if not coercible.
    """
    try:
        if isinstance(ref_answer, int):
            return ref_answer
        s = str(ref_answer).strip()
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
    except Exception:
        pass
    return None

def verify(prediction_text: str, ref_answer) -> bool:
    got = extract_boxed_int(prediction_text)
    ref = _coerce_ref_to_int(ref_answer)
    if got is None or ref is None:
        return False
    return got == ref
