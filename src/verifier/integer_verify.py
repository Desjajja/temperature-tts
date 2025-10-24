
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

def verify(prediction_text: str, ref_answer: int) -> bool:
    got = extract_boxed_int(prediction_text)
    return (got == ref_answer)
