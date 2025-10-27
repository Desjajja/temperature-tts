
import argparse, json, os, re
from typing import List, Dict, Any
from .prompts import build_prompt
from .sampling import get_sampler
from .voting import TwoStageVoter

from .verifier import integer_verify as integer_v
try:
    from .verifier import sympy_verify as sympy_v
except Exception:
    sympy_v = None

def load_dataset(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def parse_temps(spec: str) -> List[float]:
    if ":" in spec:
        start, end, step = [float(x) for x in spec.split(":")]
        n = int(round((end - start) / step)) + 1
        return [round(start + i*step, 6) for i in range(n)]
    return [float(t) for t in spec.split(",")]

def get_verifier(name: str):
    name = name.lower()
    if name == "integer":
        return integer_v.verify
    elif name == "sympy":
        if sympy_v is None:
            raise RuntimeError("sympy verifier requested but sympy not available.")
        return sympy_v.verify
    else:
        raise ValueError(f"Unknown verifier: {name}")

def ensure_dirs(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--prompt_template", type=str, default="AIME")
    ap.add_argument("--temps", type=str, default="0.0:1.2:0.1")
    ap.add_argument("--samples_per_temp", type=int, default=1024)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--verifier", type=str, default="integer")
    # OpenAI-compatible endpoint options (e.g., LiteLLM proxy)
    ap.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL (e.g., LiteLLM proxy). Defaults to http://localhost:4000/chat/v1 when using OpenAI mode.")
    ap.add_argument("--ak", type=str, default=None, help="API key for the OpenAI-compatible endpoint (falls back to OPENAI_API_KEY)")
    ap.add_argument("--use_two_stage_voting", action="store_true")
    ap.add_argument("--tau_intra", type=float, default=0.8)
    ap.add_argument("--tau_cross", type=float, default=1.0)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    # Lazy import tqdm if available; fall back to no-op wrapper
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        class _NoOpTqdm:
            def __init__(self, iterable=None, total=None, desc=None, leave=True):
                self._iterable = iterable if iterable is not None else range(total or 0)
                self.total = total
                self.n = 0
            def __iter__(self):
                for x in self._iterable:
                    yield x
            def update(self, n=1):
                self.n += n
            def set_postfix_str(self, s):
                pass
            def close(self):
                pass
        def tqdm(iterable=None, total=None, desc=None, leave=True):  # type: ignore
            return _NoOpTqdm(iterable=iterable, total=total, desc=desc, leave=leave)

    temps = parse_temps(args.temps)
    # Prefer provided --ak, then env var
    api_key = args.ak if args.ak else os.environ.get("OPENAI_API_KEY")
    sampler = get_sampler(
        args.model,
        max_tokens=args.max_tokens,
        base_url=args.base_url if args.base_url else None,
        api_key=api_key,
    )
    verify_fn = get_verifier(args.verifier)
    ensure_dirs(args.output)

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} questions from {args.dataset}")
    print(f"Temperatures: {temps} | samples_per_temp={args.samples_per_temp}")
    print(f"Voting: enabled={args.use_two_stage_voting}, tau_intra={args.tau_intra}, tau_cross={args.tau_cross}")

    for ex in tqdm(dataset, total=len(dataset), desc="Questions", leave=True):
        qid = ex["id"]
        qtext = ex["question"]
        ref = ex.get("answer", None)
        prompt = build_prompt(args.prompt_template, qtext)

        voter = TwoStageVoter(temperatures=temps, tau_intra=args.tau_intra, tau_cross=args.tau_cross) if args.use_two_stage_voting else None
        per_temp_counts = {t: {"N": 0, "C": 0} for t in temps}

        log_path = os.path.join(args.output, "logs", f"{qid}.jsonl")
        total_samples = args.samples_per_temp * len(temps)
        with open(log_path, "w") as flog:
            early_exit = False
            sb = tqdm(total=total_samples, desc=f"Q {qid}", leave=False)
            try:
                for round_idx in range(args.samples_per_temp):
                    for t in temps:
                        if early_exit:
                            break
                        text = sampler.generate_one(prompt, temperature=t)
                        correct = False
                        if ref is not None:
                            try:
                                correct = bool(verify_fn(text, ref))
                            except Exception:
                                correct = False

                        rec = {"qid": qid, "temp": t, "round": round_idx, "text": text, "correct": correct}
                        flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        flog.flush()
                        per_temp_counts[t]["N"] += 1
                        if correct:
                            per_temp_counts[t]["C"] += 1

                        if voter is not None:
                            m = re.search(r"\\boxed\{(-?\d+)\}", text or "")
                            ans_str = m.group(1) if m else (text.strip().splitlines()[-1] if text.strip() else "")
                            voter.add_answer(t, ans_str)
                            early, decided, dbg = voter.step()
                            if early:
                                early_exit = True
                                flog.write(json.dumps({"qid": qid, "early_exit": True, "decided_answer": decided, "debug": dbg}, ensure_ascii=False) + "\n")
                                flog.flush()
                        sb.set_postfix_str(f"t={t}, round={round_idx}, C/N={per_temp_counts[t]['C']}/{per_temp_counts[t]['N']}")
                        sb.update(1)
            finally:
                sb.close()

            flog.write(json.dumps({"qid": qid, "per_temp": per_temp_counts, "done": True}, ensure_ascii=False) + "\n")
            flog.flush()

    print(f"Run completed. Logs in: {args.output}/logs")

if __name__ == "__main__":
    main()
