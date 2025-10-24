
import argparse, json, os, glob
from math import comb

def nCk(n,k):
    if k<0 or k>n: return 0
    return comb(n,k)

def pass_at_k(N, C, K):
    if N == 0 or K == 0: return 0.0
    if C == 0: return 0.0
    return 1.0 - (nCk(N-C, K) / nCk(N, K))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="run output directory")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "logs", "*.jsonl")))
    per_q = {}
    total_correct = 0
    total = 0
    for fp in files:
        qid = os.path.splitext(os.path.basename(fp))[0]
        N = 0
        C = 0
        with open(fp, "r") as f:
            for line in f:
                rec = json.loads(line)
                if "temp" not in rec:  # footer or early-exit record
                    continue
                N += 1
                if rec.get("correct"):
                    C += 1
        per_q[qid] = {"N": N, "C": C, "Avg@N": (C / N) if N else 0.0}
        total_correct += C
        total += N

    Ks = [1, 2, 4, 8, 16, 32, 64]
    summary = {
        "per_question": per_q,
        "global": {"N": total, "C": total_correct, "Avg@N": (total_correct/total) if total else 0.0},
        "pass@K": {K: pass_at_k(total, total_correct, K) for K in Ks}
    }
    out_path = os.path.join(args.run_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
