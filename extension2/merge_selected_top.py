import os
import json
import glob
import argparse
from typing import Any, Dict, List


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dedup_key(sample):
    q = (sample.get("question") or "").strip()
    a = (sample.get("answer") or "").strip()
    if q:
        return q
    return json.dumps({"q": q, "a": a}, ensure_ascii=False, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rico_dir", type=str, default="outputs/rico")
    ap.add_argument("--pattern", type=str, default="rico_scores_rank*.jsonl")
    ap.add_argument("--top_frac", type=float, default=0.8)
    ap.add_argument("--out_path", type=str, default="data/train/gsm8k_teacher_rico_top80_train.jsonl")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.rico_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No files found!")

    all_rows = []
    for p in paths:
        rows = read_jsonl(p)
        all_rows.extend(rows)

    # Extract (score, sample) pairs
    scored = []
    bad = 0
    for r in all_rows:
        if "score" not in r or "sample" not in r:
            bad += 1
            continue
        try:
            s = float(r["score"])
        except Exception:
            bad += 1
            continue
        scored.append((s, r["sample"]))

    # Sort by score descending
    if not scored:
        raise RuntimeError("No valid rows with (score, sample) found.")

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top fraction
    k = max(1, int(len(scored) * args.top_frac))
    selected_samples = [sample for _, sample in scored[:k]]

    seen = set()
    deduped = []
    for smp in selected_samples:
        key = dedup_key(smp)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(smp)
    selected_samples = deduped

    write_jsonl(args.out_path, selected_samples)

    best = scored[0][0]
    cutoff = scored[k - 1][0]
    print(f"Select {len(selected_samples)} samples to {args.out_path}")


if __name__ == "__main__":
    main()
