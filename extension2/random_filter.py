import json
import random
import argparse
from typing import List, Dict


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to original dataset (jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save randomly dropped dataset")
    parser.add_argument("--keep_frac", type=float, default=0.8,
                        help="Fraction of data to keep (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    assert 0 < args.keep_frac <= 1.0

    random.seed(args.seed)

    data = read_jsonl(args.input)
    n_total = len(data)
    n_keep = int(n_total * args.keep_frac)

    indices = list(range(n_total))
    random.shuffle(indices)

    keep_indices = set(indices[:n_keep])
    kept_data = [data[i] for i in range(n_total) if i in keep_indices]

    write_jsonl(args.output, kept_data)

    print(f"Original size: {n_total}")
    print(f"Kept size: {len(kept_data)} ({args.keep_frac*100:.1f}%)")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
