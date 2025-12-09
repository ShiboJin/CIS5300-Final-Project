import json
import re
import argparse
from pathlib import Path


def extract_teacher_cot(sample) -> str:
    ans = sample.get("answer", "")
    m = re.search(r"(.*)####", ans, re.S)
    if not m:
        return ans.strip()
    return m.group(1).strip()


def extract_final_answer_text(ans: str) -> str:
    m = re.search(r"####\s*([^\n]+)", ans)
    if not m:
        return ""
    return m.group(1).strip()


def normalize_number_string(s: str) -> str:
    if not s:
        return ""

    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")

    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return ""

    num_str = m.group(0)

    if "." not in num_str:
        num_str = str(int(num_str))
    else:
        num_str = str(float(num_str))

    return num_str


BANNED_PHRASES = [
    "the user wants", "rewrite", "original solution",
    "the requirements", "let me", "let's", "okay", "ok,",
    "i need to", "i should", "as an ai", "structure",
    "break it down", "the problem is about",
    "present this", "guess", "hmm"
]


def is_low_quality_cot(cot: str) -> bool:
    cot_lower = cot.lower().strip()

    for p in BANNED_PHRASES:
        if p in cot_lower:
            return True

    if len(cot_lower.split()) > 250:
        return True

    return False


def filter_teacher_cot_only(input_path: str, output_path: str):

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    removed_low_quality = 0
    removed_empty_answer = 0
    removed_no_hash = 0
    removed_answer_mismatch = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            total += 1

            full_answer = ex.get("answer", "")
            if not full_answer or not full_answer.strip():
                removed_empty_answer += 1
                continue

            teacher_cot = extract_teacher_cot(ex)

            if is_low_quality_cot(teacher_cot):
                removed_low_quality += 1
                continue  # drop

            gold_raw = ex.get("gold_answer", "")
            gold_raw = gold_raw.strip() if isinstance(gold_raw, str) else str(gold_raw)

            if gold_raw:
                gold_norm = normalize_number_string(gold_raw)
                final_text = extract_final_answer_text(full_answer)

                if not final_text:
                    removed_no_hash += 1
                    continue

                pred_norm = normalize_number_string(final_text)
                if gold_norm and pred_norm and gold_norm != pred_norm:
                    removed_answer_mismatch += 1
                    continue

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Total samples: {total}")
    print(f"Removed useless CoT: {removed_low_quality}")
    print(f"Removed empty answer: {removed_empty_answer + removed_no_hash}")
    print(f"Removed incorrect final answer: {removed_answer_mismatch}")
    print(f"Remaining samples: {kept}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    filter_teacher_cot_only(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()