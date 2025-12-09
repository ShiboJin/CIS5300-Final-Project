import json
import re
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_gold_answer(old_answer: str) -> str:
    m = re.search(r"####\s*([^\n]+)", old_answer)
    if not m:
        return ""
    return m.group(1).strip()

def strip_think_block(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()

def build_messages(question: str, old_cot: str, gold_answer: str):
    system_prompt = (
        "You are an expert math tutor.\n"
        "Given a math word problem, an existing solution, and the correct final numeric answer, "
        "rewrite the solution into a clear, concise, step-by-step chain-of-thought suitable for "
        "training a smaller student model.\n"
        "Requirements:\n"
        "1. Keep all steps logically correct and consistent with the final answer.\n"
        "2. Just answer the question with chain-of-thought, do not add any extra information or context.\n"
        "3. Use short paragraphs or bullet-style steps if helpful.\n"
        "4. On the LAST line, output the final numeric answer in the exact format '#### x', "
        "where x is the correct answer.\n"
        "5. Do NOT write anything after '#### x' on that last line.\n"
    )

    user_prompt = (
        "Problem:\n"
        f"{question}\n\n"
        "Original solution:\n"
        f"{old_cot}\n\n"
        "Correct final answer:\n"
        f"{gold_answer}\n\n"
        "Now rewrite the solution to follow the requirements above. "
        "Make it clean and easy for a smaller model to learn from."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def regenerate_cot(
    input_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen3-8B",
    batch_size: int = 4,
    max_new_tokens: int = 512,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading teacher model: {model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
    model.eval()
    torch.set_grad_enabled(False)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)

            if "question" not in ex or "answer" not in ex:
                continue
            examples.append(ex)

    print(f"Loaded {len(examples)} examples from {input_path}")

    with output_path.open("w", encoding="utf-8") as fout:
        for start in range(0, len(examples), batch_size):
            end = min(start + batch_size, len(examples))
            batch = examples[start:end]
            print(f"Regenerating CoT for {start+1} - {end} / {len(examples)}")

            texts = []
            meta = []

            for ex in batch:
                q = ex["question"]
                old_cot = ex["answer"]
                gold_ans = extract_gold_answer(old_cot)

                if gold_ans == "":
                    print("WARNING: no gold answer found, skipping example.")
                    continue

                messages = build_messages(q, old_cot, gold_ans)
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    # enable_thinking=False
                )
                texts.append(text)
                meta.append(
                    {
                        "orig_ex": ex,
                        "gold_answer": gold_ans,
                    }
                )

            if not texts:
                continue

            model_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for i in range(len(texts)):
                input_len = model_inputs.input_ids[i].shape[0]
                output_ids = gen_ids[i][input_len:]
                new_cot = tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip()

                m = re.search(r"####\s*[^\n]+", new_cot)
                if m:
                    new_cot = new_cot[: m.end()].strip()

                ex_meta = meta[i]
                orig_ex = ex_meta["orig_ex"]
                gold_ans = ex_meta["gold_answer"]

                new_record = {
                    "question": orig_ex["question"],
                    "answer": strip_think_block(new_cot),
                    "gold_answer": gold_ans,
                    "orig_answer": orig_ex.get("answer", ""),
                    "teacher_model": model_name,
                }

                fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"Saved regenerated CoT dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="teacher 模型名称")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()
    regenerate_cot(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main()