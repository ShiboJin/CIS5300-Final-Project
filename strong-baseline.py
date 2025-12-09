import json
from pathlib import Path
from typing import List
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from data_loader import load_gsm8k_test, load_math500_test, load_aime_test, load_amc23_test, MathSample
from peft import PeftModel

def load_dataset(name, data_root="data"):
    if name == "gsm8k":
        return load_gsm8k_test(data_root)
    elif name == "math500":
        return load_math500_test(data_root)
    elif name == "aime":
        return load_aime_test(data_root)
    elif name == "amc23":
        return load_amc23_test(data_root)
    else:
        raise ValueError(f"Unknown dataset: {name}")

FEW_SHOTS = [
    {
        "question":"In five years Sam will be 3 times as old as Drew. If Drew is currently 12 years old, how old is Sam?",
        "answer":"In five years Drew will be 12+5=<<12+5=17>>17 years old.\nIn five years Sam will be 3(17)=51 years old.\nSam is currently 51-5=<<51-5=46>>46 years old.\n#### 46"
    }
]


def build_messages(sample):
    system_prompt = (
        "You are a helpful math reasoning assistant.\n"
        "Solve math word problems step by step.\n"
        "For each problem, show your reasoning, and on the last line write the final numeric answer "
        "in the exact format '#### x', where x is the answer. "
        "Do not output anything after the number on that line.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    for ex in FEW_SHOTS:
        ex_user = (
            "Problem:\n"
            f"{ex['question']}\n"
        )
        ex_assistant = ex["answer"]
        messages.append({"role": "user", "content": ex_user})
        messages.append({"role": "assistant", "content": ex_assistant})

    current_user_prompt = (
        "Now solve the following new problem.\n\n"
        "Remember:\n"
        "- On the last line, output the final numeric answer in the format '#### x'.\n"
        "- Do NOT output anything after the '#### x' on that line.\n\n"
        f"Problem:\n{sample.question}\n"
    )

    messages.append({"role": "user", "content": current_user_prompt})

    return messages


def split_prediction(output_ids, tokenizer):
    prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    thinking_content = prediction
    content = prediction

    m = re.search(r"####\s*([^\n]+)", prediction)
    if m:
        final_ans = m.group(1).strip()
        content = f"#### {final_ans}"

    return thinking_content, content


def strong_baseline(args):
    dataset = args.dataset
    data_root = args.data_root
    device = args.device
    base_model = args.base_model
    lora_path = args.lora_path
    output_path = args.output_path
    batch_size = args.batch_size

    samples = load_dataset(dataset, data_root)
    print(f"Loaded {len(samples)} samples!")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype="auto",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_path)

    print(f"Loaded base model: {base_model}")
    print(f"Loaded LoRA adapter from: {lora_path}")
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), output_path.open("w", encoding="utf-8") as fout:

        for start in range(0, len(samples), batch_size):
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]
            print(f"Inference {start+1} - {end} / {len(samples)}")

            texts = []
            ids = []
            datasets = []

            for s in batch:
                messages = build_messages(s)
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
                ids.append(s.id)
                datasets.append(s.dataset)

            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id,
            )

            batch_outputs = []
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                new_tokens = output_ids[len(input_ids):]
                batch_outputs.append(new_tokens.unsqueeze(0))

            for i, out_ids in enumerate(batch_outputs):
                thinking, content = split_prediction(out_ids, tokenizer)

                record = {
                    "id": ids[i],
                    "dataset": datasets[i],
                    "thinking": thinking,
                    "content": content
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(record["content"])

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="math500", choices=["gsm8k", "math500", "aime", "amc23"])
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="outputs/output.jsonl")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    strong_baseline(args)