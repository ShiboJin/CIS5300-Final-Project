from datasets import load_dataset

gsm8k = load_dataset("openai/gsm8k", "main")
print(gsm8k)

math500 = load_dataset("HuggingFaceH4/MATH-500")
aime = load_dataset("HuggingFaceH4/aime_2024")
amc23 = load_dataset("math-ai/amc23")


split = gsm8k["train"].train_test_split(test_size=0.1, seed=42)
split["train"].to_json("data/gsm8k/train.jsonl", lines=True)
split["test"].to_json("data/gsm8k/dev.jsonl", lines=True)
gsm8k["test"].to_json("data/gsm8k/test.jsonl", lines=True)

math500["test"].to_json("data/math500/test.jsonl", orient="records", lines=True)
aime["train"].to_json("data/aime/test.jsonl", orient="records", lines=True)
amc23["test"].to_json("data/amc23/test.jsonl", orient="records", lines=True)

print(len(split["train"]), len(split["test"]), len(gsm8k["test"]), len(math500["test"]),
      len(aime["train"]), len(amc23["test"]))