import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from data_loader import load_gsm8k_test, load_math500_test, load_aime_test, load_amc23_test, MathSample
from math_verify import parse, verify
from pydantic import BaseModel
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type
from typing import Optional

from openai import OpenAI

def read_predictions(path):
    preds = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds



def extract_final_answer(record, content_key= "content"):
    text = record.get(content_key, "")
    if not isinstance(text, str):
        text = str(text)

    if "####" in text:
        _, tail = text.rsplit("####", 1)
        candidate = tail.strip()
        if candidate:
            return candidate

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        candidate = lines[-1]
    else:
        candidate = text.strip()

    return candidate


def normalize_answer(ans):
    if ans is None:
        return ""
    s = ans.strip()
    return s.lower()


def math_verify_equal(gold_str, pred_str):
    gold_parsed = parse(gold_str)
    pred_parsed = parse(pred_str)
    return verify(gold_parsed, pred_parsed)


def compute_accuracy(gold_samples, pred_records, content_key="content"):
    pred_by_id = {}
    for rec in pred_records:
        rid = rec.get("id", None)
        if rid is not None:
            pred_by_id[rid] = rec

    correct = 0
    total = 0

    for g in gold_samples:
        gid = getattr(g, "id", None)
        if gid is None:
            continue
        if gid not in pred_by_id:
            total += 1
            continue

        pred_rec = pred_by_id[gid]
        gold_str = g.gold_answer
        pred_str = extract_final_answer(pred_rec, content_key)

        try:
            is_correct = math_verify_equal(gold_str, pred_str)
        except Exception:
            is_correct = (normalize_answer(gold_str) == normalize_answer(pred_str))

        total += 1
        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total



def compute_avg_word_count(pred_records, content_key="content"):
    if not pred_records:
        return 0.0

    if isinstance(content_key, str):
        keys = [content_key]
    else:
        keys = list(content_key)

    lengths = []

    for rec in pred_records:
        total_text = ""
        for k in keys:
            val = rec.get(k, "")
            if val:
                total_text += " " + val

        text = total_text.strip()
        if not text:
            lengths.append(0)
        else:
            lengths.append(len(text.split()))

    return sum(lengths) / len(lengths)


class CriticScore(BaseModel):
    score: float

def build_critic_prompt(sample, student_output):
    question_text = sample.question
    gold_cot = sample.cot_solution

    system_prompt = f"""
        You are an expert math reasoning evaluator.

        You will be given:
        1. A math problem
        2. A reference (gold) chain-of-thought solution.
        3. A student's chain-of-thought solution (model output).

        Your task:
        - Compare the student's reasoning with the reference reasoning.
        - Focus on:
        - Logical correctness of each step.
        - Completeness of the reasoning (are key steps included?).
        - Coherence and mathematical soundness.
        - Similarity of reasoning structure to the reference solution (not exactly the same words, but similar logical flow).

        Please output a JSON object only in the following format:

        {{"score": <a number between 1 and 5>}}

        Where:
        - score = 5 means "excellent reasoning, correct and well aligned with the reference"
        - score = 3 means "partially correct reasoning, with some issues"
        - score = 1 means "mostly incorrect or incoherent reasoning"
    """

    prompt = f"""
        Here are the inputs:

        [Problem]
        {question_text}

        [Reference (gold) solution]
        {gold_cot}

        [Student (model) solution]
        {student_output}

        Please output the result JSON now
    """
    return system_prompt, prompt


def call_llm_critic(system_prompt, prompt, model_name, client):
    parsed = client.responses.parse(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        text_format=CriticScore,
    )
    print(parsed.output_parsed.score)
    score = float(parsed.output_parsed.score)
    score = max(1.0, min(5.0, score))
    return score

def compute_llm_critic_scores(client, gold_samples, pred_contents, model_name, max_samples=50):
    candidates: List[Tuple[MathSample, str]] = []
    for s, content in zip(gold_samples, pred_contents):
        if s.cot_solution:
            candidates.append((s, content))

    if not candidates:
        return 0.0, 0

    if max_samples > 0 and len(candidates) > max_samples:
        candidates = candidates[:max_samples]

    scores = []
    for idx, (sample, student_cot) in enumerate(candidates, start=1):
        print(f"Evaluating {idx}/{len(candidates)}")

        system_prompt, prompt = build_critic_prompt(sample, student_cot)
        score = call_llm_critic(system_prompt, prompt, model_name, client)

        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0, 0

    avg_score = sum(scores) / len(scores)
    return avg_score, len(scores)


def load_gold_for_dataset(dataset: str, data_root: str) -> List[MathSample]:
    dataset = dataset.lower()
    if dataset == "gsm8k":
        return load_gsm8k_test(data_root)
    elif dataset == "math500":
        return load_math500_test(data_root)
    elif dataset == "aime":
        return load_aime_test(data_root)
    elif dataset == "amc23":
        return load_amc23_test(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main(args):
    OPENAI_API_KEY = ""
    dataset = args.dataset
    data_root = args.data_root
    pred_path = args.pred_path
    enable_critic = args.enable_critic
    content_key = args.content_key
    critic_key = args.critic_key
    critic_model = args.critic_model
    critic_max_samples = args.critic_max_samples

    # Load gold
    gold_samples = load_gold_for_dataset(dataset, data_root)
    print(f"Loaded {len(gold_samples)} gold samples!")

    # Load predictions
    pred_records = read_predictions(Path(pred_path))
    if len(pred_records) != len(gold_samples):
        print(f"Gold size {len(gold_samples)} != Pred size {len(pred_records)}.")
    min_len = min(len(gold_samples), len(pred_records))
    gold_samples = gold_samples[:min_len]
    pred_records = pred_records[:min_len]

    # Compute Accuracy
    accuracy, correct, total = compute_accuracy(gold_samples, pred_records, content_key=content_key)
    print(f"Accuracy: {accuracy:.4f}  ({correct}/{total})")

    # Compute avg word count
    avg_len = compute_avg_word_count(pred_records, content_key=[critic_key])
    print(f"Average word count: {avg_len:.2f}")

    if args.enable_critic:
        client = OpenAI(api_key=OPENAI_API_KEY)
        pred_contents = [rec.get("thinking", "") for rec in pred_records]
        avg_score, n_scored = compute_llm_critic_scores(
            client,
            gold_samples,
            pred_contents,
            model_name=critic_model,
            max_samples=critic_max_samples,
        )
        print(f"LLM Critic avg score: {avg_score:.3f} (n={n_scored})")
    else:
        print("LLM critic disabled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="math500", choices=["gsm8k", "math500", "aime", "amc23"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--pred_path", type=str, default="outputs/output.jsonl")
    parser.add_argument("--enable_critic", action="store_true")
    parser.add_argument("--content_key", type=str, default="content")
    parser.add_argument("--critic_key", type=str, default="thinking")
    parser.add_argument("--critic_model", type=str, default="gpt-5-nano-2025-08-07")
    parser.add_argument("--critic_max_samples", type=int, default=50)

    args = parser.parse_args()
    main(args)