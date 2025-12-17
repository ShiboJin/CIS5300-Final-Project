import os
import json
import random
import argparse
from typing import List, Dict, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

# As we are using distributed processing, we need to initialize the process group
def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

# Read and write JSONL files
def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Build assessment sets from Math500 and AMC datasets
# We use approximately 40 samples from Math500
def build_assessment_from_math500(rows: List[Dict], n: int) -> List[Tuple[str, str]]:
    random.shuffle(rows)
    rows = rows[:n]
    out = []
    for r in rows:
        q = r["problem"].strip()
        a = r["answer"].strip()
        prefix = f"Q: {q}\nA:"
        target = f" #### {a}"
        out.append((prefix, target))
    return out

# We use approximately 10 samples from AMC 2023
def build_assessment_from_amc(rows: List[Dict], n: int) -> List[Tuple[str, str]]:
    random.shuffle(rows)
    rows = rows[:n]
    out = []
    for r in rows:
        q = r["question"].strip()
        a = r["answer"].strip()
        prefix = f"Q: {q}\nA:"
        target = f" #### {a}"
        out.append((prefix, target))
    return out



@torch.no_grad()
def per_example_perplexity(model, tokenizer, prefixes, targets, device, max_length):
    full_texts = [p + t for p, t in zip(prefixes, targets)]

    enc_full = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc_pref = tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc_full["input_ids"].to(device)
    attn = enc_full["attention_mask"].to(device)

    labels = input_ids.clone()
    pref_lens = enc_pref["attention_mask"].sum(dim=1).tolist()
    # Mask out prefix tokens as we only compute loss on target tokens which is the final answer part
    for i, L in enumerate(pref_lens):
        labels[i, :L] = -100

    # Output logits
    outputs = model(input_ids=input_ids, attention_mask=attn)
    logits = outputs.logits

    # Shift so that tokens < n predict n
    # Reference: Homework 4
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_attn = attn[:, 1:]

    # Use CrossEntropyLoss which combines LogSoftmax and NLLLoss to get the prediction loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    vocab_size = shift_logits.size(-1)

    token_losses = loss_fct(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
    ).reshape(shift_labels.size())

    # Mask out non-target tokens
    mask = (shift_labels != -100) & (shift_attn == 1)
    token_losses = token_losses * mask

    # Per-example NLL
    denom = mask.sum(dim=1).clamp(min=1)
    nll = token_losses.sum(dim=1) / denom
    return nll.detach().cpu()

def demo_block_from_gsm8k_sample(sample):
    q = sample["question"].strip()
    a = sample["answer"].strip()
    return f"Q: {q}\nA: {a}\n\n"


def shard_list(lst, rank, world):
    return [x for i, x in enumerate(lst) if i % world == rank]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--train_candidates", type=str, default="data/train/gsm8k_teacher_filtered_train.jsonl")
    parser.add_argument("--math500_path", type=str, default="data/test/math500_test.jsonl")
    parser.add_argument("--amc_path", type=str, default="data/test/amc23_test.jsonl")
    parser.add_argument("--n_math", type=int, default=50)
    parser.add_argument("--n_amc", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--top_frac", type=float, default=0.5)
    parser.add_argument("--output_selected", type=str, default="data/train/gsm8k_teacher_rico_selected_train.jsonl")
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--rank_out_dir", type=str, default="outputs/rico_scores")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    print(f"Rank:{rank} device:{device}")

    if rank == 0:
        os.makedirs(args.rank_out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model.eval()
    model.config.use_cache = False

    # Load candidates and shard
    candidates = read_jsonl(args.train_candidates)
    my_candidates = shard_list(candidates, rank, world)

    # Build assessment set
    math_rows = read_jsonl(args.math500_path)
    amc_rows = read_jsonl(args.amc_path)

    assessment = []
    assessment += build_assessment_from_math500(math_rows, args.n_math)
    assessment += build_assessment_from_amc(amc_rows, args.n_amc)

    base_prefixes = [p for p, _ in assessment]
    targets = [t for _, t in assessment]

    # Plain PPL
    plain_perplexity = []
    for i in range(0, len(base_prefixes), args.batch_size):
        perplexity = per_example_perplexity(model, tokenizer, base_prefixes[i:i + args.batch_size], targets[i:i + args.batch_size], device, args.max_length)
        plain_perplexity.append(perplexity)
    plain_perplexity = torch.exp(torch.cat(plain_perplexity))

    # Score candidates
    rank_scores = []
    eps = 1e-6

    # For each candidate, compute the RICO score
    for idx, cand in enumerate(my_candidates):
        demo = demo_block_from_gsm8k_sample(cand)
        prefixes_T = [demo + p for p in base_prefixes]

        nll_T = []
        for i in range(0, len(prefixes_T), args.batch_size):
            nll = per_example_perplexity(model, tokenizer, prefixes_T[i:i + args.batch_size], targets[i:i + args.batch_size], device, args.max_length)
            nll_T.append(nll)

        ppl_T = torch.exp(torch.cat(nll_T))
        score_vec = (plain_perplexity - ppl_T) / (plain_perplexity + eps)
        score = float(score_vec.mean().item())

        rank_scores.append({
            "rank": rank,
            "score": score,
            "sample": cand,
        })

    # Save per-rank scores
    out_path = os.path.join(args.rank_out_dir, f"rico_scores_rank{rank}.jsonl")
    write_jsonl(out_path, rank_scores)

    if dist.is_initialized():
        dist.barrier()

    # Merge and select
    if rank == 0:
        all_scores = []
        for rank in range(world):
            p = os.path.join(args.rank_out_dir, f"rico_scores_rank{rank}.jsonl")
            all_scores.extend(read_jsonl(p))

        all_scores.sort(key=lambda x: x["score"], reverse=True)

        k = max(1, int(len(all_scores) * args.top_frac))
        selected = [x["sample"] for x in all_scores[:k]]
        write_jsonl(args.output_selected, selected)

        print(f"Total:{len(all_scores)} selected:{k}")
        print(f"Best:{all_scores[0]['score']:.6f} "
              f"Cutoff:{all_scores[k-1]['score']:.6f}")


if __name__ == "__main__":
    main()
