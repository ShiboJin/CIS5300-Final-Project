# How different distillation methods improves LLM reasoning capability

## Baseline (Qwen2-1.5B)

We a simple baseline for solving math word problems using a small open-source LLM **Qwen2-1.5b**.
The model generates step-by-step reasoning and outputs the final numeric answer in the format:

```
#### x
```

where `x` is the answer.

---

### Base Model

This baseline uses:

**Qwen/Qwen2-1.5b**

Link: [https://huggingface.co/Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)

And this model has1.5B parameters and supports chat template, and we load this model using [HuggingFace Transformers](https://huggingface.co/Qwen/Qwen2-1.5B).


You can evaluate the model on one of the following datasets:

* `gsm8k`
* `math500`
* `aime`
* `amc23`

The data loaders are defined in `data_loader.py`.

To run the baseline:

```bash
python simple_baseline.py \
    --dataset math500 \
    --model_name Qwen/Qwen2-1.5b \
    --output_path outputs/qwen2_1.5b_math500.jsonl \
    --data_root data \
    --device cuda
```

### Key Arguments

| Argument        | Description               | Default                |
| --------------- | ------------------------- | ---------------------- |
| `--dataset`     | Dataset to evaluate       | `math500`              |
| `--model_name`  | Model name    | `Qwen/Qwen2-1.5b`      |
| `--data_root`   | Dataset root directory         | `data`                 |
| `--output_path` | Where to save predictions | `outputs/output.jsonl` |
| `--device`      | Inference device          | `cuda`                 |

---

### Output Format

Each prediction is saved as a JSON line:

```json
{
  "id": "math500_1",
  "dataset": "math500",
  "thinking": "full model reasoning...",
  "content": "#### 42"
}
```