# How different distillation methods improves LLM reasoning capability

## Extension-2 (RICO): RICO-based Data Selection for Distillation

## 1. Overview: Improving Generalization via RICO Data Selection

Inspired by the paper ["RICO: Refined In-Context Contribution for Automatic Instruction-Tuning Data Selection"](https://arxiv.org/abs/2505.05327). This extension improves our distillation pipeline by introducing **RICO-style contribution scoring** to **select high-impact training samples** from the GSM8K teacher-distilled dataset. Noted that this paper does not have the open source code, so we implement and add our new understanding to get this implementation.

**Motivation:**
Even after teacher-CoT filtering (Extension-1), the dataset may still contain many “low-value” samples (too easy / redundant / unhelpful patterns). RICO aims to measure **how much a training sample helps** the model reduce loss on an **out-of-domain assessment set** (e.g., MATH500 + AMC23). By selecting only high-contribution samples for SFT, we hope to improve **cross-dataset reasoning generalization while reduce the dataset size**.

## 2. Models used in RICO scoring

This extension uses:
#### Student/Scoring Model: **Qwen/Qwen2-1.5B**

Link: [https://huggingface.co/Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)

We use Qwen2-1.5B to compute **perplexity (PPL)** on the assessment set under different prompts.
This is used to estimate each training sample’s “contribution score”.


## 3. Assessment Set Construction (Small & Fast)

We build a small assessment set from our **out-of-domain test sets**:

* `data/test/math500_test.jsonl` (sample `n_math`, we set it 40)
* `data/test/amc23_test.jsonl` (sample `n_amc`, we set it 10)

Format used for scoring (final answer with `####`):

```
Q: <question>
A: #### <answer>
```

This is intentionally small to keep the selection process fast.


## 4. RICO Scoring: Contribution-Based Selection

We compute each candidate training sample’s contribution score by measuring how much the sample reduces loss on the assessment set.

### 4.1 Candidate training set

We score samples from:

* `data/train/gsm8k_teacher_filtered_train.jsonl`

Each candidate is converted into a one-shot demonstration block:

```
Q: <gsm8k question>
A: <teacher CoT + #### answer>
```

### 4.2 RICO score definition (loss-based)

For each candidate demo `T_i`, we evaluate the assessment loss when prepending it:

* `PPL_plain`: perplexity on assessment without any demo
* `PPL_T`: perplexity on assessment with candidate demo prepended

We compute the score:

```text
ΔPPL = (PPL_plain - PPL_T) / (PPL_plain + eps)
```

* If `ΔPPL is high`: demo helps (reduces perplexity)
* If `ΔPPL is low`: demo hurts (increases perplexity)

We save a score for each training sample and then select the top fraction.


## 5. Run command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  extension2/rico_select.py \
    --model_name Qwen/Qwen2-1.5B \
    --train_candidates data/train/gsm8k_teacher_filtered_train.jsonl \
    --math500_path data/test/math500_test.jsonl \
    --amc_path data/test/amc23_test.jsonl \
    --n_math 40 \
    --n_amc 10 \
    --batch_size 4 \
    --max_length 2048 \
    --rank_out_dir outputs/rico_scores
```

## 6. Select Top 80% and Build New Training Dataset

After RICO scoring finishes, we merge rank files and keep the top fraction (80% and 50%) to build a new training set:

```bash
python extension2/select_top_rico.py \
  --rico_dir outputs/rico \
  --top_frac 0.8 \
  --output_path data/train/gsm8k_train_rico.jsonl
```


## 7. SFT & test on RICO-selected dataset

We fine-tune Qwen2-1.5B with LoRA using LLaMA-Factory:

```bash
torchrun --standalone --nproc_per_node=4 src/train.py \
  --stage sft \
  --do_train \
  --do_eval \
  --use_fast_tokenizer \
  --model_name_or_path "Qwen/Qwen2-1.5B" \
  --template qwen \
  --dataset "gsm8k_train_rico" \
  --eval_dataset "gsm8k_dev_ours" \
  --dataset_dir "data" \
  --finetuning_type lora \
  --lora_target "q_proj,v_proj" \
  --output_dir "output_qwen2_1_5b_gsm8k_lora" \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len 2048 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_steps 100 \
  --weight_decay 0.1 \
  --logging_steps 1 \
  --eval_strategy "steps" \
  --eval_steps 10 \
  --per_device_eval_batch_size 4 \
  --save_steps 1000 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_loss" \
  --greater_is_better false \
  --plot_loss \
  --num_train_epochs 3 \
  --fp16 \
  --gradient_checkpointing \
  --report_to wandb \
  --run_name "qwen2-1.5b-rico-gsm8k" \
  --ddp_timeout 9000
```