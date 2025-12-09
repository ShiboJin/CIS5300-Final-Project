# How different distillation methods improves LLM reasoning capability

## 1. Overview: Improving LLM Reasoning through Teacher-CoT Filtering

This extension extends our baseline math-reasoning system by introducing a high-precision teacher CoT filtering pipeline. Our goal is to improve the quality of the distilled training data, thereby improving the reasoning ability of a small open-source LLM (**Qwen2-1.5B**) on math word problems.

---

## 2. Model used for CoT regeneration

This extension uses:

#### Student Model: **Qwen/Qwen2-1.5b**

Link: [https://huggingface.co/Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)

And this model has 1.5B parameters and supports chat template, and we load this model using [HuggingFace Transformers](https://huggingface.co/Qwen/Qwen2-1.5B).

#### Teacher Model: **Qwen/Qwen3-8b**

Link: [https://huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

And this up to date model has 8B parameters and supports deep thingking mode for template, and we load this model using [HuggingFace Transformers](https://huggingface.co/Qwen/Qwen3-8B). And we use this model to regenerate the CoT to make it more concise and clear, which is suitable for student model to learn.
Here is he command:

```
python teacher-cot.py \
    --input_path ../data/train/gsm8k_train.jsonl \
    --output_path ../data/train/gsm8k_teacher_train.jsonl \
    --model_name Qwen/Qwen3-8B \
    --batch_size 128 \
    --max_new_tokens 512
```

## 3. Filtering: Teacher-CoT Quality Filtering

We implement a filtering script (`filter-teacher-cot.py`) that removes unreliable teacher outputs using three checks:
1. **Remove Nosiy Output**: By defing some banned words, the outputs form teacher model contain those banned words always useless and contain repeated words. So we remove them to improve data quality.

2. **Filter invalid or missing reasoning**: As we defined the outputs the final numeric answer in the format:
    ```
    #### x
    ```
    where `x` is the answer. All invalid or missing reasoning with answers are filtered out to avoid confusing the expected output for fine-tuning the student model.

3. **Cut long reasoning**: As we find that too long reasoning always contain some repeatitive outputs and radom words generation, we ignore those outputs for sft.

4. **Wrong reasoning and answer**: Get rid of wrong reasoning and answer to make the sft data clear.

After these steps, we removed **2003** samples and finally get **4722/6725** data samples. Here is he command:

```
python filter-teacher-cot.py \
  --input_path ../data/train/gsm8k_teacher_train.jsonl \
  --output_path ../data/train/gsm8k_teacher_filtered_train.jsonl
```


### **4. SFT on new dataset**

We use the LLaMa-Factory to do the sft, here is the command:

```
torchrun --standalone --nproc_per_node=4 src/train.py \
    --stage sft \
    --do_train \
    --do_eval \
    --use_fast_tokenizer \
    --model_name_or_path "Qwen/Qwen2-1.5B" \
    --template qwen \
    --dataset "gsm8k_train_distilled" \
    --eval_dataset "gsm8k_dev_ours" \
    --dataset_dir "data" \
    --finetuning_type lora \
    --lora_target "q_proj,v_proj" \
    --output_dir "output_qwen2_1_5b_gsm8k_lora_distilled_epoch3" \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.05 \
    --logging_steps 10 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --per_device_eval_batch_size 8 \
    --save_steps 500 \
    --load_best_model_at_end \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --num_train_epochs 3 \
    --fp16 \
    --gradient_checkpointing \
    --report_to wandb \
    --run_name "qwen2-1.5b-distilled-gsm8k-cleaned" \
    --ddp_timeout 9000

```

### **5. Testset inference and score the result**

We still using the similar way with baseline and strong-baseline to do the inference on the testset questions and compared with each other based on scoring matrics. Detailed about scoring matrics check the scoring.md. here is the command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  strong-baseline-parallel.py \
    --dataset (dataset_name) \
    --base_model Qwen/Qwen2-1.5B \
    --lora_path strong-baseline/LLaMA-Factory/output_qwen2_1_5b_(dataset_name)_lora_distilled_epoch3 \
    --data_root data \
    --batch_size 64 \
    --output_path outputs/qwen2_1.5b_distilled_(dataset_name)_test.jsonl \
    --device cuda
```
```
cat outputs/qwen2_1.5b_distilled_(dataset_name)_test.jsonl.rank* > outputs/qwen2_1.5b_distilled_(dataset_name)_test.jsonl
```
```
python score.py \
    --dataset (dataset_name)
    --data_root data
    --pred_path outputs/qwen2_1.5b_distilled_(dataset_name)_test.jsonl
```

### **5. Result & Comparison to baseline & strong-baseline**
   Dataset  | Metric        | Baseline |    Strong-Baseline   |    Extension-1    |
| --------- | ------------- | -------- | -------------------- | ----------------- |
| gsm8k     | **Accuracy**  | 0.5512   | 0.5390               | 0.5739            |
| gsm8k     | **Word Count**| 282.73   | 319.75               | 216.42            |
| gsm8k     | **Critic**    | 2.333    | 3.11                 | 3.89              |
| math500   | **Accuracy**  | 0.1880   | 0.1980               | 0.2000            |
| math500   | **Word Count**| 234.29   | 198.96               | 157.12            |
| math500   | **Critic**    | 2.280    | 2.360                | 2.550             |
| aime      | **Accuracy**  | 0        | 0                    | 0                 |
| aime      | **Word Count**| 257.23   | 453.63               | 245.47            |
| aime      | **Critic**    | 1.067    | 1.082                | 1.071             |
| amc23     | **Accuracy**  | 0.125    | 0.1750               | 0.1500            |
| amc23     | **Word Count**| 274.23   | 376.44               | 245.45            |
| amc23     | **Critic**    | -        | -                    |                   |