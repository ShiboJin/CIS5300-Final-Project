## Qwen2-1.5B LoRA Strong Baseline

- **Run command**: `bash sft/LLaMA-Factory/train_qwen.sh`


| Parameter | Value | Description |
| --- | --- | --- |
| `model_name_or_path` | `/Models/Qwen2-1.5B` | Pretrained base model checkpoint |
| `output_dir` | `/Models/qwen2-1.5b_lora_gsm8k` | Where LoRA adapters/checkpoints are saved |
| `dataset` | `gsm8k_train` | Local dataset entry (registered in `data/dataset_info.json`) |
| `template` | `qwen` | Chat template applied during formatting |
| `finetuning_type` | `lora` | LoRA-based supervised fine-tuning |
| `lora_target` | `q_proj,v_proj` | Attach LoRA adapters to query/value projections only |
| `per_device_train_batch_size` | `64` | Micro-batch size per GPU |
| `gradient_accumulation_steps` | `1` | Effective batch multiplier |
| `cutoff_len` | `4096` | Maximum sequence length after tokenization |
| `num_train_epochs` | `3` | Training epochs |
| `learning_rate` | `5e-5` | Peak learning rate |
| `weight_decay` | `0.1` | L2 regularization |
| `warmup_steps` | `100` | Linear warmup steps |
| `lr_scheduler_type` | `cosine` | Learning-rate schedule |
| `save_steps` | `1000` | Checkpoint interval |
| `logging_steps` | `1` | Log frequency |
| `bf16` | `True` | Mixed-precision training in bfloat16 |
| `use_fast_tokenizer` | `True` | Enable HF fast tokenizer |
| `ddp_timeout` | `9000` | Distributed timeout (seconds) |


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