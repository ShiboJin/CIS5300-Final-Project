---
library_name: peft
license: other
base_model: Qwen/Qwen2-1.5B
tags:
- base_model:adapter:Qwen/Qwen2-1.5B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: output_qwen2_1_5b_gsm8k_lora_base
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output_qwen2_1_5b_gsm8k_lora_base

This model is a fine-tuned version of [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B) on the gsm8k_train_ours dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2576

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- total_eval_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.4563        | 0.9501 | 50   | 0.4151          |
| 0.2925        | 1.8931 | 100  | 0.2771          |
| 0.2658        | 2.8361 | 150  | 0.2577          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.1