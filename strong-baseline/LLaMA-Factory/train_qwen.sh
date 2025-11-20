export NPROC_PER_NODE=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

export MODEL_PATH=/Models/Qwen2-1.5B
export OUTPUT_PATH=/Models/qwen2-1.5b_lora_gsm8k

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --model_name_or_path $MODEL_PATH \
    --dataset gsm8k_train \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --ddp_timeout 9000 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16