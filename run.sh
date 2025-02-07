ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo.py \
    --config recipes/qwen/Qwen2.5-1.5B-Instruct/grpo/confg_full.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo.py \
    --config recipes/qwen/Qwen2.5-7B-Instruct-1M/grpo/confg_full.yaml

lighteval vllm "pretrained=zwt963/Qwen2.5-1.5B-Instruct-Open-R1-GRPO,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8" "custom|finreasoner|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir workdir/evals/Qwen2.5-1.5B-Instruct-Open-R1-GRPO