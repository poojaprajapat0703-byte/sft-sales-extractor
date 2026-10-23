#!/usr/bin/env bash
# Small, reproducible forgetting check with lm-evaluation-harness (GPU, Kaggle). Same settings for every model.
# pip install lm-eval
set -e
BASE=Qwen/Qwen2.5-1.5B-Instruct
COMMON="--tasks ifeval,mmlu_high_school_geography,mmlu_high_school_computer_science,mmlu_marketing --limit 100 --seed 42 --batch_size 8 --apply_chat_template"
lm_eval --model hf --model_args pretrained=$BASE,dtype=float16 $COMMON --output_path results/regression_base
lm_eval --model hf --model_args pretrained=$BASE,dtype=float16,peft=outputs/adapter_replay $COMMON --output_path results/regression_tuned_replay
lm_eval --model hf --model_args pretrained=$BASE,dtype=float16,peft=outputs/adapter_noreplay $COMMON --output_path results/regression_tuned_noreplay
# If a task name errors, run: lm_eval --tasks list | grep -i -E "ifeval|mmlu_" and adjust. Subsets are indicative, say so in your README.
