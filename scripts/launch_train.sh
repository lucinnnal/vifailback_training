#!/usr/bin/env bash
# Launch LoRA SFT of Qwen3-VL-8B-Instruct on 4 GPUs with DeepSpeed ZeRO-3.
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_ENV=vifailback_train   # torch(cu128)/transformers/trl/peft/deepspeed/accelerate, see README
export HF_HOME=/home/hg_models
export HF_DATASETS_CACHE=/home/dataset
GPUS="${GPUS:-0,1,2,3}"

"/home/kipyokim/.conda/envs/${CONDA_ENV}/bin/deepspeed" \
    --include "localhost:${GPUS}" \
    src/train.py --config configs/train_config.yaml
