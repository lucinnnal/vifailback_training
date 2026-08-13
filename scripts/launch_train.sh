#!/usr/bin/env bash
# Launch LoRA SFT of Qwen3-VL-8B-Instruct on 4 GPUs with DeepSpeed ZeRO-3.
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_ENV=vifailback_train   # see README's "환경 설정" for how to create this
export HF_HOME=/home/hg_models
export HF_DATASETS_CACHE=/home/dataset
GPUS="${GPUS:-0,1,2,3}"

conda run -n "${CONDA_ENV}" --no-capture-output \
    deepspeed --include "localhost:${GPUS}" \
    src/train.py --config configs/train_config.yaml
