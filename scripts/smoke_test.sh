#!/usr/bin/env bash
# One optimizer step (forward+backward) through the real LoRA+DeepSpeed-ZeRO3
# pipeline, to sanity-check the setup before a full run. No checkpoints/logs
# are written to the real output_dir.
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_ENV=vifailback_train   # see README's "환경 설정" for how to create this
export HF_HOME=/home/hg_models
export HF_DATASETS_CACHE=/home/dataset
GPUS="${GPUS:-4,5,6,7}"

conda run -n "${CONDA_ENV}" --no-capture-output \
    deepspeed --include "localhost:${GPUS}" \
    src/train.py --config configs/train_config.yaml --smoke_test
