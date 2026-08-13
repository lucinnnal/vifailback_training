#!/usr/bin/env bash
# One optimizer step (forward+backward) through the real LoRA+DeepSpeed-ZeRO3
# pipeline, to sanity-check the setup before a full run. No checkpoints/logs
# are written to the real output_dir.
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_ENV=vifailback_train
export HF_HOME=/home/hg_models
export HF_DATASETS_CACHE=/home/dataset
GPUS="${GPUS:-4,5,6,7}"

"/home/kipyokim/.conda/envs/${CONDA_ENV}/bin/deepspeed" \
    --include "localhost:${GPUS}" \
    src/train.py --config configs/train_config.yaml --smoke_test
