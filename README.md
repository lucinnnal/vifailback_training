# Qwen3-VL-8B-Instruct LoRA SFT — ViFailback VQA

Training script scaffold only — **no training has been run yet**.

## Layout

```
qwen3vl_sft/
  configs/
    train_config.yaml     # model / data / lora / training hyperparameters
    deepspeed_zero3.json  # DeepSpeed ZeRO-3 config
  src/
    data.py                # dataset + collator (prompt-masked labels, image loading)
    train.py                # TRL SFTTrainer entrypoint (--smoke_test for a 1-step sanity check)
  scripts/
    launch_train.sh          # deepspeed launcher wrapper (full run)
    smoke_test.sh             # deepspeed launcher wrapper (1-step sanity check)
  requirements.txt
```

All project-level configuration is in YAML (`configs/train_config.yaml`). The
one exception is `configs/deepspeed_zero3.json`: DeepSpeed's own config
loader requires JSON, so that file can't be YAML.

Launching is done with the `deepspeed` CLI directly (`deepspeed --include
localhost:0,1,2,3 src/train.py ...`) rather than `accelerate launch` with a
`distributed_type: DEEPSPEED` config file — the two set overlapping
DeepSpeed options (`gradient_accumulation_steps`, `zero_stage`,
`mixed_precision`, ...) and `transformers`' own `TrainingArguments(deepspeed=...)`
raises a hard conflict error when both are present. Passing the DeepSpeed
config once, via `SFTConfig(deepspeed=...)` in `src/train.py`, and launching
with the plain `deepspeed` CLI avoids the double-specification.

## Method summary (per request)

- 1 epoch over `ViFailback_VQA_train.json` (52,418 single-turn samples).
- LoRA SFT via TRL's `SFTTrainer`, rank 32, alpha 64, applied only to the
  LLM backbone's attention/MLP projections
  (`model.language_model.layers.*.{self_attn,mlp}.*_proj`).
- The vision-language merger (`visual.merger`) is fully unfrozen (real
  weights, via PEFT `modules_to_save`), not LoRA.
- The vision tower itself stays frozen: PEFT auto-freezes any parameter
  that is neither in `target_modules` nor `modules_to_save`, so no manual
  freezing code is needed.
- DeepSpeed ZeRO-3, bf16.
- Per-GPU batch size 1, gradient accumulation 4 (effective batch = 16
  across 4 GPUs), learning rate 1e-5, cosine schedule.
- 4 GPUs via the `deepspeed` launcher.

## Data

- `train_json`: `/data1/dataset/ViFailback/VQA/ViFailback_VQA_train.json`
- `image_root`: `/data1/dataset/ViFailback` (the `images` field paths, e.g.
  `annotated_data/.../0.jpg`, are relative to this).
- Each record is one user/assistant turn; `<image>` placeholders in the raw
  JSON are stripped and re-inserted by the Qwen3-VL chat template based on
  the images attached to the user turn, so counts always stay in sync.
- Labels are masked so loss is computed only on the assistant response
  (the whole user turn, including all image tokens, is masked out via a
  prompt-length cutoff — see `Qwen3VLCollator` in `src/data.py`).

## Model

- `/home/hg_models/Qwen3-VL-8B-Instruct` — already a full local snapshot on
  this machine, so no HF download / token is needed (avoids the
  `/home/hg_models/token` permission error seen when resolving
  `Qwen/Qwen3-VL-8B-Instruct` through the shared HF cache).

## Environment

`vifailback_train` conda env (cloned from `qwen3_8b`, then rebuilt) has
everything needed:

- `torch==2.11.0+cu128`, `torchvision`, `torchaudio` — reinstalled from the
  `cu130` clone because the host driver (570.211.01) only supports up to
  CUDA 12.8; `cu130` wheels silently ran CPU-only (`torch.cuda.is_available()
  == False`). Verified now: `cuda available: True`, 8 GPUs visible.
- `transformers==5.7.0`, `trl==1.9.2`, `peft==0.20.0`,
  `deepspeed==0.19.5`, `accelerate==1.14.0`, `pyyaml`, `tensorboard`.

`/home` is currently at ~100% disk usage (45G free) — LoRA checkpoints are
small (adapter + merger only), but confirm free space before a long run
since `configs/train_config.yaml`'s `training.output_dir` writes there.
Point it at `/data1/...` instead if that's a concern.

## Smoke test

One optimizer step (forward+backward) through the real LoRA + DeepSpeed
ZeRO-3 pipeline, no checkpoints/logs written:

```bash
GPUS=4,5,6,7 bash scripts/smoke_test.sh
```

## Launch (full run)

```bash
GPUS=0,1,2,3 bash scripts/launch_train.sh
# or directly:
/home/kipyokim/.conda/envs/vifailback_train/bin/deepspeed \
    --include localhost:0,1,2,3 \
    src/train.py --config configs/train_config.yaml
```
