# Qwen3-VL-8B-Instruct LoRA SFT — ViFailback VQA

학습 스크립트 뼈대입니다 — **아직 실제 학습(full run)은 실행하지 않았습니다.**

## 폴더 구조

```
qwen3vl_sft/
  configs/
    train_config.yaml     # model / data / lora / training 하이퍼파라미터
    deepspeed_zero3.json  # DeepSpeed ZeRO-3 설정
  src/
    data.py                # 데이터셋 + collator (prompt 부분 라벨 마스킹, 이미지 로딩)
    train.py                # TRL SFTTrainer 진입점 (--smoke_test 로 1-step 점검 가능)
  scripts/
    launch_train.sh          # deepspeed 런처 래퍼 (전체 학습용)
    smoke_test.sh             # deepspeed 런처 래퍼 (1-step 점검용)
  requirements.txt
```

프로젝트 설정은 전부 YAML(`configs/train_config.yaml`)에 있습니다. 유일한 예외는
`configs/deepspeed_zero3.json`입니다 — DeepSpeed 자체 config 로더가 JSON을
요구하기 때문에 이 파일만은 YAML로 바꿀 수 없습니다.

## 학습 방법 요약

- `ViFailback_VQA_train.json` (52,418개 single-turn 샘플) 기준 1 epoch.
- TRL `SFTTrainer`를 이용한 LoRA SFT, rank 32, alpha 64. LLM 백본의
  attention/MLP projection에만 적용
  (`model.language_model.layers.*.{self_attn,mlp}.*_proj`).
- vision-language merger(`visual.merger`)는 LoRA가 아니라 완전히 unfreeze
  (실제 가중치, PEFT `modules_to_save`로 처리).
- vision tower 자체는 frozen 상태 유지: PEFT가 `target_modules`에도
  `modules_to_save`에도 속하지 않는 파라미터는 자동으로 freeze하므로 별도의
  freezing 코드가 필요 없습니다.
- DeepSpeed ZeRO-3, bf16.
- GPU당 batch size 1, gradient accumulation 4 (4 GPU 기준 effective batch =
  16), learning rate 1e-5, cosine 스케줄.
- `deepspeed` 런처로 4 GPU 사용.

## 데이터

- `train_json`: `/data1/dataset/ViFailback/VQA/ViFailback_VQA_train.json`
- `image_root`: `/data1/dataset/ViFailback` (`images` 필드 경로, 예:
  `annotated_data/.../0.jpg`,는 이 경로 기준 상대경로입니다).
- 각 레코드는 user/assistant 1턴짜리 대화입니다. 원본 JSON의 `<image>`
  placeholder는 제거한 뒤, Qwen3-VL chat template이 user 턴에 붙은 이미지
  개수에 맞춰 다시 삽입하므로 이미지 개수와 토큰 수가 항상 일치합니다.
- Loss는 assistant 응답 부분에서만 계산되도록 라벨을 마스킹했습니다 (user
  턴 전체와 그 안의 모든 이미지 토큰은 prompt 길이 기준으로 마스킹 — 자세한
  구현은 `src/data.py`의 `Qwen3VLCollator` 참고).

## 모델

- `/home/hg_models/Qwen3-VL-8B-Instruct` — 이 머신에 이미 로컬로 전체
  스냅샷이 존재해서 HF 다운로드/토큰이 필요 없습니다 (공유 HF 캐시로
  `Qwen/Qwen3-VL-8B-Instruct`를 resolve할 때 발생하던
  `/home/hg_models/token` 권한 에러를 피할 수 있습니다).

## 환경 설정

```bash
# qwen3_8b 환경을 clone해서 vifailback_train 생성
conda create --name vifailback_train --clone qwen3_8b -y

# 드라이버가 CUDA 12.8까지만 지원하므로 torch를 cu128 빌드로 재설치
/home/kipyokim/.conda/envs/vifailback_train/bin/pip uninstall -y torch torchvision torchaudio
/home/kipyokim/.conda/envs/vifailback_train/bin/pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 나머지 학습 패키지 설치
/home/kipyokim/.conda/envs/vifailback_train/bin/pip install trl peft deepspeed accelerate pyyaml tensorboard
```

## Smoke test

실제 LoRA + DeepSpeed ZeRO-3 파이프라인으로 optimizer step 1회(forward +
backward)만 돌려보는 점검용입니다. 체크포인트/로그는 남기지 않습니다:

```bash
GPUS=4,5,6,7 bash scripts/smoke_test.sh
```

> 참고: smoke test는 이미지 개수가 중앙값(11장) 근처인 샘플만 골라서
> 사용합니다. 전체 데이터셋에는 최대 40장짜리 샘플도 있어(30장 이상만
> 161개), 그런 샘플이 배치에 걸리면 24GB GPU에서 OOM이 날 수 있습니다.
> 본 학습 전에 이 tail-case에 대한 대응(예: flash-attention 설치, `max_pixels`
> 하향, ZeRO-3 offload 등)을 결정해야 합니다.

## 실행 (본 학습)

```bash
GPUS=0,1,2,3 bash scripts/launch_train.sh
# 또는 직접:
/home/kipyokim/.conda/envs/vifailback_train/bin/deepspeed \
    --include localhost:0,1,2,3 \
    src/train.py --config configs/train_config.yaml
```
