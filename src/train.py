"""LoRA SFT of Qwen3-VL-8B-Instruct on ViFailback VQA, via TRL SFTTrainer.

Configuration lives in configs/train_config.yaml (and configs/deepspeed_zero3.json,
which DeepSpeed itself requires to be JSON). Launch with:

    accelerate launch --config_file configs/accelerate_zero3.yaml src/train.py \
        --config configs/train_config.yaml

or simply run scripts/launch_train.sh.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

sys.path.append(str(Path(__file__).resolve().parent))
from data import Qwen3VLCollator, ViFailbackVQADataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a single forward/backward step (no checkpoint/log writes) to sanity-check the pipeline.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by the deepspeed launcher.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    processor = AutoProcessor.from_pretrained(
        model_cfg["name_or_path"],
        min_pixels=data_cfg["min_pixels"],
        max_pixels=data_cfg["max_pixels"],
    )

    dtype = getattr(torch, model_cfg["torch_dtype"])
    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["name_or_path"],
        dtype=dtype,
        attn_implementation=model_cfg["attn_implementation"],
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
        modules_to_save=lora_cfg["modules_to_save"],
    )

    train_dataset = ViFailbackVQADataset(
        json_path=data_cfg["train_json"],
        image_root=data_cfg["image_root"],
    )

    if args.smoke_test:
        # Use samples near the median image-count (~11) rather than a random
        # draw: the dataset has a long tail up to 40 images/sample, and a
        # tail sample can OOM a 24GB GPU on its own -- see README's "Known
        # memory ceiling" note. This subset only checks that forward/backward/
        # optimizer-step mechanics work; it does not clear the tail-case risk.
        counts = [len(train_dataset[i]["image_paths"]) for i in range(len(train_dataset))]
        target = sorted(range(len(counts)), key=lambda i: abs(counts[i] - 11))[:32]
        train_dataset = torch.utils.data.Subset(train_dataset, target)

    collator = Qwen3VLCollator(processor)

    output_dir = train_cfg["output_dir"]
    save_strategy = train_cfg["save_strategy"]
    report_to = train_cfg["report_to"]
    extra_args = {}
    if args.smoke_test:
        output_dir = str(Path(train_cfg["output_dir"]).parent / "smoke_test")
        save_strategy = "no"
        report_to = []
        extra_args["max_steps"] = 1

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=1 if args.smoke_test else train_cfg["logging_steps"],
        save_strategy=save_strategy,
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        report_to=report_to,
        seed=train_cfg["seed"],
        deepspeed=cfg["deepspeed"]["config_file"],
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        **extra_args,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collator,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
    )

    trainer.train()

    if args.smoke_test:
        print("[smoke_test] forward/backward step completed successfully.")
        return

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
