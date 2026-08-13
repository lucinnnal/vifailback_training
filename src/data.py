"""Dataset and collator for ViFailback VQA SFT on Qwen3-VL."""

import json
import os

from PIL import Image
from torch.utils.data import Dataset


class ViFailbackVQADataset(Dataset):
    """Wraps ViFailback_VQA_train.json.

    Each record already looks like:
        {"images": [rel_path, ...],
         "messages": [{"role": "user", "content": "<image>...<image>text"},
                      {"role": "assistant", "content": "text"}]}
    with one <image> token per entry in `images`, all placed at the start
    of the user turn. This dataset only resolves image paths and strips
    the literal "<image>" placeholders; the chat template re-inserts the
    correct vision tokens when the collator renders the conversation.
    """

    def __init__(self, json_path: str, image_root: str):
        with open(json_path, "r") as f:
            all_records = json.load(f)
        self.image_root = image_root

        # As of 2026-08-12, roughly half of ViFailback_VQA_train.json's records
        # reference frame files that were never extracted under `image_root`
        # (only a subset of episodes have annotated_data/*/frames populated).
        # Skip those here rather than crashing mid-training; see README.
        self.records = [r for r in all_records if self._images_exist(r)]
        n_dropped = len(all_records) - len(self.records)
        if n_dropped:
            print(
                f"[ViFailbackVQADataset] dropped {n_dropped}/{len(all_records)} "
                f"records with missing image files under {image_root}"
            )

    def _images_exist(self, record):
        return all(os.path.exists(os.path.join(self.image_root, p)) for p in record["images"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_paths = [os.path.join(self.image_root, p) for p in record["images"]]

        user_msg, assistant_msg = record["messages"]
        user_text = user_msg["content"].replace("<image>", "").lstrip()
        assistant_text = assistant_msg["content"]

        return {
            "image_paths": image_paths,
            "user_text": user_text,
            "assistant_text": assistant_text,
        }


def build_conversation(example):
    """Turns a dataset item into Qwen3-VL chat-template messages."""
    images = [Image.open(p).convert("RGB") for p in example["image_paths"]]
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": example["user_text"]})
    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": example["assistant_text"]}]},
    ]
    return conversation


class Qwen3VLCollator:
    """Builds padded batches and masks the prompt tokens out of the loss.

    Images only occur in the user turn, so masking everything up to (and
    including) the assistant header token is equivalent to masking every
    vision token as well -- no separate image-token masking is needed.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _encode_one(self, example):
        conversation = build_conversation(example)

        full = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
        )
        prompt_only = self.processor.apply_chat_template(
            conversation[:1],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        prompt_len = prompt_only["input_ids"].shape[1]

        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"][0],
            "labels": labels,
            # marks each token as text (0) or image (1); required by Qwen3-VL
            # to compute M-RoPE position ids.
            "mm_token_type_ids": full["mm_token_type_ids"][0],
            "pixel_values": full["pixel_values"],
            "image_grid_thw": full["image_grid_thw"],
        }

    def __call__(self, examples):
        import torch

        encoded = [self._encode_one(ex) for ex in examples]
        max_len = max(e["input_ids"].shape[0] for e in encoded)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels, mm_token_type_ids = [], [], [], []
        for e in encoded:
            pad_n = max_len - e["input_ids"].shape[0]
            input_ids.append(
                torch.nn.functional.pad(e["input_ids"], (0, pad_n), value=pad_id)
            )
            attention_mask.append(
                torch.nn.functional.pad(e["attention_mask"], (0, pad_n), value=0)
            )
            labels.append(
                torch.nn.functional.pad(e["labels"], (0, pad_n), value=-100)
            )
            mm_token_type_ids.append(
                torch.nn.functional.pad(e["mm_token_type_ids"], (0, pad_n), value=0)
            )

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "mm_token_type_ids": torch.stack(mm_token_type_ids),
            "pixel_values": torch.cat([e["pixel_values"] for e in encoded], dim=0),
            "image_grid_thw": torch.cat([e["image_grid_thw"] for e in encoded], dim=0),
        }
        return batch
