"""
Local HarMeme dataset loader (Pramanick et al. 2021).

Reads the MMF-style layout:
    <root>/
        annotations/{train,val,test}.jsonl   each row: id, image, labels[], text
        images/covid_memes_*.png

Each row's `labels` field contains one of:
    ["not harmful"]                  -> binary label 0
    ["somewhat harmful", <target>]   -> binary label 1
    ["very harmful",     <target>]   -> binary label 1
(<target> in {individual, organization, community, society}.)

We binarize "somewhat harmful" + "very harmful" into harmful=1, matching the
binary framing of the Facebook Hateful Memes benchmark. The optional
`fine_grained=True` flag instead returns a 3-class label
(not=0, somewhat=1, very=2).
"""

import os
import io
import json
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, AutoTokenizer
from PIL import Image


_DUMMY_PIXELS = torch.zeros(3, 224, 224)


LABEL_MAP_BINARY = {
    "not harmful": 0,
    "somewhat harmful": 1,
    "very harmful": 1,
}

LABEL_MAP_FINE = {
    "not harmful": 0,
    "somewhat harmful": 1,
    "very harmful": 2,
}


def _grey_placeholder():
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def to_pil_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError:
        return _grey_placeholder()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _resolve_label(label_field, label_map):
    """label_field is a list of strings, first is the harm category."""
    if isinstance(label_field, str):
        first = label_field
    elif isinstance(label_field, (list, tuple)) and label_field:
        first = label_field[0]
    else:
        raise ValueError(f"Cannot parse label {label_field!r}")
    first = first.strip().lower()
    if first not in label_map:
        raise ValueError(f"Unknown HarMeme label {first!r}")
    return label_map[first]


class HarMemeDataset(Dataset):
    def __init__(self, root_dir, split, clip_model_name, text_model_name,
                 max_length=77, use_image=True, use_text=True,
                 fine_grained=False, strict_images=True):
        """
        root_dir: path that contains annotations/ and images/ subfolders.
        split: one of "train", "val", "test".
        fine_grained: if True, returns 3-class labels (not / somewhat / very harmful).
                      if False (default), binarises to harmful vs not.
        """
        self.use_image = use_image
        self.use_text = use_text
        self.fine_grained = fine_grained
        root_dir = os.path.abspath(root_dir)
        ann_path = os.path.join(root_dir, "annotations", f"{split}.jsonl")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        self.rows = load_jsonl(ann_path)
        self.img_dir = os.path.join(root_dir, "images")
        label_map = LABEL_MAP_FINE if fine_grained else LABEL_MAP_BINARY

        # Pre-resolve labels and image paths
        for r in self.rows:
            r["_label_id"] = _resolve_label(r["labels"], label_map)
            r["_img_path"] = os.path.join(self.img_dir, r["image"])

        if use_image and strict_images:
            n_probe = min(20, len(self.rows))
            n_missing = sum(1 for r in self.rows[:n_probe] if not os.path.exists(r["_img_path"]))
            if n_missing:
                raise FileNotFoundError(
                    f"{n_missing}/{n_probe} HarMeme image paths missing under "
                    f"{self.img_dir!r}. Check root_dir."
                )

        print(f"HarMeme {split}: {len(self.rows)} rows | "
              f"label_mode={'fine' if fine_grained else 'binary'} | "
              f"img_dir={self.img_dir}")
        from collections import Counter
        dist = Counter(r["_label_id"] for r in self.rows)
        print(f"  label distribution: {dict(dist)}")

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row["text"] or ""
        label = row["_label_id"]

        if self.use_image:
            img = to_pil_rgb(row["_img_path"])
            pixel_values = self.clip_processor(
                images=img, return_tensors="pt"
            )["pixel_values"].squeeze(0)
        else:
            pixel_values = _DUMMY_PIXELS

        text_enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text,
            "id": row.get("id", idx),
        }
