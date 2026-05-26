import os
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
import io

_DUMMY_PIXELS = torch.zeros(3, 224, 224)


def _find_image_col(features):
    for name in ("img", "image"):
        if name in features:
            return name
    return None


def _get_snapshot_dir(hf_name):
    try:
        return snapshot_download(repo_id=hf_name, repo_type="dataset", local_files_only=True)
    except Exception:
        try:
            return snapshot_download(repo_id=hf_name, repo_type="dataset")
        except Exception:
            return None


def load_hf_split(hf_name, split):
    try:
        return load_dataset(hf_name, split=split)
    except Exception as e:
        raise RuntimeError(f"Could not load split '{split}' from '{hf_name}': {e}")


_GREY = None

def _grey_placeholder():
    global _GREY
    if _GREY is None:
        _GREY = Image.new("RGB", (224, 224), color=(128, 128, 128))
    return _GREY.copy()


def to_pil_rgb(img, base_dir=None):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        path = img if (base_dir is None or os.path.isabs(img)) else os.path.join(base_dir, img)
        try:
            return Image.open(path).convert("RGB")
        except FileNotFoundError:
            return _grey_placeholder()
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            p = img["path"]
            path = p if (base_dir is None or os.path.isabs(p)) else os.path.join(base_dir, p)
            try:
                return Image.open(path).convert("RGB")
            except FileNotFoundError:
                return _grey_placeholder()
    raise ValueError(f"Cannot convert {type(img)} to PIL Image")


class HatefulMemesDataset(Dataset):
    def __init__(self, hf_name, split, clip_model_name, text_model_name,
                 max_length=77, use_image=True, use_text=True):
        self.use_image = use_image
        self.use_text = use_text
        # Load raw dataset — no HFImage casting, we resolve paths ourselves
        self.examples = load_hf_split(hf_name, split)
        self._img_col = _find_image_col(self.examples.features)
        self._snapshot_dir = _get_snapshot_dir(hf_name) if use_image else None
        if use_image and self._snapshot_dir:
            print(f"Image base dir: {self._snapshot_dir}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["text"]
        label = int(ex["label"])

        if self.use_image:
            image = to_pil_rgb(ex[self._img_col], base_dir=self._snapshot_dir)
            pixel_values = self.clip_processor(
                images=image, return_tensors="pt"
            )["pixel_values"].squeeze(0)
        else:
            pixel_values = _DUMMY_PIXELS

        text_enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text,
            "id": ex.get("id", idx),
        }
