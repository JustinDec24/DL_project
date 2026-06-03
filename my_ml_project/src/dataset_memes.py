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
                 max_length=77, use_image=True, use_text=True,
                 local_img_dir=None, strict_images=True):
        self.use_image = use_image
        self.use_text = use_text
        self.strict_images = strict_images
        # Load raw dataset — no HFImage casting, we resolve paths ourselves
        self.examples = load_hf_split(hf_name, split)
        self._img_col = _find_image_col(self.examples.features)
        if use_image:
            if local_img_dir:
                self._img_base = os.path.abspath(local_img_dir)
                print(f"Image base dir (local override): {self._img_base}")
            else:
                self._img_base = _get_snapshot_dir(hf_name)
                print(f"Image base dir (HF snapshot): {self._img_base}")
            self._verify_image_loading()
        else:
            self._img_base = None
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_length = max_length

    def _verify_image_loading(self):
        """Sanity check: probe the first 20 examples and fail loudly if
        most images can't be resolved. Prevents silently training on the
        grey placeholder image."""
        n_probe = min(20, len(self.examples))
        n_missing = 0
        for i in range(n_probe):
            img_field = self.examples[i][self._img_col]
            if isinstance(img_field, str):
                rel = img_field
                base = self._img_base or ""
                full = rel if os.path.isabs(rel) else os.path.join(base, rel)
                if not os.path.exists(full):
                    n_missing += 1
        if n_missing > 0:
            msg = (
                f"WARNING: {n_missing}/{n_probe} sampled image paths do not "
                f"exist under {self._img_base!r}. Training/inference will use "
                f"grey placeholder images, making the model effectively text-only. "
                f"Set local_img_dir in the dataset config to point at the actual "
                f"Hateful Memes image folder (containing img/*.png)."
            )
            if self.strict_images:
                raise FileNotFoundError(msg)
            else:
                print(msg)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["text"]
        label = int(ex["label"])

        if self.use_image:
            image = to_pil_rgb(ex[self._img_col], base_dir=self._img_base)
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
