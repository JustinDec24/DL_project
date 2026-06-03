"""
Dispatch helper that builds the right meme dataset class based on config.

config["dataset"]["type"] selects:
    "hateful_memes" (default) - HatefulMemesDataset (HF metadata + local_img_dir)
    "harmeme"                 - HarMemeDataset (local jsonl + local images)

The dispatch is a thin wrapper so train_multimodal.py / train_crossattn.py /
threshold_tune.py don't grow several branches each.
"""

import os


def build_meme_dataset(config, split):
    """
    `split` is one of "train", "val", "test". The script picks the right
    split key from the config based on the dataset type.
    """
    ds_cfg = config["dataset"]
    ds_type = ds_cfg.get("type", "hateful_memes")
    model_cfg = config["model"]
    clip_model = model_cfg["clip_model"]
    text_model = model_cfg["text_model"]
    max_length = model_cfg.get("max_length", 77)
    use_image = model_cfg.get("use_image", True)
    use_text = model_cfg.get("use_text", True)

    if ds_type == "hateful_memes":
        from dataset_memes import HatefulMemesDataset
        split_key = {"train": "train_split", "val": "val_split", "test": "test_split"}[split]
        hf_split = ds_cfg[split_key]
        return HatefulMemesDataset(
            hf_name=ds_cfg["hf_name"],
            split=hf_split,
            clip_model_name=clip_model,
            text_model_name=text_model,
            max_length=max_length,
            use_image=use_image,
            use_text=use_text,
            local_img_dir=ds_cfg.get("local_img_dir"),
            strict_images=ds_cfg.get("strict_images", True),
        )

    if ds_type == "harmeme":
        from dataset_harmeme import HarMemeDataset
        return HarMemeDataset(
            root_dir=ds_cfg["root_dir"],
            split=split,
            clip_model_name=clip_model,
            text_model_name=text_model,
            max_length=max_length,
            use_image=use_image,
            use_text=use_text,
            fine_grained=ds_cfg.get("fine_grained", False),
            strict_images=ds_cfg.get("strict_images", True),
        )

    raise ValueError(f"Unknown dataset.type={ds_type!r}")
