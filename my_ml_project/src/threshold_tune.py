"""
Evaluate a trained multimodal model with threshold tuning.

1) Run inference on validation set, sweep thresholds in [0.05, 0.95],
   pick the one that maximizes F1 on the hateful class.
2) Run inference on test set, report metrics at default threshold (0.5)
   and at the optimal threshold tuned on val.

Supports both late-fusion (MultimodalMemeClassifier) and cross-attention
(CrossAttentionMemeClassifier) models via --model_type.
"""

import os
import json
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

from dataset_memes import HatefulMemesDataset
from dataset_factory import build_meme_dataset
from train_multimodal import collate_fn


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_type, config, device):
    if model_type == "latefusion":
        from model_multimodal import MultimodalMemeClassifier
        return MultimodalMemeClassifier(
            clip_model_name=config["model"]["clip_model"],
            text_model_name=config["model"]["text_model"],
            num_classes=2,
            dropout=config["model"]["dropout"],
            freeze_encoders=config["model"]["freeze_encoders"],
            unfreeze_last_n_layers=config["model"].get("unfreeze_last_n_layers", 0),
            use_image=config["model"]["use_image"],
            use_text=config["model"]["use_text"],
        ).to(device)
    elif model_type == "crossattn":
        from model_crossattn import CrossAttentionMemeClassifier
        return CrossAttentionMemeClassifier(
            clip_model_name=config["model"]["clip_model"],
            text_model_name=config["model"]["text_model"],
            num_classes=2,
            dropout=config["model"]["dropout"],
            freeze_encoders=config["model"]["freeze_encoders"],
            unfreeze_last_n_layers=config["model"].get("unfreeze_last_n_layers", 0),
            n_fusion_layers=config["model"].get("n_fusion_layers", 2),
            fusion_num_heads=config["model"].get("fusion_num_heads", 8),
        ).to(device)
    raise ValueError(f"unknown model_type {model_type}")


def predict_probs(model, dataloader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probs = torch.softmax(out["logits"], dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return np.array(all_labels), np.array(all_probs)


def tune_threshold(labels, probs, grid=None):
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.01)
    best_f1, best_thr = -1.0, 0.5
    for thr in grid:
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def report_at_threshold(labels, probs, thr, class_names, name=""):
    preds = (probs >= thr).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)
    auc = roc_auc_score(labels, probs)
    print(f"\n=== {name} (threshold = {thr:.3f}) ===")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 hateful: {f1:.4f}")
    print(f"  AUROC    : {auc:.4f}")
    print("  Classification report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    print("  Confusion matrix:")
    print(confusion_matrix(labels, preds))
    return {"threshold": thr, "accuracy": float(acc), "f1_hateful": float(f1), "auroc": float(auc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--model_type",
        choices=["latefusion", "crossattn"],
        required=True,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path. Defaults to results_dir/checkpoints/best_model.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    clip_model = config["model"]["clip_model"]
    text_model = config["model"]["text_model"]
    max_length = config["model"].get("max_length", 77)
    batch_size = config["training"]["batch_size"]
    class_names = config["task"]["class_names"]
    results_dir = config["paths"]["results_dir"]
    checkpoint_path = args.checkpoint or os.path.join(results_dir, "checkpoints", "best_model.pt")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Loading model:", args.model_type)
    model = build_model(args.model_type, config, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_auc={ckpt.get('val_auc', '?'):.4f})")

    print("\nLoading splits...")
    val_ds = build_meme_dataset(config, "val")
    test_ds = build_meme_dataset(config, "test")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    print("\nRunning inference on val set...")
    val_labels, val_probs = predict_probs(model, val_loader, device)
    best_thr, best_val_f1 = tune_threshold(val_labels, val_probs)
    print(f"Best threshold on val: {best_thr:.3f} (val F1 hateful = {best_val_f1:.4f})")

    print("\nRunning inference on test set...")
    test_labels, test_probs = predict_probs(model, test_loader, device)

    test_default = report_at_threshold(
        test_labels, test_probs, 0.5, class_names, name="TEST @ default 0.5"
    )
    test_tuned = report_at_threshold(
        test_labels, test_probs, best_thr, class_names, name=f"TEST @ tuned {best_thr:.3f}"
    )

    summary = {
        "model_type": args.model_type,
        "checkpoint": checkpoint_path,
        "val_best_threshold": float(best_thr),
        "val_best_f1": float(best_val_f1),
        "test_at_0.5": test_default,
        "test_at_tuned": test_tuned,
        "delta_f1": test_tuned["f1_hateful"] - test_default["f1_hateful"],
    }
    out_path = os.path.join(results_dir, "threshold_tuned_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved threshold-tuned summary to {out_path}")
    print(f"Delta F1 (tuned - default): {summary['delta_f1']:+.4f}")


if __name__ == "__main__":
    main()
