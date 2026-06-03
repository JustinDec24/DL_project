"""
Ensemble + Test-Time Augmentation (TTA) evaluation on Hateful Memes.

Combines the three models trained in this run:
  - frozen baseline (Exp 22)
  - defrost late-fusion (Exp 20)
  - cross-attention (Exp 21)

Two evaluation modes are reported for each ensemble:
  1. probs at default threshold 0.5
  2. probs at threshold tuned on validation set (maximise F1 hateful)

For TTA, the best single model (cross-attention) is evaluated on:
  - original image
  - horizontal flip
  - 5 deterministic crops (4 corners + center, resized back to 224)
The TTA probabilities are averaged.

Each ensemble entry can also include TTA on its members.
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
import torchvision.transforms.functional as TF

from dataset_memes import HatefulMemesDataset
from train_multimodal import collate_fn
from threshold_tune import tune_threshold


CKPT_FROZEN = "results/memes_multimodal/checkpoints/best_model.pt"
CKPT_DEFROST = "results/memes_multimodal_defrost/checkpoints/best_model.pt"
CKPT_CROSSATTN = "results/memes_crossattn/checkpoints/best_model.pt"

CFG_FROZEN = "configs/experiment_memes_multimodal.yaml"
CFG_DEFROST = "configs/experiment_memes_defrost.yaml"
CFG_CROSSATTN = "configs/experiment_memes_crossattn.yaml"


def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def build_latefusion(cfg, device):
    from model_multimodal import MultimodalMemeClassifier
    return MultimodalMemeClassifier(
        clip_model_name=cfg["model"]["clip_model"],
        text_model_name=cfg["model"]["text_model"],
        num_classes=2,
        dropout=cfg["model"]["dropout"],
        freeze_encoders=cfg["model"]["freeze_encoders"],
        unfreeze_last_n_layers=cfg["model"].get("unfreeze_last_n_layers", 0),
        use_image=cfg["model"]["use_image"],
        use_text=cfg["model"]["use_text"],
    ).to(device)


def build_crossattn(cfg, device):
    from model_crossattn import CrossAttentionMemeClassifier
    return CrossAttentionMemeClassifier(
        clip_model_name=cfg["model"]["clip_model"],
        text_model_name=cfg["model"]["text_model"],
        num_classes=2,
        dropout=cfg["model"]["dropout"],
        freeze_encoders=cfg["model"]["freeze_encoders"],
        unfreeze_last_n_layers=cfg["model"].get("unfreeze_last_n_layers", 0),
        n_fusion_layers=cfg["model"].get("n_fusion_layers", 2),
        fusion_num_heads=cfg["model"].get("fusion_num_heads", 8),
    ).to(device)


def load_model(model_type, cfg_path, ckpt_path, device):
    cfg = load_yaml(cfg_path)
    if model_type == "latefusion":
        m = build_latefusion(cfg, device)
    elif model_type == "crossattn":
        m = build_crossattn(cfg, device)
    else:
        raise ValueError(model_type)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, cfg


def predict_probs(model, dataloader, device, tta_image=False):
    """Return (labels, probs). If tta_image, average 7 variants per example."""
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            variants = [pixel_values]
            if tta_image:
                # horizontal flip
                variants.append(torch.flip(pixel_values, dims=[3]))
                # 5 deterministic crops (corners + center), resized back to 224
                # input is 224x224. Crop 196x196 from each corner + center, resize 224.
                B, C, H, W = pixel_values.shape
                crop = 196
                positions = [
                    (0, 0), (0, W - crop),
                    (H - crop, 0), (H - crop, W - crop),
                    ((H - crop) // 2, (W - crop) // 2),
                ]
                for (top, left) in positions:
                    cropped = pixel_values[:, :, top:top + crop, left:left + crop]
                    resized = torch.nn.functional.interpolate(
                        cropped, size=(H, W), mode="bilinear", align_corners=False
                    )
                    variants.append(resized)

            probs_acc = None
            for variant in variants:
                out = model(
                    pixel_values=variant,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                p = torch.softmax(out["logits"], dim=1)[:, 1]
                probs_acc = p if probs_acc is None else probs_acc + p
            probs = probs_acc / len(variants)

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return np.array(all_labels), np.array(all_probs)


def report(labels, probs, thr, class_names, name):
    preds = (probs >= thr).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)
    auc = roc_auc_score(labels, probs)
    print(f"\n=== {name} (thr={thr:.3f}) ===")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 hateful : {f1:.4f}")
    print(f"  AUROC      : {auc:.4f}")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    print("  Confusion:")
    print(confusion_matrix(labels, preds))
    return {"name": name, "threshold": float(thr),
            "accuracy": float(acc), "f1_hateful": float(f1), "auroc": float(auc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", action="store_true",
                        help="Apply test-time augmentation (image flip + 5 crops)")
    parser.add_argument("--out", type=str, default="results/ensemble_tta_summary.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("TTA enabled:", args.tta)

    # Load shared config for dataset settings (they're identical for memes)
    cfg = load_yaml(CFG_CROSSATTN)
    hf_name = cfg["dataset"]["hf_name"]
    val_split = cfg["dataset"]["val_split"]
    test_split = cfg["dataset"]["test_split"]
    local_img_dir = cfg["dataset"].get("local_img_dir", None)
    strict_images = cfg["dataset"].get("strict_images", True)
    batch_size = cfg["training"]["batch_size"]
    class_names = cfg["task"]["class_names"]
    max_length = cfg["model"]["max_length"]
    clip_model = cfg["model"]["clip_model"]
    text_model = cfg["model"]["text_model"]

    print("Loading datasets...")
    val_ds = HatefulMemesDataset(hf_name, val_split, clip_model, text_model, max_length, True, True,
                                 local_img_dir=local_img_dir, strict_images=strict_images)
    test_ds = HatefulMemesDataset(hf_name, test_split, clip_model, text_model, max_length, True, True,
                                  local_img_dir=local_img_dir, strict_images=strict_images)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    members = [
        ("frozen", "latefusion", CFG_FROZEN, CKPT_FROZEN),
        ("defrost", "latefusion", CFG_DEFROST, CKPT_DEFROST),
        ("crossattn", "crossattn", CFG_CROSSATTN, CKPT_CROSSATTN),
    ]

    val_probs_by_member = {}
    test_probs_by_member = {}
    val_labels = test_labels = None

    for name, mtype, cfg_path, ckpt_path in members:
        print(f"\n>>> Loading {name} ({mtype}) from {ckpt_path}")
        model, _ = load_model(mtype, cfg_path, ckpt_path, device)
        print(f"  Inference on val...")
        vl, vp = predict_probs(model, val_loader, device, tta_image=args.tta)
        print(f"  Inference on test...")
        tl, tp = predict_probs(model, test_loader, device, tta_image=args.tta)
        val_probs_by_member[name] = vp
        test_probs_by_member[name] = tp
        if val_labels is None:
            val_labels, test_labels = vl, tl
        del model
        torch.cuda.empty_cache()

    # Average ensemble (uniform weights)
    val_ens_probs = np.mean(np.stack([val_probs_by_member[n] for n in val_probs_by_member]), axis=0)
    test_ens_probs = np.mean(np.stack([test_probs_by_member[n] for n in test_probs_by_member]), axis=0)

    suffix = " + TTA" if args.tta else ""
    print("\n" + "=" * 60)
    print(f" ENSEMBLE OF 3 MODELS{suffix}")
    print("=" * 60)
    best_thr, best_val_f1 = tune_threshold(val_labels, val_ens_probs)
    print(f"Best threshold on val (ensemble): {best_thr:.3f} (val F1 hateful = {best_val_f1:.4f})")
    summary = {
        "tta": bool(args.tta),
        "members": list(val_probs_by_member.keys()),
        "val_best_threshold": float(best_thr),
        "val_best_f1": float(best_val_f1),
        "test_at_0.5": report(test_labels, test_ens_probs, 0.5, class_names, f"ENSEMBLE TEST @ 0.5{suffix}"),
        "test_at_tuned": report(test_labels, test_ens_probs, best_thr, class_names, f"ENSEMBLE TEST @ tuned {best_thr:.3f}{suffix}"),
    }

    # Individual TTA / baseline reports
    summary["per_member"] = {}
    for n in val_probs_by_member:
        thr_n, _ = tune_threshold(val_labels, val_probs_by_member[n])
        summary["per_member"][n] = {
            "tuned_threshold": float(thr_n),
            "test_at_0.5": report(test_labels, test_probs_by_member[n], 0.5, class_names, f"{n} TEST @ 0.5{suffix}"),
            "test_at_tuned": report(test_labels, test_probs_by_member[n], thr_n, class_names, f"{n} TEST @ tuned {thr_n:.3f}{suffix}"),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
