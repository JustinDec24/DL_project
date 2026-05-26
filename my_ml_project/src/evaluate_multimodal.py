import os
import json
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from dataset_memes import HatefulMemesDataset
from model_multimodal import MultimodalMemeClassifier
from train_multimodal import collate_fn


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_memes_multimodal.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    clip_model = config["model"]["clip_model"]
    text_model = config["model"]["text_model"]
    dropout = config["model"]["dropout"]
    freeze_encoders = config["model"]["freeze_encoders"]
    unfreeze_last_n_layers = config["model"].get("unfreeze_last_n_layers", 0)
    use_image = config["model"]["use_image"]
    use_text = config["model"]["use_text"]
    max_length = config["model"].get("max_length", 77)

    hf_name = config["dataset"]["hf_name"]
    test_split = config["dataset"]["test_split"]

    batch_size = config["training"]["batch_size"]
    class_names = config["task"]["class_names"]
    results_dir = config["paths"]["results_dir"]
    checkpoint_path = os.path.join(results_dir, "checkpoints", "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading test set...")
    test_dataset = HatefulMemesDataset(hf_name, test_split, clip_model, text_model, max_length,
                                       use_image=use_image, use_text=use_text)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultimodalMemeClassifier(
        clip_model_name=clip_model,
        text_model_name=text_model,
        num_classes=2,
        dropout=dropout,
        freeze_encoders=freeze_encoders,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        use_image=use_image,
        use_text=use_text,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_auc={checkpoint['val_auc']:.4f})")

    all_preds, all_labels, all_probs, all_texts, all_ids = [], [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_texts.extend(batch["texts"])
            all_ids.extend(batch["ids"])

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nTest accuracy:  {acc:.4f}")
    print(f"Test F1 (hateful): {f1:.4f}")
    print(f"Test AUROC:     {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Error analysis
    errors = []
    correct_by_class = {i: {"correct": 0, "total": 0} for i in range(2)}
    for idx, (pred, label, prob) in enumerate(zip(all_preds, all_labels, all_probs)):
        correct_by_class[label]["total"] += 1
        if pred == label:
            correct_by_class[label]["correct"] += 1
        else:
            errors.append({
                "id": all_ids[idx],
                "text": all_texts[idx],
                "true_label": class_names[label],
                "pred_label": class_names[pred],
                "p_hateful": round(prob, 4),
            })

    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        s = correct_by_class[i]
        print(f"  {name}: {s['correct']}/{s['total']} ({s['correct']/s['total']:.3f})")

    confusion_pairs = {}
    for e in errors:
        pair = (e["true_label"], e["pred_label"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    print(f"\nTotal errors: {len(errors)} / {len(all_preds)}")
    print("Top confusion pairs:")
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1]):
        print(f"  {pair[0]} -> {pair[1]}: {count}")

    os.makedirs(results_dir, exist_ok=True)
    error_path = os.path.join(results_dir, "error_analysis.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    summary = {
        "test_accuracy": round(acc, 4),
        "test_f1_hateful": round(f1, 4),
        "test_auroc": round(auc, 4),
        "total_errors": len(errors),
        "per_class_accuracy": {
            class_names[i]: round(correct_by_class[i]["correct"] / correct_by_class[i]["total"], 4)
            for i in range(2)
        },
    }
    with open(os.path.join(results_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved error analysis to {error_path}")


if __name__ == "__main__":
    main()
