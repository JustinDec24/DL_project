import os
import json
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset import HateXplainDataset
from model import TransformerClassifier


def load_config(config_path="configs/experiment.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Timestamp sub-directory created by train.py (e.g. 20260501_121012)")
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = config["model"]["backbone"]
    max_length = config["model"]["max_length"]
    dropout = config["model"]["dropout"]
    batch_size = config["training"]["batch_size"]
    class_names = config["task"]["class_names"]

    processed_dir = config["paths"]["processed_data_dir"]
    results_dir = config["paths"]["results_dir"]
    if args.run_id:
        results_dir = os.path.join(results_dir, args.run_id)

    test_file = config["data"]["test_file"]
    test_path = os.path.join(processed_dir, test_file)
    checkpoint_path = os.path.join(results_dir, "checkpoints", "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = HateXplainDataset(
        test_path,
        tokenizer_name=model_name,
        max_length=max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerClassifier(
        model_name=model_name,
        num_classes=config["task"]["num_classes"],
        dropout=dropout
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            class_logits = outputs["class_logits"]
            probs = torch.softmax(class_logits, dim=1)
            preds = torch.argmax(class_logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro-F1: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # --- Error analysis ---
    errors = []
    correct_by_class = {i: {"correct": 0, "total": 0} for i in range(len(class_names))}

    for idx, (pred, label, probs) in enumerate(zip(all_preds, all_labels, all_probs)):
        correct_by_class[label]["total"] += 1
        if pred == label:
            correct_by_class[label]["correct"] += 1
        else:
            row = test_dataset.rows[idx]
            text = row.get("text", " ".join(row.get("tokens", [])))
            errors.append({
                "idx": idx,
                "text": text,
                "true_label": class_names[label],
                "pred_label": class_names[pred],
                "model_confidence": round(max(probs), 4),
                "probs": {class_names[i]: round(p, 4) for i, p in enumerate(probs)},
                "annotator_confidence": round(row.get("label_confidence", 1.0), 4),
            })

    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        stats = correct_by_class[i]
        class_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {name}: {stats['correct']}/{stats['total']} ({class_acc:.3f})")

    # High-confidence errors (model was sure but wrong)
    high_conf_errors = [e for e in errors if e["model_confidence"] >= 0.9]
    print(f"\nTotal errors: {len(errors)} / {len(all_preds)}")
    print(f"High-confidence errors (>= 0.90): {len(high_conf_errors)}")

    # Most frequent confusion pairs
    confusion_pairs = {}
    for e in errors:
        pair = (e["true_label"], e["pred_label"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    print("\nTop confusion pairs (true -> pred):")
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1])[:5]:
        print(f"  {pair[0]} -> {pair[1]}: {count}")

    # Save error analysis
    os.makedirs(results_dir, exist_ok=True)
    error_path = os.path.join(results_dir, "error_analysis.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(errors)} misclassified examples to {error_path}")

    # Save summary metrics
    summary = {
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(f1, 4),
        "per_class_accuracy": {
            class_names[i]: round(correct_by_class[i]["correct"] / correct_by_class[i]["total"], 4)
            for i in range(len(class_names))
        },
        "total_errors": len(errors),
        "high_confidence_errors": len(high_conf_errors),
        "confusion_pairs": {f"{p[0]}->{p[1]}": c for p, c in confusion_pairs.items()},
    }
    summary_path = os.path.join(results_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved test summary to {summary_path}")


if __name__ == "__main__":
    main()
