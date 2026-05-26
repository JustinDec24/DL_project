"""
Target-group bias analysis for HateXplain.

For each target community (African, Arab, Women, Jewish, etc.),
computes classification metrics (macro-F1, per-class accuracy) to reveal
whether the model performs differently across targeted groups.
"""

import os
import json
import yaml
import torch
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

from dataset import HateXplainDataset
from model import TransformerClassifier


KNOWN_GROUPS = [
    "African", "Arab", "Asian", "Caucasian", "Christian",
    "Economic", "Hindu", "Hispanic", "Homosexual", "Islam",
    "Jewish", "Lgbtq", "Refugee", "Women", "Other",
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_targets_for_example(raw_row):
    """
    Return the set of target groups mentioned by at least 2 annotators
    (majority among the 3 annotators). Falls back to any mention if
    all annotators disagree.
    """
    targets_per_annotator = raw_row.get("targets", [])
    if not targets_per_annotator:
        return set()

    # Flatten and count mentions across annotators
    mention_count = defaultdict(int)
    for annotator_targets in targets_per_annotator:
        for t in annotator_targets:
            if t and t.lower() != "none":
                mention_count[t] += 1

    n_annotators = len(targets_per_annotator)
    majority_threshold = n_annotators / 2  # > half

    agreed = {t for t, c in mention_count.items() if c > majority_threshold}
    if agreed:
        return agreed
    # Fall back: at least one annotator mentioned it
    return set(mention_count.keys())


def run_inference(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["class_logits"]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_preds, all_labels, all_probs


def compute_group_metrics(preds, labels, class_names):
    if len(set(labels)) < 2:
        present = list(set(labels))
        acc = accuracy_score(labels, preds)
        return {
            "n": len(labels),
            "accuracy": round(acc, 4),
            "macro_f1": None,
            "per_class_f1": {class_names[i]: None for i in range(len(class_names))},
            "note": f"Only class(es) {present} present",
        }

    classes_present = sorted(set(labels))
    f1_per_class = f1_score(labels, preds, average=None, labels=list(range(len(class_names))), zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", labels=classes_present, zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        "n": len(labels),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {
            class_names[i]: round(float(f1_per_class[i]), 4)
            for i in range(len(class_names))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/experiment_v4b_hatebert_noweights.yaml")
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
    test_path = os.path.join(processed_dir, config["data"]["test_file"])
    checkpoint_path = os.path.join(results_dir, "checkpoints", "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load raw rows (for target info) and dataset (for model input)
    raw_rows = load_jsonl(test_path)
    test_dataset = HateXplainDataset(test_path, tokenizer_name=model_name, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerClassifier(
        model_name=model_name, num_classes=config["task"]["num_classes"], dropout=dropout
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")

    all_preds, all_labels, all_probs = run_inference(model, test_loader, device)

    # ── Overall metrics ──────────────────────────────────────────────────────
    overall = compute_group_metrics(all_preds, all_labels, class_names)
    print(f"\n{'='*60}")
    print(f"OVERALL  n={overall['n']}  acc={overall['accuracy']:.4f}  macro-F1={overall['macro_f1']:.4f}")
    print(f"{'='*60}")

    # ── Build per-example target sets ────────────────────────────────────────
    example_groups = [get_targets_for_example(row) for row in raw_rows]

    # Map group name → list of (pred, label) indices
    group_indices = defaultdict(list)
    for idx, groups in enumerate(example_groups):
        for g in groups:
            group_indices[g].append(idx)

    # ── Per-group analysis ───────────────────────────────────────────────────
    group_results = {}
    rows_for_table = []

    for group in KNOWN_GROUPS:
        idxs = group_indices.get(group, [])
        if len(idxs) < 10:
            continue  # skip groups with too few examples

        g_preds = [all_preds[i] for i in idxs]
        g_labels = [all_labels[i] for i in idxs]
        metrics = compute_group_metrics(g_preds, g_labels, class_names)
        group_results[group] = metrics
        rows_for_table.append((group, metrics))

    # Sort by macro-F1 ascending (worst first)
    rows_for_table.sort(key=lambda x: x[1]["macro_f1"] if x[1]["macro_f1"] is not None else 1.0)

    # Print table
    print(f"\n{'Group':<14} {'N':>5}  {'Acc':>6}  {'MacroF1':>8}  "
          f"{'F1-hate':>8}  {'F1-norm':>8}  {'F1-off':>8}")
    print("-" * 72)
    for group, m in rows_for_table:
        pf = m["per_class_f1"]
        f1_h = f"{pf.get('hatespeech', 0.0) or 0.0:.4f}"
        f1_n = f"{pf.get('normal', 0.0) or 0.0:.4f}"
        f1_o = f"{pf.get('offensive', 0.0) or 0.0:.4f}"
        mf1 = f"{m['macro_f1']:.4f}" if m["macro_f1"] is not None else "  N/A"
        print(f"{group:<14} {m['n']:>5}  {m['accuracy']:>6.4f}  {mf1:>8}  "
              f"{f1_h:>8}  {f1_n:>8}  {f1_o:>8}")

    # ── Label distribution per group ─────────────────────────────────────────
    print(f"\n{'Group':<14}  Label distribution (hate / normal / offensive)")
    print("-" * 55)
    for group, m in rows_for_table:
        idxs = group_indices[group]
        g_labels = [all_labels[i] for i in idxs]
        dist = {cn: g_labels.count(i) for i, cn in enumerate(class_names)}
        print(f"{group:<14}  hate={dist['hatespeech']:3d}  "
              f"norm={dist['normal']:3d}  off={dist['offensive']:3d}")

    # ── Gap analysis ─────────────────────────────────────────────────────────
    valid = [(g, m) for g, m in rows_for_table if m["macro_f1"] is not None]
    if valid:
        best_g, best_m = max(valid, key=lambda x: x[1]["macro_f1"])
        worst_g, worst_m = min(valid, key=lambda x: x[1]["macro_f1"])
        print(f"\nBest group:  {best_g} — macro-F1 = {best_m['macro_f1']:.4f}")
        print(f"Worst group: {worst_g} — macro-F1 = {worst_m['macro_f1']:.4f}")
        print(f"Gap: {best_m['macro_f1'] - worst_m['macro_f1']:.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "overall": overall,
        "per_group": group_results,
        "class_names": class_names,
    }
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "target_bias_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
