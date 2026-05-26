import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
from datetime import datetime

from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

from dataset import HateXplainDataset
from model import TransformerClassifier


def load_config(config_path="configs/experiment.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_group_aware_weights(dataset, max_weight=5.0):
    """
    Assign per-example weights based on (target_group, class) inverse frequency.
    Under-represented (group, class) pairs get higher weight so the model learns
    them as well as dominant ones (e.g. 'Jewish × hatespeech' at 68%).
    Weights are clipped at max_weight and normalized to mean=1 over the dataset.
    """
    from dataset import get_majority_target
    import statistics

    group_class_counts = defaultdict(int)
    for row in dataset.rows:
        g = get_majority_target(row)
        c = row["label_id"]
        group_class_counts[(g, c)] += 1

    counts = list(group_class_counts.values())
    median_count = statistics.median(counts)

    raw_weights = []
    for row in dataset.rows:
        g = get_majority_target(row)
        c = row["label_id"]
        n = group_class_counts[(g, c)]
        w = median_count / n
        raw_weights.append(w)

    # Normalize to mean=1, then clip, then re-normalize to keep mean=1
    mean_w = sum(raw_weights) / len(raw_weights)
    normalized = [w / mean_w for w in raw_weights]
    normalized = [min(w, max_weight) for w in normalized]
    mean_w2 = sum(normalized) / len(normalized)
    normalized = [w / mean_w2 for w in normalized]

    for row, w in zip(dataset.rows, normalized):
        row["group_weight"] = w

    print(f"Group-aware weights: min={min(normalized):.3f}  "
          f"max={max(normalized):.3f}  mean={sum(normalized)/len(normalized):.3f}")
    unique_groups = sorted({get_majority_target(r) for r in dataset.rows})
    print(f"Target groups found: {unique_groups}")
    return dataset


def compute_class_weights(dataset, num_classes):
    label_counts = Counter(row["label_id"] for row in dataset.rows)
    total = sum(label_counts.values())

    weights = []
    for class_id in range(num_classes):
        count = label_counts[class_id]
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    Down-weights easy examples so training focuses on hard ones (e.g. class 2 - offensive).
    Returns per-example losses (no reduction) to stay compatible with confidence weighting.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return (1 - pt) ** self.gamma * ce_loss


def compute_rationale_loss(rationale_logits, rationale_targets, attention_mask):
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss = bce(rationale_logits, rationale_targets)

    mask = attention_mask.float()
    loss = (loss * mask).sum() / mask.sum()
    return loss


def reduce_classification_loss(loss_vector, batch, device, config):
    use_label_confidence = config["training"].get("use_label_confidence", False)
    use_group_weights = config["training"].get("use_group_weights", False)

    weight = torch.ones_like(loss_vector)
    if use_label_confidence and "label_confidence" in batch:
        weight = weight * batch["label_confidence"].to(device)
    if use_group_weights and "group_weight" in batch:
        weight = weight * batch["group_weight"].to(device)

    return (loss_vector * weight).sum() / weight.sum()


def filter_dataset_to_full_agreement(dataset, min_confidence=0.999):
    original_size = len(dataset.rows)

    dataset.rows = [
        row for row in dataset.rows
        if row.get("label_confidence", 1.0) >= min_confidence
    ]

    filtered_size = len(dataset.rows)
    print(f"Filtered training set to full agreement only: {original_size} -> {filtered_size}")
    return dataset


def evaluate(model, dataloader, device, config, class_criterion):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    use_rationales = config["experiment"].get("use_rationales", False)
    rationale_weight = config["training"].get("rationale_loss_weight", 0.0)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            class_logits = outputs["class_logits"]

            class_loss_vector = class_criterion(class_logits, labels)
            loss = reduce_classification_loss(class_loss_vector, batch, device, config)

            if use_rationales and "token_rationale_mask" in batch:
                rationale_targets = batch["token_rationale_mask"].to(device)
                rationale_logits = outputs["rationale_logits"]
                rationale_loss = compute_rationale_loss(
                    rationale_logits,
                    rationale_targets,
                    attention_mask
                )
                loss = loss + rationale_weight * rationale_loss

            preds = torch.argmax(class_logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Optional run identifier appended to results_dir. "
                             "Defaults to a timestamp so reruns never overwrite each other.")
    args = parser.parse_args()

    config = load_config(args.config)

    seed = config["experiment"]["seed"]
    set_seed(seed)

    model_name = config["model"]["backbone"]
    max_length = config["model"]["max_length"]
    dropout = config["model"]["dropout"]

    batch_size = config["training"]["batch_size"]
    lr = float(config["training"]["learning_rate"])
    epochs = config["training"]["epochs"]
    weight_decay = config["training"]["weight_decay"]
    use_class_weights = config["training"].get("use_class_weights", False)
    use_focal_loss = config["training"].get("use_focal_loss", False)
    focal_gamma = config["training"].get("focal_gamma", 2.0)
    warmup_ratio = config["training"].get("warmup_ratio", 0.0)
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)

    use_group_weights = config["training"].get("use_group_weights", False)
    use_rationales = config["experiment"].get("use_rationales", False)
    rationale_weight = config["training"].get("rationale_loss_weight", 0.0)
    filter_train_to_full_agreement = config["training"].get("filter_train_to_full_agreement", False)

    processed_dir = config["paths"]["processed_data_dir"]
    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(config["paths"]["results_dir"], run_id)
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Run ID: {run_id} — saving to {results_dir}")

    train_file = config["data"]["train_file"]
    val_file = config["data"]["validation_file"]

    train_path = os.path.join(processed_dir, train_file)
    val_path = os.path.join(processed_dir, val_file)

    train_dataset = HateXplainDataset(
        train_path,
        tokenizer_name=model_name,
        max_length=max_length
    )
    val_dataset = HateXplainDataset(
        val_path,
        tokenizer_name=model_name,
        max_length=max_length
    )

    if filter_train_to_full_agreement:
        train_dataset = filter_dataset_to_full_agreement(train_dataset)

    if use_group_weights:
        print("Computing group-aware sample weights...")
        compute_group_aware_weights(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Using rationales:", use_rationales)
    print("Using focal loss:", use_focal_loss, f"(gamma={focal_gamma})" if use_focal_loss else "")
    print("Warmup ratio:", warmup_ratio)
    print("Max grad norm:", max_grad_norm)
    print("Filter train to full agreement:", filter_train_to_full_agreement)
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))

    model = TransformerClassifier(
        model_name=model_name,
        num_classes=config["task"]["num_classes"],
        dropout=dropout
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Scheduler: cosine decay over {total_steps} steps, {warmup_steps} warmup steps")

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(
            train_dataset,
            config["task"]["num_classes"]
        ).to(device)
        print("Class weights:", class_weights)

    if use_focal_loss:
        class_criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
        print(f"Loss: FocalLoss(gamma={focal_gamma})")
    else:
        class_criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        print("Loss: CrossEntropyLoss")

    best_val_f1 = -1.0
    training_log = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            class_logits = outputs["class_logits"]

            class_loss_vector = class_criterion(class_logits, labels)
            loss = reduce_classification_loss(class_loss_vector, batch, device, config)

            if use_rationales and "token_rationale_mask" in batch:
                rationale_targets = batch["token_rationale_mask"].to(device)
                rationale_logits = outputs["rationale_logits"]
                rationale_loss = compute_rationale_loss(
                    rationale_logits,
                    rationale_targets,
                    attention_mask
                )
                loss = loss + rationale_weight * rationale_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            device,
            config,
            class_criterion
        )
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "lr": current_lr,
        })

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            },
            checkpoint_path
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                },
                best_model_path
            )
            print(f"  -> New best model saved (val_f1={val_f1:.4f})")

    log_path = os.path.join(results_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print("Training finished.")
    print("Best validation F1:", best_val_f1)


if __name__ == "__main__":
    main()
