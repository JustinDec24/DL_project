import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import argparse

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset_memes import HatefulMemesDataset
from model_multimodal import MultimodalMemeClassifier


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    texts = [b["text"] for b in batch]
    ids = [b["id"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "texts": texts,
        "ids": ids,
    }


def evaluate(model, dataloader, device, criterion):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
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
            loss = criterion(logits, labels)
            if loss.dim() > 0:
                loss = loss.mean()
            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_memes_multimodal.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    set_seed(config["experiment"]["seed"])

    clip_model = config["model"]["clip_model"]
    text_model = config["model"]["text_model"]
    dropout = config["model"]["dropout"]
    freeze_encoders = config["model"]["freeze_encoders"]
    unfreeze_last_n_layers = config["model"].get("unfreeze_last_n_layers", 0)
    use_image = config["model"]["use_image"]
    use_text = config["model"]["use_text"]
    max_length = config["model"].get("max_length", 77)

    hf_name = config["dataset"]["hf_name"]
    train_split = config["dataset"]["train_split"]
    val_split = config["dataset"]["val_split"]

    batch_size = config["training"]["batch_size"]
    lr = float(config["training"]["learning_rate"])
    encoder_lr_multiplier = config["training"].get("encoder_lr_multiplier", 1.0)
    epochs = config["training"]["epochs"]
    weight_decay = config["training"]["weight_decay"]
    warmup_ratio = config["training"].get("warmup_ratio", 0.06)
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)
    use_focal_loss = config["training"].get("use_focal_loss", False)
    focal_gamma = config["training"].get("focal_gamma", 2.0)

    results_dir = config["paths"]["results_dir"]
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Modality: image={use_image}, text={use_text}")
    print(f"Freeze encoders: {freeze_encoders}, unfreeze_last_n={unfreeze_last_n_layers}")
    print(f"Focal loss: {use_focal_loss}" + (f" (gamma={focal_gamma})" if use_focal_loss else ""))

    print("Loading datasets...")
    train_dataset = HatefulMemesDataset(hf_name, train_split, clip_model, text_model, max_length,
                                        use_image=use_image, use_text=use_text)
    val_dataset = HatefulMemesDataset(hf_name, val_split, clip_model, text_model, max_length,
                                      use_image=use_image, use_text=use_text)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # Differential LR: encoder params get lower LR than classifier head
    encoder_params = []
    if use_image and hasattr(model, "clip"):
        encoder_params += [p for p in model.clip.parameters() if p.requires_grad]
    if use_text and hasattr(model, "text_encoder"):
        encoder_params += [p for p in model.text_encoder.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    if encoder_params:
        encoder_lr = lr * encoder_lr_multiplier
        print(f"Differential LR: encoder={encoder_lr:.2e}, head={lr:.2e}")
        optimizer = AdamW([
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
        ])
    else:
        optimizer = AdamW(head_params, lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Scheduler: cosine over {total_steps} steps, {warmup_steps} warmup")

    criterion = FocalLoss(gamma=focal_gamma) if use_focal_loss else torch.nn.CrossEntropyLoss()
    best_val_auc = -1.0
    training_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(outputs["logits"], labels)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, device, criterion)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"lr={current_lr:.2e}"
        )

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1_hateful": round(val_f1, 4),
            "val_auc": round(val_auc, 4),
        })

        torch.save(
            {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "val_auc": val_auc},
            os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"),
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "val_auc": val_auc},
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
            print(f"  -> New best model saved (val_auc={val_auc:.4f})")

    with open(os.path.join(results_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print("Training finished. Best val AUC:", best_val_auc)


if __name__ == "__main__":
    main()
