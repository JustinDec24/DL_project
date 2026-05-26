import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_majority_target(row):
    """Return the most-agreed-upon target group, or 'None' if no consensus."""
    targets_per_annotator = row.get("targets", [])
    if not targets_per_annotator:
        return "None"
    mention_count = defaultdict(int)
    for annotator_targets in targets_per_annotator:
        for t in annotator_targets:
            if t and t.lower() != "none":
                mention_count[t] += 1
    if not mention_count:
        return "None"
    n = len(targets_per_annotator)
    agreed = {t: c for t, c in mention_count.items() if c > n / 2}
    if agreed:
        return max(agreed, key=agreed.get)
    return max(mention_count, key=mention_count.get)


class HateXplainDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_name="roberta-base", max_length=128):
        self.rows = load_jsonl(jsonl_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # If tokens exist, use them directly for alignment
        if "tokens" in row:
            tokens = row["tokens"]
        else:
            tokens = row["text"].split()

        label_id = row["label_id"]

        label_confidence = row.get("label_confidence", 1.0)

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        group_weight = row.get("group_weight", 1.0)

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
            "label_confidence": torch.tensor(label_confidence, dtype=torch.float),
            "group_weight": torch.tensor(group_weight, dtype=torch.float),
        }

        # Optional rationale alignment
        if "rationale_mask" in row:
            word_ids = encoding.word_ids(batch_index=0)
            word_level_mask = row["rationale_mask"]

            token_level_mask = []
            for word_id in word_ids:
                if word_id is None:
                    token_level_mask.append(0)
                else:
                    token_level_mask.append(word_level_mask[word_id])

            item["token_rationale_mask"] = torch.tensor(token_level_mask, dtype=torch.float)

        return item