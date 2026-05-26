import os
import json
import yaml
from collections import Counter
from datasets import load_from_disk


def load_config(config_path="configs/experiment.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def majority_vote_with_confidence(labels):
    counts = Counter(labels)
    label_id, count = counts.most_common(1)[0]
    confidence = count / len(labels)
    return label_id, confidence, dict(counts)


def aggregate_rationales(rationales, n_tokens):
    if not rationales:
        return [0] * n_tokens

    agg = []
    for i in range(n_tokens):
        values = []
        for r in rationales:
            if i < len(r):
                values.append(r[i])
        if not values:
            agg.append(0)
        else:
            agg.append(1 if sum(values) / len(values) >= 0.5 else 0)
    return agg


def process_example(example):
    tokens = example["post_tokens"]
    text = " ".join(tokens)

    label_votes = example["annotators"]["label"]
    label_id, label_confidence, label_vote_counts = majority_vote_with_confidence(label_votes)

    rationale_mask = aggregate_rationales(example["rationales"], len(tokens))

    targets = example["annotators"].get("target", [])

    return {
        "id": example["id"],
        "text": text,
        "tokens": tokens,
        "label_id": label_id,
        "label_votes": label_votes,
        "label_confidence": label_confidence,
        "label_vote_counts": label_vote_counts,
        "rationale_mask": rationale_mask,
        "targets": targets
    }


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    config = load_config()
    raw_dir = config["paths"]["raw_data_dir"]
    processed_dir = config["paths"]["processed_data_dir"]

    os.makedirs(processed_dir, exist_ok=True)

    raw_dataset_path = os.path.join(raw_dir, "hatexplain_hf")
    dataset = load_from_disk(raw_dataset_path)

    for split_name in dataset.keys():
        processed_rows = [process_example(ex) for ex in dataset[split_name]]

        output_path = os.path.join(processed_dir, f"{split_name}.jsonl")
        save_jsonl(processed_rows, output_path)

        print(f"{split_name}: saved {len(processed_rows)} examples to {output_path}")

        label_counts = Counter(row["label_id"] for row in processed_rows)
        print(f"{split_name} label distribution: {dict(label_counts)}")

    print("Processing finished.")


if __name__ == "__main__":
    main()