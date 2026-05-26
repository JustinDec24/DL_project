import os
import json
import yaml
import random
from datasets import load_dataset


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    config = load_config("configs/implicit_binary.yaml")
    seed = config["experiment"]["seed"]
    random.seed(seed)

    processed_dir = config["paths"]["processed_data_dir"]

    # 1) Load HateXplain processed splits
    hx_train = load_jsonl(os.path.join(processed_dir, "train.jsonl"))
    hx_val = load_jsonl(os.path.join(processed_dir, "validation.jsonl"))
    hx_test = load_jsonl(os.path.join(processed_dir, "test.jsonl"))

    # Keep only HateXplain NORMAL examples (label_id = 1)
    hx_train_normal = [x for x in hx_train if x["label_id"] == 1]
    hx_val_normal = [x for x in hx_val if x["label_id"] == 1]
    hx_test_normal = [x for x in hx_test if x["label_id"] == 1]

    # Convert to binary label 0 = other
    def convert_hatexplain_negative(rows):
        out = []
        for x in rows:
            out.append({
                "text": x["text"],
                "label_id": 0,
                "source": "hatexplain_normal"
            })
        return out

    hx_train_normal = convert_hatexplain_negative(hx_train_normal)
    hx_val_normal = convert_hatexplain_negative(hx_val_normal)
    hx_test_normal = convert_hatexplain_negative(hx_test_normal)

    # 2) Load ImplicitHate
    ih = load_dataset("SALT-NLP/ImplicitHate")["train"]

    implicit_rows = []
    for x in ih:
        implicit_rows.append({
            "text": x["post"],
            "label_id": 1,
            "source": "implicit_hate",
            "implicit_class": x["implicit_class"],
            "extra_implicit_class": x["extra_implicit_class"]
        })

    random.shuffle(implicit_rows)

    n_total = len(implicit_rows)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    ih_train = implicit_rows[:n_train]
    ih_val = implicit_rows[n_train:n_train + n_val]
    ih_test = implicit_rows[n_train + n_val:]

    # 3) Sample same number of negative examples for balanced splits
    random.shuffle(hx_train_normal)
    random.shuffle(hx_val_normal)
    random.shuffle(hx_test_normal)

    hx_train_sample = hx_train_normal[:len(ih_train)]
    hx_val_sample = hx_val_normal[:len(ih_val)]
    hx_test_sample = hx_test_normal[:len(ih_test)]

    train_rows = ih_train + hx_train_sample
    val_rows = ih_val + hx_val_sample
    test_rows = ih_test + hx_test_sample

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    save_jsonl(train_rows, os.path.join(processed_dir, "implicit_binary_train.jsonl"))
    save_jsonl(val_rows, os.path.join(processed_dir, "implicit_binary_validation.jsonl"))
    save_jsonl(test_rows, os.path.join(processed_dir, "implicit_binary_test.jsonl"))

    print("Saved:")
    print(" -", os.path.join(processed_dir, "implicit_binary_train.jsonl"), len(train_rows))
    print(" -", os.path.join(processed_dir, "implicit_binary_validation.jsonl"), len(val_rows))
    print(" -", os.path.join(processed_dir, "implicit_binary_test.jsonl"), len(test_rows))


if __name__ == "__main__":
    main()