import json


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_jsonl("data/processed/train.jsonl")

    n_total = len(rows)
    n_ok = 0
    n_bad = 0

    for i, row in enumerate(rows):
        tokens = row["tokens"]
        rationale_mask = row["rationale_mask"]

        if len(tokens) == len(rationale_mask):
            n_ok += 1
        else:
            n_bad += 1
            print(f"Mismatch at example {i}: tokens={len(tokens)}, rationale_mask={len(rationale_mask)}")

    print("Total examples:", n_total)
    print("Aligned examples:", n_ok)
    print("Mismatched examples:", n_bad)

    print("\nExample:")
    sample = rows[0]
    print("TOKENS:", sample["tokens"])
    print("RATIONALE:", sample["rationale_mask"])


if __name__ == "__main__":
    main()