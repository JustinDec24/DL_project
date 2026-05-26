from dataset import HateXplainDataset


def main():
    ds = HateXplainDataset(
        "data/processed/train.jsonl",
        tokenizer_name="roberta-base",
        max_length=64
    )

    sample = ds[0]

    print("Keys:", sample.keys())
    print("input_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("labels:", sample["labels"])

    if "token_rationale_mask" in sample:
        print("token_rationale_mask shape:", sample["token_rationale_mask"].shape)
        print("token_rationale_mask:", sample["token_rationale_mask"].tolist())
    else:
        print("No token_rationale_mask found.")


if __name__ == "__main__":
    main()