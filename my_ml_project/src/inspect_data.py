from datasets import load_from_disk
import json


def main():
    dataset = load_from_disk("data/raw/hatexplain_hf")

    sample = dataset["train"][0]

    print("Top-level keys:")
    print(sample.keys())

    print("\nFull sample:")
    print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()