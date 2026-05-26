"""
Live demo for the hate speech detection models.

Two tabs in the browser:
  - Text: HateXplain 3-class (hatespeech / normal / offensive), HateBERT (Exp 14)
  - Meme: Hateful Memes binary (not_hateful / hateful), CLIP + HateBERT (Exp 17)

Run:
    pip install gradio
    python src/demo.py \\
        --text_checkpoint results/v4b_hatebert_noweights/<run_id>/checkpoints/best_model.pt \\
        --meme_checkpoint results/memes_multimodal/checkpoints/best_model.pt

If only one checkpoint is provided, the other tab is disabled with a notice.
Use --share to expose a temporary public URL (useful when projecting from another machine).
"""

import os
import argparse
import torch
import gradio as gr
from transformers import AutoTokenizer, CLIPProcessor

from model import TransformerClassifier
from model_multimodal import MultimodalMemeClassifier


HATEXPLAIN_CLASSES = ["hatespeech", "normal", "offensive"]
MEME_CLASSES = ["not_hateful", "hateful"]

TEXT_MODEL_NAME = "GroNLP/hateBERT"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def load_text_model(checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, None
    model = TransformerClassifier(
        model_name=TEXT_MODEL_NAME, num_classes=3, dropout=0.1
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    return model, tokenizer


def load_meme_model(checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, None, None
    model = MultimodalMemeClassifier(
        clip_model_name=CLIP_MODEL_NAME,
        text_model_name=TEXT_MODEL_NAME,
        num_classes=2,
        dropout=0.1,
        freeze_encoders=True,
        unfreeze_last_n_layers=0,
        use_image=True,
        use_text=True,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    return model, clip_processor, tokenizer


def predict_text(text, model, tokenizer, device):
    if not text or not text.strip():
        return {c: 0.0 for c in HATEXPLAIN_CLASSES}
    enc = tokenizer(
        text, max_length=128, truncation=True, padding=True, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(out["class_logits"], dim=1).squeeze(0).cpu().tolist()
    return {c: float(p) for c, p in zip(HATEXPLAIN_CLASSES, probs)}


def predict_meme(image, text, model, clip_processor, tokenizer, device):
    if image is None and (not text or not text.strip()):
        return {c: 0.0 for c in MEME_CLASSES}

    if image is not None:
        pixel_values = clip_processor(
            images=image.convert("RGB"), return_tensors="pt"
        )["pixel_values"].to(device)
    else:
        pixel_values = torch.zeros(1, 3, 224, 224).to(device)

    text = text if (text and text.strip()) else ""
    enc = tokenizer(
        text, max_length=77, truncation=True, padding="max_length", return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probs = torch.softmax(out["logits"], dim=1).squeeze(0).cpu().tolist()
    return {c: float(p) for c, p in zip(MEME_CLASSES, probs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_checkpoint",
        type=str,
        default="results/v4b_hatebert_noweights/checkpoints/best_model.pt",
        help="Path to HateXplain best_model.pt (Exp 14)",
    )
    parser.add_argument(
        "--meme_checkpoint",
        type=str,
        default="results/memes_multimodal/checkpoints/best_model.pt",
        help="Path to multimodal best_model.pt (Exp 17)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a temporary public Gradio URL",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading text model...")
    text_model, text_tokenizer = load_text_model(args.text_checkpoint, device)
    if text_model is None:
        print(f"  WARNING: text checkpoint not found at {args.text_checkpoint} - text tab disabled.")
    else:
        print(f"  Loaded from {args.text_checkpoint}")

    print("Loading meme model...")
    meme_model, meme_clip_processor, meme_tokenizer = load_meme_model(
        args.meme_checkpoint, device
    )
    if meme_model is None:
        print(f"  WARNING: meme checkpoint not found at {args.meme_checkpoint} - meme tab disabled.")
    else:
        print(f"  Loaded from {args.meme_checkpoint}")

    if text_model is None and meme_model is None:
        print("ERROR: no checkpoint loaded. Pass --text_checkpoint and/or --meme_checkpoint.")
        return

    with gr.Blocks(title="Hate Speech Detection - Demo") as demo:
        gr.Markdown("# Hate Speech Detection - Live Demo")
        gr.Markdown(
            "**Text tab** - HateXplain 3-class (HateBERT, Exp 14). "
            "**Meme tab** - CLIP + HateBERT multimodal (Exp 17)."
        )

        with gr.Tab("Text (HateXplain)"):
            if text_model is None:
                gr.Markdown(
                    f"Checkpoint not found at `{args.text_checkpoint}`. "
                    "Pass `--text_checkpoint <path>` when launching the demo."
                )
            else:
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Post / sentence",
                            placeholder="Type something here...",
                            lines=3,
                        )
                        text_btn = gr.Button("Predict", variant="primary")
                    with gr.Column():
                        text_output = gr.Label(
                            label="Predicted class", num_top_classes=3
                        )

                gr.Examples(
                    examples=[
                        ["I love spending time with my family on weekends."],
                        ["These idiots can never do anything right."],
                        ["What an amazing performance, congratulations!"],
                    ],
                    inputs=text_input,
                )

                text_btn.click(
                    fn=lambda t: predict_text(t, text_model, text_tokenizer, device),
                    inputs=text_input,
                    outputs=text_output,
                )

        with gr.Tab("Meme (Image + Caption)"):
            if meme_model is None:
                gr.Markdown(
                    f"Checkpoint not found at `{args.meme_checkpoint}`. "
                    "Pass `--meme_checkpoint <path>` when launching the demo."
                )
            else:
                with gr.Row():
                    with gr.Column():
                        meme_image = gr.Image(label="Meme image", type="pil")
                        meme_text = gr.Textbox(
                            label="Caption (text overlay)", lines=2
                        )
                        meme_btn = gr.Button("Predict", variant="primary")
                    with gr.Column():
                        meme_output = gr.Label(
                            label="Prediction", num_top_classes=2
                        )

                meme_btn.click(
                    fn=lambda i, t: predict_meme(
                        i, t, meme_model, meme_clip_processor, meme_tokenizer, device
                    ),
                    inputs=[meme_image, meme_text],
                    outputs=meme_output,
                )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
