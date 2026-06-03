"""
Cross-attention fusion model for Hateful Memes.

Idea: instead of concatenating final CLS embeddings (late fusion), let text
tokens attend over image patches and image patches attend over text tokens
through bidirectional cross-attention. This captures fine-grained
text<->image interactions which is precisely where hateful meme meaning
lives (e.g. innocuous text on offensive image).

Architecture:
    image patches (50, 768)  --\\
                                >-- cross-attn (bidirectional, N layers)
    text tokens   (77, 768)  --/
                |          |
            pooled_img  pooled_txt
                |          |
                  concat
                    |
              MLP classifier -> 2 logits

CLIP and HateBERT can be partially unfrozen (last N transformer blocks) just
like the late-fusion model.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModel


class BidirectionalCrossAttention(nn.Module):
    """One layer of bidirectional text<->image cross-attention with residual."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.txt_to_img = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.img_to_txt = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_txt1 = nn.LayerNorm(dim)
        self.norm_img1 = nn.LayerNorm(dim)
        self.ff_txt = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.ff_img = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.norm_txt2 = nn.LayerNorm(dim)
        self.norm_img2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, txt, img, txt_mask=None):
        # txt: (B, T, D), img: (B, I, D), txt_mask: (B, T) bool with True = pad
        txt_q = self.norm_txt1(txt)
        img_q = self.norm_img1(img)
        txt_out, _ = self.txt_to_img(query=txt_q, key=img_q, value=img_q)
        img_out, _ = self.img_to_txt(
            query=img_q, key=txt_q, value=txt_q, key_padding_mask=txt_mask
        )
        txt = txt + self.dropout(txt_out)
        img = img + self.dropout(img_out)
        txt = txt + self.dropout(self.ff_txt(self.norm_txt2(txt)))
        img = img + self.dropout(self.ff_img(self.norm_img2(img)))
        return txt, img


class CrossAttentionMemeClassifier(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        text_model_name="GroNLP/hateBERT",
        num_classes=2,
        dropout=0.1,
        freeze_encoders=True,
        unfreeze_last_n_layers=2,
        n_fusion_layers=2,
        fusion_num_heads=8,
    ):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        img_dim = self.clip.vision_model.config.hidden_size  # 768 for ViT-B/32
        txt_dim = self.text_encoder.config.hidden_size  # 768 for BERT-base

        # Project to common dim if needed (here both are 768, so identity)
        common_dim = txt_dim
        self.img_proj = (
            nn.Identity() if img_dim == common_dim else nn.Linear(img_dim, common_dim)
        )

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            if unfreeze_last_n_layers > 0:
                for layer in self.clip.vision_model.encoder.layers[
                    -unfreeze_last_n_layers:
                ]:
                    for p in layer.parameters():
                        p.requires_grad = True
                for p in self.clip.vision_model.post_layernorm.parameters():
                    p.requires_grad = True
                for layer in self.text_encoder.encoder.layer[
                    -unfreeze_last_n_layers:
                ]:
                    for p in layer.parameters():
                        p.requires_grad = True

        self.fusion_layers = nn.ModuleList(
            [
                BidirectionalCrossAttention(common_dim, fusion_num_heads, dropout)
                for _ in range(n_fusion_layers)
            ]
        )
        self.norm_final_txt = nn.LayerNorm(common_dim)
        self.norm_final_img = nn.LayerNorm(common_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * common_dim, common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Image patch features: (B, num_patches+1, hidden_size). For ViT-B/32 at 224: 50 tokens.
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        img_tokens = vision_out.last_hidden_state  # (B, 50, 768)
        img_tokens = self.img_proj(img_tokens)

        # Text token features: (B, T, 768)
        txt_out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        txt_tokens = txt_out.last_hidden_state

        # nn.MultiheadAttention key_padding_mask: True at padded positions
        txt_pad_mask = attention_mask == 0

        for layer in self.fusion_layers:
            txt_tokens, img_tokens = layer(
                txt_tokens, img_tokens, txt_mask=txt_pad_mask
            )

        txt_tokens = self.norm_final_txt(txt_tokens)
        img_tokens = self.norm_final_img(img_tokens)

        # Pool: use CLS-equivalents (first token of each sequence)
        txt_cls = txt_tokens[:, 0, :]
        img_cls = img_tokens[:, 0, :]

        fused = torch.cat([txt_cls, img_cls], dim=-1)
        logits = self.classifier(fused)
        return {"logits": logits}
