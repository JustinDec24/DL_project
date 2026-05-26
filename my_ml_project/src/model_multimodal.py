import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModel


class MultimodalMemeClassifier(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        text_model_name="GroNLP/hateBERT",
        num_classes=2,
        dropout=0.1,
        freeze_encoders=True,
        unfreeze_last_n_layers=0,
        use_image=True,
        use_text=True,
    ):
        super().__init__()
        assert use_image or use_text, "At least one of use_image or use_text must be True"
        self.use_image = use_image
        self.use_text = use_text

        image_dim = 0
        text_dim = 0

        if use_image:
            self.clip = CLIPModel.from_pretrained(clip_model_name)
            image_dim = self.clip.config.projection_dim  # 512 for ViT-B/32
            if freeze_encoders:
                for p in self.clip.parameters():
                    p.requires_grad = False
                if unfreeze_last_n_layers > 0:
                    for layer in self.clip.vision_model.encoder.layers[-unfreeze_last_n_layers:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                    for p in self.clip.vision_model.post_layernorm.parameters():
                        p.requires_grad = True
                    for p in self.clip.visual_projection.parameters():
                        p.requires_grad = True

        if use_text:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            text_dim = self.text_encoder.config.hidden_size  # 768 for BERT-base
            if freeze_encoders:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
                if unfreeze_last_n_layers > 0:
                    for layer in self.text_encoder.encoder.layer[-unfreeze_last_n_layers:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                    for p in self.text_encoder.pooler.parameters():
                        p.requires_grad = True

        fusion_dim = image_dim + text_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        features = []

        if self.use_image and pixel_values is not None:
            vision_out = self.clip.vision_model(pixel_values=pixel_values)
            img_feat = self.clip.visual_projection(vision_out.pooler_output)
            features.append(img_feat)

        if self.use_text and input_ids is not None:
            txt_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            txt_feat = txt_out.last_hidden_state[:, 0, :]  # CLS token
            features.append(txt_feat)

        fused = torch.cat(features, dim=-1)
        return {"logits": self.classifier(fused)}
