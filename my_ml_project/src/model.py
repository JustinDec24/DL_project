import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_classes=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Head for sentence-level classification
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Head for token-level rationale prediction
        self.rationale_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state
        cls_embedding = sequence_output[:, 0, :]

        class_logits = self.classifier(self.dropout(cls_embedding))
        rationale_logits = self.rationale_head(sequence_output).squeeze(-1)

        return {
            "class_logits": class_logits,
            "rationale_logits": rationale_logits
        }