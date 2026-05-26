# Implicit vs Explicit Hate Speech Detection

## Goal
Build a text classification model that distinguishes:
- non-hateful content
- explicit hate speech
- implicit hate speech

## First version (MVP)
- Text-only approach
- 3-class classification
- Pretrained transformer fine-tuning
- Standard cross-entropy loss

## Possible extension
- Use rationales to improve interpretability
- Guide attention with rationale supervision
- Add an auxiliary loss to better separate implicit vs explicit hate

## Main question
Can a pretrained language model reliably distinguish implicit hate from explicit hate?

## Notes
Datasets will be checked and cleaned before final label mapping.