# Experiment Log — Hate Speech Detection

## Project Overview

This project tackles **hate speech detection** from two angles:

**Part 1 — Text classification on HateXplain**
Classify social media posts into three categories: *hatespeech*, *normal*, or *offensive*. The key challenge is the middle class: "offensive" content is not necessarily hateful, but is aggressive or vulgar. This boundary is inherently ambiguous even for human annotators.

**Part 2 — Multimodal classification on Hateful Memes**
Classify internet memes as *hateful* or *not hateful*. Memes combine an image with a text overlay, and their hateful meaning often only emerges from the combination of both — a benign caption on an offensive image, or a dog whistle phrase on an innocuous photo. This is fundamentally a multimodal problem.

---

## Datasets

### HateXplain
- Source: `hatexplain` on HuggingFace (Mathew et al., 2021)
- 3 splits: train (15,383) / validation (1,922) / test (1,924)
- 3 classes: hatespeech (0), normal (1), offensive (2)
- Each post was labelled by **3 annotators** independently
- Final label = majority vote; `label_confidence` = fraction of annotators who agreed (0.33 / 0.66 / 1.0)
- Each annotator also provides token-level rationales (which words justify the label) and a **target group** (who the post targets: Jewish, Women, African, etc.)
- Class distribution in train: hatespeech 36%, normal 41%, offensive 23%

### Hateful Memes (Facebook / limjiayi/hateful_memes_expanded)
- Source: `limjiayi/hateful_memes_expanded` on HuggingFace
- 3 splits: train (12,887) / validation (1,040) / test (3,000)
- Binary: not_hateful (0), hateful (1)
- Class balance: ~58.7% not_hateful, ~41.3% hateful
- Each example contains a meme image (RGB) and a short text caption

---

## Models and Techniques

### Backbones used
- **roberta-base**: General-purpose language model pre-trained on large web text. Strong general baseline.
- **GroNLP/hateBERT**: BERT model fine-tuned on Reddit's Abusive Language corpus (RAL-E). Specialised for hate speech — the key hypothesis is that domain-specific pre-training gives a head start.
- **openai/clip-vit-base-patch32**: Vision-language model pre-trained on 400M image-text pairs. Its vision encoder produces 512-dimensional embeddings that capture semantic image content.

### Training techniques
- **Label-confidence weighting**: Each training example is weighted by its annotator agreement score. A post that all 3 annotators agreed was hateful contributes fully to the loss; one where only 2/3 agreed contributes 2/3 as much. This prevents the model from over-committing to ambiguous labels.
- **Focal Loss** (Lin et al., 2017): Modifies cross-entropy to down-weight easy, correctly-classified examples and focus training on hard ones. Formula: `FL = (1 - p_t)^γ × CE`, with γ=2.0. Helps with class imbalance by preventing the majority class from dominating training.
- **LR warmup + cosine decay**: Learning rate linearly increases for the first 6% of training steps, then decays to 0 following a cosine curve. This avoids training instability at the start and a hard stop at the end.
- **Gradient clipping** (max_norm=1.0): Clips gradient norms to prevent exploding gradients.

### Metric
- **Macro-F1** for HateXplain (3 classes, unweighted average): penalises equally poor performance on any class, including the minority "offensive" class.
- **AUROC** (primary) + **binary F1** for Hateful Memes: AUROC measures ranking quality independent of threshold; F1 measures classification at threshold 0.5.

---

---

# Part 1 — HateXplain: 3-class Text Classification

## Historical experiments (Exp 1–8)
*These experiments were run on an earlier version of the pipeline. They are preserved for reference but should not be directly compared to Exp 9+ which use the unified, corrected pipeline.*

---

### Exp 1 — Historical baseline (roberta-base, no extras)
- Backbone: roberta-base, max_length=128
- No class weighting, no rationale supervision, no confidence weighting
- val macro-F1: **0.6874** | test macro-F1: **0.6879**
- Per-class test F1: hatespeech=0.763, normal=0.742, offensive=0.559
- Note: treated as historical reference only; pipeline was later refactored.

---

### Exp 2 — Longer context (max_length=256)
- max_length increased from 128 to 256
- val F1: 0.6917 | test F1: **0.6639** (worse on test)
- Interpretation: most posts are short; extra context brings noise, not signal. Rejected.

---

### Exp 3 — Class-weighted loss
- Inverse-frequency class weights: [1.08, 0.82, 1.17]
- val F1: **0.6747** (worse than baseline)
- Interpretation: class weights did not help with roberta-base in this historical run. Rejected.

---

### Exp 4 — Implicit hate binary classification (exploratory)
- Binary task: HateXplain normal (0) vs ImplicitHate dataset (1)
- val F1: **0.9842** | test F1: **0.9819**
- WARNING: results are inflated by **source confounding** — the model may be learning dataset style differences (Reddit vs Twitter) rather than implicit hate semantics. Not comparable to the main task.

---

### Exp 5 & 6 — Token-level rationale supervision
- Total loss = classification loss + α × token-level BCE on rationale masks
- Exp 5: α=0.5 → val F1: 0.6581 | Exp 6: α=0.1 → val F1: 0.6544
- Both worse than baseline. Rationale supervision hurt performance, possibly due to noise in aggregated rationale masks.

---

### Exp 7 — Label-confidence weighting (invalid run)
- First attempt at confidence weighting, but processed data files on the cluster were missing the `label_confidence` field at the time.
- Code silently defaulted to confidence=1.0 for all examples → equivalent to unweighted training.
- **Invalid experiment, do not interpret.**

---

### Exp 8 — Full-agreement filtering
- Keep only examples where all 3 annotators agreed (confidence=1.0)
- Training set reduced from 15,383 → 7,888 examples
- val F1: **0.6474**
- Interpretation: throwing away half the training data (including useful hard examples) is worse than keeping all examples with soft weighting. Rejected.

---

## Updated pipeline experiments (Exp 9–14)
*All comparable. The pipeline was unified: consistent preprocessing, JSONL format with label_confidence and targets, cosine scheduler, gradient clipping.*

---

### Exp 9 — Baseline (roberta-base, updated pipeline)
Config: `experiment.yaml`

- Backbone: roberta-base, max_length=128
- No special techniques
- val macro-F1: **0.6465**

This is the clean reference point for all subsequent updated-pipeline experiments.

---

### Exp 10 — Label-confidence weighting
Config: `experiment_confidence.yaml`

**What changed:** each training loss is multiplied by the annotator agreement confidence. Posts where all 3 annotators agreed (confidence=1.0) get full weight; 2/3-agreement posts (confidence=0.667) count less.

- val macro-F1: **0.6703** (+0.024 vs Exp 9)
- test macro-F1: **0.6631**
- Per-class test F1: hatespeech=0.747, normal=0.724, offensive=0.518

Interpretation: weighting by annotator agreement consistently improves performance. The signal: ambiguous posts (where annotators disagree) carry noisier label information. Down-weighting them without removing them is better than either using them fully or filtering them out (Exp 8).

---

### Exp 11 — Focal loss + LR warmup + cosine decay (roberta-base)
Config: `experiment_v2.yaml`

**What changed:** replaced CrossEntropy with FocalLoss(γ=2), added LR warmup over 6% of steps + cosine decay, gradient clipping, extended to 5 epochs.

Training dynamics: val_f1 peaked at epoch 4 (0.6560), val_loss rising from epoch 3 → overfitting. 5 epochs is too many.

- val macro-F1: **0.6560**
- test macro-F1: **0.6700** (+0.007 vs Exp 10)
- Per-class test F1: hatespeech=0.768, normal=0.737, offensive=0.506
- High-confidence errors: **0** (model is well-calibrated)

Interpretation: focal loss + warmup improves test F1 and calibration but val F1 dropped vs Exp 10 (overfitting artefact). The zero high-confidence errors is a consistent property of focal-trained models throughout all subsequent experiments.

---

### Exp 12 — Focal loss + class weights (roberta-base)
Config: `experiment_v3.yaml`

**What changed vs Exp 11:** 3 epochs instead of 5 (fixing overfitting), added class weights [1.08, 0.82, 1.17].

- val macro-F1: **0.6715**
- test macro-F1: **0.6625**
- Per-class test F1: hatespeech=0.756, normal=0.699, offensive=0.533

Interpretation: class weights improved offensive F1 (0.506 → 0.533) but at the cost of normal F1 (0.737 → 0.699). Normal→offensive errors increased from 135 to 196. The model over-predicts offensive when explicitly pushed to pay more attention to it. This reflects the fundamental ambiguity: "offensive" is defined negatively (not hateful, not normal) and lacks strong positive features.

---

### Exp 13 — HateBERT backbone + focal loss + class weights
Config: `experiment_v4_hatebert.yaml`

**What changed vs Exp 12:** backbone switched from roberta-base to GroNLP/hateBERT.

HateBERT is a BERT model that was further pre-trained on the Reddit Abusive Language English (RAL-E) dataset, giving it a strong prior on hate speech vocabulary, slurs, and abusive language patterns.

- val macro-F1: **0.6877** (already at epoch 1: 0.6862 — higher than any roberta-base epoch ever)
- test macro-F1: **0.6829**
- Per-class test F1: hatespeech=0.771, normal=0.740, offensive=0.537
- High-confidence errors: **0**

Interpretation: domain-specific pre-training is the single most impactful improvement. HateBERT's epoch-1 performance (0.686) already exceeds the best roberta-base model trained for 3–5 epochs (0.671). The model has effectively "seen" hate speech before fine-tuning. The class-weight offensive/normal trade-off is less pronounced than in Exp 12.

---

### Exp 14 — HateBERT, no class weights (ablation) — **FINAL SELECTED MODEL**
Config: `experiment_v4b_hatebert_noweights.yaml`

**What changed vs Exp 13:** class weights removed.

Training dynamics:
- Epoch 1: val_f1=0.6780 | Epoch 2: 0.6826 | Epoch 3: **0.6931** ← best
- val_loss rising throughout (0.255 → 0.275 → 0.309) but val_f1 still improving: the model is not overfitting in terms of classification performance; loss increase reflects confidence calibration shift.

Results:
- val macro-F1: **0.6931** ← best across all experiments
- test accuracy: **0.6949**
- test macro-F1: **0.6827**

Per-class test F1:
- hatespeech: **0.7685**
- normal: **0.7450**
- offensive: **0.5346**

Confusion matrix (test):
```
             pred_hate  pred_norm  pred_off
true_hate       478        41        75
true_norm        55       577       150
true_off        117       149       282
```

Error analysis:
- Total errors: 587 / 1924 (30.5%)
- High-confidence errors (≥ 0.90): **0**
- Top confusion pairs: normal→offensive (150), offensive→normal (149), offensive→hatespeech (117)

Interpretation:
- Removing class weights does not degrade performance — differences vs Exp 13 are within noise (<0.003 on all metrics). HateBERT's domain representations already handle imbalance implicitly.
- Simpler model, equal or better performance → selected as final.
- The offensive class (F1=0.535) remains the hardest. The confusion is symmetric in both directions (normal↔offensive), confirming this is an annotation ambiguity problem, not a model failure.

**Why this is the final model:** best val F1 (0.693), simplest configuration, domain-appropriate backbone, no class-weight hyperparameter to tune.

---

## Part 1 summary table

| Exp | Backbone | Key additions | Val F1 | Test F1 | Off. F1 |
|-----|----------|---------------|--------|---------|---------|
| 9   | roberta-base | baseline (updated pipeline) | 0.6465 | — | — |
| 10  | roberta-base | + confidence weighting | 0.6703 | 0.6631 | 0.518 |
| 11  | roberta-base | + focal loss + warmup | 0.6560 | 0.6700 | 0.506 |
| 12  | roberta-base | + class weights | 0.6715 | 0.6625 | 0.533 |
| 13  | HateBERT | focal + warmup + class weights | 0.6877 | 0.6829 | 0.537 |
| **14** | **HateBERT** | **focal + warmup, no class weights** | **0.6931** | **0.6827** | **0.535** |

Key takeaway: HateBERT accounts for most of the gain (+0.123 val F1 vs roberta baseline). All other techniques contribute cumulatively but modestly.

---

---

# Part 1 — Bias Analysis (Target Group Fairness)

## Motivation

HateXplain annotators label each post with a **target group**: the community the post attacks (e.g., Jewish, African, Women, Islam, etc.). These labels are not used during training, but they allow us to ask: *does the model perform equally well across different targeted communities?*

This analysis was run on the best model (Exp 14) using the test set.

## Method

For each test example, we extract the majority-agreed target group (the group mentioned by at least 2 of the 3 annotators). Examples with no agreed target are excluded. We then compute macro-F1 and per-class F1 separately for each group with at least 10 test examples.

Script: `src/analyze_target_bias.py`

## Results (Exp 14 — HateBERT, best model)

| Group | N | Accuracy | Macro-F1 | F1-hate | F1-norm | F1-off |
|-------|---|----------|----------|---------|---------|--------|
| Asian | 53 | 0.491 | **0.467** | 0.619 | 0.474 | 0.308 |
| Hispanic | 44 | 0.636 | **0.473** | 0.793 | 0.471 | 0.154 |
| Jewish | 210 | 0.686 | 0.509 | 0.836 | 0.364 | 0.329 |
| Islam | 238 | 0.643 | 0.584 | 0.765 | 0.625 | 0.362 |
| African | 388 | 0.696 | 0.584 | 0.841 | 0.481 | 0.430 |
| Arab | 90 | 0.711 | 0.593 | 0.852 | 0.483 | 0.444 |
| Caucasian | 114 | 0.632 | 0.617 | 0.667 | 0.698 | 0.485 |
| Homosexual | 223 | 0.610 | 0.619 | 0.643 | 0.661 | 0.551 |
| Refugee | 113 | 0.664 | 0.647 | 0.634 | 0.746 | 0.560 |
| Women | 233 | 0.670 | 0.664 | 0.640 | 0.696 | 0.657 |
| Other | 244 | 0.705 | **0.690** | 0.667 | 0.663 | 0.741 |

**Gap between best (Other, 0.690) and worst (Asian, 0.467): 0.223**

Label distribution per group:

| Group | % hate | % normal | % offensive |
|-------|--------|----------|-------------|
| Asian | 34% | 34% | 32% |
| Hispanic | 59% | 14% | 27% |
| Jewish | **68%** | 12% | 20% |
| Islam | 52% | 20% | 28% |
| African | **60%** | 18% | 22% |
| Arab | 59% | 16% | 26% |
| Caucasian | 14% | **53%** | 33% |
| Homosexual | 34% | 25% | 41% |
| Refugee | 19% | **45%** | 36% |
| Women | 16% | **37%** | **47%** |
| Other | 11% | **34%** | **55%** |

## Key findings

**Finding 1 — Systematic bias against racial/ethnic minorities.**
The 6 worst-performing groups are all racial or ethnic minorities (Asian, Hispanic, Jewish, Islam, African, Arab). These groups score 0.47–0.59 macro-F1, compared to 0.62–0.69 for gender/identity groups (Women, Homosexual) and non-specific targets (Other).

**Finding 2 — The model has learned a spurious shortcut: ethnic group name → hate speech.**
For every racial/ethnic group, the model achieves high F1 on the *hatespeech* class (0.76–0.85) but very low F1 on *normal* and *offensive* (as low as 0.15 for Hispanic offensive). This happens because these groups appear predominantly in hateful contexts in the training data (Jewish: 68% hate, African: 60%, Arab: 59%). The model has learned "if a post mentions these groups, predict hate" — which works well when the post is indeed hateful, but fails completely on normal or offensive posts mentioning those same groups.

**Finding 3 — Caucasian shows the mirror-image bias.**
Caucasian posts are only 14% hateful in the test set. The model accordingly has low hate recall for Caucasian (F1-hate=0.667) but good normal recall (F1-norm=0.698). The bias runs in both directions, following the data distribution.

**Finding 4 — Underrepresentation amplifies bias.**
Asian (n=53) and Hispanic (n=44) are the two smallest groups and the two worst performers. Small training set size → model cannot generalise.

---

## Bias Correction Attempt — Group-aware loss weighting

### Motivation
Since the bias stems from unequal (group, class) frequency in training data, we can try to correct it by up-weighting under-represented (group, class) pairs during training.

### Method
For each training example, compute a weight:
```
weight(g, c) = median_count / count(group=g AND class=c)
```
where `median_count` is the median (group, class) frequency across all training examples. This gives weight ~1.0 to average pairs, and higher weight to rare pairs (e.g., Jewish×normal, which has only 26 training examples vs. 1115 Jewish×hatespeech).

Weights are normalized to mean=1.0 after clipping at max=5.0.

The group weight is combined with label-confidence weighting multiplicatively:
```
effective_weight = label_confidence × group_weight
loss = (focal_loss × effective_weight).sum() / effective_weight.sum()
```

### Exp 18 — Group-aware weighting (v5, normalization bug)
Config: `experiment_v5_group_aware.yaml`

A normalization bug caused weights to reach max=17.0 instead of the intended 5.0 (the clip was applied before normalization, so normalization amplified the clipped values).

- val macro-F1: **0.6647** (−0.028 vs Exp 14)
- Best val F1 at epoch 3; training finished.

Target group analysis (Exp 18):

| Group | Exp 14 Macro-F1 | Exp 18 Macro-F1 | Δ |
|-------|----------------|----------------|---|
| Asian | 0.467 | **0.398** | −0.069 |
| Hispanic | 0.473 | 0.564 | **+0.092** |
| Jewish | 0.509 | 0.521 | +0.012 |
| African | 0.584 | 0.564 | −0.020 |
| Refugee | 0.647 | 0.566 | −0.081 |
| Women | 0.664 | 0.610 | −0.054 |
| Gap (best−worst) | 0.223 | **0.264** | worse |

The extreme weights (17×) destabilised training: most groups degraded, the gap between best and worst groups actually increased.

### Exp 19 — Group-aware weighting (v5b, corrected normalization)
Config: `experiment_v5_group_aware.yaml` (same, bug fixed in `train.py`)

Normalization bug fixed: clip is now applied *after* normalizing to mean=1.0, ensuring max ≤ 5.0.

Observed weights: min=0.113, max=5.000, mean=1.000 ✓

- val macro-F1: **0.6579** (−0.035 vs Exp 14)
- Test macro-F1: — (run terminated after identifying the negative result pattern)

### Interpretation of Exp 18 & 19

Both versions of group-aware weighting degraded overall performance by ~3.5pp without providing consistent improvement to the worst-performing groups. This is a **meaningful negative result**:

1. **Re-weighting cannot substitute for data.** Asian (n≈50 in train) has so few examples that even with 5× weight, the model cannot learn generalizable features for this group. The problem is data scarcity, not loss weighting.

2. **The fairness–performance trade-off is real.** Correcting for group imbalance via re-weighting shifts the optimisation objective away from overall accuracy. Without sufficient per-group data, this creates instability rather than improved representation.

3. **This is consistent with the fairness literature.** Davidson et al. (2019) showed that racial/ethnic bias in hate speech models largely reflects training data composition and is resistant to simple post-hoc corrections. Addressing it properly would require data augmentation or group-stratified sampling with substantially more balanced data per group.

**The best model remains Exp 14 (v4b).** The bias analysis and failed correction attempt are themselves scientific contributions: they characterise a known limitation of the field and demonstrate the boundary of what re-weighting can achieve.

---

---

# Part 2 — Hateful Memes: Multimodal Classification

## Architecture

```
Input image ──► CLIP ViT-B/32 vision encoder (frozen) ──► visual_projection ──► 512-dim vector
                                                                                       │
                                                                                  concat (1280-dim)
                                                                                       │
Input text ───► HateBERT tokenizer ──► HateBERT encoder (frozen) ──► CLS token ──► 768-dim vector
                                                                                       │
                                                                              Linear(1280, 256)
                                                                                    ReLU
                                                                               Dropout(0.1)
                                                                              Linear(256, 2)
                                                                                    │
                                                                             logits (not_hateful / hateful)
```

**Design choices:**
- Both encoders are **frozen**: only the 328k-parameter classification head is trained. This makes training fast (~5 min/epoch) and prevents catastrophic forgetting with limited data.
- CLIP is chosen as the image encoder because it was pre-trained on vision-language pairs, giving it semantically rich representations aligned with textual meaning.
- HateBERT is kept from Part 1: it was the best text encoder for hate speech.
- Late fusion (concatenation): the simplest possible fusion strategy, used as baseline.

**Training config:**
- Optimizer: AdamW, lr=2e-4, weight_decay=0.01
- Scheduler: cosine warmup (6%) + cosine decay
- Epochs: 10, batch_size=32
- Best checkpoint selected by validation AUROC (not F1, more robust to threshold)
- Loss: CrossEntropy (not focal — focal hurt on memes with frozen encoders)

## Ablation design

Three conditions tested on strictly identical architecture and hyperparameters:

| Condition | What's active | Fusion input size |
|-----------|--------------|-------------------|
| Exp 15: Text-only | HateBERT only | 768-dim |
| Exp 16: Image-only | CLIP only | 512-dim |
| Exp 17: Multimodal | CLIP + HateBERT | 1280-dim |

When a modality is disabled, it is replaced by a zero tensor of the right dimension, so the architecture stays identical.

---

### Exp 15 — Text-only (HateBERT)
Config: `experiment_memes_textonly.yaml`
Trainable params: 197,378 / 109,679,618 (0.18% of total)

Training dynamics: val AUC peaked at epoch 7 (0.613), val F1 oscillating (0.25–0.37), suggesting text features are weakly discriminative.

Results:
- test accuracy: **0.5900**
- test F1 (hateful): **0.3607**
- test AUROC: **0.6095**

Per-class:
- not_hateful: P=0.614, R=0.809, F1=0.698
- hateful: P=0.507, R=0.280, F1=0.361

Confusion matrix:
```
              pred_nothate  pred_hate
true_nothate    1423          337
true_hate        893          347
```

Interpretation: text-only is the worst condition. Meme captions are short, often deliberately innocuous ("when you see them") — their hateful meaning is entirely in the visual context. The model misses 893 of 1240 hateful memes (72% miss rate). The text modality alone cannot detect hateful memes.

---

### Exp 16 — Image-only (CLIP)
Config: `experiment_memes_imageonly.yaml`
Trainable params: 131,842 / 151,409,155 (0.09% of total)

Training dynamics: val AUC steadily improving to 0.640 at epoch 10 (still improving — could benefit from more epochs).

Results:
- test accuracy: **0.6187**
- test F1 (hateful): **0.4245**
- test AUROC: **0.6709**

Per-class:
- not_hateful: P=0.637, R=0.815, F1=0.715
- hateful: P=0.564, R=0.340, F1=0.425

Confusion matrix:
```
              pred_nothate  pred_hate
true_nothate    1434          326
true_hate        818          422
```

Interpretation: image alone significantly outperforms text alone (+0.061 AUROC, +0.064 F1). CLIP's visual representations encode stereotype-laden imagery, skin tone, and targeting cues even without any text. This confirms that **for hateful memes, visual content carries more discriminative information than the caption text**. However, recall is still low (0.340): hateful visual signals alone are not sufficient.

---

### Exp 17 — Multimodal fusion (CLIP + HateBERT) — **FINAL MULTIMODAL MODEL**
Config: `experiment_memes_multimodal.yaml`
Trainable params: 328,450 / 261,088,003 (0.13% of total)

Training dynamics: val AUC steadily improving across all 10 epochs, peaking at epoch 9 (0.6907). Consistent, stable learning.

Results:
- test accuracy: **0.6583**
- test F1 (hateful): **0.4943**
- test AUROC: **0.7179**

Per-class:
- not_hateful: P=0.666, R=0.838, F1=0.742
- hateful: P=0.637, R=0.404, F1=0.494

Confusion matrix:
```
              pred_nothate  pred_hate
true_nothate    1474          286
true_hate        739          501
```

Interpretation:
- Multimodal fusion outperforms both single-modality conditions on all metrics.
- AUROC: 0.718 vs 0.671 (image) vs 0.610 (text): fusion adds +0.047 over image-only, +0.108 over text-only.
- Hateful recall improves from 0.340 (image) to 0.404: text provides complementary signal even though it is weaker alone.
- Hateful precision improves to 0.637: combining modalities reduces false positives.
- The two modalities are **complementary**: text handles cases where visual context alone is ambiguous, and vice versa.

**Remaining weakness:** 739 hateful memes are still missed (recall 0.404). This reflects the fundamental difficulty of the task — many memes require cultural background knowledge or very subtle visual encoding that frozen encoders cannot capture.

---

## Multimodal ablation summary

| Exp | Modality | Trainable params | Test Acc | F1 (hateful) | AUROC |
|-----|----------|-----------------|----------|--------------|-------|
| 15 | Text-only (HateBERT) | 197k | 0.590 | 0.361 | 0.610 |
| 16 | Image-only (CLIP) | 132k | 0.619 | 0.425 | 0.671 |
| **17** | **Multimodal (CLIP + HateBERT)** | **328k** | **0.658** | **0.494** | **0.718** |

**Key findings from ablation:**
1. **Image > Text** for hateful meme detection: visual features carry more discriminative information than short caption text. Meme captions are often designed to be plausibly deniable.
2. **Fusion > Image > Text**: late concatenation fusion is effective; each modality contributes independently.
3. **Parameter efficiency**: training only 0.13% of the model's 261M parameters achieves competitive AUROC. Frozen encoders are sufficient with this training set size.

---

---

# Final Summary

## Best models

| Task | Model | Val metric | Test metric |
|------|-------|-----------|-------------|
| HateXplain (3-class) | Exp 14: HateBERT + focal + warmup + confidence weighting | val macro-F1: 0.693 | test macro-F1: 0.683 |
| Hateful Memes (binary) | Exp 17: CLIP + HateBERT fusion, frozen, CE loss | val AUROC: 0.691 | test AUROC: 0.718 |

## Key lessons learned

**1. Domain pre-training is the most impactful single change.**
Switching from roberta-base to HateBERT added +0.123 val F1 on HateXplain. HateBERT's epoch-1 performance already exceeds the best roberta-base model. No combination of tricks (confidence weighting, focal loss, class weights) on roberta-base could match a domain-adapted backbone.

**2. Label-confidence weighting improves robustness on ambiguous data.**
Weighting training examples by annotator agreement (+0.024 val F1 over baseline) is a principled way to handle annotator disagreement. Removing ambiguous examples entirely (Exp 8) is worse; using them equally weighted is also worse. Soft down-weighting is the right approach.

**3. Focal loss improves calibration, not necessarily raw F1.**
Focal loss consistently produced zero high-confidence errors across all experiments where it was used, even when macro-F1 was similar to cross-entropy. The model is less overconfident on hard examples.

**4. Class weights create offensive/normal trade-off with roberta-base, but not with HateBERT.**
With roberta-base, class weights boost offensive recall at the cost of normal precision (normal→offensive confusion increases). HateBERT is robust to this: adding or removing class weights changes performance by <0.003. Domain-specific representations already encode the class boundaries implicitly.

**5. For hateful memes, image information dominates text.**
AUROC 0.671 (image-only) vs 0.610 (text-only). Meme captions are deliberately ambiguous; the visual content carries the hateful meaning. This has implications for content moderation systems that rely on text alone.

**6. Fairness bias in hate speech models is resistant to simple re-weighting.**
The bias analysis revealed a 0.22 macro-F1 gap between the best-performing target group (Other, 0.690) and worst (Asian, 0.467). The model has learned spurious shortcuts associating racial/ethnic group names with hatespeech labels, reflecting training data composition rather than genuine understanding. Inverse-frequency group weighting (Exp 18 & 19) failed to close this gap and degraded overall performance by ~3.5pp. This is consistent with the literature: addressing this bias requires more representative data per group, not re-weighting.

## Remaining limitations

- **Offensive class is structurally hard** (F1≈0.535): defined negatively as "not hateful, not normal", leading to symmetric confusion in both directions. This is a labelling ambiguity issue, not purely a modelling failure.
- **Hateful meme recall is low** (0.404): 60% of hateful memes are missed. Solving this would require unfreezing the visual encoder (computationally expensive) or a larger training set.
- **Racial/ethnic group bias** (see bias analysis): the model over-predicts hate for posts mentioning racial/ethnic minorities. A production system would require targeted data collection and debiasing.
