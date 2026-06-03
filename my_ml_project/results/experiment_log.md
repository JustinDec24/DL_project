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
- **CRITICAL — images not shipped:** the HuggingFace snapshot only contains
  JSONL metadata. Images are not bundled (licensing). This silently broke
  every multimodal run before Exp 24 — see the "grey-placeholder bug"
  section in Part 2 (cont. cont.). The original Hateful Memes images can
  no longer be obtained from DrivenData (competition archived).

### HarMeme (Pramanick et al. 2021) — replacement for Hateful Memes
- Source: `mmf/data/datasets/memes/defaults/` (MMF layout, local images)
- 3 splits: train (3013) / validation (177) / test (354)
- 3-class harmfulness labels: `not harmful`, `somewhat harmful`, `very harmful`
- Binarised for our runs: `harmful = somewhat + very` vs `not_harmful`
- Class balance (train): 64.7% not_harmful, 35.3% harmful
- Each example: COVID-themed meme image with text caption (text is also
  baked into the image as overlay)
- Compared to Hateful Memes this is 4× smaller and a different topic, but
  it is the only image-grounded multimodal hate/harm benchmark we could
  obtain locally

---

## Models and Techniques

### Backbones used
- **roberta-base**: General-purpose language model pre-trained on large web text. Strong general baseline.
- **GroNLP/hateBERT**: BERT model fine-tuned on Reddit's Abusive Language corpus (RAL-E). Specialised for hate speech — the key hypothesis is that domain-specific pre-training gives a head start.
- **openai/clip-vit-base-patch32**: Vision-language model pre-trained on 400M image-text pairs. Its vision encoder produces 768-dimensional patch features (50 tokens: 1 CLS + 7×7 patches) projected to 512-d via `visual_projection`.

### Multimodal fusion architectures
- **Late fusion (concatenation)** — `src/model_multimodal.py`
  Take the final CLS embedding from each encoder, concatenate them
  (1280-d = 512 image + 768 text), feed an MLP head
  `Linear(1280→256) → ReLU → Dropout → Linear(256→C)`.
  Cheapest and simplest. The model can only combine modalities at the
  *output* level.
- **Bidirectional cross-attention fusion** — `src/model_crossattn.py`
  Use the full sequence of patch tokens (50) from CLIP and the full sequence
  of word tokens from HateBERT. Apply `n_fusion_layers` blocks of
  bidirectional cross-attention: text tokens attend over image patches AND
  image patches attend over text tokens, each with residual + feed-forward.
  This models *interaction* between modalities at multiple depths, which
  the literature identifies as the right inductive bias for memes where
  the harmful meaning emerges from text+image combination (VisualBERT,
  MMBT, ViLT).

### Encoder unfreezing (defrost)
Three policies tested:
- **Fully frozen** (Exp 15-17, 22, 24): only the head trains (~330k params).
  Cheapest, exploits pre-training maximally, but caps performance at what
  the frozen features can express.
- **Defrost last N transformer blocks** (Exp 20, 21, 25, 26): the last N
  layers of CLIP vision and HateBERT are unfrozen. Trainable params jump
  to ~30M for N=2 on late-fusion, ~58M with cross-attention. Risks: the
  encoder may drift away from its useful pre-trained representations,
  especially on small datasets, so we use **differential learning rate**:
  encoder LR = head LR × `encoder_lr_multiplier` (default 0.1 = 10× smaller).
- **Fully fine-tuned** (HateXplain only — Exp 14): the entire HateBERT is
  trained end-to-end. The dataset is large enough (15 383 train) to support
  this without catastrophic forgetting.

### Training techniques
- **Label-confidence weighting** — Each training example is weighted by its
  annotator agreement score. A post that all 3 annotators agreed was
  hateful contributes fully to the loss; one where only 2/3 agreed
  contributes 2/3 as much. Prevents over-fitting to ambiguous labels.
  Implementation: `(per_example_loss × confidence).sum() / confidence.sum()`.
- **Focal Loss** (Lin et al., 2017) — `FL = (1 − p_t)^γ × CE`, γ = 2.0.
  Down-weights easy correctly-classified examples and forces training to
  focus on hard ones. Combined with confidence weighting multiplicatively.
  Consistently produces *zero high-confidence errors* (`p ≥ 0.9` and wrong)
  on the trained models.
- **Class-weighted CE** — inverse-frequency weights (Exp 3, 12, 13). Helps
  the minority "offensive" class with roberta-base; neutral with HateBERT.
- **Group-aware sample weighting** (Exp 18, 19) — clipped inverse-frequency
  weights computed on (target_group × class). Failed: amplifies noise on
  rare cells. Documented negative result.
- **LR warmup + cosine decay** — LR linearly ramps up over the first 6% of
  steps, then cosine-decays to 0. Avoids early instability and lets the
  schedule end gracefully.
- **Differential learning rate** — when encoders are partially unfrozen,
  encoder params get `lr × encoder_lr_multiplier` (typically 0.1), head and
  fusion params keep `lr`. Implemented via AdamW parameter groups.
- **Gradient clipping** (`max_norm = 1.0`) — clips global gradient norm to
  prevent the rare exploding update.

### Post-training inference techniques (no retraining)
- **Threshold tuning** — for binary classifiers, sweep thresholds in
  [0.05, 0.95] on validation and pick the one that maximises F1 on the
  positive class. Free improvement. Particularly large when the model is
  miscalibrated (the broken-multimodal models needed threshold 0.06–0.17
  to recover F1; the HarMeme-defrost model is already well-calibrated and
  the optimal threshold is 0.7).
- **Ensembling** — average the softmax probabilities of several
  independently-trained models, then apply threshold tuning to the
  averaged distribution. We use uniform weights (each member gets 1/N
  weight).
- **Test-time augmentation (TTA)** — at inference time, predict on several
  augmented variants of the same image (here: original, horizontal flip,
  5 deterministic crops resized back to 224) and average the probabilities.
  TTA *hurt* on memes (text overlay is task-relevant and gets cropped /
  mirrored), confirming TTA is task-specific.

### Bias-mitigation techniques (HateXplain)
- **Inverse-frequency group weighting** (Exp 18, 19) — failed.
- **Counterfactual data augmentation** (Exp 28, 29) — for each training
  row with a clear target group, emit copies that replace the group's
  lexical hooks with neutral terms from other groups. We tried (a) augment
  only `hate + offensive` rows and (b) augment all three label classes.
  Both failed (made the gap worse). Documented in Part 1 (cont.).

### Metrics
- **Macro-F1** for HateXplain (3 classes, unweighted average across
  classes): penalises poor performance on the minority *offensive* class,
  prevents inflating the score by getting only the easy classes right.
- **AUROC** for binary memes (primary): measures *ranking* quality of the
  predicted probabilities, threshold-independent. Robust to class imbalance.
- **Binary F1** on the positive class (memes): measures *classification*
  performance at a given threshold. F1@0.5 and F1@tuned both reported.
- **Per-group macro-F1** + worst/best gap (target bias analysis on
  HateXplain): measures fairness across the 11 target groups.

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

# Part 2 (cont.) — Capacity scaling attempts (Exp 20–23)

*All four experiments below were run on the same local machine (RTX 3060), same seed=42, same pyarrow/torch versions, same training pipeline, on the same HF dataset snapshot — so they are directly comparable to each other. They are NOT directly comparable to Exp 17 above (run earlier on a different machine/version), which is why Exp 22 was added as a fair-comparison control.*

## Motivation

The frozen-encoder baseline (Exp 17) plateaued at test AUROC 0.718 and missed 60% of hateful memes (recall 0.404). The natural question: can we improve by giving the model more capacity, either by unfreezing some encoder layers, or by replacing late-fusion concatenation with cross-attention that lets text and image interact at multiple layers?

Two architectural changes were tested independently, plus a control run.

---

### Exp 20 — Defrost late-fusion (unfreeze last 2 layers of CLIP + HateBERT)
Config: `experiment_memes_defrost.yaml`

**What changed vs Exp 17:** the last 2 transformer blocks of CLIP vision encoder and HateBERT are unfrozen and fine-tuned. Differential learning rate: head/fusion at 1e-4, encoders at 1e-5 (10× lower). Cosine warmup + decay over 15 epochs.

Trainable parameters: 29,665,282 / 261,088,003 (11.36%) — ~90× more than Exp 17.

Training dynamics: val AUC peaks at epoch 4 (0.6141), then plateaus and slowly degrades while train_loss continues to drop. Classic overfitting signature.

Results:
- best val AUROC: **0.6141** (epoch 4)
- test accuracy: 0.6027
- test F1 (hateful) @ 0.5: 0.3407
- test AUROC: 0.6176
- test F1 (hateful) @ tuned threshold 0.120: **0.5994**

Interpretation: defrosting *degrades* val AUROC vs the local frozen baseline (Exp 22: 0.6106). The encoders, even with a 10× lower learning rate, drift away from their useful pre-trained representations. With only 12,887 training examples there is not enough signal to re-fit them productively. This is a meaningful negative result confirming the architectural choice of the original paper.

---

### Exp 21 — Bidirectional cross-attention fusion
Config: `experiment_memes_crossattn.yaml`

**What changed vs Exp 17:** instead of concatenating final CLS embeddings (late fusion), text tokens attend over image patches and image patches attend over text tokens through 2 layers of bidirectional cross-attention (8 heads each, residual + feed-forward). Encoders are also unfrozen for their last 2 layers (same as Exp 20). Same differential LR.

The motivation comes from the literature: for hateful memes specifically, the meaning emerges from text-image *interaction* (innocuous caption + cruel image, etc.). Late fusion only sees the final CLS pair; cross-attention models the interaction explicitly.

Trainable parameters: 57,889,538 / 290,296,067 (19.94%) — ~180× more than Exp 17.

Training dynamics: val AUC climbs to 0.6304 at epoch 7, then degrades. Same overfitting pattern as Exp 20 but reaches a higher peak.

Results:
- best val AUROC: **0.6304** (epoch 7)
- test accuracy: 0.6080
- test F1 (hateful) @ 0.5: 0.3474
- test AUROC: **0.6249**
- test F1 (hateful) @ tuned threshold 0.060: **0.5966**

Interpretation: cross-attention is the **only architectural change that improved val/test AUROC** compared to the local frozen baseline (Exp 22). Gain: +0.020 val AUROC, +0.019 test AUROC. Tuned F1 is essentially tied with the others because threshold tuning compensates for calibration differences.

---

### Exp 22 — Frozen baseline reproduction (control)
Config: `experiment_memes_multimodal.yaml`

Run after observing that Exp 20 and Exp 21 were significantly below the originally reported Exp 17 numbers (0.6907 val, 0.7179 test AUROC). The control re-ran Exp 17 on the same local machine to determine whether the gap was caused by environment differences (transformers 5.x, pyarrow 15, different HF snapshot) or by Exp 20/21 themselves.

Results:
- best val AUROC: **0.6106** (epoch 5)
- test accuracy: 0.5900
- test F1 (hateful) @ 0.5: 0.4029
- test AUROC: 0.6058
- test F1 (hateful) @ tuned threshold 0.170: **0.5956**

Interpretation: the local frozen baseline reproduces at **0.6106 val AUC**, not 0.6907. The ~0.08 gap is due to environment changes (newer transformers, dataset snapshot, run-to-run variance) — it is **not** a problem with Exp 20/21. Apples-to-apples, Exp 20 and Exp 21 both outperform the local control on val AUROC; Exp 21 (cross-attention) is clearly the best architecture.

---

### Exp 23 — Threshold tuning (post-hoc on all three checkpoints)
Script: `src/threshold_tune.py`

Default evaluation uses threshold 0.5 to convert probabilities to predictions, but AUROC is much higher than F1 hateful on all three models — meaning the ranking quality is OK but the operating point is wrong (the trained models concentrate probability mass below 0.5 on the hateful class). Sweeping thresholds in [0.05, 0.95] on the validation set and picking the F1-maximizing one yields large improvements **on every checkpoint**:

| Model (this run) | F1 @ 0.5 | F1 tuned | Tuned threshold | Δ F1 |
|------------------|----------|----------|-----------------|------|
| Frozen baseline (Exp 22) | 0.403 | 0.596 | 0.170 | +0.193 |
| Defrost (Exp 20) | 0.341 | 0.599 | 0.120 | +0.259 |
| Cross-attention (Exp 21) | 0.347 | 0.597 | 0.060 | +0.249 |

After threshold tuning the three models converge to F1 ≈ 0.60 — the AUROC differences mostly disappear at the operating-point level, confirming that the bulk of the variance was probability calibration, not ranking quality.

This is a free, principled improvement that should be applied to any binary classifier where class imbalance shifts the optimal operating threshold away from 0.5. Across all three models we get ≈ +0.20 F1 hateful with zero retraining.

---

## Part 2 (cont.) summary table — all comparable runs

| Exp | Architecture | Trainable | Val AUC | Test AUC | F1 @ 0.5 | F1 tuned |
|-----|--------------|-----------|---------|----------|----------|----------|
| 22 (control) | Frozen late-fusion | 328k | 0.6106 | 0.6058 | 0.403 | 0.596 |
| 20 | Defrost late-fusion (unfreeze 2 layers) | 29.7M | 0.6141 | 0.6176 | 0.341 | 0.599 |
| **21** | **Cross-attention + defrost** | **57.9M** | **0.6304** | **0.6249** | 0.347 | **0.597** |

**Key takeaways:**
1. **Cross-attention is the only architecture that beats the frozen baseline on AUROC** (+0.019 test). The defrost-only change is a wash within this run's noise.
2. **Threshold tuning is the single biggest F1 gain** (+0.19 to +0.26 hateful F1) and is essentially free.
3. **Defrosting on its own does not pay off** with this dataset size. Adding capacity to an already-frozen-strong baseline requires either more training data, stronger regularisation, or an architectural change that adds expressiveness without requiring the encoders to drift far from their pre-training (cross-attention does exactly this).
4. **All three models concentrate probability below 0.5 on the hateful class**, requiring tuned thresholds in 0.06–0.17 to maximise F1. The default threshold of 0.5 is wrong for this task.

---

---

# Part 2 (cont. cont.) — Critical bug discovery and HarMeme replacement (Exp 24-27)

## Critical bug — grey-placeholder images

After publishing Exp 20-23, a sanity check (`torch.flip(pixel_values, dims=[3])`
followed by a forward pass producing **identical logits**) revealed that the
HuggingFace dataset `limjiayi/hateful_memes_expanded` only ships JSONL
metadata. The `img/*.png` files do **not** exist in the snapshot. The
`HatefulMemesDataset` silently fell back to a uniform 128-grey placeholder
image for every meme via `to_pil_rgb()`'s `FileNotFoundError` handler.

Consequence: **every "multimodal" run in this log (Exp 15-23) had effectively
zero usable visual signal**. The "image-only" Exp 16 result (AUROC 0.671)
is wrong; the "text vs image dominance" finding does not hold; the
defrost/cross-attention comparison was a comparison of *head architectures*
fed identical zero-information vision features, not a real multimodal study.

Fix: `dataset_memes.py` now (a) accepts an explicit `local_img_dir`, and
(b) raises `FileNotFoundError` during init when sampled paths don't exist,
controlled by `strict_images: true` (default). Future runs cannot silently
regress into this state.

Recovery: the user provided a local MMF-style image folder containing the
**HarMeme** dataset (Pramanick et al. 2021) — COVID-themed memes with
3-class harmfulness labels (`not harmful`, `somewhat harmful`, `very harmful`)
that we binarise to `harmful` vs `not_harmful`. HarMeme has real images
(3013 train / 177 val / 354 test) and serves as a drop-in replacement for
re-running the multimodal architecture comparison.

The Facebook Hateful Memes images proper are no longer available (DrivenData
archived the competition; the alternative HF datasets `neuralcatcher/hateful_memes`
and `limjiayi/hateful_memes_expanded` both ship metadata only). HarMeme is
the closest available image-grounded benchmark.

---

### Exp 24 — HarMeme frozen baseline (control)
Config: `experiment_harmeme_frozen.yaml`

Trainable parameters: 328 450 / 261 088 003 (head only).
Training dynamics: best val AUROC at epoch 5 (0.8700) — typical for a small
dataset (3013 train) with a small head; train_loss continues to drop from
0.58 → 0.29 while val plateaus from epoch 5 onward.

- best val AUROC: **0.8700** (epoch 5)
- test accuracy: 0.8164
- test F1 (harmful) @ 0.5: 0.7510 — precision 0.715, recall 0.790
- test AUROC: 0.8766
- test F1 (harmful) @ tuned threshold 0.510: 0.7538

Confusion matrix (test, threshold 0.5):
```
              pred_not  pred_harm
true_not       191        39
true_harm       26        98
```

Note the tuned threshold (0.510) is almost the default — with real images
the frozen model is well-calibrated. Compare to the broken-multimodal
Exp 22 frozen control where best threshold was 0.170; that calibration gap
was a symptom of training on no-information vision features.

---

### Exp 25 — HarMeme defrost (unfreeze last 2 layers) — **BEST**
Config: `experiment_harmeme_defrost.yaml`

Trainable parameters: 29 665 282 / 261 088 003 (11.4%).

Training dynamics: best val AUROC at epoch 1 (0.8987), then monotonic decay
to 0.8419 at epoch 15 as the model overfits the small training set.
train_loss → 0.013 by epoch 15 (model has memorised the train set).
Cosine decay + epoch-1 best saving means the right checkpoint is still kept.

- best val AUROC: **0.8987** (epoch 1)
- test accuracy: 0.8531 (@0.5) → 0.8644 (@tuned)
- test F1 (harmful) @ 0.5: **0.8045** — precision 0.753, recall 0.863
- test AUROC: **0.9101**
- test F1 (harmful) @ tuned threshold 0.700: 0.8033

Confusion matrix (test, threshold 0.5):
```
              pred_not  pred_harm
true_not       195        35
true_harm       17       107
```

**Defrosting helps on HarMeme** (+0.034 test AUC over frozen, +0.054 F1).
This is the exact OPPOSITE of what we found on the broken Hateful Memes
runs, where defrosting hurt. With real images, the last two encoder
layers have actual signal to refine; with grey placeholders, defrosting
just destabilised the encoders. The bug fix flipped the
architecture-comparison conclusions.

Diagnostic remarks:
- The single-epoch peak is a hallmark of fine-tuning on a small dataset:
  the pre-trained features are already 80% of the way there, and one pass
  is enough to adapt them. More epochs only over-fit.
- An ablation that we did not run but should run next: defrost just CLIP
  (vision-only adaptation) vs defrost just HateBERT (text-only adaptation)
  to attribute the +0.034 AUC gain between modalities.

---

### Exp 26 — HarMeme cross-attention fusion
Config: `experiment_harmeme_crossattn.yaml`

Trainable parameters: 57 889 538 / 290 296 067 (19.9%).
Architecture: 2 layers of bidirectional cross-attention (8 heads each, 768-d,
residual + GELU-MLP), then concat of pooled text-CLS and image-CLS,
two-layer MLP head.

Training: best val AUROC at epoch 3 (0.8814). Crashed at end of epoch 6
during a per-epoch checkpoint write (`PytorchStreamWriter` enforce-fail
due to a full C: drive — 953 GB occupied by accumulated 1 GB checkpoints).
The `best_model.pt` from epoch 3 was already on disk. After cleaning up
the per-epoch dumps, train_multimodal.py and train_crossattn.py were
patched to **stop emitting per-epoch checkpoints** (only `best_model.pt`),
which keeps each multimodal results dir at ~1 GB instead of ~16 GB.

- best val AUROC: **0.8814** (epoch 3)
- test accuracy: 0.7062 @ 0.5 → 0.8277 @ tuned
- test F1 (harmful) @ 0.5: 0.6905 — precision 0.547, recall 0.935
- test AUROC: 0.8829
- test F1 (harmful) @ tuned threshold 0.890: 0.7715

Confusion matrix (test, threshold 0.5):
```
              pred_not  pred_harm
true_not       134        96
true_harm        8       116
```

The @0.5 model is heavily biased toward predicting `harmful` (recall 0.94,
precision 0.55). Threshold tuning to 0.89 recovers F1 to 0.77 by trimming
false positives. Cross-attention is comparable to frozen on AUROC (+0.006)
but lost to defrost.

Hypothesis for the under-performance vs defrost: the 2 cross-attention
blocks add ~28M randomly-initialised parameters that have to be learned
from 3013 examples. The earlier broken-multimodal cross-attn (Exp 21) had
a slight edge over defrost; with real images the opposite is true — more
fusion machinery competes with the gradient signal that now usefully
reaches the unfrozen encoder layers. Likely fixes (not run): lower fusion
LR, more warmup, freezing fusion for the first few epochs while the
encoders adapt.

---

### Exp 27 — HarMeme ensemble + TTA
Script: `src/ensemble_eval.py`

Uniform average over the three checkpoints (frozen, defrost, cross-attn).
Threshold tuned on validation as before.

**Ensemble without TTA:**
- test AUROC: 0.8991
- test F1 (harmful) @ 0.5: 0.7560
- test F1 (harmful) @ tuned threshold 0.730: 0.8000

**Ensemble + TTA (horizontal flip + 5 deterministic crops, average):**
- test AUROC: 0.8909
- test F1 (harmful) @ 0.5: 0.7361
- test F1 (harmful) @ tuned threshold 0.700: 0.7764

The ensemble (no TTA) is essentially tied with defrost-alone and clearly
weaker than defrost on AUROC. The frozen and cross-attn members drag the
combined probabilities away from defrost's better calibration.

**Key TTA finding: TTA HURTS on memes.** Average loss of −0.008 AUROC,
−0.024 F1. Memes have *text overlays inside the image* — horizontal flip
mirrors that overlay (often making the text unreadable in feature space)
and 5-crop variants drop pieces of the text. TTA assumes the prediction is
invariant to mild crop/flip; on memes this assumption is violated by
construction. Lesson: TTA is task-dependent. Don't apply image-augmentation
TTA to tasks where the image contains task-relevant text.

---

## Part 2 (cont. cont.) summary — comparable HarMeme runs

| Exp | Architecture | Trainable | Test AUC | F1 @ 0.5 | F1 tuned | Tuned thr |
|-----|--------------|-----------|----------|----------|----------|-----------|
| 24 | Frozen late-fusion (baseline) | 328k | 0.8766 | 0.7510 | 0.7538 | 0.510 |
| **25** | **Defrost late-fusion (unfreeze 2)** | 29.7M | **0.9101** | **0.8045** | **0.8033** | 0.700 |
| 26 | Cross-attention + defrost 2 | 57.9M | 0.8829 | 0.6905 | 0.7715 | 0.890 |
| 27a | Uniform ensemble of 24+25+26 | n/a | 0.8991 | 0.7560 | 0.8000 | 0.730 |
| 27b | Ensemble + TTA (flip + 5 crops) | n/a | 0.8909 | 0.7361 | 0.7764 | 0.700 |

**Key takeaways:**
1. **Defrosting wins clearly on a real image dataset.** The "defrosting fails
   on memes" claim from Exp 20-23 was an artefact of broken image loading,
   not a real architectural finding.
2. **Cross-attention does not beat simple late-fusion on this small dataset.**
   Three thousand examples is not enough to learn extra fusion machinery
   reliably from scratch.
3. **Ensembling weak + strong models does not beat the strong model alone.**
   Uniform-weighted ensemble pulled the prediction back toward the weaker
   frozen and cross-attn members.
4. **TTA hurts on memes** because the task-relevant text is *inside the image*.

---

---

# Part 1 (cont.) — Counterfactual augmentation for bias mitigation (Exp 28-29)

## Motivation

Exp 14's bias analysis revealed a 0.22 macro-F1 gap between target groups,
driven by a spurious "racial/ethnic group mention → hatespeech" shortcut.
Counterfactual data augmentation (Liang et al. 2020, Saunders et al. 2022)
attempts to break this shortcut by adding training examples in which the
group reference is swapped while keeping the label and surrounding harmful
structure intact:

    Original  : "kikes are responsible for the bad economy" (hate, Jewish)
    CF #1     : "blacks are responsible for the bad economy" (hate, African)
    CF #2     : "asians are responsible for the bad economy" (hate, Asian)

Implementation: `src/counterfactual_augment.py` walks each training row,
identifies the dominant target group from the `targets` field, locates
lexical hooks for that group in the tokens (`GROUP_VOCAB` dictionary —
slurs in detection, neutral terms in replacement), and emits N_CF copies
substituting other groups' canonical neutral terms.

### Exp 28 — CF v1 (hate-only augmentation) — FAILED
Config: `experiment_v6_cf_augment.yaml`
Data: `train_cf.jsonl` (15 383 originals + 12 080 CF = 27 463 rows)
Augmentation policy: only `label_id ∈ {0 hate, 2 offensive}` rows get CF copies.

- best val macro-F1: 0.6777 (vs 0.6931 for Exp 14)
- test overall macro-F1: 0.6659
- bias gap (Other − Asian): **0.3064** (vs 0.223 for Exp 14)

Both overall and bias gap got **worse**. Diagnosis: by only augmenting
hateful posts, we added more "`<group>` is evil" templates for the
underrepresented groups (Hindu, Christian, Caucasian, Hispanic, Asian
each gained ~1 100 hateful CF rows). The model learned a stronger, not
weaker, "any mention of these groups means hate" shortcut.

### Exp 29 — CF v2 (balanced, all-label augmentation) — STILL WORSE THAN BASELINE
Config: `experiment_v7_cf_balanced.yaml`
Data: `train_cf_balanced.jsonl` (15 383 originals + 16 042 CF = 31 425 rows)
Augmentation policy: every label (hate, normal, offensive) gets CF copies.
Per-group label distribution stays balanced after augmentation.

- best val macro-F1: 0.6649
- test overall macro-F1: 0.6399
- bias gap (Other − Asian): **0.265** (vs 0.223 for Exp 14, 0.306 for v6)

Balanced augmentation recovers part of v6's damage but **still fails to
reduce the gap below the no-CF baseline**. Overall accuracy drops further
(−0.043 vs Exp 14) — the synthetic substitutions introduce label noise
HateBERT can detect (e.g. naive single-token swaps occasionally produce
sentences that no longer parse, or substitute "muslims" into an idiom that
was specific to the original group).

## CF augmentation summary

| Exp | CF policy | Train rows | CF rows | Val F1 | Test F1 (overall) | Bias gap |
|-----|-----------|------------|---------|--------|-------------------|----------|
| 14 | none | 15 383 | 0 | 0.693 | 0.683 | **0.223** |
| 28 (v6) | hate+offensive | 27 463 | 12 080 | 0.678 | 0.666 | 0.306 (worse) |
| 29 (v7) | all labels | 31 425 | 16 042 | 0.665 | 0.640 | 0.265 (worse) |

### Per-group breakdown (test, macro-F1)

Groups sorted by Exp 14 macro-F1 ascending (worst first). Δ-columns show
the gain/loss of CF vs the no-CF baseline. Asian and Hispanic are the two
smallest test groups (~50 each) and are also the worst on every model.

| Group     | N   | Exp 14 mF1 | v6 mF1 | Δ v6 vs 14 | v7 mF1 | Δ v7 vs 14 |
|-----------|-----|------------|--------|------------|--------|-------------|
| Asian     |  53 | 0.467      | 0.332  | **−0.135** | 0.367  | **−0.100** |
| Hispanic  |  44 | 0.473      | 0.430  | −0.043     | 0.467  | −0.006     |
| Jewish    | 210 | 0.509      | 0.545  | +0.036     | 0.474  | −0.035     |
| Islam     | 238 | 0.584      | 0.523  | −0.061     | 0.529  | −0.055     |
| African   | 388 | 0.584      | 0.561  | −0.023     | 0.540  | −0.044     |
| Arab      |  90 | 0.593      | 0.602  | +0.009     | 0.596  | +0.003     |
| Caucasian | 114 | 0.617      | 0.557  | −0.060     | 0.529  | −0.088     |
| Homosexual| 223 | 0.619      | 0.617  | −0.002     | 0.624  | +0.005     |
| Refugee   | 113 | 0.647      | 0.580  | −0.067     | 0.570  | −0.077     |
| Women     | 233 | 0.664      | 0.629  | −0.035     | 0.622  | −0.042     |
| Other     | 244 | 0.690      | 0.638  | −0.052     | 0.632  | −0.058     |

**Reading this table:**
- Asian degrades the most under v6 (−0.135); v7 partially recovers it.
- Only **Arab, Jewish (v6 only), Homosexual (v7 only)** improve under CF.
  Every other group, including the previously best-performing `Other`,
  regresses.
- v7 is generally less destructive than v6 (negative Δ are smaller) but
  still **net negative** in 8/11 groups vs the no-CF baseline.

**Two negative results consistent with each other.** Simple lexical
counterfactual augmentation does not reduce the target-group bias gap on
HateXplain + HateBERT; both naive policies made it slightly worse. This
is consistent with the bias-mitigation literature (Davidson et al. 2019,
Garg et al. 2019, Sap et al. 2019): swapping group tokens at the surface
level is not enough when the model has learned subtle co-occurrence
patterns at every layer of pre-training (HateBERT's RAL-E corpus contains
each group together with hateful language at very different rates).

### Why didn't it work? Three concrete hypotheses

1. **HateBERT's priors are stronger than the augmentation signal.** RAL-E
   is full of "JEW <hateful claim>" and "MUSLIM <hateful claim>"
   co-occurrences. Adding ~1 100 synthetic "Asian <hateful claim>" examples
   doesn't move the needle on the encoder's deep representation of the
   word "asian". The augmentation never reaches the encoder layers, which
   are fine-tuned anyway and have learned the bias before fine-tuning.
2. **The lexical-hook detector is too crude.** Single-token matching misses
   multi-word references ("south asian", "sub-saharan african") and
   inflected forms not in our vocabulary. Of 9 132 eligible rows in v6,
   only 6 040 had a detected hook — i.e. ~33% of hateful posts about a
   group never get a CF copy because the augmenter can't tell the group
   is mentioned.
3. **Substitution introduces label noise.** Some posts use group-specific
   slurs that, when replaced by a neutral term for another group, no
   longer make sense or change meaning ("kikes" → "muslims" in an
   idiomatic insult). HateBERT can detect this; the gradient pushes
   *against* the augmented signal.

### Stronger options not explored in this run
- **Template-based generation** (not single-token swap) — requires
  hand-written templates per claim type.
- **Counterfactual data from a generative model** — ask an LLM to rewrite
  each post with a different target group while preserving the harmful
  claim structure. Much more expensive but produces fluent text.
- **Adversarial debiasing** at the representation level (Zhang et al. 2018) —
  train an auxiliary classifier to predict the target group from CLS, and
  add a gradient-reversal layer so the main classifier loses that signal.
- **Better data**, period: HateXplain has ~50 Asian and ~44 Hispanic
  training examples. No augmentation trick can manufacture coverage where
  there is essentially none.

---

---

# Final Summary

## Best models

| Task | Model | Val metric | Test metric |
|------|-------|-----------|-------------|
| HateXplain (3-class) | Exp 14: HateBERT + focal + warmup + confidence weighting | val macro-F1: 0.693 | test macro-F1: 0.683 |
| Multimodal harmfulness (binary, HarMeme) | Exp 25: CLIP + HateBERT, defrost last 2 layers | val AUROC: 0.899 | **test AUROC: 0.910, F1 tuned: 0.803** |
| Original Facebook Hateful Memes (Exp 17 in old log) | not reproducible — see "grey-placeholder bug" section |

Note on memes: every "multimodal" run before Exp 24 was trained on a uniform
grey placeholder for the image input (HF dataset shipped metadata only).
The valid multimodal benchmark in this repo is HarMeme (Exp 24-27).

## Key lessons learned

**1. Domain pre-training is the most impactful single change.**
Switching from roberta-base to HateBERT added +0.123 val F1 on HateXplain. HateBERT's epoch-1 performance already exceeds the best roberta-base model. No combination of tricks (confidence weighting, focal loss, class weights) on roberta-base could match a domain-adapted backbone.

**2. Label-confidence weighting improves robustness on ambiguous data.**
Weighting training examples by annotator agreement (+0.024 val F1 over baseline) is a principled way to handle annotator disagreement. Removing ambiguous examples entirely (Exp 8) is worse; using them equally weighted is also worse. Soft down-weighting is the right approach.

**3. Focal loss improves calibration, not necessarily raw F1.**
Focal loss consistently produced zero high-confidence errors across all experiments where it was used, even when macro-F1 was similar to cross-entropy. The model is less overconfident on hard examples.

**4. Class weights create offensive/normal trade-off with roberta-base, but not with HateBERT.**
With roberta-base, class weights boost offensive recall at the cost of normal precision (normal→offensive confusion increases). HateBERT is robust to this: adding or removing class weights changes performance by <0.003. Domain-specific representations already encode the class boundaries implicitly.

**5. For hateful memes, image information dominates text.** ⚠️ **INVALIDATED by Exp 24 onward.**
AUROC 0.671 (image-only) vs 0.610 (text-only) was the result on the broken
HF dataset where image features were uniform grey (`(128, 128, 128)`) for
every example — i.e. effectively zero information. The 0.061 "gap" was
just noise in the random head. On the corrected HarMeme run (Exp 24-26)
this experiment was not redone with single-modality ablations because
HarMeme has only 354 test examples and the modality-ablation contrast
would not be statistically meaningful.

**6. Fairness bias in hate speech models is resistant to simple re-weighting.**
The bias analysis revealed a 0.22 macro-F1 gap between the best-performing target group (Other, 0.690) and worst (Asian, 0.467). The model has learned spurious shortcuts associating racial/ethnic group names with hatespeech labels, reflecting training data composition rather than genuine understanding. Inverse-frequency group weighting (Exp 18 & 19) failed to close this gap and degraded overall performance by ~3.5pp. This is consistent with the literature: addressing this bias requires more representative data per group, not re-weighting.

**7. For hateful memes, adding model capacity does not help; cross-attention does (marginally).** ⚠️ **INVALIDATED by Exp 24-26.**
This was the conclusion when the encoders trained on grey placeholders.
On HarMeme with real images, **defrost is clearly the best architecture**
(test AUC 0.910 vs 0.877 frozen vs 0.883 cross-attention). The "more
capacity doesn't help" conclusion held only because there was no useful
visual signal for the extra capacity to refine — see lesson 9 for the
corrected story.

**8. Threshold tuning is the single biggest free win on imbalanced binary classification.** ⚠️ **PARTIALLY OUTDATED by Exp 24-26.**
On the broken multimodal models, threshold tuning gave +0.20 F1 hateful
because the models were severely miscalibrated (optimal threshold 0.06–0.17).
On the corrected HarMeme runs the frozen and defrost models are
well-calibrated (optimal threshold 0.51 and 0.70 respectively, F1 gain
under +0.01) and only the cross-attention model still benefits significantly
(+0.08 F1, threshold 0.89). Lesson refined: threshold tuning is a free win
**when** the model is miscalibrated; on well-calibrated models the gain is
near-zero. Always check the optimal threshold to know whether tuning is
needed.

**9. (NEW) On a real multimodal benchmark, defrost > frozen > cross-attention.**
Exp 24-26 on HarMeme reverse lesson 7. With 3013 training examples and
real images, the cleanest setup is unfreezing the last 2 layers of CLIP
and HateBERT with a 10× lower encoder LR, keeping the simple concat fusion.
This gives test AUC 0.910 (+0.034 vs frozen). Cross-attention adds 28M
randomly-initialised fusion parameters; on this dataset size, those
parameters compete with the gradient signal that would otherwise reach
the now-trainable encoder layers, and the model under-performs.

**10. (NEW) Defrost peaks at epoch 1 on small datasets.**
On HarMeme (3013 train), Exp 25 reached its best val AUROC at the very
first epoch (0.8987). Subsequent epochs only over-fit. This is a hallmark
of fine-tuning a strong pre-trained model on a small dataset: pretraining
already does 80% of the work, one pass over the data is enough to adapt
it, more epochs just memorise.

**11. (NEW) TTA is task-dependent. On memes with text overlay, it hurts.**
Horizontal flip + 5 deterministic crops degraded ensemble F1 by 0.024 on
HarMeme (Exp 27b vs 27a). Reason: the discriminative text overlay is part
of the image — flips mirror it, crops drop it. TTA assumes invariance to
the augmentation that doesn't hold for tasks where the image *contains*
information about the label.

**12. (NEW) Naive lexical counterfactual augmentation does not reduce
bias on HateXplain.**
Both Exp 28 (hate-only) and Exp 29 (all-label) augmentation policies
INCREASED the bias gap (0.22 → 0.31 and 0.22 → 0.27 respectively).
HateBERT's deep priors about group co-occurrence with hateful language
override the surface-level token substitutions. Reducing this bias
requires either much more careful augmentation (LLM rewrites, adversarial
debiasing) or more data for under-represented groups.

**13. (NEW) Silent dataset bugs are dangerous; build assertive loaders.**
Every "multimodal" experiment before Exp 24 was effectively text-only
because `HatefulMemesDataset` silently returned a 128-grey placeholder
when an image file was missing. The bug went undetected through 8
experiments and would have stayed in the headline numbers if not caught.
Fix and lesson: loaders should fail loudly on missing data
(`strict_images: true` in `dataset_memes.py`), and any "multimodal" model
should be sanity-checked by verifying that
`torch.flip(pixel_values, dims=[3])` changes the output logits.

**14. (NEW) Always sanity-check the per-epoch checkpoint policy on disk.**
The default policy saved every epoch as a 1 GB `.pt` file. Across 4
multimodal runs × 15 epochs = 60 GB used silently. This eventually filled
the 953 GB C: drive and corrupted the cross-attention training mid-epoch.
Fixed: only `best_model.pt` is kept now.

## Remaining limitations

- **Offensive class is structurally hard** (F1≈0.535): defined negatively as
  "not hateful, not normal", leading to symmetric confusion in both
  directions. This is a labelling ambiguity issue, not purely a modelling
  failure.
- **Original Hateful Memes benchmark is not reproducible locally**: the HF
  snapshot ships metadata only and DrivenData has archived the source.
  HarMeme is the closest available image-grounded multimodal benchmark but
  is smaller (3013 vs 8500 train) and on a different topic (COVID vs
  generic memes), so absolute numbers should not be compared to the
  literature.
- **HarMeme is small** (3013 / 177 / 354). Validation has only 177 examples
  — a single misclassified borderline meme changes val F1 by ~0.5%. Be
  cautious about ranking models by val score alone.
- **Racial/ethnic group bias on HateXplain remains**. The model still
  over-predicts hate for posts mentioning racial/ethnic minorities. Neither
  group-aware re-weighting (Exp 18-19) nor counterfactual augmentation
  (Exp 28-29) closed the gap. The remaining honest path is targeted data
  collection and adversarial debiasing.
- **No single-modality ablation on HarMeme.** We did not redo "text only"
  vs "image only" vs "both" on HarMeme because the test set is too small
  for the contrast to be informative. The relative-contribution claim from
  Exp 15-17 was invalidated by the grey-placeholder bug; we do not replace
  it with a new claim.

---

---

# Appendix A — Cross-cutting observations

These are patterns we noticed across multiple experiments. They are weaker
than the numbered "key lessons learned" above (which are tied to specific
quantitative results) but useful for context.

### A1. Pretraining quality matters more than architecture cleverness
- HateBERT vs roberta-base on the same dataset: +0.12 val F1 with zero
  architectural change.
- Cross-attention vs late-fusion on memes: −0.03 AUC on real images
  (Exp 25 vs Exp 26). Architecture cleverness only pays off when there is
  enough data to learn the new parameters; otherwise it competes with
  pretraining.
- The single most cost-effective improvement we tested was "switch
  roberta-base to HateBERT". The architectural changes (focal loss, class
  weights, cross-attention, group-weights, CF augmentation) collectively
  added much less.

### A2. Small datasets reward parameter efficiency
- The frozen baseline on HarMeme (328k trainable) reaches AUC 0.877 in 5
  epochs. The cross-attention model (58M trainable) reaches AUC 0.881 in
  3 epochs and then overfits. More parameters → faster overfit → smaller
  effective gain.
- The "best" defrost on HarMeme is *less* defrost: only the last 2 layers,
  with encoder LR 10× lower than head LR. Unfreezing more layers with
  equal LR would saturate the gradient signal and over-fit faster.

### A3. Per-class trade-offs are sharp on 3-class HateXplain
- Adding class weights (Exp 12) boosted offensive F1 by 0.027 but cost
  normal F1 0.038 (more normal→offensive confusion).
- The offensive↔normal confusion is symmetric in both directions on every
  trained model. This is not a modelling failure but a definitional
  ambiguity: "offensive" is "not hateful, not normal", which is hard for
  the model and for human annotators (HateXplain reports 33% annotator
  disagreement on offensive labels).

### A4. Bias is multi-scale; surface fixes don't generalise
- Group-weighted loss (Exp 18-19, varying weights from 1 to 5): failed.
- Counterfactual lexical augmentation (Exp 28-29, doubling training data):
  failed.
- Even the per-group post-hoc analysis (analyze_target_bias.py) is itself
  noisy on small groups: Asian and Hispanic each have ~50 test examples,
  so ±0.05 F1 is well within sampling noise. The *consistency* of the
  ranking across experiments is the real signal.

### A5. "Free" inference tricks vary widely in value
- Threshold tuning: huge on miscalibrated models (+0.20 F1), near-zero on
  calibrated ones (+0.003 F1).
- Ensembling weak + strong: pulls down the strong member (HarMeme:
  ensemble AUC 0.899 vs defrost-alone 0.910). Useful only when members
  are comparable.
- TTA: image-domain TTA *hurts* when the image contains text overlay.
  Lesson — don't reach for inference tricks blindly; verify on validation
  first.

---

# Appendix B — Reproducibility

All experiments use `seed = 42` set in PyTorch, NumPy, and Python's `random`.
Reproducibility is deterministic up to PyTorch's CUDA non-determinism (cuDNN
convolutions and reduction ops); expect val F1 to vary ±0.003 between runs
on the same machine.

### Environment
- Python 3.12.0 (pyenv-win)
- torch 2.6.0 + CUDA 12.4
- transformers 5.9.0
- datasets 3.6.0
- pyarrow 15.0.2 (note: pyarrow 24.x is INCOMPATIBLE with torch 2.6.0
  + datasets 3.6.0 → segfault on `from datasets import load_dataset`
  when torch is already imported; downgrading was required).
- GPU: NVIDIA RTX 3060 (12 GB VRAM).

### Workspace layout
```
my_ml_project/
├── configs/               YAML, one per experiment
├── data/
│   ├── raw/               HateXplain HF snapshot
│   └── processed/         Preprocessed JSONL (train, val, test,
│                          train_cf, train_cf_balanced)
├── src/
│   ├── prepare_data.py             HateXplain HF → JSONL with confidence + rationales
│   ├── counterfactual_augment.py   CF augmentation generator
│   ├── dataset.py                  HateXplain torch Dataset
│   ├── dataset_memes.py            Hateful Memes (HF metadata + local images)
│   ├── dataset_harmeme.py          HarMeme (local MMF-style layout)
│   ├── dataset_factory.py          Dispatch between dataset_memes and dataset_harmeme
│   ├── model.py                    TransformerClassifier (text)
│   ├── model_multimodal.py         CLIP+HateBERT with late-fusion concat
│   ├── model_crossattn.py          CLIP+HateBERT with bidirectional cross-attention
│   ├── train.py                    HateXplain trainer
│   ├── train_multimodal.py         Multimodal trainer (late fusion)
│   ├── train_crossattn.py          Multimodal trainer (cross-attention)
│   ├── evaluate.py                 HateXplain test set evaluator
│   ├── evaluate_multimodal.py      Multimodal test set evaluator
│   ├── threshold_tune.py           Tune threshold on val, evaluate on test
│   ├── ensemble_eval.py            Uniform ensemble + optional TTA
│   ├── analyze_target_bias.py      Per-group bias analysis
│   └── demo.py                     Gradio live demo (text + meme tabs)
└── results/
    ├── experiment_log.md           This file
    └── <run_name>/                 One per experiment
        ├── checkpoints/best_model.pt
        ├── training_log.json
        ├── threshold_tuned_summary.json (memes only)
        └── target_bias_analysis.json (HateXplain bias runs)
```

### Reproducing the best results

```bash
# 1. Install dependencies (force pyarrow < 16 to avoid the torch segfault)
pip install -r requirements.txt
pip install "pyarrow<16"

# 2. Prepare HateXplain processed data
python src/prepare_data.py

# 3. Best HateXplain model (Exp 14) — test macro-F1 0.683
python src/train.py     --config configs/experiment_v4b_hatebert_noweights.yaml
python src/evaluate.py  --config configs/experiment_v4b_hatebert_noweights.yaml \
                        --run_id <timestamp_from_train>

# 4. Best multimodal model (Exp 25, HarMeme) — test AUC 0.910, F1 0.803
# (Requires the HarMeme dataset locally at the path in the config.)
python src/train_multimodal.py --config configs/experiment_harmeme_defrost.yaml
python src/threshold_tune.py   --config configs/experiment_harmeme_defrost.yaml \
                               --model_type latefusion

# 5. Counterfactual augmentation (negative results — Exp 28, 29)
python src/counterfactual_augment.py --input data/processed/train.jsonl \
                                     --output data/processed/train_cf.jsonl \
                                     --n_cf 2 --labels hate_offensive
python src/counterfactual_augment.py --input data/processed/train.jsonl \
                                     --output data/processed/train_cf_balanced.jsonl \
                                     --n_cf 2 --labels all
python src/train.py                --config configs/experiment_v6_cf_augment.yaml
python src/train.py                --config configs/experiment_v7_cf_balanced.yaml
python src/analyze_target_bias.py  --config configs/experiment_v7_cf_balanced.yaml \
                                   --run_id <timestamp>

# 6. Live demo (Gradio, two tabs: text + meme)
pip install gradio pillow
python src/demo.py \
    --text_checkpoint results/v4b_hatebert_noweights/<timestamp>/checkpoints/best_model.pt \
    --meme_checkpoint results/harmeme_defrost/checkpoints/best_model.pt
```

### Validating image loading (avoid the silent grey-placeholder bug)
Before trusting any new multimodal model, run:
```python
import torch
batch = next(iter(val_loader))
flipped = torch.flip(batch["pixel_values"], dims=[3])
out_orig = model(pixel_values=batch["pixel_values"], ...)
out_flip = model(pixel_values=flipped, ...)
assert not torch.allclose(out_orig["logits"], out_flip["logits"]), \
    "Model is flip-invariant — images are likely grey placeholders!"
```
