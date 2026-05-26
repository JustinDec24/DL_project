# Baseline results — RoBERTa on HateXplain

## Setup
- Model: roberta-base
- Task: 3-class classification
- Labels:
  - 0
  - 1
  - 2
- Loss: cross-entropy
- Rationales used: no
- Auxiliary loss: no

## Validation
- Best validation macro-F1: 0.6874

## Test
- Accuracy: 0.6949
- Macro-F1: 0.6879

## Per-class F1
- Class 0: 0.7631
- Class 1: 0.7420
- Class 2: 0.5585

## Notes
- Class 2 is the hardest class.
- Baseline is fully working on the GPU cluster.
- This result will be the reference point for future improvements.