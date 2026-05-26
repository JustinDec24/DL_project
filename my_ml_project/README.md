# Hate Speech Detection — Text & Multimodal

Ce projet étudie la détection de discours de haine sous deux angles :
classification textuelle multi-classe et classification multimodale image+texte.
Il inclut également une analyse de biais par groupe cible et une tentative
(documentée) de correction de ces biais.

## Tâches

### Partie 1 — HateXplain (texte, 3 classes)
Classification de posts en `hatespeech`, `normal` ou `offensive`.
La difficulté principale tient à la frontière `normal` / `offensive` :
le contenu offensant n'est pas nécessairement haineux, et cette frontière
est intrinsèquement ambiguë même pour les annotateurs humains.

### Partie 2 — Hateful Memes (image + texte, binaire)
Classification de memes en `hateful` / `not_hateful`. Le caractère haineux
émerge souvent uniquement de la *combinaison* image+texte (caption anodine
sur image offensante, ou dog-whistle textuel sur image neutre), ce qui en
fait un problème fondamentalement multimodal.

## Datasets

| Dataset | Source | Splits (train/val/test) | Classes |
|---------|--------|-------------------------|---------|
| HateXplain | `hatexplain` (HF, Mathew et al., 2021) | 15 383 / 1 922 / 1 924 | 3 |
| Hateful Memes | `limjiayi/hateful_memes_expanded` (HF) | 12 887 / 1 040 / 3 000 | 2 |

HateXplain fournit, en plus du label : la confiance d'annotation (fraction
d'annotateurs en accord), des rationales token-level, et un groupe cible
(communauté attaquée). Ces signaux sont exploités dans plusieurs expériences.

## Meilleurs résultats

| Tâche | Modèle final | Métrique val | Métrique test |
|-------|--------------|--------------|---------------|
| HateXplain (3-class) | HateBERT + focal loss + warmup + label-confidence weighting | macro-F1 = **0.693** | macro-F1 = **0.683** |
| Hateful Memes (binaire) | CLIP ViT-B/32 + HateBERT, encoders gelés, fusion par concaténation | AUROC = **0.691** | AUROC = **0.718** |

Voir [`results/experiment_log.md`](results/experiment_log.md) pour le journal
complet des 19 expériences (hypothèses, configs, dynamiques d'entraînement,
interprétations, ablations, et résultats négatifs).

## Analyse de biais

Une analyse par groupe cible (Asian, Jewish, African, Women, etc.) révèle un
**écart de 0.22 macro-F1** entre le meilleur et le pire groupe sur le modèle
final HateXplain. Le modèle a appris un raccourci spurieux
« mention de groupe ethnique → hatespeech », reflétant la composition du
jeu d'entraînement. Une tentative de correction par pondération inverse de
fréquence (group, class) a dégradé les performances globales sans réduire
l'écart — résultat négatif assumé, cohérent avec la littérature
(Davidson et al., 2019).

Détails : [`src/analyze_target_bias.py`](src/analyze_target_bias.py) et
[`results/experiment_log.md`](results/experiment_log.md) (section *Bias Analysis*).

## Structure du repo

```
my_ml_project/
├── configs/             # un YAML par expérience (backbone, loss, hyperparams)
├── data/
│   ├── raw/             # HateXplain HF snapshot
│   └── processed/       # JSONL préparés (train/val/test)
├── src/
│   ├── prepare_data.py          # HateXplain → JSONL avec confidence + rationales
│   ├── prepare_implicit_binary.py  # (exploratoire, voir Extensions)
│   ├── dataset.py / dataset_memes.py
│   ├── model.py / model_multimodal.py
│   ├── train.py / train_multimodal.py
│   ├── evaluate.py / evaluate_multimodal.py
│   └── analyze_target_bias.py
└── results/
    ├── experiment_log.md        # journal des expériences (référence principale)
    ├── baseline_results.md      # historique, conservé pour traçabilité
    └── checkpoints/
```

## Utilisation

```bash
pip install -r requirements.txt

# Préparation des données HateXplain
python src/prepare_data.py

# Entraînement (ex. modèle final HateXplain)
python src/train.py --config configs/experiment_v4b_hatebert_noweights.yaml

# Évaluation sur test
python src/evaluate.py --config configs/experiment_v4b_hatebert_noweights.yaml --run_id <timestamp>

# Analyse de biais par groupe cible
python src/analyze_target_bias.py --config configs/experiment_v4b_hatebert_noweights.yaml --run_id <timestamp>

# Pipeline multimodale (Hateful Memes)
python src/train_multimodal.py --config configs/experiment_memes_multimodal.yaml
python src/evaluate_multimodal.py --config configs/experiment_memes_multimodal.yaml
```

## Techniques utilisées

- **Label-confidence weighting** : pondération de la loss par l'accord
  inter-annotateurs (down-weight des exemples ambigus sans les jeter).
- **Focal Loss** (γ=2) : réduit le poids des exemples faciles, améliore
  la calibration (0 erreurs haute-confiance observées).
- **LR warmup + cosine decay**, gradient clipping.
- **HateBERT** (`GroNLP/hateBERT`) : pré-entraînement domaine sur RAL-E,
  amélioration la plus impactante (+0.123 val F1 vs `roberta-base`).
- **CLIP ViT-B/32** + HateBERT en *late fusion* (concaténation) avec
  encodeurs gelés pour la tâche multimodale.

## Extensions possibles

- **Implicit vs explicit hate speech.** Question initiale du projet, non
  résolue à ce stade. HateXplain ne fournit pas cette distinction
  directement. Une première tentative (Exp 4) utilisant ImplicitHate
  (Twitter) vs HateXplain `normal` (Reddit) atteignait F1 = 0.98 mais
  les résultats sont **invalidés par un confounder de source**
  (le modèle apprend le style de plateforme plutôt que la sémantique
  d'implicit hate). Une approche valide demanderait un dataset où
  implicit et explicit hate proviennent de la même source.
- **Dégeler les dernières couches** des encodeurs sur Hateful Memes
  (le hook `unfreeze_last_n_layers` existe déjà dans la config).
- **Threshold tuning + courbe PR** pour améliorer le recall hateful
  (actuellement 0.40 sur les memes).
- **Réduction du biais inter-groupes** par collecte de données plus
  équilibrée (le re-weighting seul ne suffit pas, voir Exp 18 & 19).
