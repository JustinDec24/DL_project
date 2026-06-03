"""
Counterfactual augmentation for HateXplain.

Motivation
----------
The bias analysis (results/experiment_log.md) showed the best model has
learned a spurious shortcut: "post mentions racial/ethnic group X" ->
"predict hatespeech". This happens because in the training data, posts
mentioning groups like Jewish, African, Arab are predominantly hateful
(60-68% hate rate). The model over-fits the group name itself instead of
the harmful claim around it.

Counterfactual augmentation (Liang et al. 2020, Saunders et al. 2022) breaks
this shortcut by adding training examples where the group reference is
swapped while keeping the label and the surrounding harmful structure
intact. Example:

    Original  : "kikes are responsible for the bad economy"  (hate, target=Jewish)
    CF #1     : "blacks are responsible for the bad economy" (hate, target=African)
    CF #2     : "muslims are responsible for the bad economy"(hate, target=Islam)

After training on (original + CF), the model can no longer rely on "jew"
alone to predict hate, because the same harmful template is now labelled
hate when mentioning every group equally. The signal forced to be the
*claim* ("X are responsible for the bad economy"), not the target identity.

What this script does
---------------------
1. Reads data/processed/train.jsonl
2. For each row, identifies the dominant target group (majority across
   the 3 annotators) — same logic as analyze_target_bias.py
3. Locates tokens in the text that lexically refer to that group, using
   the GROUP_VOCAB dictionary below (slurs + neutral terms collapsed to a
   single canonical group key)
4. For each successfully tagged hateful (or offensive) row, emits N_CF
   counterfactuals by replacing those tokens with neutral terms from
   *other* groups, cycling through the group list
5. Writes train_cf.jsonl = original + CF rows

A few important guardrails:
- ALL eligible labels (hate=0, normal=1, offensive=2) get CF copies. This is
  critical: if we only augment hateful posts, we add more "X is evil" for
  the underrepresented groups and *reinforce* the spurious "group name -> hate"
  association we are trying to break (verified empirically — see Exp v6 in
  the experiment log). Augmenting all labels keeps the per-group label
  distribution balanced after augmentation.
- We replace with NEUTRAL canonical terms ("muslims", "blacks", "jews",
  "asians", ...) — never with slurs. The harmful structure stays, the
  target identity changes, the slur->slur cycle that would also be valid
  is deliberately avoided to keep this code free of slur generation.
- The rationale_mask and label_confidence are copied unchanged. The
  targets field is updated to the new group so analyze_target_bias.py
  can re-measure the per-group gap on the augmented training set.

Usage
-----
    python src/counterfactual_augment.py \\
        --input data/processed/train.jsonl \\
        --output data/processed/train_cf.jsonl \\
        --n_cf 2

`n_cf` is the number of counterfactual copies per eligible original row.
Set to 2 to double the contribution of each rebalanced row; higher values
risk over-representing the synthetic distribution.
"""

import json
import argparse
import random
import re
from collections import Counter, defaultdict


# Canonical group keys → list of lexical patterns that refer to that group
# in HateXplain posts. Lowercase, no punctuation. Slurs are included as
# DETECTION patterns only — they are never produced as substitutions.
GROUP_VOCAB = {
    "Jewish":     ["jew", "jews", "jewish", "kike", "kikes", "yid", "yids",
                   "hebrew", "hebrews", "zionist", "zionists"],
    "African":    ["black", "blacks", "african", "africans", "negro", "negroes",
                   "nigga", "niggas", "nigger", "niggers"],
    "Arab":       ["arab", "arabs", "arabic"],
    "Asian":      ["asian", "asians", "chinese", "chink", "chinks",
                   "gook", "gooks", "jap", "japs", "japanese", "korean"],
    "Hispanic":   ["mexican", "mexicans", "hispanic", "hispanics",
                   "latino", "latina", "spic", "spics", "wetback", "wetbacks"],
    "Islam":      ["muslim", "muslims", "islam", "islamic", "muzzie", "muzzies",
                   "raghead", "ragheads"],
    "Caucasian":  ["white", "whites", "caucasian", "honky", "honkies",
                   "cracker", "crackers"],
    "Homosexual": ["gay", "gays", "homosexual", "homosexuals", "homo", "homos",
                   "fag", "fags", "faggot", "faggots", "queer", "queers",
                   "lesbian", "lesbians", "dyke", "dykes",
                   "tranny", "trannies", "transsexual"],
    "Women":      ["woman", "women", "girl", "girls", "bitch", "bitches",
                   "slut", "sluts", "whore", "whores", "feminist", "feminists"],
    "Refugee":    ["refugee", "refugees", "immigrant", "immigrants"],
    "Hindu":      ["hindu", "hindus", "indian", "indians"],
    "Christian":  ["christian", "christians"],
}

# Neutral canonical replacements emitted by the augmenter (NOT slurs).
NEUTRAL_REPLACEMENT = {
    "Jewish":     "jews",
    "African":    "blacks",
    "Arab":       "arabs",
    "Asian":      "asians",
    "Hispanic":   "mexicans",
    "Islam":      "muslims",
    "Caucasian":  "whites",
    "Homosexual": "gays",
    "Women":      "women",
    "Refugee":    "refugees",
    "Hindu":      "hindus",
    "Christian":  "christians",
}

# Reverse map: token -> group it refers to. Built once at module load.
TOKEN_TO_GROUP = {}
for group, patterns in GROUP_VOCAB.items():
    for p in patterns:
        TOKEN_TO_GROUP[p.lower()] = group


def get_majority_target(targets_per_annotator):
    """Return the most-agreed-upon target group, or None if no consensus."""
    if not targets_per_annotator:
        return None
    mention = defaultdict(int)
    for annot_targets in targets_per_annotator:
        for t in annot_targets:
            if t and t.lower() != "none":
                mention[t] += 1
    if not mention:
        return None
    n = len(targets_per_annotator)
    agreed = {t: c for t, c in mention.items() if c > n / 2}
    if agreed:
        return max(agreed, key=agreed.get)
    return max(mention, key=mention.get)


def find_group_token_positions(tokens, group):
    """Return list of indices in `tokens` whose lowercased form refers
    to `group`. Multi-word patterns are not handled — HateXplain tokens
    are already word-split, so single-token matching is enough in practice."""
    positions = []
    for i, tok in enumerate(tokens):
        clean = re.sub(r"[^\w]", "", tok.lower())
        if not clean:
            continue
        if TOKEN_TO_GROUP.get(clean) == group:
            positions.append(i)
    return positions


def make_counterfactual(row, source_group, target_group, source_positions):
    """Produce a new row where source-group tokens are replaced with the
    canonical neutral term for target_group."""
    new_tokens = list(row["tokens"])
    replacement = NEUTRAL_REPLACEMENT[target_group]
    for i in source_positions:
        new_tokens[i] = replacement

    new_row = dict(row)  # shallow copy is fine: we don't mutate sub-lists below
    new_row["tokens"] = new_tokens
    new_row["text"] = " ".join(new_tokens)
    new_row["targets"] = [[target_group] for _ in row.get("targets", [])]
    new_row["id"] = f"{row['id']}__cf_{target_group}"
    new_row["is_counterfactual"] = True
    new_row["cf_source_group"] = source_group
    return new_row


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/train_cf.jsonl")
    parser.add_argument("--n_cf", type=int, default=2,
                        help="Counterfactual copies per eligible row")
    parser.add_argument("--labels", type=str, default="all",
                        choices=["all", "hate_only", "hate_offensive"],
                        help="Which label classes are eligible for CF augmentation. "
                             "'all' = augment every label class (recommended, balanced bias). "
                             "'hate_only' = augment label=0 only. "
                             "'hate_offensive' = augment labels 0 and 2 (the v1 broken policy).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_jsonl(args.input)
    print(f"Loaded {len(rows)} original rows from {args.input}")

    if args.labels == "all":
        eligible_labels = {0, 1, 2}
    elif args.labels == "hate_offensive":
        eligible_labels = {0, 2}
    else:
        eligible_labels = {0}
    print(f"Eligible labels for CF augmentation: {sorted(eligible_labels)}")

    augmented = list(rows)  # start with all originals
    n_eligible = 0
    n_tagged = 0
    n_cf_emitted = 0
    cf_by_source_group = Counter()
    cf_by_target_group = Counter()

    all_groups = sorted(NEUTRAL_REPLACEMENT.keys())

    for row in rows:
        if row["label_id"] not in eligible_labels:
            continue
        n_eligible += 1
        target = get_majority_target(row.get("targets", []))
        if target not in NEUTRAL_REPLACEMENT:
            continue  # can't CF this group (Other, or no consensus)
        positions = find_group_token_positions(row["tokens"], target)
        if not positions:
            continue  # group mentioned in annotation but no lexical hook
        n_tagged += 1

        candidate_targets = [g for g in all_groups if g != target]
        random.shuffle(candidate_targets)
        for new_target in candidate_targets[: args.n_cf]:
            cf_row = make_counterfactual(row, target, new_target, positions)
            augmented.append(cf_row)
            n_cf_emitted += 1
            cf_by_source_group[target] += 1
            cf_by_target_group[new_target] += 1

    save_jsonl(augmented, args.output)
    print(f"Saved {len(augmented)} total rows to {args.output} "
          f"(orig={len(rows)}, cf={n_cf_emitted})")
    print(f"Eligible rows by label   : {n_eligible}")
    print(f"Rows with lexical hook   : {n_tagged}")
    print(f"CF copies emitted        : {n_cf_emitted}")
    print(f"\nCF rows by source group (where original was):")
    for g, n in cf_by_source_group.most_common():
        print(f"  {g:<12s} {n}")
    print(f"\nCF rows by target group (new label-target):")
    for g, n in cf_by_target_group.most_common():
        print(f"  {g:<12s} {n}")


if __name__ == "__main__":
    main()
