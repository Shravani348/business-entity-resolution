#!/usr/bin/env python3
"""
local_eval.py — Local F0.5 macro-average scorer for the Business Entity
Resolution challenge.

Lets the team score any candidate model locally, on a held-out slice of the
*training* data (which has labels), before spending a leaderboard submission.

Implements the leaderboard's exact metric:
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    per source1_entity_id, then macro-averaged across all entities
    (singletons included; a correctly-predicted empty match list scores 1.0,
     any predicted match on a true singleton scores 0.0).

Reusable function (import this in other scripts, e.g. the model trainer):
    make_validation_split(source1_ids, val_fraction=0.2, seed=42)
        -> (train_ids: list[str], val_ids: list[str])
    Deterministic given the same ids/fraction/seed — every teammate who
    calls it against the same train_ground_truth.tsv gets the identical
    split, so results are comparable across runs and people.

Stdlib only (no pandas/numpy dependency), matching validate_submission.py's
style so this runs anywhere the raw TSVs are readable.

---------------------------------------------------------------------------
CLI
---------------------------------------------------------------------------

1) Generate a deterministic validation split of Source 1 entity ids:

    python3 local_eval.py split \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --val-fraction 0.2 --seed 42 \
        --out-dir splits/

    Writes splits/train_ids.txt and splits/val_ids.txt (one entity_id per
    line). Use the val_ids.txt entities as your held-out set: run your
    pipeline on those Source 1 entities only (against the *train* source2/3
    pools) and score the resulting predictions file with `score` below.

2) Score a predictions file (matching_results.tsv format) against ground
   truth, restricted to a set of entity ids (typically your val split):

    python3 local_eval.py score \
        --predictions output/matching_results.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --entity-ids splits/val_ids.txt

    (--entity-ids is optional; omit it to score against every entity_id
    present in the ground-truth file.)

3) Sanity-check the scorer + split themselves, with zero model involved:
   generates the trivial "predict all singletons" baseline (empty match
   list for every entity) over a given id set, writes it out, and scores
   it immediately. This should score close to the train-set singleton rate
   (~5.6% get F0.5 = 1.0, everything else scores 0.0 for that baseline) —
   run this FIRST, before trusting any real model's number.

    python3 local_eval.py baseline \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --entity-ids splits/val_ids.txt \
        --out output/baseline_singleton_predictions.tsv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_two_col_tsv(path):
    """Read a {source1_entity_id}\\t{comma-separated ids} TSV.

    Works for train_ground_truth.tsv, matching_results.tsv and
    candidate_pairs.tsv alike — the second column's *name* differs
    (matched_entity_ids / candidate_entity_ids) but the shape doesn't, so
    we just use column position, not header name.

    Returns dict: source1_entity_id -> set(matched/candidate entity_ids)
    Missing / empty second column -> empty set (singleton / no candidates).
    """
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return result
        for row_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < 1:
                continue
            source1_id = row[0].strip()
            if not source1_id:
                continue
            if source1_id in result:
                print(
                    f"WARNING: duplicate source1_entity_id '{source1_id}' "
                    f"at {path}:{row_num} — keeping the first occurrence",
                    file=sys.stderr,
                )
                continue
            raw = row[1].strip() if len(row) > 1 else ""
            ids = {x.strip() for x in raw.split(",") if x.strip()} if raw else set()
            result[source1_id] = ids
    return result


def _read_ids_file(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_ids_file(path, ids):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(i + "\n")


def _write_predictions_tsv(path, predictions, header_col_name="matched_entity_ids"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", header_col_name])
        for source1_id in predictions:
            writer.writerow([source1_id, ",".join(sorted(predictions[source1_id]))])


# ---------------------------------------------------------------------------
# Reusable: deterministic validation split
# ---------------------------------------------------------------------------

def make_validation_split(source1_ids, val_fraction=0.2, seed=42):
    """Deterministically split Source 1 entity ids into train/val subsets.

    Same (ids, val_fraction, seed) always yields the same split — so every
    teammate scoring locally, and the model-training script, all hold out
    the identical validation entities.

    Args:
        source1_ids: iterable of source1_entity_id strings (e.g. every key
            in train_ground_truth.tsv, or every id in train_source1.tsv).
        val_fraction: fraction (0, 1) held out for validation.
        seed: fixed random seed for reproducibility.

    Returns:
        (train_ids, val_ids): two sorted lists of entity_id strings,
        disjoint, together covering exactly the input ids.
    """
    if not (0 < val_fraction < 1):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    ids = sorted(set(source1_ids))  # sort first so shuffle is order-independent
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_entity(predicted_ids, true_ids, beta=0.5):
    """F_beta (default 0.5) for one Source 1 entity. Returns (precision, recall, f_beta).

    Singleton edge cases (explicit, per the challenge's own scoring rule):
      - true empty, predicted empty  -> correctly predicted singleton: F = 1.0
      - true empty, predicted non-empty -> false merge on a singleton:  F = 0.0
      - true non-empty, predicted empty -> missed every match:          F = 0.0
      - true non-empty, predicted non-empty -> standard P/R/F_beta
    """
    predicted_ids = set(predicted_ids)
    true_ids = set(true_ids)

    if not true_ids and not predicted_ids:
        return 1.0, 1.0, 1.0
    if not true_ids and predicted_ids:
        return 0.0, 0.0, 0.0
    if true_ids and not predicted_ids:
        return 0.0, 0.0, 0.0

    tp = len(predicted_ids & true_ids)
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    if precision == 0 and recall == 0:
        return precision, recall, 0.0
    beta2 = beta * beta
    f_beta = ((1 + beta2) * precision * recall) / (beta2 * precision + recall)
    return precision, recall, f_beta


def macro_score(predictions, ground_truth, entity_ids=None, beta=0.5):
    """Macro-average F_beta over entity_ids (default: every ground-truth id).

    Entities present in ground_truth but absent from predictions are scored
    as an empty prediction (with a warning) — this mirrors what actually
    happens on the leaderboard if your pipeline drops an entity, so don't
    silently skip them.

    Returns a dict: {
        'macro_f_beta', 'macro_precision', 'macro_recall', 'n_entities',
        'n_singletons_true', 'n_missing_from_predictions', 'per_entity'
    }
    per_entity maps source1_entity_id -> (precision, recall, f_beta).
    """
    ids = sorted(entity_ids) if entity_ids is not None else sorted(ground_truth.keys())

    missing_from_gt = [i for i in ids if i not in ground_truth]
    if missing_from_gt:
        print(
            f"WARNING: {len(missing_from_gt)} entity id(s) requested for "
            f"scoring have no ground-truth row at all (e.g. "
            f"{missing_from_gt[:3]}) — skipping them.",
            file=sys.stderr,
        )
    ids = [i for i in ids if i in ground_truth]

    missing_from_pred = [i for i in ids if i not in predictions]
    if missing_from_pred:
        print(
            f"WARNING: {len(missing_from_pred)} entity id(s) missing from "
            f"predictions (e.g. {missing_from_pred[:3]}) — scored as empty "
            f"prediction lists, same as the real leaderboard would treat "
            f"a missing row's absence of matches.",
            file=sys.stderr,
        )

    per_entity = {}
    n_singletons_true = 0
    for source1_id in ids:
        true_ids = ground_truth[source1_id]
        pred_ids = predictions.get(source1_id, set())
        if not true_ids:
            n_singletons_true += 1
        per_entity[source1_id] = score_entity(pred_ids, true_ids, beta=beta)

    n = len(per_entity)
    if n == 0:
        raise ValueError("No entities to score — check your id/ground-truth overlap.")

    macro_p = sum(v[0] for v in per_entity.values()) / n
    macro_r = sum(v[1] for v in per_entity.values()) / n
    macro_f = sum(v[2] for v in per_entity.values()) / n

    return {
        "macro_f_beta": macro_f,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "n_entities": n,
        "n_singletons_true": n_singletons_true,
        "n_missing_from_predictions": len(missing_from_pred),
        "per_entity": per_entity,
    }


def _print_report(result, beta=0.5):
    print(f"\n=== F{beta} macro-average score ===")
    print(f"Entities scored:        {result['n_entities']}")
    print(f"  of which true singletons: {result['n_singletons_true']} "
          f"({100 * result['n_singletons_true'] / result['n_entities']:.1f}%)")
    print(f"Missing from predictions: {result['n_missing_from_predictions']}")
    print(f"Macro precision:        {result['macro_precision']:.4f}")
    print(f"Macro recall:           {result['macro_recall']:.4f}")
    print(f"Macro F{beta}:              {result['macro_f_beta']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_split(args):
    gt = _read_two_col_tsv(args.ground_truth)
    train_ids, val_ids = make_validation_split(
        gt.keys(), val_fraction=args.val_fraction, seed=args.seed
    )
    _write_ids_file(Path(args.out_dir) / "train_ids.txt", train_ids)
    _write_ids_file(Path(args.out_dir) / "val_ids.txt", val_ids)
    print(f"Total entities: {len(gt)}")
    print(f"Train split:    {len(train_ids)} ({100 * len(train_ids) / len(gt):.1f}%)")
    print(f"Val split:      {len(val_ids)} ({100 * len(val_ids) / len(gt):.1f}%)  "
          f"(seed={args.seed}, val_fraction={args.val_fraction})")
    print(f"Written to {args.out_dir}/train_ids.txt and {args.out_dir}/val_ids.txt")


def cmd_score(args):
    predictions = _read_two_col_tsv(args.predictions)
    ground_truth = _read_two_col_tsv(args.ground_truth)
    entity_ids = _read_ids_file(args.entity_ids) if args.entity_ids else None
    result = macro_score(predictions, ground_truth, entity_ids=entity_ids, beta=args.beta)
    _print_report(result, beta=args.beta)


def cmd_baseline(args):
    ground_truth = _read_two_col_tsv(args.ground_truth)
    entity_ids = _read_ids_file(args.entity_ids) if args.entity_ids else list(ground_truth.keys())
    entity_ids = [i for i in entity_ids if i in ground_truth]
    baseline_predictions = {i: set() for i in entity_ids}  # predict all singletons
    _write_predictions_tsv(args.out, baseline_predictions)
    print(f"Wrote trivial 'predict all singletons' baseline for "
          f"{len(entity_ids)} entities to {args.out}")
    result = macro_score(baseline_predictions, ground_truth, entity_ids=entity_ids, beta=args.beta)
    _print_report(result, beta=args.beta)
    print(
        "\nSanity check: this should land close to the entities' true "
        "singleton rate (entities with 0 true matches score 1.0 here, "
        "everything else scores 0.0). If this number looks wrong, the bug "
        "is in the scorer or the split — fix that before trusting any real "
        "model's F0.5."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Local F0.5 macro-average scorer for the entity resolution challenge."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="Generate a deterministic train/val id split.")
    p_split.add_argument("--ground-truth", required=True,
                          help="Path to train_ground_truth.tsv")
    p_split.add_argument("--val-fraction", type=float, default=0.2)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--out-dir", required=True)
    p_split.set_defaults(func=cmd_split)

    p_score = sub.add_parser("score", help="Score a predictions file against ground truth.")
    p_score.add_argument("--predictions", required=True,
                          help="Path to a matching_results.tsv-format file")
    p_score.add_argument("--ground-truth", required=True,
                          help="Path to train_ground_truth.tsv")
    p_score.add_argument("--entity-ids", default=None,
                          help="Optional path to a newline-separated id file "
                               "(e.g. val_ids.txt) to restrict scoring to")
    p_score.add_argument("--beta", type=float, default=0.5)
    p_score.set_defaults(func=cmd_score)

    p_baseline = sub.add_parser(
        "baseline", help="Generate + score the trivial 'predict all singletons' baseline."
    )
    p_baseline.add_argument("--ground-truth", required=True,
                             help="Path to train_ground_truth.tsv")
    p_baseline.add_argument("--entity-ids", default=None,
                             help="Optional path to a newline-separated id file "
                                  "(e.g. val_ids.txt); default is all ground-truth ids")
    p_baseline.add_argument("--out", required=True,
                             help="Where to write the baseline predictions TSV")
    p_baseline.add_argument("--beta", type=float, default=0.5)
    p_baseline.set_defaults(func=cmd_baseline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()