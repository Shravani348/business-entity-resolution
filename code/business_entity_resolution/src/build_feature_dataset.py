import argparse
from pathlib import Path

import pandas as pd

from candidate_generation import generate_candidates


SEED = 42

N_POSITIVES = 5000
N_NEGATIVES = 15000


# ----------------------------------------------------------------------
# Load source data
# ----------------------------------------------------------------------

def load_data(dataset_dir):

    dataset_dir = Path(dataset_dir)

    source1 = pd.read_csv(
        dataset_dir / "train_source1.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        dtype=str,
    )

    source2 = pd.read_csv(
        dataset_dir / "train_source2.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        dtype=str,
    )

    source3 = pd.read_csv(
        dataset_dir / "train_source3.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        dtype=str,
    )

    ground_truth = pd.read_csv(
        dataset_dir / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
    )

    return (
        source1,
        source2,
        source3,
        ground_truth,
    )


# ----------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------

def parse_ground_truth(ground_truth):

    matches = {}

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if pd.isna(matched) or not str(matched).strip():

            matches[s1_id] = set()

        else:

            matches[s1_id] = {
                x.strip()
                for x in str(matched).split(",")
                if x.strip()
            }

    return matches


# ----------------------------------------------------------------------
# Build candidate labels
# ----------------------------------------------------------------------

def label_candidates(
    candidates,
    ground_truth_matches,
):
    """
    Label generated candidate pairs.

    1 = true match
    0 = candidate but not a true match
    """

    labels = []

    for _, row in candidates.iterrows():

        s1_id = row["source1_entity_id"]
        candidate_id = row["candidate_entity_id"]

        true_matches = ground_truth_matches.get(
            s1_id,
            set(),
        )

        if candidate_id in true_matches:
            labels.append(1)
        else:
            labels.append(0)

    candidates = candidates.copy()

    candidates["label"] = labels

    return candidates


# ----------------------------------------------------------------------
# Attach actual records
# ----------------------------------------------------------------------

def build_lookup(df):

    return df.set_index(
        "entity_id"
    ).to_dict("index")


def attach_records(
    candidates,
    source1_lookup,
    source2_lookup,
    source3_lookup,
):
    """
    Convert candidate IDs into actual business records.
    """

    rows = []

    for _, candidate in candidates.iterrows():

        s1_id = candidate["source1_entity_id"]
        other_id = candidate["candidate_entity_id"]
        source = candidate["candidate_source"]

        s1 = source1_lookup[s1_id]

        if source == "S2":
            other = source2_lookup[other_id]
        else:
            other = source3_lookup[other_id]

        rows.append(
            {
                "s1_id": s1_id,
                "other_id": other_id,
                "other_source": source,

                "s1_name": s1["business_name"],
                "other_name": other["business_name"],

                "s1_address": s1["business_address"],
                "other_address": other["business_address"],

                "s1_country": s1["country"],
                "other_country": other["country"],

                "matched_on_name": candidate[
                    "matched_on_name"
                ],

                "matched_on_address": candidate[
                    "matched_on_address"
                ],

                "label": candidate["label"],
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Sample balanced prototype dataset
# ----------------------------------------------------------------------

def sample_training_pairs(
    candidates,
    n_positive,
    n_negative,
):
    """
    Sample positives and negatives from the generated
    candidate pool.
    """

    positives = candidates[
        candidates["label"] == 1
    ]

    negatives = candidates[
        candidates["label"] == 0
    ]

    print(
        f"Available candidate positives: "
        f"{len(positives):,}"
    )

    print(
        f"Available candidate negatives: "
        f"{len(negatives):,}"
    )

    if len(positives) < n_positive:

        raise ValueError(
            f"Only {len(positives):,} positive candidates "
            f"available; need {n_positive:,}."
        )

    if len(negatives) < n_negative:

        raise ValueError(
            f"Only {len(negatives):,} negative candidates "
            f"available; need {n_negative:,}."
        )

    positive_sample = positives.sample(
        n=n_positive,
        random_state=SEED,
    )

    negative_sample = negatives.sample(
        n=n_negative,
        random_state=SEED,
    )

    result = pd.concat(
        [
            positive_sample,
            negative_sample,
        ],
        ignore_index=True,
    )

    result = result.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to train dataset directory",
    )

    parser.add_argument(
        "--out",
        default="prototype_data/feature_pairs.csv",
        help="Output feature-pair CSV path",
    )

    parser.add_argument(
        "--max-block-size",
        type=int,
        default=2000,
        help="Maximum allowed block size",
    )

    args = parser.parse_args()

    print("Loading dataset...")

    (
        source1,
        source2,
        source3,
        ground_truth,
    ) = load_data(args.dataset_dir)

    print(
        f"Source1 rows: {len(source1):,}"
    )

    print(
        f"Source2 rows: {len(source2):,}"
    )

    print(
        f"Source3 rows: {len(source3):,}"
    )

    print(
        f"Ground truth rows: {len(ground_truth):,}"
    )

    # --------------------------------------------------------------
    # Ground truth lookup
    # --------------------------------------------------------------

    ground_truth_matches = parse_ground_truth(
        ground_truth
    )

    # --------------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------------

    print("\nGenerating candidates...")

    candidates = generate_candidates(
        source1,
        source2,
        source3,
        max_block_size=args.max_block_size,
    )

    print(
        f"\nTotal generated candidates: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------------
    # Label candidates
    # --------------------------------------------------------------

    print("\nLabeling candidates...")

    candidates = label_candidates(
        candidates,
        ground_truth_matches,
    )

    print(
        f"Positive candidates: "
        f"{(candidates['label'] == 1).sum():,}"
    )

    print(
        f"Negative candidates: "
        f"{(candidates['label'] == 0).sum():,}"
    )

    # --------------------------------------------------------------
    # Sample prototype dataset
    # --------------------------------------------------------------

    print("\nSampling prototype dataset...")

    candidates = sample_training_pairs(
        candidates,
        n_positive=N_POSITIVES,
        n_negative=N_NEGATIVES,
    )

    print(
        f"Selected prototype pairs: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------------
    # Lookups
    # --------------------------------------------------------------

    source1_lookup = build_lookup(source1)
    source2_lookup = build_lookup(source2)
    source3_lookup = build_lookup(source3)

    # --------------------------------------------------------------
    # Attach records
    # --------------------------------------------------------------

    print("\nAttaching source records...")

    result = attach_records(
        candidates,
        source1_lookup,
        source2_lookup,
        source3_lookup,
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    output_path = Path(args.out)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\nDone.")

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Total pairs: {len(result):,}"
    )

    print("\nLabel distribution:")

    print(
        result["label"].value_counts()
    )

    print("\nSource distribution:")

    print(
        result["other_source"].value_counts()
    )

    print("\nCandidate block information:")

    print(
        result[
            [
                "matched_on_name",
                "matched_on_address",
            ]
        ].value_counts()
    )


if __name__ == "__main__":
    main()