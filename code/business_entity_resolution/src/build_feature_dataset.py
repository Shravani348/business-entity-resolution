import argparse
import random
from pathlib import Path

import pandas as pd


SEED = 42
N_POSITIVES = 5000
N_NEGATIVES = 15000


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

    return source1, source2, source3, ground_truth


def build_lookup(df):
    """
    Convert a source dataframe into:
        entity_id -> record
    """
    return df.set_index("entity_id").to_dict("index")


def parse_ground_truth(ground_truth):
    """
    Convert ground truth into:

        S1 ID -> set of matched S2/S3 IDs
    """

    matches = {}

    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if pd.isna(matched) or not str(matched).strip():
            matches[s1_id] = set()
        else:
            matches[s1_id] = set(
                str(matched).split(",")
            )

    return matches


def build_positive_pairs(
    ground_truth,
    source1_lookup,
    source2_lookup,
    source3_lookup,
    n_positive,
    rng,
):
    """
    Build real Source1 -> Source2/Source3 matches.
    """

    all_positive_pairs = []

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            continue

        for other_id in str(matched_ids).split(","):

            other_id = other_id.strip()

            if other_id.startswith("S2-"):
                other_lookup = source2_lookup
                other_source = "S2"

            elif other_id.startswith("S3-"):
                other_lookup = source3_lookup
                other_source = "S3"

            else:
                continue

            if s1_id not in source1_lookup:
                continue

            if other_id not in other_lookup:
                continue

            all_positive_pairs.append(
                (
                    s1_id,
                    other_id,
                    other_source,
                )
            )

    print(f"Total available positive pairs: {len(all_positive_pairs):,}")

    if len(all_positive_pairs) < n_positive:
        raise ValueError(
            f"Only {len(all_positive_pairs)} positive pairs available."
        )

    selected = rng.sample(
        all_positive_pairs,
        n_positive,
    )

    return selected


def build_country_index(source2, source3):
    """
    country -> list of S2/S3 entity IDs
    """

    country_index = {}

    for df in [source2, source3]:

        for _, row in df.iterrows():

            country = row["country"]

            if pd.isna(country):
                continue

            country = str(country).strip().lower()

            if not country:
                continue

            country_index.setdefault(country, []).append(
                row["entity_id"]
            )

    return country_index


def build_negative_pairs(
    source1,
    source2,
    source3,
    ground_truth_matches,
    n_negative,
    rng,
):
    """
    Generate same-country pairs that are NOT true matches.
    """

    source2_lookup = build_lookup(source2)
    source3_lookup = build_lookup(source3)

    country_index = build_country_index(
        source2,
        source3,
    )

    s1_lookup = build_lookup(source1)

    # Only Source1 entities whose country has candidate records.
    eligible_s1 = []

    for s1_id, record in s1_lookup.items():

        country = record["country"]

        if pd.isna(country):
            continue

        country = str(country).strip().lower()

        if country in country_index:
            eligible_s1.append(
                (s1_id, country)
            )

    print(
        f"Eligible Source1 entities for negatives: "
        f"{len(eligible_s1):,}"
    )

    negatives = []
    seen = set()

    attempts = 0
    max_attempts = n_negative * 20

    while len(negatives) < n_negative:

        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Could not generate enough valid negative pairs."
            )

        s1_id, country = rng.choice(eligible_s1)

        candidate_ids = country_index[country]

        other_id = rng.choice(candidate_ids)

        pair_key = (s1_id, other_id)

        # Avoid duplicate negative pairs.
        if pair_key in seen:
            continue

        # Make sure this is not an actual match.
        true_matches = ground_truth_matches.get(
            s1_id,
            set(),
        )

        if other_id in true_matches:
            continue

        if other_id.startswith("S2-"):
            other_source = "S2"
        elif other_id.startswith("S3-"):
            other_source = "S3"
        else:
            continue

        seen.add(pair_key)

        negatives.append(
            (
                s1_id,
                other_id,
                other_source,
            )
        )

    return negatives


def pairs_to_dataframe(
    pairs,
    source1_lookup,
    source2_lookup,
    source3_lookup,
    label,
):
    rows = []

    for s1_id, other_id, other_source in pairs:

        s1 = source1_lookup[s1_id]

        if other_source == "S2":
            other = source2_lookup[other_id]
        else:
            other = source3_lookup[other_id]

        rows.append(
            {
                "s1_id": s1_id,
                "other_id": other_id,
                "other_source": other_source,
                "s1_name": s1["business_name"],
                "other_name": other["business_name"],
                "s1_address": s1["business_address"],
                "other_address": other["business_address"],
                "s1_country": s1["country"],
                "other_country": other["country"],
                "label": label,
            }
        )

    return pd.DataFrame(rows)


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
        help="Output CSV path",
    )

    args = parser.parse_args()

    rng = random.Random(SEED)

    print("Loading dataset...")

    source1, source2, source3, ground_truth = load_data(
        args.dataset_dir
    )

    print(f"Source1 rows: {len(source1):,}")
    print(f"Source2 rows: {len(source2):,}")
    print(f"Source3 rows: {len(source3):,}")
    print(f"Ground truth rows: {len(ground_truth):,}")

    source1_lookup = build_lookup(source1)
    source2_lookup = build_lookup(source2)
    source3_lookup = build_lookup(source3)

    ground_truth_matches = parse_ground_truth(
        ground_truth
    )

    print("\nBuilding positive pairs...")

    positives = build_positive_pairs(
        ground_truth,
        source1_lookup,
        source2_lookup,
        source3_lookup,
        N_POSITIVES,
        rng,
    )

    print(
        f"Selected positive pairs: "
        f"{len(positives):,}"
    )

    print("\nBuilding negative pairs...")

    negatives = build_negative_pairs(
        source1,
        source2,
        source3,
        ground_truth_matches,
        N_NEGATIVES,
        rng,
    )

    print(
        f"Selected negative pairs: "
        f"{len(negatives):,}"
    )

    positive_df = pairs_to_dataframe(
        positives,
        source1_lookup,
        source2_lookup,
        source3_lookup,
        label=1,
    )

    negative_df = pairs_to_dataframe(
        negatives,
        source1_lookup,
        source2_lookup,
        source3_lookup,
        label=0,
    )

    result = pd.concat(
        [positive_df, negative_df],
        ignore_index=True,
    )

    # Shuffle the final dataset.
    result = result.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

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
    print(f"Saved: {output_path}")
    print(f"Total pairs: {len(result):,}")
    print("\nLabel distribution:")
    print(result["label"].value_counts())
    print("\nSource distribution:")
    print(result["other_source"].value_counts())


if __name__ == "__main__":
    main()
    