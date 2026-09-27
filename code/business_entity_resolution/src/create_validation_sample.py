from pathlib import Path
import pandas as pd


DATASET_DIR = Path(
    r"C:\Users\ASUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
)

OUTPUT_DIR = Path("prototype_data/test_5000")

N_S1 = 5000
SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source files...")

    source1 = pd.read_csv(
        DATASET_DIR / "train_source1.tsv",
        sep="\t",
        dtype=str,
    )

    source2 = pd.read_csv(
        DATASET_DIR / "train_source2.tsv",
        sep="\t",
        dtype=str,
    )

    source3 = pd.read_csv(
        DATASET_DIR / "train_source3.tsv",
        sep="\t",
        dtype=str,
    )

    ground_truth = pd.read_csv(
        DATASET_DIR / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
    )

    print(f"Source 1: {len(source1):,}")
    print(f"Source 2: {len(source2):,}")
    print(f"Source 3: {len(source3):,}")
    print(f"Ground truth: {len(ground_truth):,}")

    # ---------------------------------------------------------
    # Select 5,000 Source-1 records
    # ---------------------------------------------------------

    selected_s1 = source1.sample(
        n=min(N_S1, len(source1)),
        random_state=SEED,
    )

    selected_s1_ids = set(
        selected_s1["entity_id"]
    )

    print(
        f"\nSelected Source-1 records: "
        f"{len(selected_s1):,}"
    )

    # ---------------------------------------------------------
    # Keep only corresponding ground-truth rows
    # ---------------------------------------------------------

    selected_gt = ground_truth[
        ground_truth["source1_entity_id"].isin(
            selected_s1_ids
        )
    ].copy()

    # ---------------------------------------------------------
    # Find all S2/S3 records that are true matches
    # ---------------------------------------------------------

    matched_ids = set()

    for value in selected_gt["matched_entity_ids"]:
        if pd.isna(value):
            continue

        for entity_id in str(value).split(","):
            entity_id = entity_id.strip()

            if entity_id:
                matched_ids.add(entity_id)

    print(
        f"Ground-truth matched candidate IDs: "
        f"{len(matched_ids):,}"
    )

    # ---------------------------------------------------------
    # Keep all matching S2/S3 records.
    #
    # Add extra random records so candidate generation
    # still has negative candidates.
    # ---------------------------------------------------------

    source2_matches = source2[
        source2["entity_id"].isin(matched_ids)
    ]

    source3_matches = source3[
        source3["entity_id"].isin(matched_ids)
    ]

    # Add random records from S2/S3.
    # This keeps the validation set manageable while
    # preserving all true matches.
    remaining_s2 = source2[
        ~source2["entity_id"].isin(
            source2_matches["entity_id"]
        )
    ]

    remaining_s3 = source3[
        ~source3["entity_id"].isin(
            source3_matches["entity_id"]
        )
    ]

    extra_s2_n = min(
        max(5000 - len(source2_matches), 0),
        len(remaining_s2),
    )

    extra_s3_n = min(
        max(5000 - len(source3_matches), 0),
        len(remaining_s3),
    )

    extra_s2 = remaining_s2.sample(
        n=extra_s2_n,
        random_state=SEED,
    )

    extra_s3 = remaining_s3.sample(
        n=extra_s3_n,
        random_state=SEED,
    )

    selected_s2 = pd.concat(
        [source2_matches, extra_s2],
        ignore_index=True,
    ).drop_duplicates(
        subset=["entity_id"]
    )

    selected_s3 = pd.concat(
        [source3_matches, extra_s3],
        ignore_index=True,
    ).drop_duplicates(
        subset=["entity_id"]
    )

    # ---------------------------------------------------------
    # Save validation files
    # ---------------------------------------------------------

    selected_s1.to_csv(
        OUTPUT_DIR / "train_source1.tsv",
        sep="\t",
        index=False,
    )

    selected_s2.to_csv(
        OUTPUT_DIR / "train_source2.tsv",
        sep="\t",
        index=False,
    )

    selected_s3.to_csv(
        OUTPUT_DIR / "train_source3.tsv",
        sep="\t",
        index=False,
    )

    selected_gt.to_csv(
        OUTPUT_DIR / "train_ground_truth.tsv",
        sep="\t",
        index=False,
    )

    print("\nValidation dataset created.")

    print(
        f"S1: {len(selected_s1):,}"
    )

    print(
        f"S2: {len(selected_s2):,}"
    )

    print(
        f"S3: {len(selected_s3):,}"
    )

    print(
        f"Ground truth: {len(selected_gt):,}"
    )

    print(
        f"\nSaved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()