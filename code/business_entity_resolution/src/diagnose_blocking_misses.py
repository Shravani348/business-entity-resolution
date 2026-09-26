import argparse
from pathlib import Path

import pandas as pd  # type: ignore

from normalize import normalize_name, normalize_address
from blocking import TokenBlocker


# Must match the experimental value in blocking.py
MAX_FREQ = 50000


def inspect_tokens(blocker, country, tokens, index, target_id, field):
    c_key = str(country).lower().strip()

    rows = []

    for token in sorted(tokens):
        key = (c_key, token)
        bucket = index.get(key, set())
        bucket_size = len(bucket)

        rows.append({
            "field": field,
            "token": token,
            "bucket_size": bucket_size,
            "usable": 0 < bucket_size <= MAX_FREQ,
            "target_in_bucket": target_id in bucket,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-misses", type=int, default=20)

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    source1 = pd.read_csv(
        dataset_dir / "train_source1.tsv",
        sep="\t"
    )

    source2 = pd.read_csv(
        dataset_dir / "train_source2.tsv",
        sep="\t"
    )

    source3 = pd.read_csv(
        dataset_dir / "train_source3.tsv",
        sep="\t"
    )

    ground_truth = pd.read_csv(
        dataset_dir / "train_ground_truth.tsv",
        sep="\t"
    )

    # ---------------------------------------------------------
    # Same deterministic sample as recall checker
    # ---------------------------------------------------------
    sample_ids = source1["entity_id"].sample(
        n=args.sample_size,
        random_state=args.seed
    ).tolist()

    sample_set = set(sample_ids)

    sample_gt = ground_truth[
        ground_truth["source1_entity_id"].isin(sample_set)
    ]

    gt_lookup = (
        sample_gt
        .set_index("source1_entity_id")["matched_entity_ids"]
        .to_dict()
    )

    sample_countries = set(
        source1[
            source1["entity_id"].isin(sample_set)
        ]["country"].dropna()
    )

    s2_pool = source2[
        source2["country"].isin(sample_countries)
    ].copy()

    s3_pool = source3[
        source3["country"].isin(sample_countries)
    ].copy()

    # ---------------------------------------------------------
    # Build blocker
    # ---------------------------------------------------------
    blocker = TokenBlocker()

    for df in (s2_pool, s3_pool):
        for row in df.itertuples(index=False):
            blocker.index_record(
                row.entity_id,
                row.country,
                normalize_name(row.business_name),
                normalize_address(row.business_address)
            )

    # ---------------------------------------------------------
    # Fast lookup for Source 2 / Source 3
    # ---------------------------------------------------------
    s2_lookup = source2.set_index("entity_id")
    s3_lookup = source3.set_index("entity_id")

    # ---------------------------------------------------------
    # Find misses
    # ---------------------------------------------------------
    misses = []

    s1_sample_df = source1[
        source1["entity_id"].isin(sample_set)
    ]

    for row in s1_sample_df.itertuples(index=False):

        matched_ids = gt_lookup.get(row.entity_id)

        if pd.isna(matched_ids):
            continue

        true_matches = set(str(matched_ids).split(","))

        norm_name = normalize_name(row.business_name)
        norm_address = normalize_address(row.business_address)

        candidates = blocker.get_candidates(
            row.country,
            norm_name,
            norm_address
        )

        missed = true_matches - candidates

        for target_id in missed:
            misses.append(
                (
                    row,
                    target_id,
                    norm_name,
                    norm_address
                )
            )

            if len(misses) >= args.max_misses:
                break

        if len(misses) >= args.max_misses:
            break

    # ---------------------------------------------------------
    # Diagnose each miss
    # ---------------------------------------------------------
    print(
        f"\nDiagnosing {len(misses)} missed true matches "
        f"with MAX_FREQ={MAX_FREQ} and count >= 1...\n"
    )

    for i, (
        s1_row,
        target_id,
        s1_norm_name,
        s1_norm_address
    ) in enumerate(misses, start=1):

        if target_id.startswith("S2-"):
            target_row = s2_lookup.loc[target_id]
            target_source = "Source 2"

        elif target_id.startswith("S3-"):
            target_row = s3_lookup.loc[target_id]
            target_source = "Source 3"

        else:
            continue

        target_norm_name = normalize_name(
            target_row["business_name"]
        )

        target_norm_address = normalize_address(
            target_row["business_address"]
        )

        print("=" * 90)
        print(f"MISS #{i}")
        print(f"S1 ID:     {s1_row.entity_id}")
        print(f"Target ID: {target_id}")
        print(f"Target:    {target_source}")
        print(f"Country:   {s1_row.country}")

        print("\nS1")
        print(f"  Name:        {s1_row.business_name}")
        print(f"  Address:     {s1_row.business_address}")
        print(f"  Norm name:   {s1_norm_name}")
        print(f"  Norm address:{s1_norm_address}")

        print(f"\n{target_source}")
        print(f"  Name:        {target_row['business_name']}")
        print(f"  Address:     {target_row['business_address']}")
        print(f"  Norm name:   {target_norm_name}")
        print(f"  Norm address:{target_norm_address}")

        # -----------------------------------------------------
        # Inspect every S1 token against the actual index
        # -----------------------------------------------------
        name_details = inspect_tokens(
            blocker,
            s1_row.country,
            blocker.tokenize(s1_norm_name),
            blocker.name_index,
            target_id,
            "name"
        )

        address_details = inspect_tokens(
            blocker,
            s1_row.country,
            blocker.tokenize(s1_norm_address),
            blocker.address_index,
            target_id,
            "address"
        )

        details = name_details + address_details

        print("\nTOKEN DIAGNOSTICS")

        for d in details:
            status = []

            if d["usable"]:
                status.append("USABLE")
            else:
                status.append("IGNORED")

            if d["target_in_bucket"]:
                status.append("TARGET_IN_BUCKET")

            print(
                f"  {d['field']:8s} "
                f"{d['token']!r:20s} "
                f"bucket={d['bucket_size']:6d} "
                f"{', '.join(status)}"
            )

        # -----------------------------------------------------
        # Calculate actual blocker overlap
        # -----------------------------------------------------
        usable_target_hits = [
            d for d in details
            if d["usable"] and d["target_in_bucket"]
        ]

        usable_buckets = [
            d for d in details
            if d["usable"]
        ]

        print("\nSUMMARY")
        print(f"  Usable buckets:     {len(usable_buckets)}")
        print(f"  Target bucket hits: {len(usable_target_hits)}")

        if len(usable_target_hits) == 0:
            print(
                "  FAILURE: target is not present in any usable bucket."
            )

        else:
            print(
                "  SAFE RULE: target has at least one usable bucket hit "
                "and should be returned with count >= 1."
            )

        print()


if __name__ == "__main__":
    main()