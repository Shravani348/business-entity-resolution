import argparse
import statistics
from pathlib import Path

import pandas as pd  # type: ignore

from normalize import normalize_name, normalize_address
from blocking import TokenBlocker


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate blocking recall on a deterministic Source 1 sample."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the train dataset directory."
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Number of Source 1 entities to sample."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling."
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    # 1. Load data
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

    # 2. Sample Source 1 entities deterministically
    sample_ids = source1["entity_id"].sample(
        n=args.sample_size,
        random_state=args.seed
    ).tolist()

    sample_set = set(sample_ids)

    sample_gt = ground_truth[
        ground_truth["source1_entity_id"].isin(sample_set)
    ]

    # Faster lookup: source1_entity_id -> matched_entity_ids
    gt_lookup = (
        sample_gt
        .set_index("source1_entity_id")["matched_entity_ids"]
        .to_dict()
    )

    # 3. Restrict Source 2/3 to countries present in the sample
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

    print(f"Sample size: {len(sample_ids)} Source 1 entities")

    print(
        f"Honest candidate pool: "
        f"{len(s2_pool)} from Source 2, "
        f"{len(s3_pool)} from Source 3"
    )

    # 4. Build the real inverted index on the honest pool
    blocker = TokenBlocker()

    for df in (s2_pool, s3_pool):
        for row in df.itertuples(index=False):
            norm_name = normalize_name(row.business_name)
            norm_address = normalize_address(row.business_address)

            blocker.index_record(
                row.entity_id,
                row.country,
                norm_name,
                norm_address
            )

    print("Index built. Retrieving candidates and checking recall...")

    # 5. Evaluate blocking recall
    total_true = 0
    total_found = 0
    zero_candidate_entities = 0
    candidate_counts = []
    missed_matches = []

    s1_sample_df = source1[
        source1["entity_id"].isin(sample_set)
    ]

    for row in s1_sample_df.itertuples(index=False):
        norm_name = normalize_name(row.business_name)
        norm_address = normalize_address(row.business_address)

        candidates = blocker.get_candidates(
            row.country,
            norm_name,
            norm_address
        )

        candidate_counts.append(len(candidates))

        if len(candidates) == 0:
            zero_candidate_entities += 1

        matched_ids = gt_lookup.get(row.entity_id)

        # Singleton / no ground-truth matches
        if matched_ids is None or pd.isna(matched_ids):
            continue

        true_matches = set(
            str(matched_ids).split(",")
        )

        found = true_matches & candidates
        missed = true_matches - candidates

        total_true += len(true_matches)
        total_found += len(found)

        # Keep only the first 100 missed true matches
        if missed and len(missed_matches) < 20:
            for missed_id in missed:
                missed_matches.append({
                    "source1_entity_id": row.entity_id,
                    "missed_entity_id": missed_id,
                    "country": row.country,
                    "source1_name": row.business_name,
                    "source1_address": row.business_address,
                    "source1_norm_name": norm_name,
                    "source1_norm_address": norm_address,
                })

                if len(missed_matches) >= 20:
                    break

    # 6. Report results
    recall = (
        total_found / total_true
        if total_true > 0
        else 0
    )

    print(
        f"\nBlocking recall: {recall:.4f} "
        f"({total_found}/{total_true} true matches found)"
    )

    print(
        f"Entities with zero candidates: "
        f"{zero_candidate_entities}"
    )

    print(
        f"Candidates per entity - "
        f"min: {min(candidate_counts)}, "
        f"median: {statistics.median(candidate_counts)}, "
        f"max: {max(candidate_counts)}, "
        f"avg: {sum(candidate_counts) / len(candidate_counts):.1f}"
    )

    # 7. Show sample of missed true matches
    print("\n--- Sample missed true matches ---")

    # Build quick lookup tables for the missed IDs
    s2_lookup = (
        source2.set_index("entity_id")
        .to_dict("index")
    )

    s3_lookup = (
        source3.set_index("entity_id")
        .to_dict("index")
    )

    for i, item in enumerate(missed_matches, start=1):
        missed_id = item["missed_entity_id"]

        if missed_id.startswith("S2-"):
            target = s2_lookup.get(missed_id)
            source_name = "Source 2"
        else:
            target = s3_lookup.get(missed_id)
            source_name = "Source 3"

        print(f"\nMiss #{i}")
        print(f"Source 1 ID: {item['source1_entity_id']}")
        print(f"Missed ID:   {missed_id}")
        print(f"Country:     {item['country']}")

        print("\nSOURCE 1")
        print(f"Name:        {item['source1_name']}")
        print(f"Address:     {item['source1_address']}")
        print(f"Norm name:   {item['source1_norm_name']}")
        print(f"Norm address:{item['source1_norm_address']}")

        if target is not None:
            target_name = normalize_name(target["business_name"])
            target_address = normalize_address(target["business_address"])

            print(f"\n{source_name}")
            print(f"Name:        {target['business_name']}")
            print(f"Address:     {target['business_address']}")
            print(f"Norm name:   {target_name}")
            print(f"Norm address:{target_address}")

            s1_name_tokens = set(item["source1_norm_name"].split())
            s1_addr_tokens = set(item["source1_norm_address"].split())

            target_name_tokens = set(target_name.split())
            target_addr_tokens = set(target_address.split())

            shared_name = s1_name_tokens & target_name_tokens
            shared_address = s1_addr_tokens & target_addr_tokens

            print(f"\nShared name tokens:    {sorted(shared_name)}")
            print(f"Shared address tokens: {sorted(shared_address)}")

if __name__ == "__main__":
    main()