import pandas as pd
from normalize import normalize_name, normalize_address
from blocking import TokenBlocker

# 1. Load data (parquet caches if available)
source1 = pd.read_parquet("dataset/train/train_source1.parquet")
source2 = pd.read_parquet("dataset/train/train_source2.parquet")
source3 = pd.read_parquet("dataset/train/train_source3.parquet")
ground_truth = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

# 2. Sample 20,000 Source 1 entities honestly
sample_ids = source1["entity_id"].sample(n=20000, random_state=42).tolist()
sample_set = set(sample_ids)
sample_gt = ground_truth[ground_truth["source1_entity_id"].isin(sample_set)]

# 3. Restrict Source 2/3 to countries present in the sample - honest restriction
sample_countries = set(source1[source1["entity_id"].isin(sample_set)]["country"])
s2_pool = source2[source2["country"].isin(sample_countries)].copy()
s3_pool = source3[source3["country"].isin(sample_countries)].copy()

print(f"Sample size: {len(sample_ids)} Source 1 entities")
print(f"Honest candidate pool: {len(s2_pool)} from Source 2, {len(s3_pool)} from Source 3")

# 4. Build the REAL inverted index on this honest pool
blocker = TokenBlocker()

for df in (s2_pool, s3_pool):
    for row in df.itertuples(index=False):
        norm_name = normalize_name(row.business_name)
        norm_address = normalize_address(row.business_address)
        blocker.index_record(row.entity_id, row.country, norm_name, norm_address)

print("Index built. Retrieving candidates and checking recall...")

# 5. For each sampled Source 1 entity, get candidates and check recall
total_true = 0
total_found = 0
zero_candidate_entities = 0
candidate_counts = []

s1_sample_df = source1[source1["entity_id"].isin(sample_set)]

for row in s1_sample_df.itertuples(index=False):
    norm_name = normalize_name(row.business_name)
    norm_address = normalize_address(row.business_address)
    candidates = blocker.get_candidates(row.country, norm_name, norm_address)
    candidate_counts.append(len(candidates))

    if len(candidates) == 0:
        zero_candidate_entities += 1

    gt_row = sample_gt[sample_gt["source1_entity_id"] == row.entity_id]
    if gt_row.empty or pd.isna(gt_row.iloc[0]["matched_entity_ids"]):
        continue  # singleton, nothing to check for recall

    true_matches = set(gt_row.iloc[0]["matched_entity_ids"].split(","))
    found = true_matches & candidates

    total_true += len(true_matches)
    total_found += len(found)

recall = total_found / total_true if total_true > 0 else 0
print(f"\nBlocking recall: {recall:.4f} ({total_found}/{total_true} true matches found)")
print(f"Entities with zero candidates: {zero_candidate_entities}")

import statistics
print(f"Candidates per entity - min: {min(candidate_counts)}, "
      f"median: {statistics.median(candidate_counts)}, "
      f"max: {max(candidate_counts)}, "
      f"avg: {sum(candidate_counts)/len(candidate_counts):.1f}")
