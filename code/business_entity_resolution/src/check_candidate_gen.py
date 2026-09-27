import statistics
from pathlib import Path

import pandas as pd

from candidate_generation import generate_candidates

dataset_dir = Path("dataset/train")

source1 = pd.read_csv(dataset_dir / "train_source1.tsv", sep="\t")
source2 = pd.read_csv(dataset_dir / "train_source2.tsv", sep="\t")
source3 = pd.read_csv(dataset_dir / "train_source3.tsv", sep="\t")
ground_truth = pd.read_csv(dataset_dir / "train_ground_truth.tsv", sep="\t")

sample_ids = source1["entity_id"].sample(n=20000, random_state=42).tolist()
sample_set = set(sample_ids)

sample_gt = ground_truth[ground_truth["source1_entity_id"].isin(sample_set)]
gt_lookup = sample_gt.set_index("source1_entity_id")["matched_entity_ids"].to_dict()

sample_countries = set(source1[source1["entity_id"].isin(sample_set)]["country"].dropna())

s2_pool = source2[source2["country"].isin(sample_countries)].copy()
s3_pool = source3[source3["country"].isin(sample_countries)].copy()

print(f"Sample size: {len(sample_ids)} Source 1 entities")
print(f"Honest candidate pool: {len(s2_pool):,} from Source 2, {len(s3_pool):,} from Source 3")

# We must pass normalized columns because candidate_generation uses name_norm, country_norm, etc.
# Wait, candidate_generation.py calls add_normalized_columns inside generate_candidates!
# Let's verify that. I will just pass the raw dataframes since that's what build_feature_dataset does.

s1_sample = source1[source1["entity_id"].isin(sample_set)].copy()

print("\nRunning generate_candidates...")
candidates = generate_candidates(s1_sample, s2_pool, s3_pool)
print("Finished generation.")

# 1. Candidates per entity
candidates_per_entity = candidates.groupby("source1_entity_id").size()

# Note: Some entities might have 0 candidates, so we must fill them with 0
for s1_id in sample_ids:
    if s1_id not in candidates_per_entity:
        candidates_per_entity.loc[s1_id] = 0

vals = list(candidates_per_entity.values)
min_c = min(vals)
max_c = max(vals)
avg_c = sum(vals) / len(vals)
med_c = statistics.median(vals)

print(f"\nCandidates per entity: min={min_c}, median={med_c}, avg={avg_c:.1f}, max={max_c}")
print(f"Total candidate pairs: {len(candidates):,}")

# 2. Recall check
print("\nCalculating recall...")
cand_dict = candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

total_true_matches = 0
found_matches = 0
missed = 0

for s1_id, matched_str in gt_lookup.items():
    if not isinstance(matched_str, str) or not matched_str.strip():
        continue
    
    true_matches = set(matched_str.split(","))
    total_true_matches += len(true_matches)
    
    generated_for_s1 = cand_dict.get(s1_id, set())
    found = len(true_matches & generated_for_s1)
    
    found_matches += found
    missed += len(true_matches) - found

recall = (found_matches / total_true_matches) * 100 if total_true_matches > 0 else 0
print(f"Recall: {recall:.2f}% ({found_matches:,} / {total_true_matches:,} true matches found)")
