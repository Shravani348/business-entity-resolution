import pandas as pd
import pickle
import os
import time
import statistics
from normalize import normalize_name, normalize_address
from blocking import TokenBlocker

def load_or_build_index():
    cache_path = "blocking_index_cache.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached index from {cache_path}...")
        with open(cache_path, "rb") as f:
            blocker = pickle.load(f)
        return blocker
        
    print("No cached index found. Building index...")
    # Load data
    source1 = pd.read_parquet("dataset/train/train_source1.parquet")
    source2 = pd.read_parquet("dataset/train/train_source2.parquet")
    source3 = pd.read_parquet("dataset/train/train_source3.parquet")
    
    # Sample 20,000 Source 1 entities honestly
    sample_ids = source1["entity_id"].sample(n=20000, random_state=42).tolist()
    sample_set = set(sample_ids)
    
    # Restrict Source 2/3 to countries present in the sample
    sample_countries = set(source1[source1["entity_id"].isin(sample_set)]["country"])
    s2_pool = source2[source2["country"].isin(sample_countries)].copy()
    s3_pool = source3[source3["country"].isin(sample_countries)].copy()
    
    print(f"Sample size: {len(sample_ids)} Source 1 entities")
    print(f"Honest candidate pool: {len(s2_pool)} from Source 2, {len(s3_pool)} from Source 3")
    
    blocker = TokenBlocker()
    start_time = time.time()
    
    for df_name, df in [("Source 2", s2_pool), ("Source 3", s3_pool)]:
        count = 0
        for row in df.itertuples(index=False):
            norm_name = normalize_name(row.business_name)
            norm_address = normalize_address(row.business_address)
            blocker.index_record(row.entity_id, row.country, norm_name, norm_address)
            count += 1
            if count % 500000 == 0:
                print(f"Indexed {count} rows from {df_name}... Elapsed: {time.time() - start_time:.2f}s")
                
    print(f"Index built in {time.time() - start_time:.2f}s. Saving to cache...")
    with open(cache_path, "wb") as f:
        pickle.dump(blocker, f)
        
    return blocker

def evaluate_params(blocker, max_freq, min_overlap):
    print(f"\n--- Evaluating MAX_FREQ={max_freq}, MIN_OVERLAP={min_overlap} ---")
    source1 = pd.read_parquet("dataset/train/train_source1.parquet")
    ground_truth = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
    
    sample_ids = source1["entity_id"].sample(n=20000, random_state=42).tolist()
    sample_set = set(sample_ids)
    sample_gt = ground_truth[ground_truth["source1_entity_id"].isin(sample_set)]
    s1_sample_df = source1[source1["entity_id"].isin(sample_set)]
    
    total_true = 0
    total_found = 0
    zero_candidate_entities = 0
    candidate_counts = []
    
    start_time = time.time()
    for row in s1_sample_df.itertuples(index=False):
        norm_name = normalize_name(row.business_name)
        norm_address = normalize_address(row.business_address)
        candidates = blocker.get_candidates(row.country, norm_name, norm_address, max_freq=max_freq, min_overlap=min_overlap)
        candidate_counts.append(len(candidates))

        if len(candidates) == 0:
            zero_candidate_entities += 1

        gt_row = sample_gt[sample_gt["source1_entity_id"] == row.entity_id]
        if gt_row.empty or pd.isna(gt_row.iloc[0]["matched_entity_ids"]):
            continue

        true_matches = set(gt_row.iloc[0]["matched_entity_ids"].split(","))
        found = true_matches & candidates

        total_true += len(true_matches)
        total_found += len(found)

    recall = total_found / total_true if total_true > 0 else 0
    eval_time = time.time() - start_time
    
    print(f"Recall: {recall:.4f} ({total_found}/{total_true} true matches found)")
    print(f"Entities with zero candidates: {zero_candidate_entities}")
    print(f"Candidates per entity - min: {min(candidate_counts)}, median: {statistics.median(candidate_counts)}, max: {max(candidate_counts)}, avg: {sum(candidate_counts)/len(candidate_counts):.1f}")
    print(f"Evaluation time: {eval_time:.2f}s")
    
    return {
        "max_freq": max_freq,
        "min_overlap": min_overlap,
        "recall": recall,
        "found": total_found,
        "true_matches": total_true,
        "zero_cands": zero_candidate_entities,
        "min_c": min(candidate_counts),
        "median_c": statistics.median(candidate_counts),
        "max_c": max(candidate_counts),
        "avg_c": sum(candidate_counts)/len(candidate_counts)
    }

def main():
    blocker = load_or_build_index()
    
    experiments = [
        (10000, 2), # baseline
        (50000, 1), # current experiment
        (20000, 2),
        (30000, 1)
    ]
    
    results = []
    for freq, overlap in experiments:
        results.append(evaluate_params(blocker, freq, overlap))
        
    print("\n\n=== FINAL COMPARISON ===")
    print(f"{'Max Freq':<10} {'Min Overlap':<12} {'Recall':<8} {'Matches':<15} {'Zero Cands':<12} {'Median Cands':<15} {'Avg Cands':<10} {'Min Cands':<10} {'Max Cands':<10}")
    print("-" * 110)
    for r in results:
        recall_pct = f"{r['recall']*100:.2f}%"
        matches_str = f"{r['found']}/{r['true_matches']}"
        print(f"{r['max_freq']:<10} {r['min_overlap']:<12} {recall_pct:<8} {matches_str:<15} {r['zero_cands']:<12} {r['median_c']:<15.1f} {r['avg_c']:<10.1f} {r['min_c']:<10} {r['max_c']:<10}")

if __name__ == '__main__':
    main()
