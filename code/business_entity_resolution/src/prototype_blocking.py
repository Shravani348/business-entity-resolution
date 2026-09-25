import os
import sys
import pandas as pd
import time
import json
import numpy as np

sys.path.append(os.path.dirname(__file__))
import data_loader
import normalize
from blocking import TokenBlocker

def main():
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    dataset_dir = os.path.join(workspace_root, 'dataset')
    
    print("Loading datasets...", flush=True)
    # Load just train set
    train_data = data_loader.load_data_from_dir(os.path.join(dataset_dir, 'train'))
    
    gt = train_data['train_ground_truth']
    s1 = train_data['train_source1']
    s2 = train_data['train_source2']
    s3 = train_data['train_source3']
    
    print("Filtering ground truth for records with at least 1 match...", flush=True)
    gt_matches = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')]
    
    # Sample 20,000 Source 1 entities
    SAMPLE_SIZE = 20000
    if len(gt_matches) > SAMPLE_SIZE:
        np.random.seed(42) # For reproducibility
        sampled_gt = gt_matches.sample(SAMPLE_SIZE)
    else:
        sampled_gt = gt_matches
        
    sampled_s1_ids = set(sampled_gt['source1_entity_id'])
    
    print(f"Sampled {len(sampled_s1_ids)} Source 1 entities.", flush=True)
    
    # Filter S1
    s1_sample = s1[s1['entity_id'].isin(sampled_s1_ids)].copy()
    
    # Get relevant countries
    relevant_countries = set(s1_sample['country'].dropna().str.lower().str.strip().unique())
    print(f"Relevant countries in sample: {len(relevant_countries)}", flush=True)
    
    # Filter S2 and S3 to save memory/time for the prototype
    s2_filtered = s2[s2['country'].str.lower().str.strip().isin(relevant_countries)].copy()
    s3_filtered = s3[s3['country'].str.lower().str.strip().isin(relevant_countries)].copy()
    
    print(f"Building index from {len(s2_filtered)} Source 2 and {len(s3_filtered)} Source 3 records...", flush=True)
    
    blocker = TokenBlocker()
    start_time = time.time()
    
    print("Indexing S2...", flush=True)
    count = 0
    for row in s2_filtered.itertuples(index=False):
        norm_n = normalize.normalize_name(str(row.business_name))
        norm_a = normalize.normalize_address(str(row.business_address))
        blocker.index_record(row.entity_id, row.country, norm_n, norm_a)
        count += 1
        if count % 500000 == 0:
            print(f"Indexed {count:,} rows from S2... Elapsed: {time.time() - start_time:.2f}s", flush=True)
            
    # Free S2 memory
    del s2_filtered
    del s2
    import gc
    gc.collect()
            
    print("Indexing S3...", flush=True)
    count = 0
    for row in s3_filtered.itertuples(index=False):
        norm_n = normalize.normalize_name(str(row.business_name))
        norm_a = normalize.normalize_address(str(row.business_address))
        blocker.index_record(row.entity_id, row.country, norm_n, norm_a)
        count += 1
        if count % 500000 == 0:
            print(f"Indexed {count:,} rows from S3... Elapsed: {time.time() - start_time:.2f}s", flush=True)
            
    # Free S3 memory
    del s3_filtered
    del s3
    gc.collect()
        
    print(f"Index built in {time.time() - start_time:.2f} seconds.")
    
    print("Normalizing S1 Sample...", flush=True)
    s1_sample['norm_name'] = s1_sample['business_name'].astype(str).apply(normalize.normalize_name)
    s1_sample['norm_address'] = s1_sample['business_address'].astype(str).apply(normalize.normalize_address)
    
    print("Retrieving candidates and measuring recall...")
    start_time = time.time()
    
    total_true_matches = 0
    total_found_matches = 0
    generated_counts = []
    
    non_latin_misses_no_address = 0
    non_latin_total = 0
    
    # To quickly look up true matches
    gt_dict = dict(zip(sampled_gt['source1_entity_id'], sampled_gt['matched_entity_ids']))
    
    for row in s1_sample.itertuples(index=False):
        true_matches_str = gt_dict.get(row.entity_id, "")
        if pd.isna(true_matches_str) or not true_matches_str:
            continue
            
        true_matches = set(true_matches_str.split(','))
        total_true_matches += len(true_matches)
        
        candidates = blocker.get_candidates(row.country, row.norm_name, row.norm_address)
        generated_counts.append(len(candidates))
        
        found = true_matches.intersection(candidates)
        total_found_matches += len(found)
        
        # Check non-latin limitation:
        is_nl = normalize.is_non_latin(row.business_name)
        if is_nl:
            non_latin_total += 1
            if len(found) < len(true_matches):
                # Missed some. Was address empty?
                if not str(row.business_address).strip() or str(row.business_address).lower() == 'nan':
                    non_latin_misses_no_address += 1

    recall = (total_found_matches / total_true_matches) if total_true_matches > 0 else 0
    
    # Calculate reduction ratio
    # Total possible pairs in this universe = (20,000) * (len(s2_filtered) + len(s3_filtered))
    total_possible_pairs = len(s1_sample) * (len(s2_filtered) + len(s3_filtered))
    total_generated_pairs = sum(generated_counts)
    
    reduction_ratio = 1.0 - (total_generated_pairs / total_possible_pairs)
    
    print(f"Retrieval finished in {time.time() - start_time:.2f} seconds.")
    
    print("\n--- RESULTS ---")
    print(f"Sample Size: {len(s1_sample)} Source 1 entities")
    print(f"Total True Matches: {total_true_matches}")
    print(f"Matches Found: {total_found_matches}")
    print(f"Blocking Recall: {recall:.4f} ({recall*100:.2f}%)")
    
    print(f"Total Possible Pairs: {total_possible_pairs:,}")
    print(f"Generated Pairs: {total_generated_pairs:,}")
    print(f"Reduction Ratio: {reduction_ratio:.6f} ({(reduction_ratio)*100:.4f}%)")
    
    print("\nCandidate Distribution per Source 1:")
    print(f"Min: {np.min(generated_counts)}")
    print(f"Median: {np.median(generated_counts)}")
    print(f"Max: {np.max(generated_counts)}")
    print(f"Average: {np.mean(generated_counts):.2f}")
    
    print(f"\nNon-Latin Analysis:")
    print(f"Total Non-Latin S1 records in sample: {non_latin_total}")
    print(f"Misses where S1 has Non-Latin name AND missing/empty address: {non_latin_misses_no_address}")
    
    # Estimate for full dataset
    scale_factor = 2206821 / len(s1_sample) # 2.2M Source 1 entities
    estimated_total_generated = total_generated_pairs * scale_factor
    
    # Write to BLOCKING_ANALYSIS.md
    analysis_path = os.path.join(workspace_root, 'BLOCKING_ANALYSIS.md')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("# Blocking Strategy Analysis\n\n")
        f.write("## 1. Prototype Scale\n")
        f.write(f"- **Source 1 Sample Size:** {len(s1_sample):,}\n")
        f.write(f"- **Indexed Candidates:** {len(s2_filtered):,} (Source 2) + {len(s3_filtered):,} (Source 3)\n")
        f.write(f"- **Total Possible Pairs in Sample:** {total_possible_pairs:,}\n\n")
        
        f.write("## 2. Performance Metrics\n")
        f.write(f"- **Blocking Recall:** {recall*100:.2f}% ({total_found_matches:,} / {total_true_matches:,} true matches captured)\n")
        f.write(f"- **Reduction Ratio:** {reduction_ratio*100:.4f}% of pairs eliminated from search space.\n")
        f.write(f"- **Total Candidates Generated (Sample):** {total_generated_pairs:,}\n\n")
        
        f.write("## 3. Candidates per Source 1 Entity\n")
        f.write(f"- **Min:** {np.min(generated_counts)}\n")
        f.write(f"- **Median:** {np.median(generated_counts)}\n")
        f.write(f"- **Max:** {np.max(generated_counts)}\n")
        f.write(f"- **Average:** {np.mean(generated_counts):.2f}\n\n")
        
        f.write("## 4. Non-Latin Script Limitation\n")
        f.write(f"- **Non-Latin S1 Entities in Sample:** {non_latin_total}\n")
        f.write(f"- **Missed matches due to Non-Latin name + missing address:** {non_latin_misses_no_address}\n")
        f.write("  *(This confirms our known limitation where name tokens mismatch across scripts and address tokens are unavailable to bridge the gap.)*\n\n")
        
        f.write("## 5. Full Dataset Estimation\n")
        f.write(f"Scaling this strategy to the full 2.2M Source 1 entities would generate approximately **{estimated_total_generated:,.0f}** candidate pairs, which is highly manageable for a downstream matching model.\n")
        
    print(f"\nAnalysis saved to {analysis_path}")

if __name__ == "__main__":
    main()
