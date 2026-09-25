import os
import sys
import pandas as pd
import json

# Add current dir to path to import data_loader
sys.path.append(os.path.dirname(__file__))
import data_loader

def main():
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    dataset_dir = os.path.join(workspace_root, 'dataset')
    
    print(f"Looking for dataset in: {dataset_dir}")
    train_data = data_loader.load_data_from_dir(os.path.join(dataset_dir, 'train'))
    test_data = data_loader.load_data_from_dir(os.path.join(dataset_dir, 'test'))
    
    if not test_data:
        print("WARNING: No valid TSV files were found in the test directory (skipped macOS metadata files).")

    
    analysis = []
    analysis.append("# Dataset Exploration Analysis\n")
    analysis.append("Welcome to your first hackathon! Below are the insights derived from exploring the dataset.\n")

    # 1. Row/column counts and Missing values
    analysis.append("## 1. File Statistics (Rows, Columns, and Missing Values)\n")
    
    all_data = {}
    for name, df in train_data.items():
        all_data[name] = df
    for name, df in test_data.items():
        all_data[name] = df
        
    for name, df in all_data.items():
        if df.empty and df.shape[1] == 1:
            analysis.append(f"### {name}.tsv\n")
            analysis.append(f"**Warning**: This file appears to be corrupted or a macOS metadata file (0 rows, 1 col). Skipped.\n")
            continue
            
        rows, cols = df.shape
        analysis.append(f"### {name}.tsv\n")
        analysis.append(f"- **Rows**: {rows:,}\n")
        analysis.append(f"- **Columns**: {cols}\n")
        
        missing = df.isna().sum()
        empty_strings = (df == "").sum()
        total_missing = missing + empty_strings
        
        analysis.append("- **Missing Values per Column**:\n")
        for col in df.columns:
            analysis.append(f"  - `{col}`: {total_missing[col]:,} missing\n")
        analysis.append("\n")
        
    # 2. Match/class distribution
    analysis.append("## 2. Match / Class Distribution\n")
    if 'train_ground_truth' in train_data:
        gt = train_data['train_ground_truth'].copy()
        # Vectorized counting
        matches_str = gt['matched_entity_ids'].fillna('')
        gt['num_matches'] = (matches_str.str.count(',') + 1).where(matches_str != '', 0)
        
        zero_matches = (gt['num_matches'] == 0).sum()
        one_match = (gt['num_matches'] == 1).sum()
        two_to_five = ((gt['num_matches'] >= 2) & (gt['num_matches'] <= 5)).sum()
        five_plus = (gt['num_matches'] > 5).sum()
        
        analysis.append("How many Source 2/3 records match each Source 1 business in the training set?\n")
        analysis.append(f"- **0 Matches**: {zero_matches:,}\n")
        analysis.append(f"- **1 Match**: {one_match:,}\n")
        analysis.append(f"- **2 to 5 Matches**: {two_to_five:,}\n")
        analysis.append(f"- **More than 5 Matches**: {five_plus:,}\n\n")

    # 3. France Check
    analysis.append("## 3. 'France' Country Check\n")
    france_in_train = False
    for name, df in train_data.items():
        if 'country' in df.columns:
            if df['country'].str.contains('France', case=False, na=False).any():
                france_in_train = True
                break
                
    france_in_test = False
    for name, df in test_data.items():
        if 'country' in df.columns:
            if df['country'].str.contains('France', case=False, na=False).any():
                france_in_test = True
                break
                
    if not france_in_train:
        analysis.append("- **Confirmed**: 'France' does NOT appear in any training file.\n")
    else:
        analysis.append("- **Alert**: 'France' appears in the training data.\n")
        
    if france_in_test:
        analysis.append("- **Confirmed**: 'France' appears in the test data.\n\n")
    else:
        analysis.append("- **Alert**: 'France' does NOT appear in the test data (or test data is unreadable).\n\n")

    # 4. Leakage Checks
    analysis.append("## 4. Leakage Check\n")
    leakage_found = False
    for name, df in test_data.items():
        if 'matched_entity_ids' in df.columns:
            leakage_found = True
            analysis.append(f"- **LEAKAGE DETECTED**: File `{name}` contains the `matched_entity_ids` column.\n")
            
    if not leakage_found:
        analysis.append("- **Confirmed**: No ground-truth or label columns (`matched_entity_ids`) exist in the test files.\n\n")
        
    # 5. 10-15 Real Example Pairs
    analysis.append("## 5. Real Example Pairs (Noise Inspection)\n")
    analysis.append("Here are some real examples from the dataset showing a Source 1 business alongside its matched Source 2/Source 3 records:\n\n")
    
    if 'train_ground_truth' in train_data and 'train_source1' in train_data:
        gt_examples = train_data['train_ground_truth'].copy()
        
        # Vectorized match counting
        matches_str = gt_examples['matched_entity_ids'].fillna('')
        gt_examples['num_matches'] = (matches_str.str.count(',') + 1).where(matches_str != '', 0)
        
        s1 = train_data['train_source1'].set_index('entity_id')
        s2 = train_data.get('train_source2', pd.DataFrame()).set_index('entity_id') if 'train_source2' in train_data else pd.DataFrame()
        s3 = train_data.get('train_source3', pd.DataFrame()).set_index('entity_id') if 'train_source3' in train_data else pd.DataFrame()
        
        # Get 15 examples that have at least 1 match
        examples = gt_examples[gt_examples['num_matches'] > 0].head(15)
        
        count = 1
        for idx, row in examples.iterrows():
            s1_id = row['source1_entity_id']
            matches = str(row['matched_entity_ids']).split(',')
            
            if s1_id in s1.index:
                s1_row = s1.loc[s1_id]
                analysis.append(f"### Example {count}\n")
                analysis.append(f"**Source 1 Reference:**\n")
                analysis.append(f"- **Name**: `{s1_row.get('business_name', '')}`\n")
                analysis.append(f"- **Address**: `{s1_row.get('business_address', '')}`\n\n")
                
                analysis.append(f"**Matches in Source 2 / Source 3:**\n")
                for match_id in matches:
                    if match_id in s2.index:
                        m_row = s2.loc[match_id]
                        src_type = "Source 2"
                    elif match_id in s3.index:
                        m_row = s3.loc[match_id]
                        src_type = "Source 3"
                    else:
                        continue
                        
                    analysis.append(f"- [{src_type}] **Name**: `{m_row.get('business_name', '')}`\n")
                    analysis.append(f"- [{src_type}] **Address**: `{m_row.get('business_address', '')}`\n")
                analysis.append("\n---\n\n")
                count += 1
                
    # Write to file
    out_path = os.path.join(workspace_root, 'DATASET_ANALYSIS.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(analysis)
        
    print(f"Analysis successfully written to {out_path}")
    print("\n--- QUICK SUMMARY ---")
    print(f"Train Zero matches: {zero_matches if 'zero_matches' in locals() else 'N/A'}")
    print(f"France in train: {france_in_train}")
    print(f"France in test: {france_in_test}")
    print(f"Leakage found: {leakage_found}")

if __name__ == "__main__":
    main()
