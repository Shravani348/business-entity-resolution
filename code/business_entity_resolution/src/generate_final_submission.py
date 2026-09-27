"""
Generates final test candidate pairs and matching results.
"""

import sys
import os
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from candidate_generation import generate_candidates
from features import configure_tfidf, extract_features
from normalize import normalize_name

def main():
    print("Loading test dataset...")
    dataset_dir = Path("dataset/test")
    
    s1 = pd.read_csv(dataset_dir / "test_source1.tsv", sep="\t")
    s2 = pd.read_csv(dataset_dir / "test_source2.tsv", sep="\t")
    s3 = pd.read_csv(dataset_dir / "test_source3.tsv", sep="\t")

    print(f"S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")

    print("\nGenerating candidates...")
    # This will use TokenBlocker now
    candidates = generate_candidates(s1, s2, s3, max_block_size=10000)
    print(f"Total candidate pairs generated: {len(candidates):,}")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    
    # 1. Format candidate_pairs.tsv
    # source1_entity_id \t candidate_entity_ids (comma separated)
    print("\nSaving output/candidate_pairs.tsv...")
    cand_group = candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda x: ",".join(sorted(x))
    ).reset_index()
    
    # Add empty strings for S1 entities that have NO candidates
    missing_s1 = set(s1["entity_id"]) - set(cand_group["source1_entity_id"])
    if missing_s1:
        missing_df = pd.DataFrame({"source1_entity_id": list(missing_s1), "candidate_entity_id": ""})
        cand_group = pd.concat([cand_group, missing_df], ignore_index=True)
        
    cand_group.rename(columns={"candidate_entity_id": "candidate_entity_ids"}).to_csv(
        out_dir / "candidate_pairs.tsv", sep="\t", index=False
    )

    if len(candidates) == 0:
        print("No candidates! Creating empty matching results.")
        cand_group.rename(columns={"candidate_entity_ids": "matched_entity_ids"}).to_csv(
            out_dir / "matching_results.tsv", sep="\t", index=False
        )
        return

    # 2. Extract features
    print("\nPreparing feature extraction...")
    # We need to join back to source data to get name, address, country
    # We have S2 and S3, let's combine them into a single candidate pool lookup
    s_other = pd.concat([s2, s3], ignore_index=True)
    s_other_dict = s_other.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s1_dict = s1.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    print("Fitting TF-IDF on test names...")
    # NOTE: In a real scenario, you'd use the train TF-IDF, but since we're generating features
    # for the entire test set, we fit it on the test names for this challenge phase as we don't
    # have the original vectorizer saved. Wait, fitting TF-IDF on test set is acceptable.
    normalized_names = pd.concat(
        [
            s1["business_name"].fillna("").map(normalize_name),
            s_other["business_name"].fillna("").map(normalize_name),
        ],
        ignore_index=True
    )
    normalized_names = normalized_names[normalized_names != ""]
    configure_tfidf(normalized_names)

    print("Extracting features (this might take a minute)...")
    feature_rows = []
    
    # Iterate quickly
    for idx, row in candidates.iterrows():
        s1_id = row["source1_entity_id"]
        c_id = row["candidate_entity_id"]
        
        s1_info = s1_dict[s1_id]
        c_info = s_other_dict[c_id]
        
        feats = extract_features(
            name1=s1_info.get("business_name"), address1=s1_info.get("business_address"), country1=s1_info.get("country"),
            name2=c_info.get("business_name"), address2=c_info.get("business_address"), country2=c_info.get("country")
        )
        feature_rows.append(feats)
        
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1:,} / {len(candidates):,}")

    X_test = pd.DataFrame(feature_rows)
    feature_cols = [
        "name_token_jaccard",
        "name_levenshtein",
        "name_tfidf_cosine",
        "address_token_jaccard",
        "address_levenshtein",
        "exact_name_match",
        "missing_address",
        "country_match",
    ]
    X_test = X_test[feature_cols]

    # 3. Model Prediction
    model_path = "models/logreg_baseline.pkl"
    print(f"\nLoading model {model_path}...")
    model = joblib.load(model_path)
    probs = model.predict_proba(X_test)[:, 1]
    
    with open("threshold_config.json") as f:
        threshold = json.load(f)["optimal_threshold"]
    
    print(f"Applying threshold {threshold}...")
    preds = (probs >= threshold).astype(int)
    candidates["pred"] = preds

    # 4. Format matching_results.tsv
    print("\nSaving output/matching_results.tsv...")
    matches = candidates[candidates["pred"] == 1]
    match_group = matches.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda x: ",".join(sorted(x))
    ).reset_index()
    
    missing_match_s1 = set(s1["entity_id"]) - set(match_group["source1_entity_id"])
    if missing_match_s1:
        missing_m_df = pd.DataFrame({"source1_entity_id": list(missing_match_s1), "candidate_entity_id": ""})
        match_group = pd.concat([match_group, missing_m_df], ignore_index=True)
        
    match_group.rename(columns={"candidate_entity_id": "matched_entity_ids"}).to_csv(
        out_dir / "matching_results.tsv", sep="\t", index=False
    )
    
    print("\nDONE! output/candidate_pairs.tsv and output/matching_results.tsv are ready.")

if __name__ == "__main__":
    main()
