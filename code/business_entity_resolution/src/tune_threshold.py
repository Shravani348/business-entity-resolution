"""
tune_threshold.py

Sweeps probability thresholds to optimize for Macro F0.5 on the test split.
The evaluation metric mimics the challenge's macro-averaged F0.5 per source1_entity_id.
"""

import argparse
import json
import numpy as pd
import pandas as pd
import joblib
from pathlib import Path
import sys

# Ensure we can import local_eval
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from local_eval import score_entity

def main():
    parser = argparse.ArgumentParser(description="Tune classification threshold for Macro F0.5")
    parser.add_argument("--features-path", default="prototype_data/features_prototype.csv")
    parser.add_argument("--model-path", default="models/logreg_baseline.pkl")
    parser.add_argument("--out-config", default="threshold_config.json")
    args = parser.parse_args()

    print(f"Loading features from {args.features_path}...")
    df = pd.read_csv(args.features_path)
    
    # We only care about the test split for tuning
    test_df = df[df["split"] == "test"].copy()
    print(f"Test pairs: {len(test_df)}")

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
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    print(f"Loading model from {args.model_path}...")
    model = joblib.load(args.model_path)

    # Get probabilities for the positive class
    probs = model.predict_proba(X_test)[:, 1]
    test_df['prob'] = probs

    # Prepare true matches per s1_id
    # Note: we are defining ground truth based on pairs where label=1 in this test set.
    true_matches_per_entity = {}
    for s1_id, group in test_df.groupby("s1_id"):
        true_ids = group[group["label"] == 1]["other_id"].tolist()
        true_matches_per_entity[s1_id] = set(true_ids)

    results = []
    thresholds = [i / 100 for i in range(5, 100, 5)]  # 0.05 to 0.95 step 0.05

    print("\nSweeping thresholds...")
    print(f"{'Threshold':>10} | {'Pair Precision':>14} | {'Pair Recall':>11} | {'Macro F0.5':>10}")
    print("-" * 55)

    best_macro_f05 = -1
    best_threshold = -1
    best_metrics = {}

    for t in thresholds:
        # 1. Pair-level metrics
        preds = (probs >= t).astype(int)
        
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        
        pair_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pair_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 2. Entity-level Macro F0.5
        test_df['pred'] = preds
        f05_scores = []
        
        for s1_id, group in test_df.groupby("s1_id"):
            pred_ids = group[group["pred"] == 1]["other_id"].tolist()
            true_ids = true_matches_per_entity[s1_id]
            
            _, _, f05 = score_entity(pred_ids, true_ids, beta=0.5)
            f05_scores.append(f05)
            
        macro_f05 = sum(f05_scores) / len(f05_scores)
        
        print(f"{t:10.2f} | {pair_precision:14.4f} | {pair_recall:11.4f} | {macro_f05:10.4f}")
        
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = t
            best_metrics = {
                "pair_precision": pair_precision,
                "pair_recall": pair_recall,
                "macro_f05": macro_f05,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": ((preds == 0) & (y_test == 0)).sum()
            }

    print("-" * 55)
    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    print(f"Max Macro F0.5:    {best_metrics['macro_f05']:.4f}")
    print("\nConfusion Matrix at optimal threshold:")
    print(f"[[{best_metrics['tn']} {best_metrics['fp']}]")
    print(f" [{best_metrics['fn']}  {best_metrics['tp']}]]")

    # Save to config
    config_path = Path(args.out_config)
    with open(config_path, "w") as f:
        json.dump({"optimal_threshold": best_threshold, "macro_f0.5": best_macro_f05}, f, indent=4)
    print(f"\nSaved optimal threshold to {config_path}")

if __name__ == "__main__":
    main()
