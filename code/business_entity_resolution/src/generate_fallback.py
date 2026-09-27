import pandas as pd
from pathlib import Path

def main():
    print("Generating fallback submission (all singletons)...")
    dataset_dir = Path("dataset/test")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    
    # Load test source 1
    s1 = pd.read_csv(dataset_dir / "test_source1.tsv", sep="\t", usecols=["entity_id"])
    
    # Generate empty predictions
    submission = pd.DataFrame({
        "source1_entity_id": s1["entity_id"],
        "matched_entity_ids": ""
    })
    
    candidates = pd.DataFrame({
        "source1_entity_id": s1["entity_id"],
        "candidate_entity_ids": ""
    })
    
    print(f"Total entities: {len(submission):,}")
    
    # Save
    candidates.to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False)
    submission.to_csv(out_dir / "matching_results.tsv", sep="\t", index=False)
    
    print("Fallback complete. Files saved to output/candidate_pairs.tsv and output/matching_results.tsv")

if __name__ == "__main__":
    main()
