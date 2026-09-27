from pathlib import Path
import argparse
import pandas as pd

from candidate_generation import generate_candidates, evaluate_recall


def load_data(dataset_dir):
    dataset_dir = Path(dataset_dir)

    source1 = pd.read_csv(
        dataset_dir / "train_source1.tsv",
        sep="\t",
        dtype=str,
    )

    source2 = pd.read_csv(
        dataset_dir / "train_source2.tsv",
        sep="\t",
        dtype=str,
    )

    source3 = pd.read_csv(
        dataset_dir / "train_source3.tsv",
        sep="\t",
        dtype=str,
    )

    ground_truth = pd.read_csv(
        dataset_dir / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
    )

    return source1, source2, source3, ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Test candidate generation and recall."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the train dataset directory.",
    )

    parser.add_argument(
        "--max-block-size",
        type=int,
        default=2000,
        help="Maximum allowed block size.",
    )

    args = parser.parse_args()

    print("Loading data...")

    source1, source2, source3, ground_truth = load_data(
        args.dataset_dir
    )

    print(f"Source 1:     {len(source1):,}")
    print(f"Source 2:     {len(source2):,}")
    print(f"Source 3:     {len(source3):,}")
    print(f"Ground truth: {len(ground_truth):,}")

    print("\nGenerating candidates...")

    candidates = generate_candidates(
        source1,
        source2,
        source3,
        max_block_size=args.max_block_size,
    )

    print(f"Candidates generated: {len(candidates):,}")

    print("\nEvaluating recall...")

    recall = evaluate_recall(
        candidates,
        ground_truth,
    )

    print(f"Candidate recall: {recall:.4%}")

    print("\nCandidate distribution:")
    print(candidates["candidate_source"].value_counts())

    print("\nBlocking distribution:")
    print(
        candidates[
            ["matched_on_name", "matched_on_address"]
        ].value_counts()
    )

    print("\nAverage candidates per Source-1 record:")
    print(
        f"{len(candidates) / len(source1):.2f}"
    )


if __name__ == "__main__":
    main()