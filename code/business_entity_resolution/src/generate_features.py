from pathlib import Path

import pandas as pd

from features import (
    configure_tfidf,
    extract_features,
)

from normalize import normalize_name


INPUT_PATH = Path("prototype_data/feature_pairs.csv")
OUTPUT_PATH = Path("prototype_data/features_prototype.csv")


def main():

    print("Loading prototype pairs...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df):,} pairs.")

    # ---------------------------------------------------------
    # 0. Entity-level split
    # ---------------------------------------------------------
    print("Performing entity-level split...")
    
    from sklearn.model_selection import train_test_split
    
    # Get unique entities
    unique_s1_ids = df['s1_id'].unique()
    
    # Split entities (using seed=42 for reproducibility)
    train_ids, test_ids = train_test_split(unique_s1_ids.tolist(), test_size=0.2, random_state=42)
    
    # Assign split
    df['split'] = 'test'
    df.loc[df['s1_id'].isin(train_ids), 'split'] = 'train'
    
    train_df = df[df['split'] == 'train']
    print(f"Train pairs: {len(train_df):,}")
    print(f"Test pairs: {len(df) - len(train_df):,}")

    # ---------------------------------------------------------
    # 1. Prepare TF-IDF corpus (from TRAIN only)
    # ---------------------------------------------------------
    print("Preparing TF-IDF corpus (train split only)...")

    normalized_names = pd.concat(
        [
            train_df["s1_name"].fillna("").map(normalize_name),
            train_df["other_name"].fillna("").map(normalize_name),
        ],
        ignore_index=True,
    )

    normalized_names = normalized_names[
        normalized_names != ""
    ]

    print(
        f"TF-IDF training corpus size: "
        f"{len(normalized_names):,}"
    )

    # ---------------------------------------------------------
    # 2. Fit TF-IDF
    # ---------------------------------------------------------
    print("Fitting TF-IDF...")

    configure_tfidf(normalized_names)

    print("TF-IDF fitted.")

    # ---------------------------------------------------------
    # 3. Extract features (for both train and test)
    # ---------------------------------------------------------
    print("Extracting features...")

    feature_rows = []

    for i, row in df.iterrows():

        features = extract_features(
            name1=row["s1_name"],
            address1=row["s1_address"],
            name2=row["other_name"],
            address2=row["other_address"],
            country1=row["s1_country"],
            country2=row["other_country"],
        )

        feature_rows.append(features)

        if (i + 1) % 5000 == 0:
            print(
                f"Processed {i + 1:,} / "
                f"{len(df):,}"
            )

    features_df = pd.DataFrame(feature_rows)

    # ---------------------------------------------------------
    # 4. Combine IDs, labels, split, and features
    # ---------------------------------------------------------
    output = pd.concat(
        [
            df[
                [
                    "s1_id",
                    "other_id",
                    "other_source",
                    "label",
                    "split",
                ]
            ].reset_index(drop=True),

            features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDone.")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(output):,}")


if __name__ == "__main__":
    main()