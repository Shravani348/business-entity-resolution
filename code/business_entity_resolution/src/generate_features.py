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
    # 1. Normalize all names for TF-IDF fitting
    # ---------------------------------------------------------

    print("Preparing TF-IDF corpus...")

    normalized_names = pd.concat(
        [
            df["s1_name"].fillna("").map(normalize_name),
            df["other_name"].fillna("").map(normalize_name),
        ],
        ignore_index=True,
    )

    # Remove completely empty names.
    normalized_names = normalized_names[
        normalized_names != ""
    ]

    print(
        f"TF-IDF corpus size: "
        f"{len(normalized_names):,}"
    )

    # ---------------------------------------------------------
    # 2. Fit TF-IDF
    # ---------------------------------------------------------

    print("Fitting TF-IDF...")

    configure_tfidf(normalized_names)

    print("TF-IDF fitted.")

    # ---------------------------------------------------------
    # 3. Extract features
    # ---------------------------------------------------------

    print("Extracting features...")

    feature_rows = []

    for i, row in df.iterrows():

        features = extract_features(
            name1=row["s1_name"],
            address1=row["s1_address"],
            name2=row["other_name"],
            address2=row["other_address"],
            country=(
                row["s1_country"],
                row["other_country"],
            ),
        )

        feature_rows.append(features)

        if (i + 1) % 1000 == 0:
            print(
                f"Processed {i + 1:,} / "
                f"{len(df):,}"
            )

    features_df = pd.DataFrame(feature_rows)

    # ---------------------------------------------------------
    # 4. Fix country_match
    # ---------------------------------------------------------

    features_df["country_match"] = (
        df["s1_country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        df["other_country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    ).astype(int)

    # ---------------------------------------------------------
    # 5. Combine IDs, labels and features
    # ---------------------------------------------------------

    output = pd.concat(
        [
            df[
                [
                    "s1_id",
                    "other_id",
                    "other_source",
                    "label",
                ]
            ].reset_index(drop=True),
            features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # 6. Save
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

    print("\nFeature columns:")
    for column in features_df.columns:
        print(f"  - {column}")


if __name__ == "__main__":
    main()