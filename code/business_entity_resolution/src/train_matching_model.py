"""
train_matching_model.py

This script trains a baseline matching classifier (Logistic Regression) on the 
prototype feature dataset. It acts as the modeling stage of the pipeline, sitting 
after feature generation and before threshold tuning/evaluation. 

Inputs: 
    --features-path: Path to the generated feature dataset (e.g., prototype_data/features_prototype.csv)

Outputs:
    Saves a trained model to the specified --model-out path.
    Prints evaluation metrics (Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix) 
    and feature importances (coefficients) to the console.
"""

import argparse
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)

def main():
    # Set up command-line arguments so we avoid hardcoding paths.
    parser = argparse.ArgumentParser(description="Train a baseline matching model.")
    parser.add_argument(
        "--features-path",
        required=True,
        help="Path to the prototype feature CSV",
    )
    parser.add_argument(
        "--model-out",
        default="models/logreg_baseline.pkl",
        help="Path to save the trained model (default: models/logreg_baseline.pkl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    print(f"Loading features from {args.features_path}...")
    # Load the feature dataset
    df = pd.read_csv(args.features_path)

    # Separate the columns into IDs, target, and features
    id_cols = ["s1_id", "other_id", "other_source"]
    target_col = "label"
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

    # Use the pre-computed entity-level split.
    # WHY THIS MATTERS:
    # We now perform an entity-level split before generating features to avoid 
    # TF-IDF leakage, so we simply use the 'split' column provided in the dataset.
    print("Using pre-computed entity-level train/test split...")
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Initialize and train a Logistic Regression model.
    # WHY WE START SIMPLE (Logistic Regression):
    # Before trying complex models like XGBoost or Neural Networks, a linear model 
    # provides an interpretable baseline. It allows us to clearly see the weight 
    # assigned to each feature, and helps us quickly verify whether our engineered 
    # features actually carry predictive signal for the matching task.
    print("Training Logistic Regression baseline...")
    model = LogisticRegression(random_state=args.seed, max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions on the test set for evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    print("\n--- Evaluation on Test Split ---")
    
    # Calculate precision, recall, and F1 specifically for the positive (match) class
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Calculate ROC-AUC and PR-AUC
    # WHY PR-AUC MATTERS MORE THAN ROC-AUC HERE:
    # ROC-AUC can be overly optimistic when dealing with imbalanced classes because 
    # the large number of true negatives dominates the calculation. PR-AUC (Average 
    # Precision) focuses solely on the positive class and the trade-off between 
    # precision and recall. Since the real challenge evaluates using an F0.5 score 
    # (which weighs precision higher than recall), precision-focused metrics like 
    # PR-AUC give us a more accurate picture of how well we are doing.
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Feature Coefficients ---")
    # INTERPRETING COEFFICIENTS:
    # - Sign (+/-): A positive coefficient means that as the feature value increases, 
    #   the model is more likely to predict a match (label 1). A negative coefficient 
    #   means the opposite.
    # - Magnitude: A larger absolute value means the feature has a stronger influence 
    #   on the final prediction (assuming features are on a roughly similar scale).
    
    # Pair feature names with their learned coefficients
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_[0]
    })
    
    # Sort by absolute value to see the most influential features at the top
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by="Abs_Coefficient", ascending=False).drop(columns=["Abs_Coefficient"])
    
    # Print the sorted coefficients
    for _, row in coef_df.iterrows():
        print(f"{row['Feature']:25s}: {row['Coefficient']:.4f}")

    # Save the trained model
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\nModel saved to {out_path}")

if __name__ == "__main__":
    main()
