"""Merge customer data, train the product model, and make recommendations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOCIAL_PATH = ROOT / "dataset" / "customer_social_profiles - customer_social_profiles.csv"
TRANSACTIONS_PATH = ROOT / "dataset" / "customer_transactions - customer_transactions.csv"
MERGED_PATH = ROOT / "data" / "merged_customer_data.csv"
MODEL_PATH = ROOT / "models" / "product_model.joblib"
METRICS_PATH = ROOT / "data" / "product_metrics.json"
PLOT_PATH = ROOT / "data" / "product_confusion_matrix.png"


def load_merged_data(
    social_path: Path = SOCIAL_PATH,
    transactions_path: Path = TRANSACTIONS_PATH,
) -> pd.DataFrame:
    """Aggregate repeated social records and merge them with transactions."""
    social = pd.read_csv(social_path)
    transactions = pd.read_csv(transactions_path)
    social_duplicate_rows = int(social.duplicated().sum())
    transaction_duplicate_rows = int(transactions.duplicated().sum())
    social["customer_id"] = social["customer_id_new"].str.extract(r"(\d+)").astype(int)
    transactions["customer_id"] = transactions["customer_id_legacy"].astype(int)

    social_summary = (
        social.groupby("customer_id", as_index=False)
        .agg(
            social_media_platform=("social_media_platform", lambda values: values.mode().iat[0]),
            engagement_score=("engagement_score", "mean"),
            purchase_interest_score=("purchase_interest_score", "mean"),
            review_sentiment=("review_sentiment", lambda values: values.mode().iat[0]),
        )
    )
    merged = transactions.merge(
        social_summary, on="customer_id", how="inner", validate="many_to_one"
    )
    merged["purchase_date"] = pd.to_datetime(merged["purchase_date"])
    merged["customer_rating"] = merged["customer_rating"].fillna(
        merged["customer_rating"].median()
    )
    if merged.empty:
        raise ValueError("The tabular merge produced no rows.")
    merged.attrs["transaction_rows"] = len(transactions)
    merged.attrs["unmatched_transaction_rows"] = len(transactions) - len(merged)
    merged.attrs["social_duplicate_rows"] = social_duplicate_rows
    merged.attrs["transaction_duplicate_rows"] = transaction_duplicate_rows
    return merged


def train_product_model(merged: pd.DataFrame) -> dict[str, object]:
    """Compare leakage-safe baselines and save the best product classifier."""
    features = [
        "purchase_amount",
        "customer_rating",
        "engagement_score",
        "purchase_interest_score",
        "social_media_platform",
        "review_sentiment",
    ]
    target = "product_category"
    x_train, x_test, y_train, y_test = train_test_split(
        merged[features],
        merged[target],
        test_size=0.25,
        random_state=42,
        stratify=merged[target],
    )
    categorical = ["social_media_platform", "review_sentiment"]
    numeric = [column for column in features if column not in categorical]
    numeric_transformer = Pipeline([("scale", StandardScaler())])
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", numeric_transformer, numeric),
        ]
    )
    candidates = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    evaluations: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, classifier in candidates.items():
        candidate = Pipeline(
            [("features", preprocessor), ("classifier", classifier)]
        )
        candidate.fit(x_train, y_train)
        predictions = candidate.predict(x_test)
        probabilities = candidate.predict_proba(x_test)
        evaluations.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision_macro": float(
                    precision_score(y_test, predictions, average="macro", zero_division=0)
                ),
                "recall_macro": float(
                    recall_score(y_test, predictions, average="macro", zero_division=0)
                ),
                "f1_macro": float(
                    f1_score(y_test, predictions, average="macro", zero_division=0)
                ),
                "log_loss": float(
                    log_loss(y_test, probabilities, labels=candidate.classes_)
                ),
            }
        )
        fitted_models[name] = candidate
    selected_evaluation = max(
        evaluations,
        key=lambda evaluation: (
            evaluation["f1_macro"],
            evaluation["accuracy"],
            -evaluation["log_loss"],
        ),
    )
    selected_name = str(selected_evaluation["model"])
    model = fitted_models[selected_name]
    metrics: dict[str, object] = {
        "selected_model": selected_name,
        "model_comparison": evaluations,
        "rows_after_merge": int(len(merged)),
        "transaction_rows_before_merge": int(
            merged.attrs.get("transaction_rows", len(merged))
        ),
        "unmatched_transaction_rows": int(
            merged.attrs.get("unmatched_transaction_rows", 0)
        ),
        "social_duplicate_rows": int(merged.attrs.get("social_duplicate_rows", 0)),
        "transaction_duplicate_rows": int(
            merged.attrs.get("transaction_duplicate_rows", 0)
        ),
        "missing_values_after_cleaning": int(
            merged[features + [target]].isna().sum().sum()
        ),
        "target_distribution": merged[target].value_counts().to_dict(),
    }

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    predictions = model.predict(x_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=model.classes_, xticks_rotation=45, ax=axis
    )
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": features, "classes": list(model.classes_)},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def recommend_product(
    customer_id: str | int,
    model_path: Path = MODEL_PATH,
    merged_path: Path = MERGED_PATH,
) -> tuple[str, float]:
    """Return the predicted product category and its probability."""
    customer_number = int(str(customer_id).lstrip("A"))
    merged = pd.read_csv(merged_path)
    row = merged.loc[merged["customer_id"].eq(customer_number)]
    if row.empty:
        raise ValueError(f"Customer {customer_id} is not present in the merged dataset.")
    bundle = joblib.load(model_path)
    probabilities = bundle["model"].predict_proba(row[bundle["features"]])[0]
    index = int(probabilities.argmax())
    return str(bundle["classes"][index]), float(probabilities[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge tabular data and train product model")
    parser.add_argument("--customer-id", help="Also print a recommendation for this customer")
    args = parser.parse_args()
    merged = load_merged_data()
    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_PATH, index=False)
    metrics = train_product_model(merged)
    print("Merge validation:", metrics)
    if args.customer_id:
        product, confidence = recommend_product(args.customer_id)
        print(f"Recommendation: {product} ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()
