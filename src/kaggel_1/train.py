from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split


TARGET = "class"
ID_COLUMN = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
    )

    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default="CPU",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.10,
    )

    return parser.parse_args()


def prepare_categoricals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    CatBoost requires categorical values to be strings/integers rather than
    arbitrary mixtures of strings, floats, and NaN.

    We deliberately do not ordinal-encode them because the category codes
    themselves have no meaningful numeric ordering.
    """

    for column in categorical_columns:
        train[column] = (
            train[column]
            .fillna("__MISSING__")
            .astype(str)
            .str.strip()
        )

        test[column] = (
            test[column]
            .fillna("__MISSING__")
            .astype(str)
            .str.strip()
        )

    return train, test


def find_best_threshold(
    probabilities: np.ndarray,
    labels: pd.Series,
) -> tuple[float, float]:
    """
    CatBoost learns probabilities using Logloss, but Kaggle evaluates MCC.

    Instead of blindly using probability >= 0.5, search for the decision
    threshold that maximizes validation MCC.
    """

    best_threshold = 0.5
    best_score = -1.0

    thresholds = np.linspace(0.30, 0.70, 81)

    for threshold in thresholds:
        predictions = np.where(
            probabilities >= threshold,
            "p",
            "e",
        )

        score = matthews_corrcoef(labels, predictions)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    submission_path = args.data_dir / "sample_submission.csv"

    print("Loading data...")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(submission_path)

    print()
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print()

    y = train[TARGET].copy()

    test_ids = test[ID_COLUMN].copy()

    X = train.drop(
        columns=[
            TARGET,
            ID_COLUMN,
        ]
    ).copy()

    X_test = test.drop(
        columns=[ID_COLUMN]
    ).copy()

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    print(
        f"Categorical features: {len(categorical_columns)}"
    )

    print(
        f"Numerical features:   {len(numerical_columns)}"
    )

    X, X_test = prepare_categoricals(
        X,
        X_test,
        categorical_columns,
    )

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = train_test_split(
        X,
        y,
        test_size=args.validation_size,
        random_state=42,
        stratify=y,
    )

    print()
    print("Training CatBoost...")
    print()

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=8,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="Logloss",
        l2_leaf_reg=5.0,
        random_seed=42,
        task_type=args.task_type,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    print()
    print("Evaluating validation set...")

    class_names = list(model.classes_)

    poison_index = class_names.index("p")

    valid_probabilities = model.predict_proba(
        X_valid
    )[:, poison_index]

    best_threshold, best_mcc = find_best_threshold(
        valid_probabilities,
        y_valid,
    )

    print()
    print("=" * 60)
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Validation MCC: {best_mcc:.6f}")
    print("=" * 60)

    print()
    print("Generating test predictions...")

    test_probabilities = model.predict_proba(
        X_test
    )[:, poison_index]

    test_predictions = np.where(
        test_probabilities >= best_threshold,
        "p",
        "e",
    )

    submission = sample_submission.copy()

    if not np.array_equal(
        submission[ID_COLUMN].to_numpy(),
        test_ids.to_numpy(),
    ):
        raise RuntimeError(
            "sample_submission.csv IDs do not match test.csv IDs"
        )

    submission[TARGET] = test_predictions

    output_file = (
        args.output_dir / "submission.csv"
    )

    submission.to_csv(
        output_file,
        index=False,
    )

    model_file = (
        args.output_dir / "catboost_model.cbm"
    )

    model.save_model(model_file)

    print()
    print(f"Model:      {model_file}")
    print(f"Submission: {output_file}")

    print()
    print(submission[TARGET].value_counts())

    print()
    print("Done.")


if __name__ == "__main__":
    main()
