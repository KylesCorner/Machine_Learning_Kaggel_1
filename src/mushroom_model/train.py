from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from mushroom_model.config import (
    ARTIFACT_DIR,
    ID_COLUMN,
    MODEL_FILE,
    RANDOM_SEED,
    SUBMISSION_FILE,
    TARGET_COLUMN,
)
from mushroom_model.data import (
    get_categorical_columns,
    load_data,
    prepare_categorical_columns,
    split_features_target,
)
from mushroom_model.metrics import find_best_threshold
from mushroom_model.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the mushroom classification model."
    )

    parser.add_argument(
        "--task-type",
        choices=[
            "CPU",
            "GPU",
        ],
        default="CPU",
        help="CatBoost compute backend.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Maximum number of CatBoost iterations.",
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.10,
        help="Fraction of training data used for validation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading data...")

    train, test, sample_submission = load_data()

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")

    X, y, X_test, test_ids = split_features_target(
        train,
        test,
    )

    categorical_columns = get_categorical_columns(X)

    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    print()
    print(
        f"Categorical features: {len(categorical_columns)}"
    )
    print(
        f"Numerical features:   {len(numerical_columns)}"
    )

    X, X_test = prepare_categorical_columns(
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
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print()
    print("Training CatBoost...")

    model = create_model(
        task_type=args.task_type,
        iterations=args.iterations,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    poison_index = list(
        model.classes_
    ).index("p")

    valid_probabilities = model.predict_proba(
        X_valid
    )[:, poison_index]

    best_threshold, best_mcc = find_best_threshold(
        valid_probabilities,
        y_valid,
    )

    print()
    print("=" * 60)
    print(
        f"Best threshold: {best_threshold:.4f}"
    )
    print(
        f"Validation MCC: {best_mcc:.6f}"
    )
    print("=" * 60)

    print()
    print("Predicting test data...")

    test_probabilities = model.predict_proba(
        X_test
    )[:, poison_index]

    test_predictions = np.where(
        test_probabilities >= best_threshold,
        "p",
        "e",
    )

    if not np.array_equal(
        sample_submission[ID_COLUMN].to_numpy(),
        test_ids.to_numpy(),
    ):
        raise RuntimeError(
            "Submission IDs do not match test dataset IDs."
        )

    submission = sample_submission.copy()

    submission[TARGET_COLUMN] = test_predictions

    submission.to_csv(
        SUBMISSION_FILE,
        index=False,
    )

    model.save_model(
        MODEL_FILE
    )

    print()
    print(f"Model:      {MODEL_FILE}")
    print(f"Submission: {SUBMISSION_FILE}")

    print()
    print(
        submission[TARGET_COLUMN].value_counts()
    )


if __name__ == "__main__":
    main()
