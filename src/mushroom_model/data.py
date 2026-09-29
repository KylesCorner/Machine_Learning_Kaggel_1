from __future__ import annotations

import pandas as pd

from mushroom_model.config import (
    ID_COLUMN,
    SAMPLE_SUBMISSION_FILE,
    TARGET_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)

    return train, test, sample_submission


def get_categorical_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    return (
        dataframe.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )


def prepare_categorical_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

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


def split_features_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    y = train[TARGET_COLUMN].copy()

    test_ids = test[ID_COLUMN].copy()

    X = train.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ]
    ).copy()

    X_test = test.drop(
        columns=[ID_COLUMN]
    ).copy()

    return X, y, X_test, test_ids
