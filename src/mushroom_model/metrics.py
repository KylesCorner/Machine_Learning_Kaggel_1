from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


def find_best_threshold(
    probabilities: np.ndarray,
    labels: pd.Series,
    minimum: float = 0.30,
    maximum: float = 0.70,
    steps: int = 81,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    thresholds = np.linspace(
        minimum,
        maximum,
        steps,
    )

    for threshold in thresholds:
        predictions = np.where(
            probabilities >= threshold,
            "p",
            "e",
        )

        score = matthews_corrcoef(
            labels,
            predictions,
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
