from catboost import CatBoostClassifier

from mushroom_model.config import RANDOM_SEED


def create_model(
    task_type: str = "CPU",
    iterations: int = 1000,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="Logloss",
        l2_leaf_reg=5.0,
        random_seed=RANDOM_SEED,
        task_type=task_type,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )
