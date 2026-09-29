from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

SUBMISSION_FILE = ARTIFACT_DIR / "submission.csv"
MODEL_FILE = ARTIFACT_DIR / "catboost_model.cbm"

TARGET_COLUMN = "class"
ID_COLUMN = "id"

RANDOM_SEED = 42
