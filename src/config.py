"""
Central Configuration for Health Condition Prediction.
Defines column mappings, model hyper-parameters, file paths, and random seeds.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODELS_DIR / "ensemble_pipeline.joblib"
METADATA_SAVE_PATH = MODELS_DIR / "metadata.json"
SAMPLE_DATA_PATH = DATA_DIR / "sample_patients.csv"

# Target & Feature definitions
TARGET_COL = "health_condition"
ID_COL = "id"

NUMERICAL_COLS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

CATEGORICAL_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

ENGINEERED_NUMERICAL_COLS = [
    "calorie_per_step",
    "active_to_sleep_ratio",
    "hydration_index",
    "cardio_metabolic_risk",
    "lifestyle_score",
    "cardio_strain_ratio",
    "sleep_debt_factor",
]

ENGINEERED_CATEGORICAL_COLS = [
    "bmi_category",
]

ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# Target Classes
CLASS_NAMES = ["at-risk", "fit", "unhealthy"]

# Model Hyperparameters
RANDOM_STATE = 42
N_SPLITS = 5

XGB_PARAMS = {
    "objective": "multi:softprob",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 7,
    "loss_function": "MultiClass",
    "verbose": 0,
    "random_seed": RANDOM_STATE,
}

LGBM_PARAMS = {
    "objective": "multiclass",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}
