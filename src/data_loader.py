"""
Data Loading, Preprocessing Pipeline, and Synthetic Dataset Generator.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.config import (
    CATEGORICAL_COLS,
    ENGINEERED_CATEGORICAL_COLS,
    ENGINEERED_NUMERICAL_COLS,
    ID_COL,
    N_SPLITS,
    NUMERICAL_COLS,
    RANDOM_STATE,
    SAMPLE_DATA_PATH,
    TARGET_COL,
)
from src.features import HealthFeatureEngineer


def get_preprocessor():
    """
    Constructs a comprehensive Scikit-Learn Preprocessor with Feature Engineering.
    """
    all_num_cols = NUMERICAL_COLS + ENGINEERED_NUMERICAL_COLS
    all_cat_cols = CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, all_num_cols),
            ("cat", cat_transformer, all_cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineer", HealthFeatureEngineer()),
            ("preprocessor", preprocessor),
        ]
    )

    return full_pipeline


def get_feature_names():
    """Returns the ordered list of all processed feature names."""
    return (
        NUMERICAL_COLS
        + ENGINEERED_NUMERICAL_COLS
        + CATEGORICAL_COLS
        + ENGINEERED_CATEGORICAL_COLS
    )


def create_stratified_folds(df, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """
    Generates Stratified K-Fold train/validation indices for the dataset.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(df, df[TARGET_COL])


def generate_synthetic_patients(n_samples=2500, random_state=RANDOM_STATE):
    """
    Generates a realistic synthetic patient dataset mirroring the S6E7 Kaggle distribution.
    Enables local pipeline training, unit testing, and offline app demonstration.
    """
    rng = np.random.default_rng(random_state)

    # Demographic & Lifestyle Distributions
    gender = rng.choice(["male", "female", "other"], size=n_samples, p=[0.48, 0.48, 0.04])
    diet = rng.choice(["veg", "non-veg", "vegan"], size=n_samples, p=[0.35, 0.55, 0.10])
    stress = rng.choice(["low", "moderate", "high"], size=n_samples, p=[0.25, 0.45, 0.30])
    sleep_quality = rng.choice(["good", "average", "poor"], size=n_samples, p=[0.30, 0.50, 0.20])
    activity = rng.choice(["sedentary", "moderate", "active"], size=n_samples, p=[0.35, 0.45, 0.20])
    smoking_alcohol = rng.choice(["no", "occasional", "yes"], size=n_samples, p=[0.50, 0.30, 0.20])

    # Physiological Distributions
    bmi = np.clip(rng.normal(loc=26.5, scale=5.0, size=n_samples), 16.0, 45.0)
    heart_rate = np.clip(rng.normal(loc=74.0, scale=9.0, size=n_samples), 50.0, 110.0)
    sleep_duration = np.clip(rng.normal(loc=6.8, scale=1.3, size=n_samples), 3.5, 10.5)
    exercise_duration = np.clip(rng.exponential(scale=35.0, size=n_samples), 0.0, 150.0)
    calorie_exp = np.clip(rng.normal(loc=2150.0, scale=400.0, size=n_samples), 1200.0, 3800.0)
    step_count = np.clip(rng.normal(loc=6500.0, scale=3200.0, size=n_samples), 500.0, 22000.0)
    water_intake = np.clip(rng.normal(loc=2.1, scale=0.7, size=n_samples), 0.5, 5.0)

    # Determine risk category based on composite factors + noise
    risk_score = (
        (bmi - 22.0) * 0.15
        + (heart_rate - 70.0) * 0.08
        - (sleep_duration - 7.0) * 0.4
        - (exercise_duration / 30.0) * 0.5
        - (step_count / 5000.0) * 0.4
        + (stress == "high") * 0.8
        + (stress == "moderate") * 0.3
        + (sleep_quality == "poor") * 0.7
        + (activity == "sedentary") * 0.6
        + (smoking_alcohol == "yes") * 1.0
        + rng.normal(0, 0.4, size=n_samples)
    )

    # Target Mapping with realistic class proportions (~85% at-risk, ~8% unhealthy, ~7% fit)
    conditions = [
        risk_score < -0.8,
        (risk_score >= -0.8) & (risk_score < 1.8),
        risk_score >= 1.8,
    ]
    target_classes = ["fit", "at-risk", "unhealthy"]
    target = np.select(conditions, target_classes, default="at-risk")

    df = pd.DataFrame({
        "id": np.arange(n_samples),
        "health_condition": target,
        "sleep_duration": np.round(sleep_duration, 2),
        "heart_rate": np.round(heart_rate, 1),
        "bmi": np.round(bmi, 2),
        "calorie_expenditure": np.round(calorie_exp, 0),
        "step_count": np.round(step_count, 0),
        "exercise_duration": np.round(exercise_duration, 1),
        "water_intake": np.round(water_intake, 2),
        "diet_type": diet,
        "stress_level": stress,
        "sleep_quality": sleep_quality,
        "physical_activity_level": activity,
        "smoking_alcohol": smoking_alcohol,
        "gender": gender,
    })

    # Add realistic ~2% missing values in random columns
    for col in ["bmi", "sleep_duration", "water_intake", "diet_type", "smoking_alcohol"]:
        mask = rng.random(n_samples) < 0.02
        df.loc[mask, col] = np.nan

    return df


def load_data(filepath=None):
    """
    Loads training dataset from filepath, or creates/loads sample benchmark data if not found.
    """
    if filepath and pd.io.common.file_exists(filepath):
        return pd.read_csv(filepath)

    if SAMPLE_DATA_PATH.exists():
        return pd.read_csv(SAMPLE_DATA_PATH)

    # Generate and save benchmark sample dataset
    df = generate_synthetic_patients(n_samples=3000)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    return df
