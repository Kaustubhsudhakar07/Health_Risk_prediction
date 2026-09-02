"""
Unit Tests for Health Condition Prediction Pipeline.
Validates Feature Engineering, Preprocessing, Ensembling, and Inference Engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CLASS_NAMES,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
)
from src.data_loader import (
    generate_synthetic_patients,
    get_feature_names,
    get_preprocessor,
)
from src.features import HealthFeatureEngineer
from src.models import SoftVotingEnsemble
from src.predict import HealthRiskPredictor


@pytest.fixture
def sample_raw_df():
    """Provides a small sample dataframe for testing."""
    return generate_synthetic_patients(n_samples=50, random_state=42)


def test_feature_engineer_columns(sample_raw_df):
    """Verifies that domain feature engineering produces all expected columns."""
    engineer = HealthFeatureEngineer()
    df_transformed = engineer.transform(sample_raw_df)

    expected_cols = [
        "bmi_category",
        "calorie_per_step",
        "active_to_sleep_ratio",
        "hydration_index",
        "cardio_metabolic_risk",
        "lifestyle_score",
        "cardio_strain_ratio",
        "sleep_debt_factor",
    ]
    for col in expected_cols:
        assert col in df_transformed.columns
        assert not df_transformed[col].isna().all()


def test_preprocessor_pipeline(sample_raw_df):
    """Verifies that the ColumnTransformer pipeline handles raw data with missing values."""
    X = sample_raw_df.drop(columns=["id", "health_condition"])
    preprocessor = get_preprocessor()

    X_transformed = preprocessor.fit_transform(X)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(sample_raw_df)
    assert X_transformed.shape[1] == len(get_feature_names())
    assert not np.isnan(X_transformed).any()


def test_synthetic_data_generator():
    """Verifies synthetic dataset generation integrity and class validity."""
    df = generate_synthetic_patients(n_samples=100)
    assert len(df) == 100
    assert "health_condition" in df.columns
    unique_classes = set(df["health_condition"].unique())
    assert unique_classes.issubset(set(CLASS_NAMES))


def test_soft_voting_ensemble():
    """Tests probability aggregation logic of SoftVotingEnsemble."""
    class DummyModel:
        def __init__(self, prob_matrix):
            self.prob_matrix = prob_matrix

        def predict_proba(self, X):
            return self.prob_matrix

    dummy_m1 = DummyModel(np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]))
    dummy_m2 = DummyModel(np.array([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]))

    ensemble = SoftVotingEnsemble(models=[dummy_m1, dummy_m2])
    probs = ensemble.predict_proba(np.zeros((2, 5)))

    expected_p0 = (0.8 + 0.6) / 2.0
    expected_p1 = (0.7 + 0.8) / 2.0

    assert np.isclose(probs[0, 0], expected_p0)
    assert np.isclose(probs[1, 1], expected_p1)

    preds = ensemble.predict(np.zeros((2, 5)))
    assert preds[0] == 0
    assert preds[1] == 1


def test_inference_engine_single(sample_raw_df):
    """Tests single-patient inference and clinical recommendation output."""
    predictor = HealthRiskPredictor()
    single_record = sample_raw_df.iloc[0].to_dict()

    result = predictor.predict_single(single_record)

    assert "predicted_condition" in result
    assert result["predicted_condition"] in CLASS_NAMES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == len(CLASS_NAMES)
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0


def test_inference_engine_batch(sample_raw_df):
    """Tests batch prediction DataFrame generation."""
    predictor = HealthRiskPredictor()
    result_df = predictor.predict_batch(sample_raw_df)

    assert "predicted_health_condition" in result_df.columns
    assert "prob_at_risk" in result_df.columns
    assert "prob_fit" in result_df.columns
    assert "prob_unhealthy" in result_df.columns
    assert len(result_df) == len(sample_raw_df)
