"""
Model Factory and Soft Voting Ensemble Implementation.
Supports XGBoost, CatBoost, and LightGBM with probability aggregation.
"""

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from src.config import (
    CATBOOST_PARAMS,
    LGBM_PARAMS,
    RANDOM_STATE,
    XGB_PARAMS,
)


def get_xgb_model(num_classes=3, **kwargs):
    """Initializes and returns a configured XGBoost Classifier."""
    params = XGB_PARAMS.copy()
    params["num_class"] = num_classes
    params.update(kwargs)
    return XGBClassifier(**params)


def get_catboost_model(num_classes=3, **kwargs):
    """Initializes and returns a configured CatBoost Classifier."""
    params = CATBOOST_PARAMS.copy()
    params.update(kwargs)
    return CatBoostClassifier(**params)


def get_xgb_model(num_classes=3, **kwargs):
    """Initializes and returns a configured XGBoost Classifier."""
    params = XGB_PARAMS.copy()
    params["num_class"] = num_classes
    params.update(kwargs)
    return XGBClassifier(**params)


def get_catboost_model(num_classes=3, **kwargs):
    """Initializes and returns a configured CatBoost Classifier with auto class weighting."""
    params = CATBOOST_PARAMS.copy()
    params["auto_class_weights"] = "Balanced"
    params.update(kwargs)
    return CatBoostClassifier(**params)


def get_lgbm_model(num_classes=3, **kwargs):
    """Initializes and returns a configured LightGBM Classifier with balanced class weight."""
    params = LGBM_PARAMS.copy()
    params["num_class"] = num_classes
    params["class_weight"] = "balanced"
    params.update(kwargs)
    return LGBMClassifier(**params)


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Soft-Voting Ensemble combining predictions from multiple cross-validation folds
    and heterogeneous gradient-boosting architectures (XGBoost, CatBoost, LightGBM)
    with post-processed optimal probability thresholding.
    """

    def __init__(self, models=None, weights=None, class_multipliers=None):
        self.models = models if models is not None else []
        self.weights = weights
        self.class_multipliers = list(class_multipliers) if class_multipliers is not None else [1.0, 1.0, 1.0]
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y):
        """Fit all underlying models on the dataset."""
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Computes weighted average class probabilities across all member models.
        """
        if not self.models:
            raise ValueError("Ensemble has no fitted models.")

        n_samples = len(X)
        n_classes = len(self.classes_)
        combined_probs = np.zeros((n_samples, n_classes), dtype=float)

        weights = self.weights if self.weights is not None else [1.0] * len(self.models)
        total_weight = sum(weights)

        for model, weight in zip(self.models, weights):
            probs = model.predict_proba(X)
            # Ensure shape is 2D (n_samples, n_classes)
            if probs.ndim == 1:
                probs = np.vstack([1 - probs, probs]).T
            combined_probs += (probs * weight)

        avg_probs = combined_probs / total_weight
        # Normalize to guarantee exact sum to 1.0 per row
        row_sums = avg_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return avg_probs / row_sums

    def predict(self, X):
        """Predicts class labels applying optimal multi-class threshold multipliers."""
        probs = self.predict_proba(X)
        multipliers = np.array(self.class_multipliers)
        weighted_probs = probs * multipliers
        return np.argmax(weighted_probs, axis=1)
