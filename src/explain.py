"""
Model Explainability (XAI) using SHAP (SHapley Additive exPlanations).
Provides global and local feature contribution insights for clinical interpretability.
"""

import numpy as np
import pandas as pd
import shap


def get_tree_explainer(model):
    """
    Initializes a SHAP TreeExplainer for an underlying tree-based estimator.
    """
    # If passed an ensemble or pipeline, extract base XGBoost or LightGBM model
    base_model = model
    if hasattr(model, "models") and len(model.models) > 0:
        base_model = model.models[0]

    explainer = shap.TreeExplainer(base_model)
    return explainer, base_model


def explain_patient_prediction(model, patient_transformed, feature_names, top_n=6):
    """
    Computes local feature contributions (SHAP values) for a single patient's prediction.
    Returns sorted positive (risk-increasing) and negative (health-protective) drivers.
    """
    explainer, base_model = get_tree_explainer(model)
    
    # Compute SHAP values
    shap_vals = explainer.shap_values(patient_transformed)

    # Handle multi-class output (list of arrays or 3D array)
    if isinstance(shap_vals, list):
        # Class 0: at-risk, Class 1: fit, Class 2: unhealthy
        # Focus on at-risk / unhealthy risk attribution
        risk_shap = shap_vals[0][0] if len(shap_vals[0]) > 0 else np.zeros(len(feature_names))
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        risk_shap = shap_vals[0, :, 0]
    else:
        risk_shap = np.array(shap_vals).flatten()[:len(feature_names)]

    contributions = []
    for fname, val, sval in zip(feature_names, patient_transformed[0], risk_shap):
        contributions.append({
            "feature": fname,
            "feature_value": float(val),
            "shap_impact": float(sval),
            "effect": "Risk Increasing" if sval > 0 else "Protective/Neutral",
        })

    # Sort by absolute SHAP impact
    contributions = sorted(contributions, key=lambda x: abs(x["shap_impact"]), reverse=True)[:top_n]
    return contributions
