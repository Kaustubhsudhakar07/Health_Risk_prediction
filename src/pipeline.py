"""
End-to-End Training, Cross-Validation, Ensembling, and Artifact Serialization Pipeline.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import (
    CLASS_NAMES,
    METADATA_SAVE_PATH,
    MODEL_SAVE_PATH,
    N_SPLITS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
)
from src.data_loader import (
    get_feature_names,
    get_preprocessor,
    load_data,
)
from src.evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.models import (
    SoftVotingEnsemble,
    get_catboost_model,
    get_lgbm_model,
    get_xgb_model,
)


from scipy.optimize import minimize
from sklearn.utils.class_weight import compute_sample_weight


def optimize_threshold_multipliers(oof_probs, y_true):
    """
    Finds optimal probability multipliers using Nelder-Mead optimization
    to maximize Out-of-Fold Balanced Accuracy.
    """
    def objective(weights):
        w = np.array(weights)
        weighted_probs = oof_probs * w
        preds = np.argmax(weighted_probs, axis=1)
        return -balanced_accuracy_score(y_true, preds)

    init_weights = [1.0, 1.0, 1.0]
    result = minimize(objective, init_weights, method="Nelder-Mead", options={"maxiter": 500})
    return [float(x) for x in result.x]


def train_and_evaluate_pipeline(data_path=None, save_model=True, n_splits=N_SPLITS, fast_mode=False):
    """
    Executes full end-to-end training pipeline with 5-Fold Stratified CV,
    balanced sample weighting, post-processed threshold optimization, and artifact persistence.
    """
    print("=" * 60)
    print("🚀 Starting Optimized Health Condition Prediction Pipeline")
    print("=" * 60)

    # 1. Load Data
    df = load_data(data_path)
    print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop ID column if present
    X = df.drop(columns=[col for col in ["id", TARGET_COL] if col in df.columns])
    y_raw = df[TARGET_COL]

    # Target Encoding
    label_encoder = LabelEncoder()
    # Explicitly fit on standard class names to maintain consistent label order
    label_encoder.fit(CLASS_NAMES)
    y = label_encoder.transform(y_raw)

    num_classes = len(CLASS_NAMES)
    print(f"🎯 Target Distribution:\n{pd.Series(y_raw).value_counts(normalize=True).round(4)}")

    # 2. Fit Preprocessing Pipeline
    preprocessor = get_preprocessor()
    print("🔄 Fitting preprocessing and domain feature engineering...")
    X_transformed = preprocessor.fit_transform(X)
    feature_names = get_feature_names()
    print(f"✨ Feature matrix ready: {X_transformed.shape[1]} features")

    # 3. Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_xgb = np.zeros((len(df), num_classes))
    oof_cat = np.zeros((len(df), num_classes))
    oof_lgb = np.zeros((len(df), num_classes))

    xgb_models = []
    cat_models = []
    lgb_models = []

    xgb_fold_scores = []
    cat_fold_scores = []
    lgb_fold_scores = []

    n_estimators = 100 if fast_mode else 300

    print(f"\n📈 Running {n_splits}-Fold Stratified Cross-Validation with Balanced Weighting...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_transformed, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_train, y_train = X_transformed[train_idx], y[train_idx]
        X_val, y_val = X_transformed[val_idx], y[val_idx]

        # Calculate balanced sample weights to penalize minority class errors
        sample_weights_train = compute_sample_weight("balanced", y_train)

        # A. XGBoost
        xgb = get_xgb_model(num_classes=num_classes, n_estimators=n_estimators)
        xgb.fit(X_train, y_train, sample_weight=sample_weights_train)
        xgb_val_prob = xgb.predict_proba(X_val)
        oof_xgb[val_idx] = xgb_val_prob
        xgb_score = balanced_accuracy_score(y_val, np.argmax(xgb_val_prob, axis=1))
        xgb_fold_scores.append(xgb_score)
        xgb_models.append(xgb)
        print(f"  ⚡ XGBoost Fold {fold} Balanced Accuracy: {xgb_score:.4f}")

        # B. CatBoost
        cat = get_catboost_model(num_classes=num_classes, iterations=n_estimators)
        cat.fit(X_train, y_train, sample_weight=sample_weights_train)
        cat_val_prob = cat.predict_proba(X_val)
        oof_cat[val_idx] = cat_val_prob
        cat_score = balanced_accuracy_score(y_val, np.argmax(cat_val_prob, axis=1))
        cat_fold_scores.append(cat_score)
        cat_models.append(cat)
        print(f"  🐱 CatBoost Fold {fold} Balanced Accuracy: {cat_score:.4f}")

        # C. LightGBM
        lgb = get_lgbm_model(num_classes=num_classes, n_estimators=n_estimators)
        lgb.fit(X_train, y_train, sample_weight=sample_weights_train)
        lgb_val_prob = lgb.predict_proba(X_val)
        oof_lgb[val_idx] = lgb_val_prob
        lgb_score = balanced_accuracy_score(y_val, np.argmax(lgb_val_prob, axis=1))
        lgb_fold_scores.append(lgb_score)
        lgb_models.append(lgb)
        print(f"  💡 LightGBM Fold {fold} Balanced Accuracy: {lgb_score:.4f}")

    # 4. Out-of-Fold Ensemble & Threshold Multiplier Optimization
    oof_ensemble_probs = (oof_xgb + oof_cat + oof_lgb) / 3.0
    
    print("\n🔍 Optimizing multi-class probability threshold multipliers on OOF predictions...")
    optimal_multipliers = optimize_threshold_multipliers(oof_ensemble_probs, y)
    print(f"✨ Optimal Threshold Multipliers: {np.round(optimal_multipliers, 4)}")

    oof_weighted_probs = oof_ensemble_probs * np.array(optimal_multipliers)
    oof_ensemble_preds = np.argmax(oof_weighted_probs, axis=1)

    print("\n" + "=" * 60)
    print("🏆 Optimized Out-of-Fold Validation Summary")
    print("=" * 60)
    print(f"XGBoost Mean CV Balanced Accuracy:  {np.mean(xgb_fold_scores):.4f} (± {np.std(xgb_fold_scores):.4f})")
    print(f"CatBoost Mean CV Balanced Accuracy: {np.mean(cat_fold_scores):.4f} (± {np.std(cat_fold_scores):.4f})")
    print(f"LightGBM Mean CV Balanced Accuracy: {np.mean(lgb_fold_scores):.4f} (± {np.std(lgb_fold_scores):.4f})")

    ensemble_metrics = evaluate_predictions(y, oof_ensemble_preds, oof_ensemble_probs, CLASS_NAMES)
    print(f"\n🌟 OPTIMIZED SOFT-VOTING ENSEMBLE BALANCED ACCURACY: {ensemble_metrics['balanced_accuracy']:.4f}")
    print(f"   Accuracy: {ensemble_metrics['accuracy']:.4f} | Macro F1: {ensemble_metrics['macro_f1']:.4f}")

    # 5. Extract Feature Importance
    avg_importances = np.mean([m.feature_importances_ for m in xgb_models], axis=0)
    importance_dict = {fname: float(score) for fname, score in zip(feature_names, avg_importances)}

    # 6. Save Reports & Visualizations
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    fi_path = REPORTS_DIR / "feature_importance.png"
    plot_confusion_matrix(np.array(ensemble_metrics["confusion_matrix"]), CLASS_NAMES, save_path=cm_path)
    plot_feature_importance(importance_dict, top_n=12, save_path=fi_path)

    # 7. Package and Serialize Complete Pipeline
    all_ensemble_models = xgb_models + cat_models + lgb_models
    ensemble = SoftVotingEnsemble(models=all_ensemble_models, class_multipliers=optimal_multipliers)

    pipeline_bundle = {
        "preprocessor": preprocessor,
        "ensemble": ensemble,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "class_multipliers": optimal_multipliers,
    }

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(feature_names),
        "metrics": {
            "xgb_mean_cv": float(np.mean(xgb_fold_scores)),
            "cat_mean_cv": float(np.mean(cat_fold_scores)),
            "lgb_mean_cv": float(np.mean(lgb_fold_scores)),
            "ensemble_oof_balanced_accuracy": float(ensemble_metrics["balanced_accuracy"]),
            "ensemble_oof_accuracy": float(ensemble_metrics["accuracy"]),
            "ensemble_oof_macro_f1": float(ensemble_metrics["macro_f1"]),
        },
        "feature_importances": importance_dict,
    }

    if save_model:
        print(f"\n💾 Saving model bundle to {MODEL_SAVE_PATH}...")
        joblib.dump(pipeline_bundle, MODEL_SAVE_PATH)
        with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print("✅ Pipeline and metadata successfully serialized!")

    return pipeline_bundle, metadata
