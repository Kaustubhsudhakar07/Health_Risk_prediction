"""
Production Inference Engine for Single-Patient and Batch Health Condition Prediction.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    METADATA_SAVE_PATH,
    MODEL_SAVE_PATH,
)
from src.explain import explain_patient_prediction


class HealthRiskPredictor:
    """
    Inference service encapsulating preprocessor, soft-voting ensemble,
    and clinical recommendation generation.
    """

    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model_path = Path(model_path)
        self.bundle = None
        self.preprocessor = None
        self.ensemble = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = CLASS_NAMES
        self._load()

    def _load(self):
        """Loads serialized pipeline or triggers training if missing."""
        if not self.model_path.exists():
            print(f"⚠️ Model bundle not found at {self.model_path}. Training default pipeline...")
            from src.pipeline import train_and_evaluate_pipeline
            self.bundle, _ = train_and_evaluate_pipeline(save_model=True, fast_mode=True)
        else:
            self.bundle = joblib.load(self.model_path)

        self.preprocessor = self.bundle["preprocessor"]
        self.ensemble = self.bundle["ensemble"]
        self.label_encoder = self.bundle["label_encoder"]
        self.feature_names = self.bundle["feature_names"]
        self.class_names = self.bundle.get("class_names", CLASS_NAMES)

    def predict_single(self, patient_data: dict):
        """
        Processes a single patient record and returns detailed risk diagnosis.
        """
        df_input = pd.DataFrame([patient_data])
        X_trans = self.preprocessor.transform(df_input)
        probs = self.ensemble.predict_proba(X_trans)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]

        prob_dict = {c: float(p) for c, p in zip(self.class_names, probs)}

        # Explain prediction via SHAP
        try:
            explanations = explain_patient_prediction(
                self.ensemble, X_trans, self.feature_names
            )
        except Exception:
            explanations = []

        # Generate Clinical Actionable Advice
        recommendations = self._generate_recommendations(patient_data, pred_class)

        return {
            "predicted_condition": pred_class,
            "confidence": float(probs[pred_idx]),
            "probabilities": prob_dict,
            "top_risk_drivers": explanations,
            "recommendations": recommendations,
        }

    def predict_batch(self, df_input: pd.DataFrame):
        """
        Processes a batch DataFrame and returns predictions with probability scores.
        """
        df_clean = df_input.copy()
        drop_cols = [c for c in ["id", "health_condition"] if c in df_clean.columns]
        X_eval = df_clean.drop(columns=drop_cols)

        X_trans = self.preprocessor.transform(X_eval)
        probs = self.ensemble.predict_proba(X_trans)
        preds = np.argmax(probs, axis=1)

        result_df = df_input.copy()
        result_df["predicted_health_condition"] = [self.class_names[p] for p in preds]

        for i, c in enumerate(self.class_names):
            result_df[f"prob_{c.replace('-', '_')}"] = probs[:, i].round(4)

        return result_df

    def _generate_recommendations(self, data: dict, pred_class: str):
        """Generates patient-tailored lifestyle & clinical suggestions."""
        tips = []
        bmi = float(data.get("bmi", 24.0))
        hr = float(data.get("heart_rate", 72.0))
        sleep = float(data.get("sleep_duration", 7.0))
        exercise = float(data.get("exercise_duration", 30.0))
        water = float(data.get("water_intake", 2.0))
        stress = str(data.get("stress_level", "low")).lower()
        smoke_alc = str(data.get("smoking_alcohol", "no")).lower()

        if bmi >= 25.0:
            tips.append("⚖️ **Weight Management:** BMI indicates overweight/obesity. Aim for a caloric deficit and structured physical training.")
        if hr > 85:
            tips.append("❤️ **Cardio Monitoring:** Elevated resting heart rate detected. Consider a comprehensive cardiovascular screening.")
        if sleep < 6.5:
            tips.append("😴 **Sleep Optimization:** Sleep duration is below recommended 7-9 hours. Target regular sleep schedules to reduce systemic inflammation.")
        if exercise < 20:
            tips.append("🏃 **Physical Activity:** Increase moderate-to-vigorous aerobic activity to at least 150 minutes per week.")
        if water < 2.0:
            tips.append("💧 **Hydration:** Increase daily water intake to 2.5–3.0 liters to support metabolic clearance.")
        if stress in ["high", "moderate"]:
            tips.append("🧘 **Stress Reduction:** Implement mindfulness, meditation, or breathing exercises to lower cortisol levels.")
        if smoke_alc in ["yes", "occasional"]:
            tips.append("🚭 **Substance Cessation:** Reducing or eliminating tobacco and alcohol consumption will drastically lower multi-system risk.")

        if not tips:
            tips.append("🌟 **Maintain Habits:** Excellent overall markers! Continue balanced nutrition, consistent hydration, and active lifestyle.")

        return tips
