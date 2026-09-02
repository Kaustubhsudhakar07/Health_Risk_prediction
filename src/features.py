"""
Domain-driven Feature Engineering Transformer for Health Condition Prediction.
Derives physiological interactions, metabolic metrics, and lifestyle composite scores.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HealthFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for generating clinical and lifestyle features.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms input DataFrame by adding domain-engineered features.
        """
        df = X.copy()

        # 1. BMI Category (WHO Classification)
        if "bmi" in df.columns:
            bmi = pd.to_numeric(df["bmi"], errors="coerce").fillna(24.0)
            conditions = [
                bmi < 18.5,
                (bmi >= 18.5) & (bmi < 25.0),
                (bmi >= 25.0) & (bmi < 30.0),
                bmi >= 30.0,
            ]
            choices = ["underweight", "normal", "overweight", "obese"]
            df["bmi_category"] = np.select(conditions, choices, default="normal")
        else:
            df["bmi_category"] = "normal"

        # 2. Calorie per Step (Metabolic efficiency)
        if "calorie_expenditure" in df.columns and "step_count" in df.columns:
            cal = pd.to_numeric(df["calorie_expenditure"], errors="coerce").fillna(2000.0)
            steps = pd.to_numeric(df["step_count"], errors="coerce").fillna(5000.0)
            df["calorie_per_step"] = cal / (steps + 100.0)
        else:
            df["calorie_per_step"] = 0.4

        # 3. Active-to-Sleep Ratio (Exertion vs. Rest Balance)
        if "exercise_duration" in df.columns and "sleep_duration" in df.columns:
            exercise_hrs = pd.to_numeric(df["exercise_duration"], errors="coerce").fillna(30.0) / 60.0
            sleep_hrs = pd.to_numeric(df["sleep_duration"], errors="coerce").fillna(7.0)
            df["active_to_sleep_ratio"] = exercise_hrs / (sleep_hrs + 0.1)
        else:
            df["active_to_sleep_ratio"] = 0.1

        # 4. Hydration Index (Water intake relative to metabolic expenditure)
        if "water_intake" in df.columns and "calorie_expenditure" in df.columns:
            water = pd.to_numeric(df["water_intake"], errors="coerce").fillna(2.0)
            cal_k = pd.to_numeric(df["calorie_expenditure"], errors="coerce").fillna(2000.0) / 1000.0
            df["hydration_index"] = water / (cal_k + 0.5)
        else:
            df["hydration_index"] = 0.8

        # 5. Cardio-Metabolic Risk Index (Interaction between Heart Rate and BMI)
        if "heart_rate" in df.columns and "bmi" in df.columns:
            hr = pd.to_numeric(df["heart_rate"], errors="coerce").fillna(72.0)
            bmi = pd.to_numeric(df["bmi"], errors="coerce").fillna(24.0)
            df["cardio_metabolic_risk"] = (hr / 70.0) * (bmi / 22.0)
        else:
            df["cardio_metabolic_risk"] = 1.0

        # 6. Composite Lifestyle Score (Higher = healthier habits)
        lifestyle = np.zeros(len(df), dtype=float)

        stress_numeric = np.zeros(len(df), dtype=float)
        if "stress_level" in df.columns:
            stress_map = {"low": 2.0, "moderate": 1.0, "high": 0.0}
            stress_num_map = {"low": 0.0, "moderate": 1.0, "high": 2.0}
            lifestyle += df["stress_level"].astype(str).str.lower().map(stress_map).fillna(1.0)
            stress_numeric = df["stress_level"].astype(str).str.lower().map(stress_num_map).fillna(1.0)

        if "sleep_quality" in df.columns:
            sleep_map = {"good": 2.0, "average": 1.0, "poor": 0.0}
            lifestyle += df["sleep_quality"].astype(str).str.lower().map(sleep_map).fillna(1.0)

        if "physical_activity_level" in df.columns:
            activity_map = {"active": 2.0, "moderate": 1.0, "sedentary": 0.0}
            lifestyle += df["physical_activity_level"].astype(str).str.lower().map(activity_map).fillna(1.0)

        if "smoking_alcohol" in df.columns:
            smoke_map = {"no": 2.0, "occasional": 1.0, "yes": 0.0}
            lifestyle += df["smoking_alcohol"].astype(str).str.lower().map(smoke_map).fillna(1.0)

        if "diet_type" in df.columns:
            diet_map = {"vegan": 1.5, "veg": 1.0, "non-veg": 0.5}
            lifestyle += df["diet_type"].astype(str).str.lower().map(diet_map).fillna(0.5)

        df["lifestyle_score"] = lifestyle

        # 7. Cardiovascular Strain Ratio (Heart Rate relative to Step Activity)
        if "heart_rate" in df.columns and "step_count" in df.columns:
            hr = pd.to_numeric(df["heart_rate"], errors="coerce").fillna(72.0)
            steps = pd.to_numeric(df["step_count"], errors="coerce").fillna(5000.0)
            df["cardio_strain_ratio"] = hr / ((steps / 1000.0) + 1.0)
        else:
            df["cardio_strain_ratio"] = 12.0

        # 8. Sleep Debt Factor (Deficit under 7.5h scaled by stress)
        if "sleep_duration" in df.columns:
            sleep = pd.to_numeric(df["sleep_duration"], errors="coerce").fillna(7.0)
            sleep_deficit = np.maximum(0.0, 7.5 - sleep)
            df["sleep_debt_factor"] = sleep_deficit * (stress_numeric + 1.0)
        else:
            df["sleep_debt_factor"] = 0.5

        return df
