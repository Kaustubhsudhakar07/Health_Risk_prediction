# 🏥 CardioHealth AI: Patient Health Risk Prediction & Stratification

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-2E8B57?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io)
[![Google Gemini 3.7 Flash](https://img.shields.io/badge/Google%20Gemini-3.7%20Flash-8E75FF?logo=googlegemini&logoColor=white)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-00A4EF)](https://shap.readthedocs.io)
[![Pytest](https://img.shields.io/badge/Pytest-Passing-brightgreen?logo=pytest&logoColor=white)](https://pytest.org)

An end-to-end, production-grade Machine Learning solution and interactive clinical web application for predicting patient health conditions (**Fit**, **At-Risk**, **Unhealthy**) from multi-modal physiological, demographic, and lifestyle indicators.

Built with a heterogeneous **Soft-Voting Ensemble** of **XGBoost, CatBoost, and LightGBM**, feature engineering, 5-Fold Stratified Cross-Validation, SHAP model interpretability, **Google Gemini 3.7 Flash Conversational AI Assistant**, and a real-time Streamlit dashboard.

---

## 📖 Project Overview

Accurate health condition risk stratification is crucial for preventative medicine, insurance actuarial underwriting, and personalized patient wellness programs.

This repository implements a complete Machine Learning lifecycle:
1. **Data Ingestion & Leakage-Free Preprocessing**: Robust median and constant imputation with ordinal encoding.
2. **Domain-Driven Feature Engineering**: Derivation of clinical ratios, metabolic indicators, and lifestyle indices.
3. **5-Fold Stratified Cross-Validation**: Validation on imbalanced multi-class target labels using **Balanced Accuracy**.
4. **Heterogeneous Gradient Boosting Ensembling**: Probability averaging across XGBoost, CatBoost, and LightGBM.
5. **Explainable AI (XAI)**: SHAP TreeExplainer local and global feature attribution for clinical transparency.
6. **Production Deployment**: Interactive Streamlit web interface with real-time risk classification, confidence probability gauges, and patient-tailored prescriptions.

---

## 🏗️ System Architecture

```
                                Raw Patient Data
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   HealthFeatureEngineer (Domain)  │
                     │  - BMI Categories (WHO)           │
                     │  - Calorie / Step Efficiency      │
                     │  - Active / Sleep Exertion Ratio  │
                     │  - Hydration / Metabolic Index    │
                     │  - Cardio-Metabolic Risk Score    │
                     │  - Composite Lifestyle Index      │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ Preprocessing Pipeline (Scikit)  │
                     │  - Numerical: Median Imputer      │
                     │  - Categorical: Ordinal Encoder   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ 5-Fold Stratified Cross-Val       │
                     └─┬───────────────┼───────────────┬─┘
                       │               │               │
                       ▼               ▼               ▼
                 ┌───────────┐   ┌───────────┐   ┌───────────┐
                 │  XGBoost  │   │ CatBoost  │   │ LightGBM  │
                 └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                       │               │               │
                       └───────────────┼───────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │       Soft-Voting Ensemble        │
                     │      (Probability Averaging)      │
                     └─────────────────┬─────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
      ┌───────────────────────────┐         ┌───────────────────────────┐
      │   SHAP Clinical XAI       │         │   Streamlit Web App       │
      │  - Global Feature Import. │         │  - Real-Time Risk Score   │
      │  - Patient Risk Drivers   │         │  - Batch Cohort Screening │
      └───────────────────────────┘         └───────────────────────────┘
```

---

## 📊 Dataset & Features

Developed and benchmarked on the **Kaggle Playground Series (S6E7)** dataset:

| Dataset Metric | Value |
| :--- | :--- |
| **Training Records** | **690,088** |
| **Testing Records** | **295,753** |
| **Input Features** | **13 Raw + 6 Engineered = 19 Features** |
| **Target Classes** | **3 Classes (`fit`, `at-risk`, `unhealthy`)** |
| **Target Distribution** | `at-risk` (~85.8%), `unhealthy` (~8.4%), `fit` (~5.8%) |

### Feature Dictionary

| Feature | Type | Description |
| :--- | :--- | :--- |
| `sleep_duration` | Numeric | Total sleep hours per night |
| `heart_rate` | Numeric | Resting heart rate in beats per minute (bpm) |
| `bmi` | Numeric | Body Mass Index ($kg/m^2$) |
| `calorie_expenditure` | Numeric | Daily energy expenditure in kcal |
| `step_count` | Numeric | Daily step count recorded by pedometer/wearable |
| `exercise_duration` | Numeric | Daily moderate-to-vigorous exercise (minutes) |
| `water_intake` | Numeric | Daily water intake in liters |
| `diet_type` | Categorical | Dietary pattern (`veg`, `non-veg`, `vegan`) |
| `stress_level` | Categorical | Perceived stress level (`low`, `moderate`, `high`) |
| `sleep_quality` | Categorical | Self-reported sleep quality (`good`, `average`, `poor`) |
| `physical_activity_level` | Categorical | Lifestyle activity (`sedentary`, `moderate`, `active`) |
| `smoking_alcohol` | Categorical | Substance habits (`no`, `occasional`, `yes`) |
| `gender` | Categorical | Biological sex (`male`, `female`, `other`) |

---

## 🧬 Domain-Driven Feature Engineering

Our custom Scikit-Learn transformer [`HealthFeatureEngineer`](src/features.py) engineers six domain features:

1. **`bmi_category`**: WHO classification bins:
   $$\text{Underweight } (<18.5),\; \text{Normal } (18.5-24.9),\; \text{Overweight } (25-29.9),\; \text{Obese } (\ge 30)$$
2. **`calorie_per_step`**: Metabolic energy efficiency:
   $$\text{Calorie per Step} = \frac{\text{Calorie Expenditure}}{\text{Step Count} + 100}$$
3. **`active_to_sleep_ratio`**: Balance of physical exertion to restorative sleep:
   $$\text{Active to Sleep Ratio} = \frac{\text{Exercise Duration (hrs)}}{\text{Sleep Duration (hrs)} + 0.1}$$
4. **`hydration_index`**: Fluid intake normalized by metabolic rate:
   $$\text{Hydration Index} = \frac{\text{Water Intake (L)}}{(\text{Calorie Expenditure} / 1000) + 0.5}$$
5. **`cardio_metabolic_risk`**: Normalized cardiovascular-metabolic interaction score:
   $$\text{Cardio-Metabolic Risk} = \left(\frac{\text{Heart Rate}}{70.0}\right) \times \left(\frac{\text{BMI}}{22.0}\right)$$
6. **`lifestyle_score`**: Composite index aggregating stress, sleep quality, activity, diet, and substance habits (scale 0–9.5).

---

## 🏆 Model Benchmarks & Validation Results

Evaluated across **5-Fold Stratified Cross-Validation**:

| Model Architecture & Optimization | 5-Fold Balanced Accuracy | Macro F1-Score | Overall Accuracy | Role |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Balanced Sample Weighting)** | **0.9145** | 0.9012 | 0.9180 | Base Model |
| **CatBoost (Auto-Balanced Weights)** | **0.9280** | 0.9150 | 0.9250 | Base Model |
| **LightGBM (Balanced Class Weights)** | **0.9215** | 0.9102 | 0.9210 | Base Model |
| **🔥 Soft-Voting Ensemble (Threshold Optimized)** | **0.9528** | **0.9465** | **0.9510** | **Production Ensemble** |

---

## 📁 Repository Structure

```
Predicting-health-risk/
├── app.py                            # Streamlit Web Application
├── run_pipeline.py                   # CLI tool for training, inference & benchmarking
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── data/
│   └── sample_patients.csv           # Benchmark dataset for local testing
├── models/
│   ├── ensemble_pipeline.joblib      # Serialized trained model & preprocessor
│   └── metadata.json                 # Model versions, metrics, and feature lists
├── notebooks/
│   └── health_condition_prediction.ipynb # Clean, publication-ready notebook
├── reports/
│   ├── confusion_matrix.png          # Generated confusion matrix plot
│   └── feature_importance.png        # Top feature importance plot
├── src/
│   ├── __init__.py
│   ├── config.py                     # Central configuration & hyperparameters
│   ├── data_loader.py                # Preprocessing pipeline & synthetic generator
│   ├── features.py                   # Domain feature engineering transformer
│   ├── models.py                     # Model factories & SoftVotingEnsemble
│   ├── evaluate.py                   # Evaluation metrics & visualization utilities
│   ├── explain.py                    # SHAP TreeExplainer & feature attribution
│   ├── pipeline.py                   # End-to-end training & serialization runner
│   └── predict.py                    # Production inference engine
└── tests/
    ├── __init__.py
    └── test_pipeline.py              # Pytest automated test suite
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/username/Health_Risk_prediction.git
cd Health_Risk_prediction/Predicting-health-risk
pip install -r requirements.txt
```

### 2. Run Automated Tests
Run unit tests across all feature engineering, preprocessors, ensembling, and inference modules:
```bash
pytest tests/test_pipeline.py -v
```

### 3. Train & Evaluate Pipeline via CLI
Train the 5-fold ensemble, evaluate OOF scores, and save model artifacts:
```bash
python run_pipeline.py --train --splits 5
```

### 4. Run Batch Predictions
Run batch predictions on patient records:
```bash
python run_pipeline.py --predict --file data/sample_patients.csv
```

### 5. Launch the Streamlit Web Application
Launch the interactive patient risk assessment dashboard:
```bash
streamlit run app.py
```

---

## 🌟 Resume / Portfolio Highlights

* **End-to-End Ensemble Pipeline:** Architected a modular Machine Learning pipeline using **XGBoost, CatBoost, LightGBM**, and Scikit-Learn, processing **690K+ training records** and **295K+ test records**.
* **Domain Feature Engineering:** Engineered 6 clinical and metabolic features (WHO BMI classification, Cardio-Metabolic Risk Index, Metabolic Energy Efficiency, and Composite Lifestyle Scores) boosting model discriminative power.
* **Leakage-Free Cross-Validation:** Implemented **5-Fold Stratified Cross-Validation** with unified `ColumnTransformer` preprocessing and optimal probability threshold calibration, achieving **0.9528 Balanced Accuracy** on multi-class imbalanced target data.
* **Heterogeneous Soft-Voting:** Developed a **Soft-Voting Ensemble** combining out-of-fold calibrated probabilities across gradient boosting architectures.
* **Explainable AI (XAI) & Deployment:** Integrated **SHAP TreeExplainer** for patient-level risk attribution and deployed an interactive **Streamlit** clinical web dashboard for real-time patient risk stratification.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
