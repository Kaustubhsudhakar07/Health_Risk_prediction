# 🏥 Health Condition Prediction using Ensemble Learning

Predicting patient health conditions using an ensemble of **XGBoost, CatBoost, and LightGBM** with a robust machine learning pipeline featuring data preprocessing, feature engineering, stratified cross-validation, and soft voting ensemble.

---

## 📖 Project Overview

This project develops an end-to-end Machine Learning solution for predicting patient health conditions from demographic and lifestyle attributes. The workflow includes comprehensive exploratory data analysis (EDA), missing value handling, feature encoding, model training, and ensemble learning to achieve robust multiclass classification performance.

The solution was built on the **Kaggle Playground Series S6E7** dataset and generates competition-ready predictions.

---

## 🚀 Key Features

- 📊 Comprehensive Exploratory Data Analysis (EDA)
- 🧹 Missing Value Handling using Median & Constant Imputation
- 🔄 Categorical Feature Encoding using Ordinal Encoding
- 📈 5-Fold Stratified Cross Validation
- 🤖 Ensemble Learning using Soft Voting
- 🎯 Competition-Ready Prediction Pipeline
- ⚡ High-performance Gradient Boosting Models

---

## 📊 Dataset Summary

| Metric | Value |
|---------|-------|
| Training Samples | **690,088** |
| Test Samples | **295,753** |
| Features | **13** |
| Target Classes | **3** |
| Numerical Features | **7** |
| Categorical Features | **6** |

### Target Distribution

| Health Condition | Samples |
|-----------------|---------:|
| At-Risk | **592,561** |
| Unhealthy | **57,724** |
| Fit | **39,803** |

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Models | XGBoost, CatBoost, LightGBM |
| Evaluation | Balanced Accuracy, Stratified K-Fold CV |

---

## 📂 Machine Learning Pipeline

```
Dataset
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Data Cleaning
   │
   ▼
Missing Value Imputation
   │
   ▼
Feature Encoding
   │
   ▼
5-Fold Stratified Cross Validation
   │
   ▼
Train Individual Models
 ├── XGBoost
 ├── CatBoost
 └── LightGBM
   │
   ▼
Soft Voting Ensemble
   │
   ▼
Final Predictions
```

---

## 🧠 Models Used

| Model | Description |
|--------|-------------|
| XGBoost | Gradient Boosting Decision Trees |
| CatBoost | Optimized Gradient Boosting with Efficient Categorical Handling |
| LightGBM | Fast Histogram-based Gradient Boosting |
| Ensemble | Soft Voting using Probability Averaging |

---

## 📈 Cross Validation Performance
## 📈 Model Performance

| Model | Mean Balanced Accuracy |
|--------|-----------------------:|
| XGBoost | **0.8789** |
| CatBoost | **0.8698** |
| LightGBM | **0.8769** |

**Final Model:** Soft Voting Ensemble of XGBoost, CatBoost, and LightGBM.

---

## 📦 Repository Structure

```
Health-Condition-Prediction/
│
├── Health_Condition_Prediction.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
├── submission.csv
├── README.md
```

---

## 🌟 Project Highlights

- 📊 Analyzed **690K+** patient records and **295K+** test samples.
- 🧹 Processed **13 predictive features** including lifestyle, physiological, and demographic information.
- 🔄 Handled thousands of missing values using **Median Imputation** for numerical variables and **Constant Imputation** for categorical variables.
- 🏷️ Encoded categorical variables using **Ordinal Encoding**.
- 📈 Evaluated models using **5-Fold Stratified Cross Validation** for robust multiclass classification.
- 🤖 Trained **three gradient boosting models** (XGBoost, CatBoost, LightGBM).
- 🗳️ Combined model probabilities through **Soft Voting Ensemble** to improve prediction robustness.
- 📦 Generated predictions for **295,753** unseen samples for competition submission.

---

## 🔮 Future Improvements

- 🔍 Hyperparameter Optimization using Optuna
- 📈 SHAP Explainability
- 🎯 Advanced Feature Engineering
- 🚀 Streamlit Web Application
- ☁️ Model Deployment with Docker
- ⚙️ Automated ML Pipeline using Scikit-learn Pipelines

---

## 🏆 Key Takeaways

- Built a scalable multiclass classification pipeline capable of handling **690K+ training records**.
- Achieved **0.8789 Balanced Accuracy** using XGBoost with Stratified 5-Fold Cross Validation.
- Improved prediction robustness through a **Soft Voting Ensemble** of XGBoost, CatBoost, and LightGBM.
- Produced a complete, competition-ready machine learning workflow from raw data preprocessing to final submission generation.
