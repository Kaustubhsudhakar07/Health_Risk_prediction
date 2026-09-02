"""
Command Line Interface (CLI) for Health Risk Prediction Pipeline.
Usage:
    python run_pipeline.py --train
    python run_pipeline.py --predict --file data/sample_patients.csv
    python run_pipeline.py --benchmark
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Configure UTF-8 encoding for Windows stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import SAMPLE_DATA_PATH
from src.data_loader import generate_synthetic_patients
from src.pipeline import train_and_evaluate_pipeline
from src.predict import HealthRiskPredictor


def main():
    parser = argparse.ArgumentParser(description="Health Risk Prediction ML Pipeline")
    parser.add_argument("--train", action="store_true", help="Train 5-fold ensemble model and save artifacts")
    parser.add_argument("--data", type=str, default=None, help="Path to custom training CSV dataset")
    parser.add_argument("--splits", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--fast", action="store_true", help="Run fast training mode (100 estimators)")
    parser.add_argument("--predict", action="store_true", help="Run batch inference on a dataset")
    parser.add_argument("--file", type=str, default=None, help="CSV file for batch inference")
    parser.add_argument("--benchmark", action="store_true", help="Generate fresh synthetic benchmark dataset")

    args = parser.parse_args()

    if args.benchmark:
        print("📊 Generating 3,000 synthetic patient benchmark samples...")
        df = generate_synthetic_patients(n_samples=3000)
        df.to_csv(SAMPLE_DATA_PATH, index=False)
        print(f"✅ Benchmark data saved to: {SAMPLE_DATA_PATH}")
        return

    if args.predict:
        input_path = args.file if args.file else SAMPLE_DATA_PATH
        print(f"🔮 Running inference on: {input_path}")
        df = pd.read_csv(input_path)
        predictor = HealthRiskPredictor()
        results = predictor.predict_batch(df)
        output_path = ROOT / "predictions.csv"
        results.to_csv(output_path, index=False)
        print(f"✅ Predictions generated for {len(results)} patients and saved to: {output_path}")
        print("\nFirst 5 Predictions:")
        print(results[["id", "predicted_health_condition", "prob_at_risk", "prob_fit", "prob_unhealthy"]].head())
        return

    # Default action: Train and evaluate
    print("🚀 Running full pipeline training & validation...")
    train_and_evaluate_pipeline(
        data_path=args.data,
        n_splits=args.splits,
        save_model=True,
        fast_mode=args.fast,
    )

    # Run quick single-patient verification
    print("\n🔬 Testing single-patient inference:")
    predictor = HealthRiskPredictor()
    sample_patient = {
        "sleep_duration": 5.2,
        "heart_rate": 84.0,
        "bmi": 29.8,
        "calorie_expenditure": 2100.0,
        "step_count": 3500.0,
        "exercise_duration": 15.0,
        "water_intake": 1.4,
        "diet_type": "non-veg",
        "stress_level": "high",
        "sleep_quality": "poor",
        "physical_activity_level": "sedentary",
        "smoking_alcohol": "yes",
        "gender": "male",
    }
    result = predictor.predict_single(sample_patient)
    print(f"  Predicted Condition: {result['predicted_condition'].upper()} (Confidence: {result['confidence']*100:.1f}%)")
    print(f"  Probabilities: {result['probabilities']}")
    print("  Actionable Tips:")
    for tip in result["recommendations"]:
        print(f"    - {tip}")


if __name__ == "__main__":
    main()
