"""
Comprehensive Evaluation Metrics, Confusion Matrix, and Reporting Utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.config import CLASS_NAMES, REPORTS_DIR


def evaluate_predictions(y_true, y_pred, y_prob=None, class_names=CLASS_NAMES):
    """
    Computes a complete suite of multi-class classification metrics.
    """
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    if y_prob is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics["log_loss"] = None

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def plot_confusion_matrix(cm, class_names=CLASS_NAMES, title="Confusion Matrix", save_path=None):
    """
    Renders and optionally saves a formatted Confusion Matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Health Condition", fontsize=11, labelpad=8)
    ax.set_ylabel("True Health Condition", fontsize=11, labelpad=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_importance(importance_dict, top_n=15, title="Top Feature Importances", save_path=None):
    """
    Renders and optionally saves horizontal feature importance bar chart.
    """
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [item[0] for item in reversed(sorted_items)]
    scores = [item[1] for item in reversed(sorted_items)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(features, scores, color="#2563EB", alpha=0.85, edgecolor="#1E40AF")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Score", fontsize=11, labelpad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
