"""
src/fraud_engine/explain.py
────────────────────────────
SHAP explainability layer for the fraud model.

Produces:
  1. Global: beeswarm plot (top 20 features)
  2. Global: bar chart of mean |SHAP|
  3. Per-prediction: waterfall plot (used by Streamlit dashboard)
  4. Saves explainer as artifact for Streamlit inference

Usage:
    python src/fraud_engine/explain.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "mlflow_artifacts"
PLOTS_DIR     = os.path.join(PROCESSED_DIR, "shap_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

SAMPLE_SIZE = 2000   # SHAP on full 590k is slow — 2k rows gives stable global importance


def load_artifacts() -> tuple:
    model    = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.joblib"))
    X        = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    y        = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet"))["isFraud"]
    with open(os.path.join(ARTIFACTS_DIR, "threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    return model, X, y, threshold


def compute_shap_values(model, X_sample: pd.DataFrame):
    """
    TreeExplainer is exact (not approximate) for tree-based models.
    Returns shap_values array shaped (n_samples, n_features).
    """
    print(f"Computing SHAP values on {len(X_sample):,} samples …")
    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer(X_sample)   # returns Explanation object
    print("  Done.")
    return explainer, shap_values


def plot_beeswarm(shap_values, X_sample: pd.DataFrame, max_display: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm — Top Feature Impacts (Fraud Model)", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "beeswarm.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm saved → {out}")


def plot_bar(shap_values, max_display: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("Mean |SHAP| — Global Feature Importance", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar plot saved → {out}")


def plot_waterfall_single(shap_values, idx: int = 0) -> None:
    """Waterfall for a single prediction — this is what the Streamlit app calls."""
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title(f"SHAP Waterfall — Transaction #{idx}", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"waterfall_sample_{idx}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Waterfall saved → {out}")


def get_shap_waterfall_fig(explainer, X_row: pd.DataFrame) -> plt.Figure:
    """
    Called by Streamlit app for live per-transaction explanation.
    Returns a matplotlib Figure object (rendered by st.pyplot).
    """
    sv = explainer(X_row)
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(sv[0], show=False)
    plt.tight_layout()
    return fig


def save_explainer(explainer) -> None:
    out = os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib")
    joblib.dump(explainer, out)
    print(f"  Explainer saved → {out}")


if __name__ == "__main__":
    model, X, y, threshold = load_artifacts()

    # Sample balanced for SHAP (oversample fraud so waterfall examples exist)
    fraud_idx = y[y == 1].index[:SAMPLE_SIZE // 2]
    legit_idx = y[y == 0].index[:SAMPLE_SIZE // 2]
    sample_idx = fraud_idx.tolist() + legit_idx.tolist()
    X_sample = X.loc[sample_idx].reset_index(drop=True)

    explainer, shap_values = compute_shap_values(model, X_sample)

    print("\nGenerating global plots …")
    plot_beeswarm(shap_values, X_sample)
    plot_bar(shap_values)

    print("\nGenerating waterfall examples …")
    for i in range(3):   # save 3 sample waterfalls
        plot_waterfall_single(shap_values, idx=i)

    save_explainer(explainer)

    print("\n✓  explain.py complete. Next: python src/routing_engine/bandit.py")