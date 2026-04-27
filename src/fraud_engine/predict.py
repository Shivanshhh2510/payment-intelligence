"""
src/fraud_engine/predict.py
────────────────────────────
Inference helpers for Streamlit and batch scoring.
Loads model + explainer once, exposes clean predict() and explain() functions.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "mlflow_artifacts"


class FraudPredictor:
    """
    Wraps the trained XGBoost model + SHAP explainer.
    Designed for low-latency single-transaction scoring.
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        model_path     = os.path.join(artifacts_dir, "fraud_model.joblib")
        explainer_path = os.path.join(artifacts_dir, "shap_explainer.joblib")
        threshold_path = os.path.join(artifacts_dir, "threshold.json")
        features_path  = os.path.join(artifacts_dir, "feature_names.json")

        self.model     = joblib.load(model_path)
        self.explainer = joblib.load(explainer_path) if os.path.exists(explainer_path) else None

        with open(threshold_path) as f:
            self.threshold = json.load(f)["threshold"]

        with open(features_path) as f:
            self.feature_names = json.load(f)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability for each row."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary label using tuned threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def score_transaction(self, X_row: pd.DataFrame) -> dict:
        """
        Score a single transaction row.
        Returns dict with probability, label, confidence, risk_level.
        """
        prob  = float(self.predict_proba(X_row)[0])
        label = int(prob >= self.threshold)

        if prob < 0.3:
            risk = "LOW"
        elif prob < 0.6:
            risk = "MEDIUM"
        elif prob < 0.8:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        return {
            "probability":  round(prob, 4),
            "label":        label,
            "label_str":    "FRAUD" if label else "LEGITIMATE",
            "risk_level":   risk,
            "threshold":    self.threshold,
        }

    def explain_transaction(self, X_row: pd.DataFrame) -> plt.Figure:
        """Return SHAP waterfall figure for a single transaction."""
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not available. Run explain.py first.")
        sv = self.explainer(X_row)
        fig, _ = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv[0], show=False)
        plt.tight_layout()
        return fig

    def top_shap_features(self, X_row: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Return top N features driving this prediction (signed SHAP values)."""
        if self.explainer is None:
            return pd.DataFrame()
        sv = self.explainer(X_row)
        vals   = sv.values[0]
        feats  = X_row.columns.tolist()
        df = pd.DataFrame({
            "feature":    feats,
            "shap_value": vals,
            "abs_shap":   np.abs(vals),
            "feature_val": X_row.values[0],
        }).sort_values("abs_shap", ascending=False).head(n)
        df["direction"] = df["shap_value"].apply(lambda x: "↑ increases fraud" if x > 0 else "↓ decreases fraud")
        return df[["feature", "feature_val", "shap_value", "direction"]]