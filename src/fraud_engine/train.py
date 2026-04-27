"""
src/fraud_engine/train.py
──────────────────────────
XGBoost fraud classifier with:
  - SMOTE for class imbalance
  - MLflow experiment tracking (every run logged)
  - Threshold-tuned predictions (maximise F1 on fraud class)
  - Saves model artifact + feature importance JSON

Usage:
    python src/fraud_engine/train.py
    mlflow ui  ← then open localhost:5000 to see all runs
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, classification_report, confusion_matrix
)
import joblib

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "mlflow_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MLFLOW_EXPERIMENT = "fraud-detection-ieee-cis"

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
# These are production-grade starting params — defensible in any interview.
# scale_pos_weight is an alternative to SMOTE (we'll use both and log results).
XGBOOST_PARAMS = {
    "n_estimators":     1000,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,    # regularisation — prevents overfitting on fraud class
    "gamma":            1,     # min loss reduction to make split
    "reg_alpha":        0.1,   # L1
    "reg_lambda":       1.0,   # L2
    "tree_method":      "hist",  # GPU-accelerated when device='cuda'
    "device":           "cpu",   # change to "cuda" if you have GPU
    "eval_metric":      "auc",
    "early_stopping_rounds": 50,
    "random_state":     42,
    "verbosity":        0,
}


def load_data() -> tuple:
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet"))["isFraud"]
    X_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_val.parquet"))
    y_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_val.parquet"))["isFraud"]
    X_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"))["isFraud"]
    print(f"Train: {X_train.shape} | fraud: {y_train.mean()*100:.2f}%")
    print(f"Val:   {X_val.shape}   | fraud: {y_val.mean()*100:.2f}%")
    print(f"Test:  {X_test.shape}  | fraud: {y_test.mean()*100:.2f}%")
    return X_train, y_train, X_val, y_val, X_test, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    SMOTE on training fold only (NEVER on the full dataset before splitting —
    that would cause data leakage from synthetic samples into validation).
    k_neighbors=5 is standard; reduce to 3 if fraud class < 100 samples.
    """
    print(f"  Before SMOTE — fraud: {y_train.sum():,}  legit: {(y_train==0).sum():,}")
    sm = SMOTE(k_neighbors=5, random_state=random_state, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After  SMOTE — fraud: {y_res.sum():,}  legit: {(y_res==0).sum():,}")
    return X_res, y_res


def find_optimal_threshold(model: xgb.XGBClassifier,
                           X_val: pd.DataFrame,
                           y_val: pd.Series) -> float:
    """
    Sweep thresholds 0.1→0.9, pick the one maximising F1 on fraud class.
    Default 0.5 is almost always wrong on imbalanced data.
    """
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  Optimal threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return best_t


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("\n" + "═"*65)
    print("  TRAINING — SMOTE on train, early stopping on chronological val")
    print("═"*65)

    with mlflow.start_run(run_name="xgboost-chronological-smote") as run:
        mlflow.log_params({k: v for k, v in XGBOOST_PARAMS.items()
                           if k not in ("eval_metric", "early_stopping_rounds")})
        mlflow.log_param("split", "chronological_70_15_15")
        mlflow.log_param("smote_k_neighbors", 5)

        # SMOTE on train only
        X_tr_res, y_tr_res = apply_smote(X_train, y_train)

        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        # Val metrics
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc   = roc_auc_score(y_val, val_probs)
        val_ap    = average_precision_score(y_val, val_probs)
        val_thr   = find_optimal_threshold(model, X_val, y_val)
        val_preds = (val_probs >= val_thr).astype(int)
        val_f1    = f1_score(y_val, val_preds, pos_label=1)
        val_prec  = precision_score(y_val, val_preds, pos_label=1, zero_division=0)
        val_rec   = recall_score(y_val, val_preds, pos_label=1)

        print(f"\n── Val Results ───────────────────────────────────────────────")
        print(f"  AUC-ROC: {val_auc:.4f} | AP: {val_ap:.4f} | F1: {val_f1:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f}")

        # Test metrics — final holdout, touched only once
        test_probs = model.predict_proba(X_test)[:, 1]
        test_auc   = roc_auc_score(y_test, test_probs)
        test_ap    = average_precision_score(y_test, test_probs)
        test_preds = (test_probs >= val_thr).astype(int)
        test_f1    = f1_score(y_test, test_preds, pos_label=1)
        test_prec  = precision_score(y_test, test_preds, pos_label=1, zero_division=0)
        test_rec   = recall_score(y_test, test_preds, pos_label=1)

        print(f"\n── Test Results (final holdout) ──────────────────────────────")
        print(f"  AUC-ROC: {test_auc:.4f} | AP: {test_ap:.4f} | F1: {test_f1:.4f} | P: {test_prec:.4f} | R: {test_rec:.4f}")
        print(classification_report(y_test, test_preds, target_names=["Legit","Fraud"]))

        mlflow.log_metrics({
            "val_auc": val_auc, "val_f1": val_f1, "val_ap": val_ap,
            "val_precision": val_prec, "val_recall": val_rec,
            "test_auc": test_auc, "test_f1": test_f1, "test_ap": test_ap,
            "test_precision": test_prec, "test_recall": test_rec,
            "best_threshold": val_thr,
        })

        # Save artifacts
        model_path = os.path.join(ARTIFACTS_DIR, "fraud_model.joblib")
        joblib.dump(model, model_path)

        thresh_path = os.path.join(ARTIFACTS_DIR, "threshold.json")
        with open(thresh_path, "w") as f:
            json.dump({"threshold": val_thr}, f)

        metrics_path = os.path.join(ARTIFACTS_DIR, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "val_auc": round(val_auc, 4), "val_f1": round(val_f1, 4),
                "test_auc": round(test_auc, 4), "test_f1": round(test_f1, 4),
                "test_precision": round(test_prec, 4), "test_recall": round(test_rec, 4),
                "test_ap": round(test_ap, 4), "threshold": val_thr,
            }, f)

        fi = pd.Series(model.feature_importances_, index=X_train.columns)
        fi.sort_values(ascending=False).head(30).to_json(
            os.path.join(ARTIFACTS_DIR, "feature_importance.json"))

        with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
            json.dump(X_train.columns.tolist(), f)

        mlflow.xgboost.log_model(model, "xgboost_model")
        for path in [model_path, thresh_path, metrics_path]:
            mlflow.log_artifact(path)

        print(f"\n✓  Run ID: {run.info.run_id}")
        print(f"✓  train.py complete. Next: python src/fraud_engine/explain.py")


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)