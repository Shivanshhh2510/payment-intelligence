"""
src/fraud_engine/calibrate.py
──────────────────────────────
Probability calibration + business impact layer.

Why calibration matters:
  Raw XGBoost scores are not probabilities — a score of 0.7 does NOT mean
  70% chance of fraud. Isotonic regression maps raw scores to true probabilities
  by fitting a non-parametric monotonic function on held-out val data.
  Proven superior to Platt scaling for tree-based models (Niculescu-Mizil 2005).

Business impact layer:
  Translates ML metrics into rupee-denominated ROI.
  Cost parameters sourced from RBI and industry reports:
    - Average chargeback cost (India): ₹2,400
    - Average legitimate transaction value: ₹1,200
    - False positive cost = lost transaction revenue + customer friction
    - False negative cost = full chargeback + operational cost
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, brier_score_loss
)

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "mlflow_artifacts"
PLOTS_DIR     = os.path.join(PROCESSED_DIR, "calibration_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Business cost parameters (RBI-sourced, India e-commerce)
CHARGEBACK_COST     = 2400   # ₹ — cost of missing one fraud (FN)
LOST_REVENUE        = 1200   # ₹ — cost of blocking one legitimate tx (FP)
OPERATIONAL_COST    = 200    # ₹ — manual review cost per flagged transaction


class IsotonicCalibrator:
    """
    Fits isotonic regression on val set scores → val labels.
    Transforms raw XGBoost probabilities into calibrated probabilities.
    """
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        self.iso.fit(probs, y_true)
        self.fitted = True

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted.")
        return self.iso.predict(probs).astype(np.float32)

    def fit_transform(self, probs, y_true):
        self.fit(probs, y_true)
        return self.transform(probs)


def compute_calibration_metrics(probs_raw, probs_cal, y_true):
    """Brier score before/after — lower is better."""
    brier_raw = brier_score_loss(y_true, probs_raw)
    brier_cal = brier_score_loss(y_true, probs_cal)
    print(f"  Brier Score — Raw: {brier_raw:.4f}  Calibrated: {brier_cal:.4f}  "
          f"({'↓ improved' if brier_cal < brier_raw else '↑ degraded'})")
    return brier_raw, brier_cal


def plot_calibration_curve(probs_raw, probs_cal, y_true) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration line
    ax.plot([0,1], [0,1], "k--", linewidth=1, label="Perfect calibration")

    # Raw XGBoost
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, probs_raw, n_bins=20)
    ax.plot(mean_pred_raw, frac_pos_raw, "s-", color="crimson",
            linewidth=2, markersize=5, label="XGBoost (uncalibrated)")

    # Calibrated
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, probs_cal, n_bins=20)
    ax.plot(mean_pred_cal, frac_pos_cal, "s-", color="steelblue",
            linewidth=2, markersize=5, label="XGBoost (isotonic calibrated)")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (True Fraud Rate)")
    ax.set_title("Reliability Diagram — Probability Calibration", fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "reliability_diagram.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_pr_curve(probs_cal, y_true) -> None:
    """
    Precision-Recall curve — more informative than ROC on imbalanced data.
    ROC is optimistic because it uses TN which is huge on 3.5% fraud rate.
    PR curve shows the real precision-recall tradeoff.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs_cal)
    ap = average_precision_score(y_true, probs_cal)

    # F1 at each threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx  = np.argmax(f1_scores)
    best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color="steelblue", linewidth=2,
            label=f"PR Curve (AP={ap:.3f})")
    ax.axhline(y_true.mean(), color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (fraud rate={y_true.mean()*100:.1f}%)")
    ax.scatter(recall[best_idx], precision[best_idx],
               color="crimson", s=100, zorder=5,
               label=f"Best F1={f1_scores[best_idx]:.3f} @ thr={best_thr:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (calibrated probabilities)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pr_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_roc_curve(probs_cal, y_true) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs_cal)
    auc = roc_auc_score(y_true, probs_cal)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (calibrated)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def compute_business_impact(probs_cal, y_true, threshold,
                             chargeback_cost=CHARGEBACK_COST,
                             lost_revenue=LOST_REVENUE,
                             operational_cost=OPERATIONAL_COST) -> dict:
    """
    At given threshold, compute ₹ impact vs no-model baseline.
    """
    preds = (probs_cal >= threshold).astype(int)
    tp = int(((preds==1) & (y_true==1)).sum())   # caught fraud
    fp = int(((preds==1) & (y_true==0)).sum())   # blocked legitimate
    fn = int(((preds==0) & (y_true==1)).sum())   # missed fraud
    tn = int(((preds==0) & (y_true==0)).sum())   # correct legitimate

    # Costs
    cost_fp = fp * lost_revenue
    cost_fn = fn * chargeback_cost
    cost_review = (tp + fp) * operational_cost

    # Baseline: no model — all fraud missed
    baseline_cost = int(y_true.sum()) * chargeback_cost

    # Savings
    fraud_prevented_value = tp * chargeback_cost
    net_benefit = fraud_prevented_value - cost_fp - cost_review

    result = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fraud_caught": tp,
        "fraud_missed": fn,
        "legitimate_blocked": fp,
        "cost_false_positives_inr":    cost_fp,
        "cost_false_negatives_inr":    cost_fn,
        "cost_manual_review_inr":      cost_review,
        "baseline_total_fraud_cost_inr": baseline_cost,
        "fraud_value_prevented_inr":   fraud_prevented_value,
        "net_benefit_inr":             net_benefit,
        "roi_percent":                 round(100 * net_benefit / max(baseline_cost, 1), 1),
        "threshold_used":              threshold,
    }

    print(f"\n── Business Impact (Test Set — {len(y_true):,} transactions) ───")
    print(f"  Fraud caught:              {tp:,}  (₹{fraud_prevented_value:,} protected)")
    print(f"  Fraud missed:              {fn:,}  (₹{cost_fn:,} loss)")
    print(f"  Legitimate blocked (FP):   {fp:,}  (₹{cost_fp:,} lost revenue)")
    print(f"  Manual review cost:        ₹{cost_review:,}")
    print(f"  Baseline (no model) cost:  ₹{baseline_cost:,}")
    print(f"  Net benefit vs baseline:   ₹{net_benefit:,}")
    print(f"  ROI vs no-model baseline:  {result['roi_percent']}%")

    return result


def plot_cost_curve(probs_cal, y_true) -> None:
    """
    Expected cost at every threshold — shows optimal operating point.
    Different from F1-optimal — business optimal depends on cost ratio.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    net_benefits, f1s = [], []

    for t in thresholds:
        preds = (probs_cal >= t).astype(int)
        tp = ((preds==1) & (y_true==1)).sum()
        fp = ((preds==1) & (y_true==0)).sum()
        fn = ((preds==0) & (y_true==1)).sum()
        nb = tp * CHARGEBACK_COST - fp * LOST_REVENUE - (tp+fp) * OPERATIONAL_COST
        net_benefits.append(nb)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        f1s.append(f1)

    best_t_biz  = thresholds[np.argmax(net_benefits)]
    best_t_f1   = thresholds[np.argmax(f1s)]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, net_benefits, color="steelblue", linewidth=2, label="Net Benefit (₹)")
    ax2.plot(thresholds, f1s,          color="crimson",   linewidth=2,
             linestyle="--", label="F1 Score")

    ax1.axvline(best_t_biz, color="steelblue", linestyle=":", linewidth=1.5,
                label=f"Business optimal: {best_t_biz:.2f}")
    ax1.axvline(best_t_f1,  color="crimson",   linestyle=":", linewidth=1.5,
                label=f"F1 optimal: {best_t_f1:.2f}")

    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Net Benefit (₹)", color="steelblue")
    ax2.set_ylabel("F1 Score",        color="crimson")
    ax1.set_title("Business Cost Curve — Optimal Threshold by Objective", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cost_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")

    return best_t_biz


if __name__ == "__main__":
    print("Loading data and model …")
    X_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_val.parquet"))
    y_val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_val.parquet"))["isFraud"]
    X_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"))["isFraud"]
    model   = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "threshold.json")) as f:
        threshold = json.load(f)["threshold"]

    # Raw probabilities
    val_probs_raw  = model.predict_proba(X_val)[:, 1]
    test_probs_raw = model.predict_proba(X_test)[:, 1]

    # Fit calibrator on val set, apply to test set
    print("\nFitting isotonic calibrator on val set …")
    calibrator = IsotonicCalibrator()
    calibrator.fit(val_probs_raw, y_val.values)
    test_probs_cal = calibrator.transform(test_probs_raw)

    print("\n── Calibration Metrics ───────────────────────────────────────")
    brier_raw, brier_cal = compute_calibration_metrics(
        test_probs_raw, test_probs_cal, y_test)

    # Business impact
    impact = compute_business_impact(test_probs_cal, y_test, threshold)

    # Plots
    print("\nGenerating plots …")
    plot_calibration_curve(test_probs_raw, test_probs_cal, y_test)
    plot_pr_curve(test_probs_cal, y_test)
    plot_roc_curve(test_probs_cal, y_test)
    best_biz_threshold = plot_cost_curve(test_probs_cal, y_test)

    # Save calibrator and results
    joblib.dump(calibrator, os.path.join(ARTIFACTS_DIR, "calibrator.joblib"))
    impact["brier_raw"] = round(brier_raw, 4)
    impact["brier_cal"] = round(brier_cal, 4)
    impact["business_optimal_threshold"] = round(float(best_biz_threshold), 2)
    with open(os.path.join(ARTIFACTS_DIR, "business_impact.json"), "w") as f:
        json.dump(impact, f, indent=2)

    print(f"\n  Calibrator saved → {ARTIFACTS_DIR}/calibrator.joblib")
    print(f"  Business impact saved → {ARTIFACTS_DIR}/business_impact.json")
    print("\n✓  calibrate.py complete. Next: python src/fraud_engine/drift_monitor.py")