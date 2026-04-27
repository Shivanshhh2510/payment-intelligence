"""
src/fraud_engine/drift_monitor.py
───────────────────────────────────
Temporal drift monitoring — three drift types:
  1. Covariate drift    — input feature distributions shifted (PSI)
  2. Prior probability shift — fraud rate changed over time
  3. Concept drift      — model performance degrading over time

PSI (Population Stability Index) industry thresholds:
  PSI < 0.10  → No significant drift
  PSI 0.10–0.20 → Moderate drift, monitor closely
  PSI > 0.20  → Significant drift, retrain recommended

Data is split into time-ordered batches from the test set to simulate
real temporal monitoring — not random sampling.
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
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.fraud_engine.calibrate import IsotonicCalibrator

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "mlflow_artifacts"
PLOTS_DIR     = os.path.join(PROCESSED_DIR, "drift_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

N_BATCHES      = 10     # split test set into 10 time-ordered batches
PSI_WARN       = 0.10
PSI_RETRAIN    = 0.20
TOP_FEATURES   = 20     # monitor top N features by importance


def compute_psi(expected: np.ndarray,
                actual: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.
    Uses quantile binning on expected (training) distribution.
    """
    # Build bins from expected
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Avoid zeros
    expected_pct = np.where(expected_counts == 0, 1e-4,
                            expected_counts / len(expected))
    actual_pct   = np.where(actual_counts   == 0, 1e-4,
                            actual_counts   / len(actual))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def get_top_features(n: int = TOP_FEATURES) -> list:
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.json")
    fi = pd.read_json(fi_path, typ="series").sort_values(ascending=False)
    return fi.head(n).index.tolist()


def monitor_covariate_drift(X_train: pd.DataFrame,
                             X_test_batches: list[pd.DataFrame],
                             top_features: list) -> pd.DataFrame:
    """
    Compute PSI for each top feature across each time batch.
    Returns DataFrame: rows=features, cols=batches.
    """
    results = {}
    for feat in top_features:
        if feat not in X_train.columns:
            continue
        train_vals = X_train[feat].dropna().values
        batch_psis = []
        for batch in X_test_batches:
            if feat not in batch.columns:
                batch_psis.append(0.0)
                continue
            batch_vals = batch[feat].dropna().values
            psi = compute_psi(train_vals, batch_vals)
            batch_psis.append(round(psi, 4))
        results[feat] = batch_psis

    df = pd.DataFrame(results,
                      index=[f"Batch_{i+1}" for i in range(len(X_test_batches))])
    return df.T   # features as rows, batches as columns


def monitor_prior_shift(y_train: pd.Series,
                         y_test_batches: list[pd.Series]) -> pd.Series:
    """Track fraud rate per batch vs training baseline."""
    baseline = y_train.mean()
    rates = [b.mean() for b in y_test_batches]
    print(f"\n── Prior Probability Shift ───────────────────────────────────")
    print(f"  Training fraud rate: {baseline*100:.2f}%")
    for i, r in enumerate(rates):
        shift = (r - baseline) / baseline * 100
        print(f"  Batch {i+1:2d}: {r*100:.2f}%  (shift: {shift:+.1f}%)")
    return pd.Series(rates, index=[f"Batch_{i+1}" for i in range(len(rates))])


def monitor_concept_drift(model,
                           calibrator,
                           X_test_batches: list[pd.DataFrame],
                           y_test_batches: list[pd.Series],
                           threshold: float) -> pd.DataFrame:
    """Track AUC, F1, Precision, Recall per batch."""
    records = []
    for i, (X_b, y_b) in enumerate(zip(X_test_batches, y_test_batches)):
        if y_b.sum() < 5:
            continue
        probs_raw = model.predict_proba(X_b)[:, 1]
        probs_cal = calibrator.transform(probs_raw)
        preds     = (probs_cal >= threshold).astype(int)
        records.append({
            "batch":     f"Batch_{i+1}",
            "auc":       round(roc_auc_score(y_b, probs_cal), 4),
            "f1":        round(f1_score(y_b, preds, pos_label=1, zero_division=0), 4),
            "precision": round(precision_score(y_b, preds, pos_label=1, zero_division=0), 4),
            "recall":    round(recall_score(y_b, preds, pos_label=1), 4),
            "fraud_rate": round(y_b.mean(), 4),
        })

    df = pd.DataFrame(records).set_index("batch")
    print(f"\n── Concept Drift (Model Performance per Batch) ───────────────")
    print(df.to_string())
    return df


def generate_retraining_recommendation(psi_df: pd.DataFrame,
                                        concept_df: pd.DataFrame,
                                        prior_series: pd.Series) -> dict:
    """
    Rule-based retraining recommendation engine.
    Fires when: PSI > 0.20 on 3+ features, OR AUC drops > 0.05, OR fraud rate shifts > 30%.
    """
    last_batch = psi_df.columns[-1]
    high_psi_features = psi_df[last_batch][psi_df[last_batch] > PSI_RETRAIN].index.tolist()

    auc_drop = 0.0
    if len(concept_df) >= 2:
        auc_drop = float(concept_df["auc"].iloc[0] - concept_df["auc"].iloc[-1])

    baseline_fraud_rate = prior_series.iloc[0]
    latest_fraud_rate   = prior_series.iloc[-1]
    fraud_rate_shift    = abs(latest_fraud_rate - baseline_fraud_rate) / baseline_fraud_rate

    should_retrain = (
        len(high_psi_features) >= 3 or
        auc_drop > 0.05 or
        fraud_rate_shift > 0.30
    )

    recommendation = {
        "should_retrain":       should_retrain,
        "high_psi_features":    high_psi_features,
        "n_high_psi_features":  len(high_psi_features),
        "auc_degradation":      round(auc_drop, 4),
        "fraud_rate_shift_pct": round(fraud_rate_shift * 100, 1),
        "recommendation":       "RETRAIN" if should_retrain else "MONITOR",
        "reason": (
            f"PSI > 0.20 on {len(high_psi_features)} features: {high_psi_features[:5]}. "
            f"AUC degradation: {auc_drop:.4f}. "
            f"Fraud rate shift: {fraud_rate_shift*100:.1f}%."
        ) if should_retrain else (
            f"No significant drift detected. "
            f"Max PSI: {psi_df[last_batch].max():.3f}. "
            f"AUC stable at {concept_df['auc'].iloc[-1]:.4f}."
        )
    }

    print(f"\n── Retraining Recommendation ─────────────────────────────────")
    print(f"  Status:     {recommendation['recommendation']}")
    print(f"  Reason:     {recommendation['reason']}")

    return recommendation


def plot_psi_heatmap(psi_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    mask_data = psi_df.values

    # Custom colormap: green → yellow → red
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(mask_data, cmap=cmap, aspect="auto", vmin=0, vmax=0.3)

    ax.set_xticks(range(len(psi_df.columns)))
    ax.set_xticklabels(psi_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(psi_df.index)))
    ax.set_yticklabels(psi_df.index, fontsize=8)
    ax.set_title("Covariate Drift Heatmap (PSI per Feature per Time Batch)",
                 fontweight="bold")

    plt.colorbar(im, ax=ax, label="PSI Value")

    # Annotate threshold lines conceptually via cell color
    for i in range(len(psi_df.index)):
        for j in range(len(psi_df.columns)):
            val = mask_data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if val > 0.15 else "black")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "psi_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_concept_drift(concept_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["auc", "f1", "precision", "recall"]
    colors  = ["steelblue", "crimson", "seagreen", "darkorange"]
    titles  = ["AUC-ROC", "F1 Score", "Precision", "Recall"]

    for ax, metric, color, title in zip(axes.flat, metrics, colors, titles):
        ax.plot(range(len(concept_df)), concept_df[metric],
                "o-", color=color, linewidth=2, markersize=6)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time Batch")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(range(len(concept_df)))
        ax.set_xticklabels(concept_df.index, rotation=45, fontsize=7)
        ax.axhline(concept_df[metric].iloc[0], color="gray",
                   linestyle="--", linewidth=1, label="Batch 1 baseline")
        ax.legend(fontsize=7)

    plt.suptitle("Concept Drift — Model Performance Across Time Batches",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "concept_drift.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_prior_shift(prior_series: pd.Series, y_train_rate: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(prior_series)), prior_series.values * 100,
            "o-", color="crimson", linewidth=2, markersize=7, label="Batch fraud rate")
    ax.axhline(y_train_rate * 100, color="steelblue", linestyle="--",
               linewidth=2, label=f"Training baseline ({y_train_rate*100:.2f}%)")
    ax.fill_between(range(len(prior_series)),
                    (y_train_rate * 1.3) * 100,
                    (y_train_rate * 0.7) * 100,
                    alpha=0.1, color="orange", label="±30% warning band")
    ax.set_xlabel("Time Batch")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xticks(range(len(prior_series)))
    ax.set_xticklabels(prior_series.index, rotation=45, fontsize=8)
    ax.set_title("Prior Probability Shift — Fraud Rate Over Time", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "prior_shift.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


if __name__ == "__main__":
    print("Loading data and artifacts …")
    X_train  = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test   = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train  = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet"))["isFraud"]
    y_test   = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"))["isFraud"]
    model      = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.joblib"))
    calibrator = joblib.load(os.path.join(ARTIFACTS_DIR, "calibrator.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "threshold.json")) as f:
        threshold = json.load(f)["threshold"]

    # Split test into N time-ordered batches
    batch_size = len(X_test) // N_BATCHES
    X_batches  = [X_test.iloc[i*batch_size:(i+1)*batch_size] for i in range(N_BATCHES)]
    y_batches  = [y_test.iloc[i*batch_size:(i+1)*batch_size] for i in range(N_BATCHES)]

    top_features = get_top_features(TOP_FEATURES)

    print(f"\nMonitoring {TOP_FEATURES} top features across {N_BATCHES} time batches …")

    # 1. Covariate drift
    psi_df = monitor_covariate_drift(X_train, X_batches, top_features)

    # 2. Prior shift
    prior_series = monitor_prior_shift(y_train, y_batches)

    # 3. Concept drift
    concept_df = monitor_concept_drift(model, calibrator, X_batches, y_batches, threshold)

    # 4. Retraining recommendation
    recommendation = generate_retraining_recommendation(psi_df, concept_df, prior_series)

    # Save
    psi_df.to_csv(os.path.join(ARTIFACTS_DIR, "psi_report.csv"))
    concept_df.to_csv(os.path.join(ARTIFACTS_DIR, "concept_drift.csv"))
    with open(os.path.join(ARTIFACTS_DIR, "drift_recommendation.json"), "w") as f:
        json.dump(recommendation, f, indent=2)

    # Plots
    print("\nGenerating drift plots …")
    plot_psi_heatmap(psi_df)
    plot_concept_drift(concept_df)
    plot_prior_shift(prior_series, y_train.mean())

    print("\n✓  drift_monitor.py complete. Next: python src/fraud_engine/velocity.py")