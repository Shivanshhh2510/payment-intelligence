"""
src/fraud_engine/anomaly.py
────────────────────────────
Stage 1 of two-stage fraud detection: Isolation Forest anomaly detector.

Why two stages:
  - XGBoost is supervised — it only catches fraud patterns seen in training
  - Isolation Forest is unsupervised — it flags statistically unusual transactions
    regardless of whether that pattern exists in labeled data
  - Together: XGBoost catches known fraud, IsoForest catches novel/zero-day fraud
  - When both agree → high confidence decision
  - When they disagree → flag for human review

Production parallel: This mirrors how Stripe and Razorpay layer unsupervised
anomaly detection on top of supervised classifiers in their fraud pipelines.
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "mlflow_artifacts"
PLOTS_DIR     = os.path.join(PROCESSED_DIR, "anomaly_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def train_isolation_forest(X_train: pd.DataFrame,
                            contamination: float = 0.035) -> IsolationForest:
    """
    contamination = expected fraud rate in training data (3.5%).
    Trained on ALL training transactions — no labels used.
    n_estimators=200 for stability; max_samples='auto' = min(256, n_samples).
    n_jobs=-1 uses all cores.
    """
    print("Training Isolation Forest …")
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )
    iso.fit(X_train)
    print("  Done.")
    return iso


def get_anomaly_scores(iso: IsolationForest,
                        X: pd.DataFrame) -> np.ndarray:
    """
    decision_function returns negative anomaly scores.
    We negate and normalise to [0,1] so higher = more anomalous.
    """
    raw = iso.decision_function(X)
    # Negate: more negative raw score = more anomalous
    negated = -raw
    # Normalise to [0,1]
    min_s, max_s = negated.min(), negated.max()
    normalised = (negated - min_s) / (max_s - min_s + 1e-9)
    return normalised.astype(np.float32)


def evaluate_isolation_forest(iso: IsolationForest,
                               X_test: pd.DataFrame,
                               y_test: pd.Series) -> dict:
    scores = get_anomaly_scores(iso, X_test)
    auc    = roc_auc_score(y_test, scores)
    ap     = average_precision_score(y_test, scores)
    # IsoForest binary predictions (-1 = anomaly, 1 = normal)
    preds  = (iso.predict(X_test) == -1).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1     = f1_score(y_test, preds, pos_label=1, zero_division=0)
    prec   = precision_score(y_test, preds, pos_label=1, zero_division=0)
    rec    = recall_score(y_test, preds, pos_label=1, zero_division=0)

    print(f"\n── Isolation Forest Test Results ─────────────────────────────")
    print(f"  AUC-ROC: {auc:.4f} | AP: {ap:.4f} | F1: {f1:.4f} | P: {prec:.4f} | R: {rec:.4f}")

    return {"iso_auc": round(auc,4), "iso_ap": round(ap,4),
            "iso_f1": round(f1,4), "iso_precision": round(prec,4),
            "iso_recall": round(rec,4)}


def build_two_stage_signals(iso: IsolationForest,
                             xgb_model,
                             X: pd.DataFrame,
                             threshold: float) -> pd.DataFrame:
    """
    Combine IsoForest anomaly score + XGBoost fraud probability.
    Returns DataFrame with both signals and a combined decision.

    Decision logic:
      CONFIRMED FRAUD    — both models agree (xgb >= threshold AND iso >= 0.5)
      NOVEL ANOMALY      — iso flags but xgb doesn't (potential zero-day fraud)
      KNOWN FRAUD        — xgb flags but iso doesn't (known pattern, low anomaly)
      LEGITIMATE         — neither flags
    """
    iso_scores = get_anomaly_scores(iso, X)
    xgb_probs  = xgb_model.predict_proba(X)[:, 1]
    xgb_preds  = (xgb_probs >= threshold).astype(int)
    iso_preds  = (iso_scores >= 0.5).astype(int)

    conditions = []
    for xp, ip in zip(xgb_preds, iso_preds):
        if xp == 1 and ip == 1:
            conditions.append("CONFIRMED_FRAUD")
        elif xp == 0 and ip == 1:
            conditions.append("NOVEL_ANOMALY")
        elif xp == 1 and ip == 0:
            conditions.append("KNOWN_FRAUD")
        else:
            conditions.append("LEGITIMATE")

    return pd.DataFrame({
        "xgb_prob":       xgb_probs.astype(np.float32),
        "iso_score":      iso_scores,
        "xgb_pred":       xgb_preds,
        "iso_pred":       iso_preds,
        "decision":       conditions,
    })


def plot_score_distributions(signals_df: pd.DataFrame,
                              y_true: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # XGBoost probability distribution
    ax = axes[0]
    ax.hist(signals_df.loc[y_true==0, "xgb_prob"], bins=80,
            alpha=0.6, color="steelblue", label="Legitimate", density=True)
    ax.hist(signals_df.loc[y_true==1, "xgb_prob"], bins=80,
            alpha=0.6, color="crimson",   label="Fraud",      density=True)
    ax.set_title("XGBoost Fraud Probability Distribution", fontweight="bold")
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Density")
    ax.legend()

    # Isolation Forest anomaly score distribution
    ax = axes[1]
    ax.hist(signals_df.loc[y_true==0, "iso_score"], bins=80,
            alpha=0.6, color="steelblue", label="Legitimate", density=True)
    ax.hist(signals_df.loc[y_true==1, "iso_score"], bins=80,
            alpha=0.6, color="crimson",   label="Fraud",      density=True)
    ax.set_title("Isolation Forest Anomaly Score Distribution", fontweight="bold")
    ax.set_xlabel("Anomaly Score (normalised)")
    ax.set_ylabel("Density")
    ax.legend()

    plt.suptitle("Two-Stage Detection: Score Distributions", fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "score_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_decision_matrix(signals_df: pd.DataFrame,
                          y_true: pd.Series) -> None:
    """
    2x2 matrix showing decision category breakdown for fraud vs legit.
    """
    categories = ["CONFIRMED_FRAUD", "NOVEL_ANOMALY", "KNOWN_FRAUD", "LEGITIMATE"]
    fraud_counts = [((signals_df["decision"]==c) & (y_true==1)).sum() for c in categories]
    legit_counts = [((signals_df["decision"]==c) & (y_true==0)).sum() for c in categories]

    x      = np.arange(len(categories))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, fraud_counts, width, label="Actual Fraud",      color="crimson",   alpha=0.85)
    ax.bar(x + width/2, legit_counts, width, label="Actual Legitimate",  color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("Transaction Count")
    ax.set_title("Two-Stage Decision Matrix: How Each Category Breaks Down", fontweight="bold")
    ax.legend()

    # Annotate fraud capture rate per category
    for i, (fc, lc) in enumerate(zip(fraud_counts, legit_counts)):
        total = fc + lc
        if total > 0:
            ax.text(i, max(fc, lc) + total*0.01,
                    f"{100*fc/max(total,1):.1f}%\nfraud", ha="center", fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "decision_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_combined_scatter(signals_df: pd.DataFrame,
                           y_true: pd.Series,
                           sample_n: int = 3000) -> None:
    """
    Scatter: XGBoost prob (x) vs IsoForest score (y), coloured by true label.
    Shows agreement/disagreement regions visually.
    """
    idx = np.random.choice(len(signals_df), min(sample_n, len(signals_df)), replace=False)
    s   = signals_df.iloc[idx]
    yt  = y_true.iloc[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(s.loc[yt==0, "xgb_prob"], s.loc[yt==0, "iso_score"],
               alpha=0.3, s=8, color="steelblue", label="Legitimate")
    ax.scatter(s.loc[yt==1, "xgb_prob"], s.loc[yt==1, "iso_score"],
               alpha=0.6, s=12, color="crimson",   label="Fraud")

    # Quadrant lines
    ax.axvline(0.41, color="orange", linestyle="--", linewidth=1, label="XGB threshold")
    ax.axhline(0.50, color="green",  linestyle="--", linewidth=1, label="ISO threshold")

    # Quadrant labels
    ax.text(0.75, 0.75, "CONFIRMED\nFRAUD",    transform=ax.transAxes, ha="center", color="crimson",   fontsize=9, fontweight="bold")
    ax.text(0.15, 0.75, "NOVEL\nANOMALY",      transform=ax.transAxes, ha="center", color="darkorange",fontsize=9, fontweight="bold")
    ax.text(0.75, 0.25, "KNOWN\nFRAUD",        transform=ax.transAxes, ha="center", color="tomato",    fontsize=9, fontweight="bold")
    ax.text(0.15, 0.25, "LEGITIMATE",          transform=ax.transAxes, ha="center", color="steelblue", fontsize=9, fontweight="bold")

    ax.set_xlabel("XGBoost Fraud Probability")
    ax.set_ylabel("Isolation Forest Anomaly Score")
    ax.set_title("Two-Stage Detection: Agreement Map", fontweight="bold")
    ax.legend(loc="upper left", markerscale=2)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "agreement_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


if __name__ == "__main__":
    print("Loading data and models …")
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet"))["isFraud"]
    y_test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"))["isFraud"]
    xgb_model = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "threshold.json")) as f:
        threshold = json.load(f)["threshold"]

    # Train Isolation Forest
    iso = train_isolation_forest(X_train, contamination=0.035)
    joblib.dump(iso, os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
    print(f"  IsoForest saved → {ARTIFACTS_DIR}/isolation_forest.joblib")

    # Evaluate
    metrics = evaluate_isolation_forest(iso, X_test, y_test)

    # Build two-stage signals on test set
    print("\nBuilding two-stage decision signals …")
    signals_df = build_two_stage_signals(iso, xgb_model, X_test, threshold)

    print("\n── Decision Category Breakdown ───────────────────────────────")
    print(signals_df["decision"].value_counts().to_string())

    # Novel anomaly capture — the key metric
    novel_fraud = ((signals_df["decision"] == "NOVEL_ANOMALY") & (y_test == 1)).sum()
    novel_total = (signals_df["decision"] == "NOVEL_ANOMALY").sum()
    print(f"\n  NOVEL_ANOMALY transactions that are actually fraud: {novel_fraud} / {novel_total}")
    print(f"  → IsoForest caught {novel_fraud} fraud cases XGBoost missed entirely")

    # Save signals and metrics
    signals_df.to_parquet(os.path.join(PROCESSED_DIR, "two_stage_signals.parquet"), index=False)
    metrics["novel_fraud_caught"] = int(novel_fraud)
    with open(os.path.join(ARTIFACTS_DIR, "anomaly_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    print("\nGenerating plots …")
    plot_score_distributions(signals_df, y_test)
    plot_decision_matrix(signals_df, y_test)
    plot_combined_scatter(signals_df, y_test)

    print("\n✓  anomaly.py complete. Next: python src/fraud_engine/calibrate.py")