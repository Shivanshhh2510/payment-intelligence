"""
src/utils/eda.py
────────────────
Data loading, merging, and EDA for the IEEE-CIS Fraud Detection dataset.

Usage:
    python src/utils/eda.py

Outputs:
    - data/processed/merged_train.parquet   (merged + basic cleaning)
    - data/processed/eda_report.txt         (printed stats)
    - Console plots via matplotlib (save manually if needed)
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — safe on any machine
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

TRANSACTION_PATH = os.path.join(RAW_DIR, "train_transaction.csv")
IDENTITY_PATH    = os.path.join(RAW_DIR, "train_identity.csv")

# ── helpers ───────────────────────────────────────────────────────────────────

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to smallest safe dtype.
    Cuts RAM by ~40–60% on this dataset — essential for 590k × 400+ cols.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object:
            df[col] = df[col].astype("category")
        elif str(col_type)[:3] == "int":
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min  and c_max <= np.iinfo(np.int8).max:  df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
        elif str(col_type)[:5] == "float":
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  Memory: {start_mem:.1f} MB → {end_mem:.1f} MB  ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    return df


def load_data(transaction_path: str = TRANSACTION_PATH,
              identity_path: str   = IDENTITY_PATH) -> pd.DataFrame:
    """
    Load both CSVs, left-join on TransactionID, reduce memory.
    Returns merged DataFrame.
    """
    print("[1/4] Loading transaction data …")
    tx = pd.read_csv(transaction_path)
    print(f"      transaction shape: {tx.shape}")
    tx = reduce_mem_usage(tx)

    print("[2/4] Loading identity data …")
    id_ = pd.read_csv(identity_path)
    print(f"      identity shape:     {id_.shape}")
    id_ = reduce_mem_usage(id_)

    print("[3/4] Merging on TransactionID (left join) …")
    df = tx.merge(id_, on="TransactionID", how="left")
    print(f"      merged shape:       {df.shape}")

    return df


def run_eda(df: pd.DataFrame) -> None:
    """Print structured EDA report and save plots."""

    print("\n" + "═"*70)
    print("  IEEE-CIS FRAUD DETECTION — EDA REPORT")
    print("═"*70)

    # ── 1. Basic info ──────────────────────────────────────────────────────
    print("\n── 1. SHAPE & TARGET ─────────────────────────────────────────────")
    total = len(df)
    fraud = df["isFraud"].sum()
    legit = total - fraud
    print(f"  Rows:         {total:,}")
    print(f"  Columns:      {df.shape[1]}")
    print(f"  Fraud txns:   {fraud:,}  ({100*fraud/total:.2f}%)")
    print(f"  Legit txns:   {legit:,}  ({100*legit/total:.2f}%)")
    print(f"  Imbalance ratio (legit:fraud): {legit/fraud:.1f}:1")

    # ── 2. Missing values ──────────────────────────────────────────────────
    print("\n── 2. MISSING VALUES (cols > 40% null) ──────────────────────────")
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    high_null = null_pct[null_pct > 40]
    print(f"  Columns with >40% nulls: {len(high_null)}")
    print(high_null.head(20).to_string())

    # ── 3. TransactionAmt distribution ────────────────────────────────────
    print("\n── 3. TRANSACTION AMOUNT ─────────────────────────────────────────")
    print(df["TransactionAmt"].describe().to_string())
    print(f"\n  Fraud  — mean: ${df.loc[df.isFraud==1,'TransactionAmt'].mean():.2f}  median: ${df.loc[df.isFraud==1,'TransactionAmt'].median():.2f}")
    print(f"  Legit  — mean: ${df.loc[df.isFraud==0,'TransactionAmt'].mean():.2f}  median: ${df.loc[df.isFraud==0,'TransactionAmt'].median():.2f}")

    # ── 4. ProductCD ───────────────────────────────────────────────────────
    print("\n── 4. PRODUCT CODE FRAUD RATES ───────────────────────────────────")
    pcd = df.groupby("ProductCD")["isFraud"].agg(["count","mean"]).rename(columns={"count":"n","mean":"fraud_rate"})
    pcd["fraud_rate"] = (pcd["fraud_rate"]*100).round(2)
    print(pcd.sort_values("fraud_rate", ascending=False).to_string())

    # ── 5. Card features ───────────────────────────────────────────────────
    print("\n── 5. CARD4 (card network) FRAUD RATES ───────────────────────────")
    c4 = df.groupby("card4")["isFraud"].agg(["count","mean"]).rename(columns={"count":"n","mean":"fraud_rate"})
    c4["fraud_rate"] = (c4["fraud_rate"]*100).round(2)
    print(c4.sort_values("fraud_rate", ascending=False).to_string())

    print("\n── 6. CARD6 (card type) FRAUD RATES ─────────────────────────────")
    c6 = df.groupby("card6")["isFraud"].agg(["count","mean"]).rename(columns={"count":"n","mean":"fraud_rate"})
    c6["fraud_rate"] = (c6["fraud_rate"]*100).round(2)
    print(c6.sort_values("fraud_rate", ascending=False).to_string())

    # ── 7. DeviceType ──────────────────────────────────────────────────────
    print("\n── 7. DEVICE TYPE FRAUD RATES ───────────────────────────────────")
    if "DeviceType" in df.columns:
        dt = df.groupby("DeviceType")["isFraud"].agg(["count","mean"]).rename(columns={"count":"n","mean":"fraud_rate"})
        dt["fraud_rate"] = (dt["fraud_rate"]*100).round(2)
        print(dt.to_string())

    # ── 8. Email domains ───────────────────────────────────────────────────
    print("\n── 8. TOP P_EMAILDOMAIN FRAUD RATES ─────────────────────────────")
    em = df.groupby("P_emaildomain")["isFraud"].agg(["count","mean"]).rename(columns={"count":"n","mean":"fraud_rate"})
    em["fraud_rate"] = (em["fraud_rate"]*100).round(2)
    print(em[em["n"] > 500].sort_values("fraud_rate", ascending=False).head(10).to_string())

    # ── 9. C-columns (count features) correlation with isFraud ────────────
    print("\n── 9. C-FEATURE CORRELATION WITH isFraud ─────────────────────────")
    c_cols = [f"C{i}" for i in range(1,15) if f"C{i}" in df.columns]
    corr = df[c_cols + ["isFraud"]].corr()["isFraud"].drop("isFraud").sort_values(ascending=False)
    print(corr.round(4).to_string())

    print("\n" + "═"*70)
    print("  EDA COMPLETE")
    print("═"*70 + "\n")


def save_plots(df: pd.DataFrame, out_dir: str = PROCESSED_DIR) -> None:
    """Save key EDA plots as PNG files."""
    plots_dir = os.path.join(out_dir, "eda_plots")
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style="darkgrid", palette="muted")

    # ── Plot 1: Class imbalance ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["isFraud"].value_counts()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["steelblue","crimson"], edgecolor="white")
    ax.set_title("Class Distribution (isFraud)", fontweight="bold")
    ax.set_ylabel("Transaction Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max()*0.01, f"{v:,}\n({100*v/len(df):.1f}%)", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "01_class_distribution.png"), dpi=150)
    plt.close()

    # ── Plot 2: TransactionAmt log distribution ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    log_amt_fraud = np.log1p(df.loc[df.isFraud==1, "TransactionAmt"])
    log_amt_legit = np.log1p(df.loc[df.isFraud==0, "TransactionAmt"])
    ax.hist(log_amt_legit, bins=80, alpha=0.6, color="steelblue", label="Legitimate", density=True)
    ax.hist(log_amt_fraud, bins=80, alpha=0.6, color="crimson",   label="Fraud",     density=True)
    ax.set_xlabel("log(1 + TransactionAmt)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Amount Distribution (log scale)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "02_transaction_amt_dist.png"), dpi=150)
    plt.close()

    # ── Plot 3: Fraud rate by ProductCD ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    pcd = df.groupby("ProductCD")["isFraud"].mean().sort_values(ascending=False) * 100
    ax.bar(pcd.index.astype(str), pcd.values, color="crimson", edgecolor="white", alpha=0.85)
    ax.set_xlabel("ProductCD")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Product Code", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "03_fraud_by_productcd.png"), dpi=150)
    plt.close()

    # ── Plot 4: Null heatmap (sampled columns) ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    top_null = null_pct.head(50)
    ax.bar(range(len(top_null)), top_null.values, color="darkorange", edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(top_null)))
    ax.set_xticklabels(top_null.index, rotation=90, fontsize=7)
    ax.set_ylabel("% Missing")
    ax.set_title("Top 50 Columns by Missing Rate", fontweight="bold")
    ax.axhline(80, color="red",    linestyle="--", linewidth=1, label="80% threshold")
    ax.axhline(40, color="orange", linestyle="--", linewidth=1, label="40% threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "04_null_rates.png"), dpi=150)
    plt.close()

    print(f"  Plots saved → {plots_dir}/")


def save_processed(df: pd.DataFrame) -> None:
    """Save merged dataframe as parquet for fast downstream loading."""
    out = os.path.join(PROCESSED_DIR, "merged_train.parquet")
    df.to_parquet(out, index=False)
    size_mb = os.path.getsize(out) / 1024**2
    print(f"  Merged data saved → {out}  ({size_mb:.1f} MB)")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    print("\n[4/4] Running EDA …")
    run_eda(df)

    print("[5/5] Saving plots and processed data …")
    save_plots(df)
    save_processed(df)

    print("\n✓  eda.py complete. Next: python src/fraud_engine/features.py")