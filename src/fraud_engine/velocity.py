"""
src/fraud_engine/velocity.py
─────────────────────────────
Velocity encoding — stateful fraud features.

A transaction that looks 40% fraud in isolation looks 90% fraud
if the same card had 3 failed attempts in the last hour.
This is the single biggest signal real fraud systems use that
academic models miss entirely.

Features computed:
  - tx_count_1h:     transactions from this card in last 1 hour
  - tx_count_24h:    transactions from this card in last 24 hours
  - tx_count_7d:     transactions from this card in last 7 days
  - amt_sum_1h:      total spend from this card in last 1 hour
  - amt_sum_24h:     total spend from this card in last 24 hours
  - failed_1h:       failed transactions from this device in last 1 hour
  - unique_emails_24h: distinct email domains used by this card in 24h
  - amt_zscore:      how many std devs this amount is from card's history

Note: TransactionDT is seconds from reference point, not Unix timestamp.
All time windows are in seconds.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"

HOUR  = 3_600
DAY   = 86_400
WEEK  = 604_800


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity features on time-sorted DataFrame.
    Must be sorted by TransactionDT before calling.
    Uses expanding window per card — O(n log n) via groupby + merge_asof logic.
    """
    print("Computing velocity features …")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # We'll compute rolling counts using a vectorised approach per card
    # groupby + transform with a custom window is too slow for 500k rows
    # Instead: for each transaction, count prior transactions within window
    # using searchsorted on the sorted TransactionDT per card group

    card_col   = "card1"
    time_col   = "TransactionDT"
    amt_col    = "TransactionAmt"

    # Pre-allocate output arrays
    n = len(df)
    tx_1h    = np.zeros(n, dtype=np.int32)
    tx_24h   = np.zeros(n, dtype=np.int32)
    tx_7d    = np.zeros(n, dtype=np.int32)
    amt_1h   = np.zeros(n, dtype=np.float32)
    amt_24h  = np.zeros(n, dtype=np.float32)
    amt_zscore = np.zeros(n, dtype=np.float32)
    unique_emails_24h = np.zeros(n, dtype=np.int16)

    # Group by card
    grouped = df.groupby(card_col)

    for card_id, group in grouped:
        idx   = group.index.values
        times = group[time_col].values
        amts  = group[amt_col].values

        for i, (pos, t, a) in enumerate(zip(idx, times, amts)):
            # Count transactions within each window (exclude current)
            tx_1h[pos]   = np.searchsorted(times[:i], t - HOUR,  side="left")
            tx_24h[pos]  = np.searchsorted(times[:i], t - DAY,   side="left")
            tx_7d[pos]   = np.searchsorted(times[:i], t - WEEK,  side="left")

            # i - searchsorted gives count in window
            start_1h  = np.searchsorted(times[:i], t - HOUR,  side="left")
            start_24h = np.searchsorted(times[:i], t - DAY,   side="left")

            tx_1h[pos]  = i - start_1h
            tx_24h[pos] = i - start_24h
            tx_7d[pos]  = i  # all prior transactions

            amt_1h[pos]  = amts[start_1h:i].sum()
            amt_24h[pos] = amts[start_24h:i].sum()

            # Amount z-score vs card history
            hist_amts = amts[:i]
            if len(hist_amts) >= 2:
                mu  = hist_amts.mean()
                std = hist_amts.std() + 1e-6
                amt_zscore[pos] = (a - mu) / std
            else:
                amt_zscore[pos] = 0.0

    # Unique email domains per card in 24h — computed separately
    if "P_emaildomain" in df.columns:
        email_24h = (
            df.groupby([card_col, pd.cut(df[time_col] // DAY, bins=50)])
              ["P_emaildomain"]
              .transform("nunique")
              .fillna(1)
              .astype(np.int16)
        )
        unique_emails_24h = email_24h.values

    df["vel_tx_1h"]          = tx_1h
    df["vel_tx_24h"]         = tx_24h
    df["vel_tx_7d"]          = tx_7d
    df["vel_amt_1h"]         = amt_1h
    df["vel_amt_24h"]        = amt_24h
    df["vel_amt_zscore"]     = amt_zscore
    df["vel_unique_emails"]  = unique_emails_24h

    print(f"  Velocity features added: vel_tx_1h, vel_tx_24h, vel_tx_7d, "
          f"vel_amt_1h, vel_amt_24h, vel_amt_zscore, vel_unique_emails")
    print(f"  High velocity cards (>5 tx in 24h): "
          f"{(df['vel_tx_24h'] > 5).sum():,}")
    print(f"  Suspicious amount spikes (zscore>3): "
          f"{(df['vel_amt_zscore'] > 3).sum():,}")

    return df


def compute_velocity_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Faster approximate velocity using time-bucketed aggregations.
    Used for large datasets where exact per-transaction windows are slow.
    Accuracy tradeoff: windows are approximate (bucket-based not exact).
    """
    print("Computing velocity features (fast approximation) …")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Create time buckets
    df["_hour_bucket"] = df["TransactionDT"] // HOUR
    df["_day_bucket"]  = df["TransactionDT"] // DAY

    # Cumulative count per card per day bucket
    df["vel_tx_24h"] = df.groupby(["card1", "_day_bucket"]).cumcount()
    df["vel_amt_24h"] = (
        df.groupby(["card1", "_day_bucket"])["TransactionAmt"]
          .transform(lambda x: x.expanding().sum().shift(1).fillna(0))
    )

    # Hour-level
    df["vel_tx_1h"] = df.groupby(["card1", "_hour_bucket"]).cumcount()
    df["vel_amt_1h"] = (
        df.groupby(["card1", "_hour_bucket"])["TransactionAmt"]
          .transform(lambda x: x.expanding().sum().shift(1).fillna(0))
    )

    # 7-day: total prior transactions per card
    df["vel_tx_7d"] = df.groupby("card1").cumcount()

    # Amount z-score vs card history
    df["vel_amt_zscore"] = (
        df.groupby("card1")["TransactionAmt"]
          .transform(lambda x: (x - x.expanding().mean().shift(1)) /
                                (x.expanding().std().shift(1) + 1e-6))
          .fillna(0)
    ).astype(np.float32)

    # Unique emails per card per day
    if "P_emaildomain" in df.columns:
        df["vel_unique_emails"] = (
            df.groupby(["card1", "_day_bucket"])["P_emaildomain"]
              .transform("nunique")
              .astype(np.int16)
        )
    else:
        df["vel_unique_emails"] = 0

    # Cleanup temp columns
    df.drop(columns=["_hour_bucket", "_day_bucket"], inplace=True)

    # Cast
    for col in ["vel_tx_1h", "vel_tx_24h", "vel_tx_7d"]:
        df[col] = df[col].astype(np.int32)
    for col in ["vel_amt_1h", "vel_amt_24h"]:
        df[col] = df[col].astype(np.float32)

    print(f"  High velocity cards (>5 tx in 24h): {(df['vel_tx_24h'] > 5).sum():,}")
    print(f"  Suspicious amount spikes (zscore>3): {(df['vel_amt_zscore'] > 3).sum():,}")

    return df


def analyze_velocity_vs_fraud(df: pd.DataFrame) -> None:
    """Print fraud rates at different velocity levels."""
    if "isFraud" not in df.columns:
        return

    print("\n── Velocity vs Fraud Rate ────────────────────────────────────")

    bins = [0, 1, 3, 5, 10, 999]
    labels = ["0", "1-2", "3-4", "5-9", "10+"]
    df["_vel_bucket"] = pd.cut(df["vel_tx_24h"], bins=bins, labels=labels)
    analysis = df.groupby("_vel_bucket")["isFraud"].agg(["mean","count"])
    analysis.columns = ["fraud_rate", "count"]
    analysis["fraud_rate"] = (analysis["fraud_rate"] * 100).round(2)
    print("  Fraud rate by 24h transaction velocity:")
    print(analysis.to_string())
    df.drop(columns=["_vel_bucket"], inplace=True)


if __name__ == "__main__":
    print("Loading raw train data …")
    # Use raw train (pre-feature-engineering) to compute velocity on original data
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "raw_train.parquet"))

    print(f"  Loaded {len(df):,} rows")

    df = compute_velocity_features_fast(df)
    analyze_velocity_vs_fraud(df)

    # Save velocity features only (will be merged with X_train later)
    vel_cols = ["TransactionID"] + [c for c in df.columns if c.startswith("vel_")]
    vel_df = df[vel_cols]
    out = os.path.join(PROCESSED_DIR, "velocity_features.parquet")
    vel_df.to_parquet(out, index=False)
    print(f"\n  Velocity features saved → {out}")
    print(f"  Shape: {vel_df.shape}")
    print("\n✓  velocity.py complete.")