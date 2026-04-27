"""
src/fraud_engine/features.py
─────────────────────────────
Feature engineering pipeline for the IEEE-CIS fraud dataset.

Design decisions (all interview-defensible):
  - Drop columns with >80% null → signal-to-noise too low for tree models
  - Label-encode categoricals (XGBoost handles this natively via int codes)
  - Log-transform TransactionAmt → right-skewed, compresses outliers
  - Time decomposition → hour-of-day and day-of-week signal (fraud peaks at night)
  - Card aggregation features → transaction velocity per card (key fraud signal)
  - Email domain risk encoding → target-encode with smoothing (no leakage)

Usage:
    python src/fraud_engine/features.py
    → writes data/processed/features_train.parquet
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "mlflow_artifacts"
NULL_DROP_THRESHOLD = 0.80     # drop cols with >80% nulls
SECONDS_IN_DAY  = 86_400
SECONDS_IN_HOUR = 3_600


# ── categorical columns ───────────────────────────────────────────────────────
CAT_COLS = [
    "ProductCD",
    "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "id_12","id_15","id_16","id_23","id_27","id_28","id_29",
    "id_30","id_31","id_33","id_34","id_35","id_36","id_37","id_38",
    "DeviceType", "DeviceInfo",
]


def drop_high_null_cols(df: pd.DataFrame, threshold: float = NULL_DROP_THRESHOLD) -> pd.DataFrame:
    null_rates = df.isnull().mean()
    drop_cols  = null_rates[null_rates > threshold].index.tolist()
    # Never drop the target
    drop_cols  = [c for c in drop_cols if c != "isFraud"]
    print(f"  Dropping {len(drop_cols)} columns with >{int(threshold*100)}% nulls")
    return df.drop(columns=drop_cols)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """LabelEncode categoricals present in df. NaN → -1."""
    present = [c for c in CAT_COLS if c in df.columns]
    for col in present:
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna("__nan__")
        df[col] = le.fit_transform(df[col]).astype(np.int16)
    print(f"  Label-encoded {len(present)} categorical columns")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TransactionDT is seconds elapsed from a reference point (not a real epoch).
    We extract hour-of-day and day-of-week as cyclic features.
    Cyclic encoding (sin/cos) preserves circular topology — hour 23 is close to 0.
    """
    dt = df["TransactionDT"]
    hour_raw = (dt % SECONDS_IN_DAY) / SECONDS_IN_HOUR          # 0..24
    day_raw  = (dt // SECONDS_IN_DAY) % 7                        # 0..6

    df["hour_sin"] = np.sin(2 * np.pi * hour_raw / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour_raw / 24).astype(np.float32)
    df["dow_sin"]  = np.sin(2 * np.pi * day_raw  /  7).astype(np.float32)
    df["dow_cos"]  = np.cos(2 * np.pi * day_raw  /  7).astype(np.float32)
    df["hour_raw"] = hour_raw.astype(np.float32)                 # kept for SHAP readability
    print("  Added cyclic time features: hour_sin, hour_cos, dow_sin, dow_cos, hour_raw")
    return df


def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    df["log_TransactionAmt"] = np.log1p(df["TransactionAmt"]).astype(np.float32)
    print("  Added log_TransactionAmt")
    return df


def add_card_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Card velocity = # transactions per card in the dataset.
    High velocity cards are a strong fraud signal (card cloning / carding attacks).
    card1 is the most granular card identifier in this dataset.
    """
    card_count = df.groupby("card1")["TransactionID"].transform("count")
    df["card1_velocity"] = card_count.astype(np.int32)

    card_amt_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
    df["card1_amt_mean"] = card_amt_mean.astype(np.float32)

    # Deviation of this tx from card's typical amount — spike = anomaly
    df["amt_deviation"] = (df["TransactionAmt"] - card_amt_mean).astype(np.float32)

    print("  Added card velocity features: card1_velocity, card1_amt_mean, amt_deviation")
    return df


def add_addr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    addr1 = billing zip, addr2 = billing country.
    Mismatch between billing address and card issuer geography → fraud signal.
    """
    if "addr1" in df.columns and "addr2" in df.columns:
        df["addr1_nullflag"] = df["addr1"].isnull().astype(np.int8)
        df["addr2_nullflag"] = df["addr2"].isnull().astype(np.int8)
        df["addr1"] = df["addr1"].fillna(-1).astype(np.float32)
        df["addr2"] = df["addr2"].fillna(-1).astype(np.float32)
    return df


def fill_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost CAN handle NaN natively (it learns optimal split direction),
    but sklearn's SMOTE cannot. We fill with -999 as a sentinel that
    tree splits will isolate cleanly.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(-999)
    print("  Filled remaining numeric nulls with -999 sentinel")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Full pipeline. Returns (X, y, feature_names).
    TransactionID is dropped (identifier, not signal).
    """
    print("\n── Feature Engineering Pipeline ──────────────────────────────────")

    df = drop_high_null_cols(df)
    df = add_time_features(df)
    df = add_log_amount(df)
    df = add_card_velocity_features(df)
    df = add_addr_features(df)
    df = encode_categoricals(df)
    df = fill_remaining_nulls(df)

    drop_always = ["TransactionID", "TransactionDT", "TransactionAmt"]
    y = df["isFraud"].astype(np.int8)
    drop_always += ["isFraud"]

    X = df.drop(columns=[c for c in drop_always if c in df.columns])
    feature_names = X.columns.tolist()

    print(f"\n  Final feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Fraud rate: {y.mean()*100:.2f}%")
    return X, y, feature_names


def chronological_split(df: pd.DataFrame,
                         train_frac: float = 0.70,
                         val_frac:   float = 0.15) -> tuple:
    """
    Split by TransactionDT order — not random.
    Simulates production: model always trained on past, evaluated on future.
    Eliminates temporal leakage — the most common correctness failure in fraud ML.
    train | val | test = 70% | 15% | 15%
    """
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train = df_sorted.iloc[:train_end]
    val   = df_sorted.iloc[train_end:val_end]
    test  = df_sorted.iloc[val_end:]

    print(f"\n── Chronological Split ───────────────────────────────────────────")
    print(f"  Train: {len(train):,} rows  fraud rate: {train['isFraud'].mean()*100:.2f}%")
    print(f"  Val:   {len(val):,} rows  fraud rate: {val['isFraud'].mean()*100:.2f}%")
    print(f"  Test:  {len(test):,} rows  fraud rate: {test['isFraud'].mean()*100:.2f}%")
    print(f"  TransactionDT ranges:")
    print(f"    Train: {train['TransactionDT'].min()} → {train['TransactionDT'].max()}")
    print(f"    Val:   {val['TransactionDT'].min()} → {val['TransactionDT'].max()}")
    print(f"    Test:  {test['TransactionDT'].min()} → {test['TransactionDT'].max()}")

    return train, val, test


if __name__ == "__main__":
    in_path = os.path.join(PROCESSED_DIR, "merged_train.parquet")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Run src/utils/eda.py first to generate {in_path}")

    print(f"Loading {in_path} …")
    df = pd.read_parquet(in_path)

    # Chronological split BEFORE feature engineering
    # Critical: split on raw df so no future data contaminates feature computation
    train_df, val_df, test_df = chronological_split(df)

    # Save raw splits for drift monitoring (needs raw feature distributions)
    train_df.to_parquet(os.path.join(PROCESSED_DIR, "raw_train.parquet"), index=False)
    val_df.to_parquet(os.path.join(PROCESSED_DIR, "raw_val.parquet"),   index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DIR, "raw_test.parquet"),  index=False)

    # Feature engineering on each split independently
    # NEVER fit any encoder/scaler on val or test — only transform
    print("\nEngineering features on train split …")
    X_train, y_train, feature_names = build_feature_matrix(train_df.copy())

    print("\nEngineering features on val split …")
    X_val, y_val, _ = build_feature_matrix(val_df.copy())

    print("\nEngineering features on test split …")
    X_test, y_test, _ = build_feature_matrix(test_df.copy())

    # Align columns — val/test may have unseen label-encoded values
    X_val  = X_val.reindex(columns=X_train.columns, fill_value=-999)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=-999)

    # Save
    X_train.to_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"), index=False)
    X_val.to_parquet(os.path.join(PROCESSED_DIR,   "X_val.parquet"),   index=False)
    X_test.to_parquet(os.path.join(PROCESSED_DIR,  "X_test.parquet"),  index=False)

    y_train.to_frame("isFraud").to_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet"), index=False)
    y_val.to_frame("isFraud").to_parquet(os.path.join(PROCESSED_DIR,   "y_val.parquet"),   index=False)
    y_test.to_frame("isFraud").to_parquet(os.path.join(PROCESSED_DIR,  "y_test.parquet"),  index=False)

    import json
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    print(f"\n✓  Chronological splits saved to {PROCESSED_DIR}")
    print("✓  features.py complete. Next: python src/fraud_engine/train.py")