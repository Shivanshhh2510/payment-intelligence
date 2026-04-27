"""
src/fraud_engine/feature_store.py
───────────────────────────────────
Feature store — real-time feature computation for inference.

Solves training-serving skew: the exact same transformations applied
at training time must be applied identically at inference time.
A mismatch between training features and serving features is one of
the most common production ML failures.

This module:
  1. Loads all fitted transformers (label encoders, velocity stats)
  2. Accepts a raw transaction dict (as would arrive from an API)
  3. Returns a feature vector identical to what the model was trained on

In production this would be backed by Redis for sub-millisecond lookups.
Here we simulate it with in-memory dicts loaded from parquet.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.fraud_engine.calibrate import IsotonicCalibrator  # noqa — needed for joblib unpickling

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "mlflow_artifacts"

HOUR = 3_600
DAY  = 86_400

# Columns dropped during training
DROP_ALWAYS = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]

CAT_COLS = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "id_12","id_15","id_16","id_23","id_27","id_28","id_29",
    "id_30","id_31","id_33","id_34","id_35","id_36","id_37","id_38",
    "DeviceType","DeviceInfo",
]


class FeatureStore:
    """
    Lightweight feature store for real-time inference.
    Loads once, scores many.
    """

    def __init__(self):
        self.feature_names: list = []
        self.card_stats:    dict = {}   # card1 → {mean_amt, count}
        self.cat_encodings: dict = {}   # col → {value → int code}
        self._loaded = False

    def load(self) -> None:
        """Load all artifacts needed for inference."""
        # Feature names
        with open(os.path.join(ARTIFACTS_DIR, "feature_names.json")) as f:
            self.feature_names = json.load(f)

        # Card statistics from training data (simulates Redis cache)
        X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"),
                                   columns=["card1_velocity", "card1_amt_mean"])
        raw_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "raw_train.parquet"),
                                     columns=["card1", "TransactionAmt"])

        card_grp = raw_train.groupby("card1")["TransactionAmt"]
        self.card_stats = {
            int(k): {"mean": float(v.mean()), "std": float(v.std() + 1e-6), "count": int(len(v))}
            for k, v in card_grp
        }

        # Categorical encodings — rebuild from training data
        raw = pd.read_parquet(os.path.join(PROCESSED_DIR, "raw_train.parquet"))
        for col in CAT_COLS:
            if col in raw.columns:
                unique_vals = raw[col].astype(str).fillna("__nan__").unique()
                self.cat_encodings[col] = {v: i for i, v in enumerate(sorted(unique_vals))}

        self._loaded = True
        print(f"  FeatureStore loaded: {len(self.feature_names)} features, "
              f"{len(self.card_stats):,} card profiles")

    def _encode_cat(self, col: str, val) -> int:
        enc = self.cat_encodings.get(col, {})
        s   = str(val) if val is not None else "__nan__"
        return enc.get(s, -1)   # -1 for unseen values

    def transform(self, raw_tx: dict) -> pd.DataFrame:
        """
        Transform a single raw transaction dict into model-ready feature vector.
        Returns DataFrame with exactly the columns the model expects.
        """
        if not self._loaded:
            raise RuntimeError("Call feature_store.load() first.")

        tx = dict(raw_tx)   # copy

        # ── time features ──────────────────────────────────────────────
        dt       = float(tx.get("TransactionDT", 0))
        hour_raw = (dt % DAY) / HOUR
        day_raw  = (dt // DAY) % 7

        tx["hour_sin"] = float(np.sin(2 * np.pi * hour_raw / 24))
        tx["hour_cos"] = float(np.cos(2 * np.pi * hour_raw / 24))
        tx["dow_sin"]  = float(np.sin(2 * np.pi * day_raw  /  7))
        tx["dow_cos"]  = float(np.cos(2 * np.pi * day_raw  /  7))
        tx["hour_raw"] = float(hour_raw)

        # ── amount features ────────────────────────────────────────────
        amt = float(tx.get("TransactionAmt", 0))
        tx["log_TransactionAmt"] = float(np.log1p(amt))

        # ── card velocity features ─────────────────────────────────────
        card1 = int(tx.get("card1", -1))
        stats = self.card_stats.get(card1, {"mean": amt, "std": 1.0, "count": 1})
        tx["card1_velocity"]  = stats["count"]
        tx["card1_amt_mean"]  = stats["mean"]
        tx["amt_deviation"]   = amt - stats["mean"]

        # ── address features ───────────────────────────────────────────
        tx["addr1_nullflag"] = int(tx.get("addr1") is None)
        tx["addr2_nullflag"] = int(tx.get("addr2") is None)
        tx["addr1"] = float(tx.get("addr1") or -1)
        tx["addr2"] = float(tx.get("addr2") or -1)

        # ── categorical encoding ───────────────────────────────────────
        for col in CAT_COLS:
            tx[col] = self._encode_cat(col, tx.get(col))

        # ── build feature vector ───────────────────────────────────────
        row = {}
        for feat in self.feature_names:
            val = tx.get(feat, -999)
            row[feat] = float(val) if val is not None else -999.0

        return pd.DataFrame([row])[self.feature_names]

    def get_card_profile(self, card1: int) -> dict:
        """Return card's historical transaction profile."""
        return self.card_stats.get(card1, {})


# ── singleton ─────────────────────────────────────────────────────────────────
_store = None

def get_feature_store() -> FeatureStore:
    """Return singleton FeatureStore, loading on first call."""
    global _store
    if _store is None:
        _store = FeatureStore()
        _store.load()
    return _store


if __name__ == "__main__":
    print("Initialising feature store …")
    store = get_feature_store()

    # Test with a synthetic transaction
    test_tx = {
        "TransactionDT":  9_000_000,
        "TransactionAmt": 299.99,
        "ProductCD":      "W",
        "card1":          2755,
        "card4":          "visa",
        "card6":          "debit",
        "P_emaildomain":  "gmail.com",
        "R_emaildomain":  "gmail.com",
        "addr1":          299.0,
        "addr2":          87.0,
        "DeviceType":     "desktop",
    }

    print("\nTransforming test transaction …")
    X = store.transform(test_tx)
    print(f"  Output shape: {X.shape}")
    print(f"  Non-sentinel features: {(X.values[0] != -999).sum()} / {X.shape[1]}")

    # Load model and score
    model      = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.joblib"))
    calibrator = joblib.load(os.path.join(ARTIFACTS_DIR, "calibrator.joblib"))

    prob_raw = model.predict_proba(X)[0, 1]
    prob_cal = calibrator.transform(np.array([prob_raw]))[0]

    print(f"\n  Raw fraud probability:       {prob_raw:.4f}")
    print(f"  Calibrated fraud probability: {prob_cal:.4f}")
    print(f"  Decision: {'FRAUD' if prob_cal >= 0.41 else 'LEGITIMATE'}")
    print("\n✓  feature_store.py complete.")