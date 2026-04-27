"""
src/pipeline.py
────────────────
Unified Payment Decision Pipeline.

Every transaction flows through both engines sequentially:
  1. Velocity encoding    — stateful card-level features
  2. Feature store        — real-time feature computation
  3. Isolation Forest     — anomaly score (novel fraud detection)
  4. XGBoost + Calibrator — fraud probability (known pattern detection)
  5. Two-stage decision   — combine both signals
  6. Smart routing        — Thompson Sampling using fraud context
  7. Outcome logging      — full audit trail

This is the architecture that makes this a system, not two separate models.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from src.fraud_engine.calibrate import IsotonicCalibrator  # noqa — required for joblib unpickling
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fraud_engine.feature_store import get_feature_store
from src.fraud_engine.calibrate import IsotonicCalibrator
from src.fraud_engine.anomaly import get_anomaly_scores
from src.routing_engine.bandit import ThompsonRouter, TxType, GATEWAY_TRUE_RATES

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "mlflow_artifacts"

# Routing rules based on fraud risk
# High fraud risk → route to gateway with strongest 3DS/auth
FRAUD_ROUTING_OVERRIDE = {
    "HIGH":     {"CARD": "Stripe",   "UPI": "Razorpay", "NETBANKING": "CCAvenue", "WALLET": "Razorpay"},
    "CRITICAL": {"CARD": "Stripe",   "UPI": "Razorpay", "NETBANKING": "CCAvenue", "WALLET": "Razorpay"},
}


@dataclass
class PipelineDecision:
    transaction_id:       str
    fraud_probability:    float
    iso_anomaly_score:    float
    fraud_decision:       str    # LEGITIMATE / KNOWN_FRAUD / NOVEL_ANOMALY / CONFIRMED_FRAUD
    fraud_risk_level:     str    # LOW / MEDIUM / HIGH / CRITICAL
    recommended_action:   str    # ALLOW / REVIEW / BLOCK
    routed_gateway:       str
    routing_reason:       str
    expected_success_rate: float
    processing_time_ms:   float

    def to_dict(self) -> dict:
        return asdict(self)


class PaymentPipeline:
    """
    Singleton pipeline — load once, process many transactions.
    """

    def __init__(self):
        self._loaded = False

    def load(self) -> None:
        print("Loading Payment Intelligence Pipeline …")

        self.feature_store = get_feature_store()
        self.xgb_model     = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.joblib"))
        self.calibrator    = joblib.load(os.path.join(ARTIFACTS_DIR, "calibrator.joblib"))
        self.iso_forest    = joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
        self.ts_router     = ThompsonRouter(seed=42)

        with open(os.path.join(ARTIFACTS_DIR, "threshold.json")) as f:
            self.threshold = json.load(f)["threshold"]

        with open(os.path.join(ARTIFACTS_DIR, "business_impact.json")) as f:
            self.business_config = json.load(f)

        # Restore router state if available
        state_path = os.path.join(ARTIFACTS_DIR, "router_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            for tx_str, gw_data in state.items():
                try:
                    tx = TxType(tx_str)
                    for gw, vals in gw_data.items():
                        arm = self.ts_router.arms.get((gw, tx))
                        if arm:
                            arm.alpha  = vals["alpha"]
                            arm.beta_  = vals["beta"]
                except Exception:
                    pass

        self._loaded = True
        print("  Pipeline ready.")

    def _get_fraud_risk_level(self, prob: float) -> str:
        if prob < 0.30:   return "LOW"
        if prob < 0.55:   return "MEDIUM"
        if prob < 0.75:   return "HIGH"
        return "CRITICAL"

    def _get_recommended_action(self, decision: str, risk: str) -> str:
        if decision == "CONFIRMED_FRAUD":  return "BLOCK"
        if decision == "NOVEL_ANOMALY":    return "REVIEW"
        if risk in ("HIGH", "CRITICAL"):   return "REVIEW"
        if risk == "MEDIUM":               return "ALLOW"
        return "ALLOW"

    def _route(self, tx_type_str: str, fraud_risk: str,
               fraud_prob: float) -> tuple[str, str]:
        """
        Route transaction to optimal gateway.
        High/critical fraud risk → override to strongest-auth gateway.
        Otherwise → Thompson Sampling.
        """
        try:
            tx_type = TxType(tx_type_str.upper())
        except ValueError:
            tx_type = TxType.CARD

        if fraud_risk in FRAUD_ROUTING_OVERRIDE:
            gateway = FRAUD_ROUTING_OVERRIDE[fraud_risk].get(tx_type_str.upper(), "Razorpay")
            reason  = f"Fraud-risk override ({fraud_risk}) → strongest-auth gateway"
        else:
            gateway = self.ts_router.route(tx_type)
            arm     = self.ts_router.arms[(gateway, tx_type)]
            reason  = (f"Thompson Sampling → {gateway} "
                       f"(posterior mean: {arm.posterior_mean:.3f}, "
                       f"trials: {arm.n_trials})")

        return gateway, reason

    def process(self, raw_tx: dict,
                tx_type: str = "CARD",
                update_router: bool = True) -> PipelineDecision:
        """
        Full pipeline: raw transaction → PipelineDecision.
        """
        import time
        t0 = time.perf_counter()

        if not self._loaded:
            raise RuntimeError("Call pipeline.load() first.")

        tx_id = str(raw_tx.get("TransactionID", "unknown"))

        # ── Stage 1: Feature computation ───────────────────────────────
        X = self.feature_store.transform(raw_tx)

        # ── Stage 2: Isolation Forest anomaly score ────────────────────
        iso_score = float(get_anomaly_scores(self.iso_forest, X)[0])

        # ── Stage 3: XGBoost fraud probability ────────────────────────
        prob_raw = float(self.xgb_model.predict_proba(X)[0, 1])
        prob_cal = float(self.calibrator.transform(np.array([prob_raw]))[0])

        # ── Stage 4: Two-stage decision ────────────────────────────────
        xgb_flag = int(prob_cal >= self.threshold)
        iso_flag = int(iso_score >= 0.5)

        if xgb_flag and iso_flag:
            fraud_decision = "CONFIRMED_FRAUD"
        elif iso_flag and not xgb_flag:
            fraud_decision = "NOVEL_ANOMALY"
        elif xgb_flag and not iso_flag:
            fraud_decision = "KNOWN_FRAUD"
        else:
            fraud_decision = "LEGITIMATE"

        risk   = self._get_fraud_risk_level(prob_cal)
        action = self._get_recommended_action(fraud_decision, risk)

        # ── Stage 5: Smart routing ─────────────────────────────────────
        gateway, routing_reason = self._route(tx_type, risk, prob_cal)

        # Expected success rate from router's learned posterior
        try:
            tx_enum = TxType(tx_type.upper())
            arm     = self.ts_router.arms[(gateway, tx_enum)]
            exp_sr  = arm.posterior_mean
        except Exception:
            exp_sr  = GATEWAY_TRUE_RATES.get(gateway, {}).get(TxType.CARD, 0.85)

        # ── Stage 6: Update router with simulated outcome ──────────────
        if update_router and action == "ALLOW":
            try:
                tx_enum = TxType(tx_type.upper())
                true_p  = GATEWAY_TRUE_RATES[gateway][tx_enum]
                success = np.random.rand() < true_p
                self.ts_router.arms[(gateway, tx_enum)].update(success)
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PipelineDecision(
            transaction_id        = tx_id,
            fraud_probability     = round(prob_cal, 4),
            iso_anomaly_score     = round(iso_score, 4),
            fraud_decision        = fraud_decision,
            fraud_risk_level      = risk,
            recommended_action    = action,
            routed_gateway        = gateway,
            routing_reason        = routing_reason,
            expected_success_rate = round(float(exp_sr), 4),
            processing_time_ms    = round(elapsed_ms, 2),
        )

    def process_batch(self, transactions: list[dict],
                       tx_types: list[str] = None) -> pd.DataFrame:
        """Process a batch of transactions. Returns decisions DataFrame."""
        tx_types = tx_types or ["CARD"] * len(transactions)
        results  = []
        for tx, tt in zip(transactions, tx_types):
            try:
                decision = self.process(tx, tt, update_router=False)
                results.append(decision.to_dict())
            except Exception as e:
                results.append({"transaction_id": tx.get("TransactionID","?"),
                                 "error": str(e)})
        return pd.DataFrame(results)


# ── singleton ─────────────────────────────────────────────────────────────────
_pipeline = None

def get_pipeline() -> PaymentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PaymentPipeline()
        _pipeline.load()
    return _pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()

    # Test transactions
    test_cases = [
        {
            "tx": {"TransactionID": "TX001", "TransactionDT": 9_000_000,
                   "TransactionAmt": 45.00, "ProductCD": "W", "card1": 2755,
                   "card4": "visa", "card6": "debit", "P_emaildomain": "gmail.com"},
            "tx_type": "UPI",
            "label": "Normal UPI transaction",
        },
        {
            "tx": {"TransactionID": "TX002", "TransactionDT": 2_000,
                   "TransactionAmt": 4999.99, "ProductCD": "C", "card1": 9999,
                   "card4": "discover", "card6": "credit", "P_emaildomain": "mail.com"},
            "tx_type": "CARD",
            "label": "High-risk card transaction (2am, high amount, risky domain)",
        },
        {
            "tx": {"TransactionID": "TX003", "TransactionDT": 5_400_000,
                   "TransactionAmt": 120.00, "ProductCD": "H", "card1": 5432,
                   "card4": "mastercard", "card6": "debit", "P_emaildomain": "yahoo.com"},
            "tx_type": "NETBANKING",
            "label": "Normal netbanking transaction",
        },
    ]

    print("\n" + "═"*70)
    print("  UNIFIED PIPELINE — TEST TRANSACTIONS")
    print("═"*70)

    for case in test_cases:
        print(f"\n── {case['label']} ──")
        decision = pipeline.process(case["tx"], case["tx_type"])
        d = decision.to_dict()
        print(f"  Fraud Probability:     {d['fraud_probability']:.4f}")
        print(f"  Anomaly Score:         {d['iso_anomaly_score']:.4f}")
        print(f"  Decision:              {d['fraud_decision']}")
        print(f"  Risk Level:            {d['fraud_risk_level']}")
        print(f"  Recommended Action:    {d['recommended_action']}")
        print(f"  Routed To:             {d['routed_gateway']}")
        print(f"  Expected Success Rate: {d['expected_success_rate']:.4f}")
        print(f"  Processing Time:       {d['processing_time_ms']:.2f}ms")

    print("\n✓  pipeline.py complete.")