"""
src/fraud_engine/llm_explainer.py
───────────────────────────────────
Action-oriented LLM explanation layer.

The ML models make all decisions. The LLM only explains them in plain English
and recommends actions. This distinction is critical — LLM is the interface,
not the decision-maker.

Three explanation types:
  1. Transaction explanation  — why this fraud score, what action to take
  2. Routing explanation      — why this gateway, counterfactual analysis
  3. Drift incident report    — plain English alert for model health events

Fallback: if LLM call fails, rule-based template generates a decent explanation.
Never breaks the demo.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "mlflow_artifacts"

# Groq API — fast, free tier available
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"


def _call_groq(prompt: str, system: str, api_key: str,
               max_tokens: int = 300) -> str:
    """Call Groq API. Returns text or raises exception."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": prompt},
        ],
        "max_tokens":   max_tokens,
        "temperature":  0.3,
    }
    resp = requests.post(GROQ_API_URL, headers=headers,
                         json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Fallback templates ────────────────────────────────────────────────────────

def _fallback_fraud_explanation(decision: dict, shap_features: list = None) -> str:
    prob    = decision.get("fraud_probability", 0)
    risk    = decision.get("fraud_risk_level", "UNKNOWN")
    action  = decision.get("recommended_action", "ALLOW")
    d_type  = decision.get("fraud_decision", "LEGITIMATE")

    top_features = ""
    if shap_features:
        top_3 = shap_features[:3]
        top_features = " Key drivers: " + ", ".join(
            f"{f['feature']} ({'+' if f['shap_value']>0 else ''}{f['shap_value']:.3f})"
            for f in top_3
        )

    if d_type == "CONFIRMED_FRAUD":
        return (f"Both fraud detection systems flagged this transaction. "
                f"Calibrated fraud probability: {prob*100:.1f}%. "
                f"Risk level: {risk}.{top_features} "
                f"Recommended action: {action}.")
    elif d_type == "NOVEL_ANOMALY":
        return (f"This transaction shows unusual patterns not seen in historical fraud data. "
                f"Fraud probability: {prob*100:.1f}%. The anomaly detector flagged it as "
                f"statistically unusual.{top_features} "
                f"Recommended action: {action} — manual review advised.")
    elif d_type == "KNOWN_FRAUD":
        return (f"This transaction matches known fraud patterns. "
                f"Fraud probability: {prob*100:.1f}%.{top_features} "
                f"Recommended action: {action}.")
    else:
        return (f"Transaction appears legitimate. "
                f"Fraud probability: {prob*100:.1f}% (below threshold).{top_features} "
                f"Recommended action: ALLOW.")


def _fallback_routing_explanation(decision: dict) -> str:
    gateway = decision.get("routed_gateway", "Unknown")
    sr      = decision.get("expected_success_rate", 0)
    tx_type = decision.get("tx_type", "transaction")
    risk    = decision.get("fraud_risk_level", "LOW")

    if risk in ("HIGH", "CRITICAL"):
        return (f"Routed to {gateway} due to elevated fraud risk ({risk}). "
                f"This gateway provides strongest authentication for {tx_type} transactions. "
                f"Expected success rate: {sr*100:.1f}%.")
    else:
        return (f"Routed to {gateway} via Thompson Sampling policy. "
                f"Learned success rate for {tx_type}: {sr*100:.1f}%. "
                f"This gateway is currently optimal for this transaction type.")


def _fallback_drift_report(drift_data: dict) -> str:
    rec       = drift_data.get("recommendation", "MONITOR")
    n_drifted = drift_data.get("n_high_psi_features", 0)
    auc_drop  = drift_data.get("auc_degradation", 0)
    fr_shift  = drift_data.get("fraud_rate_shift_pct", 0)

    if rec == "RETRAIN":
        return (f"Model drift detected. {n_drifted} features show significant distribution shift "
                f"(PSI > 0.20). Model AUC has degraded by {auc_drop:.4f}. "
                f"Fraud rate has shifted {fr_shift:+.1f}% from training baseline. "
                f"Recommendation: retrain on recent data (last 90 days).")
    else:
        return (f"Model health nominal. No significant drift detected. "
                f"AUC stable, feature distributions within acceptable bounds.")


# ── LLM explanations ──────────────────────────────────────────────────────────

def explain_fraud_decision(decision: dict,
                            shap_features: list = None,
                            api_key: str = None) -> dict:
    """
    Generate action-oriented fraud explanation.
    Returns dict with explanation, counterfactual, action.
    """
    prob    = decision.get("fraud_probability", 0)
    risk    = decision.get("fraud_risk_level", "LOW")
    d_type  = decision.get("fraud_decision", "LEGITIMATE")
    action  = decision.get("recommended_action", "ALLOW")
    amt     = decision.get("TransactionAmt", "unknown")
    gateway = decision.get("routed_gateway", "unknown")

    shap_summary = ""
    if shap_features:
        shap_summary = "Top SHAP features:\n" + "\n".join(
            f"  {f['feature']}: {f['shap_value']:+.4f} ({f['direction']})"
            for f in shap_features[:5]
        )

    if api_key:
        system = (
            "You are a fraud analyst AI at a payment company. "
            "You explain ML model decisions in plain English for fraud analysts. "
            "Be concise (3-4 sentences max). Be specific about numbers. "
            "Always end with a clear recommended action."
        )
        prompt = f"""
Fraud detection result:
- Calibrated fraud probability: {prob*100:.1f}%
- Risk level: {risk}
- Decision type: {d_type}
- Recommended action: {action}
- Transaction amount: ₹{amt}
- Routed to gateway: {gateway}
{shap_summary}

Explain this decision to a fraud analyst in plain English.
Include: (1) why this score, (2) what the analyst should do, (3) one counterfactual
(what would need to change for the decision to flip).
Keep it under 4 sentences.
"""
        try:
            explanation = _call_groq(prompt, system, api_key, max_tokens=200)
            source = "llm"
        except Exception as e:
            explanation = _fallback_fraud_explanation(decision, shap_features)
            source = f"fallback ({str(e)[:50]})"
    else:
        explanation = _fallback_fraud_explanation(decision, shap_features)
        source = "fallback (no api key)"

    return {
        "explanation": explanation,
        "source":      source,
        "action":      action,
        "risk_level":  risk,
    }


def explain_routing_decision(decision: dict,
                              router_state: dict = None,
                              api_key: str = None) -> dict:
    """
    Explain routing decision with counterfactual:
    what would happen if second-best gateway was chosen instead?
    """
    gateway  = decision.get("routed_gateway", "Unknown")
    sr       = decision.get("expected_success_rate", 0)
    tx_type  = decision.get("tx_type", "CARD")
    reason   = decision.get("routing_reason", "")

    # Build counterfactual from router state
    counterfactual = ""
    if router_state and tx_type in router_state:
        gw_data = router_state[tx_type]
        sorted_gws = sorted(gw_data.items(),
                            key=lambda x: x[1].get("posterior_mean", 0),
                            reverse=True)
        if len(sorted_gws) >= 2:
            second_gw, second_data = sorted_gws[1]
            second_sr = second_data.get("posterior_mean", 0)
            sr_diff   = (sr - second_sr) * 100
            counterfactual = (f"If routed to {second_gw} instead, "
                              f"expected success rate would be {second_sr*100:.1f}% "
                              f"({sr_diff:+.1f}% difference).")

    if api_key:
        system = (
            "You are a payments infrastructure engineer. "
            "Explain routing decisions concisely. 2-3 sentences max."
        )
        prompt = f"""
Routing decision:
- Transaction type: {tx_type}
- Selected gateway: {gateway}
- Expected success rate: {sr*100:.1f}%
- Routing reason: {reason}
- Counterfactual: {counterfactual}

Explain why this gateway was selected and what the business impact is.
"""
        try:
            explanation = _call_groq(prompt, system, api_key, max_tokens=150)
            source = "llm"
        except Exception as e:
            explanation = _fallback_routing_explanation(decision)
            source = f"fallback ({str(e)[:50]})"
    else:
        explanation = _fallback_routing_explanation(decision)
        source = "fallback (no api key)"

    return {
        "explanation":     explanation,
        "counterfactual":  counterfactual,
        "source":          source,
    }


def generate_drift_report(drift_data: dict, api_key: str = None) -> dict:
    """Generate plain-English drift incident report."""
    rec       = drift_data.get("recommendation", "MONITOR")
    features  = drift_data.get("high_psi_features", [])
    auc_drop  = drift_data.get("auc_degradation", 0)
    fr_shift  = drift_data.get("fraud_rate_shift_pct", 0)

    if api_key and rec == "RETRAIN":
        system = (
            "You are an MLOps engineer. Write a concise incident report "
            "about model drift. 3 sentences max. Be specific about numbers."
        )
        prompt = f"""
Model drift alert:
- Recommendation: {rec}
- Features with PSI > 0.20: {features[:5]}
- AUC degradation: {auc_drop:.4f}
- Fraud rate shift: {fr_shift:+.1f}%

Write a brief incident report and retraining recommendation.
"""
        try:
            report = _call_groq(prompt, system, api_key, max_tokens=150)
            source = "llm"
        except Exception as e:
            report = _fallback_drift_report(drift_data)
            source = f"fallback ({str(e)[:50]})"
    else:
        report = _fallback_drift_report(drift_data)
        source = "fallback"

    return {"report": report, "source": source, "status": rec}


if __name__ == "__main__":
    # Test fallback (no API key needed)
    print("Testing LLM explainer (fallback mode) …\n")

    test_decision = {
        "fraud_probability":    0.91,
        "iso_anomaly_score":    0.72,
        "fraud_decision":       "CONFIRMED_FRAUD",
        "fraud_risk_level":     "CRITICAL",
        "recommended_action":   "BLOCK",
        "routed_gateway":       "Stripe",
        "expected_success_rate": 0.95,
        "tx_type":              "CARD",
        "TransactionAmt":       4999.99,
    }

    test_shap = [
        {"feature": "log_TransactionAmt", "shap_value": 0.82,  "direction": "↑ increases fraud"},
        {"feature": "hour_raw",           "shap_value": 0.61,  "direction": "↑ increases fraud"},
        {"feature": "P_emaildomain",      "shap_value": 0.44,  "direction": "↑ increases fraud"},
        {"feature": "dow_sin",            "shap_value": -0.31, "direction": "↓ decreases fraud"},
    ]

    fraud_exp   = explain_fraud_decision(test_decision, test_shap)
    routing_exp = explain_routing_decision(test_decision)

    with open(os.path.join(ARTIFACTS_DIR, "drift_recommendation.json")) as f:
        drift_data = json.load(f)
    drift_report = generate_drift_report(drift_data)

    print("── Fraud Explanation ─────────────────────────────────────────")
    print(fraud_exp["explanation"])
    print(f"\n  Source: {fraud_exp['source']}")

    print("\n── Routing Explanation ───────────────────────────────────────")
    print(routing_exp["explanation"])
    print(f"\n  Counterfactual: {routing_exp['counterfactual']}")

    print("\n── Drift Report ──────────────────────────────────────────────")
    print(drift_report["report"])

    print("\n✓  llm_explainer.py complete.")