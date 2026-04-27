"""
api/main.py
────────────
PAISA FastAPI Backend
Exposes all ML engines as clean REST endpoints.
Run: uvicorn api.main:app --reload --port 8001
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv


from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Preload pipeline on startup ───────────────────────────────────────────────
pipeline_instance = None
facts_cache       = None
metrics_cache     = None
impact_cache      = None
router_state_cache = None
drift_cache       = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, facts_cache, metrics_cache
    global impact_cache, router_state_cache, drift_cache

    logger.info("Loading PAISA pipeline...")
    try:
        from src.fraud_engine.calibrate import IsotonicCalibrator  # noqa
        from src.pipeline import get_pipeline
        pipeline_instance = get_pipeline()
        if not pipeline_instance._loaded:
            pipeline_instance.load()
        logger.info("Pipeline loaded.")
    except Exception as e:
        logger.error(f"Pipeline load failed: {e}")

    # Cache static data
    ARTIFACTS = os.path.join(ROOT, "mlflow_artifacts")
    PROCESSED = os.path.join(ROOT, "data", "processed")

    for path, key in [
        (os.path.join(ARTIFACTS, "test_metrics.json"),          "metrics"),
        (os.path.join(ARTIFACTS, "business_impact.json"),       "impact"),
        (os.path.join(ARTIFACTS, "router_state.json"),          "router"),
        (os.path.join(ARTIFACTS, "drift_recommendation.json"),  "drift"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if key == "metrics":   metrics_cache       = data
            if key == "impact":    impact_cache        = data
            if key == "router":    router_state_cache  = data
            if key == "drift":     drift_cache         = data

    # Compute dataset facts
    try:
        import pandas as pd
        import numpy as np
        raw = pd.read_parquet(os.path.join(PROCESSED, "raw_train.parquet"))
        facts = {
            "total":       int(len(raw)),
            "fraud_count": int(raw["isFraud"].sum()),
            "fraud_pct":   round(float(raw["isFraud"].mean()) * 100, 2),
            "legit_count": int(len(raw) - raw["isFraud"].sum()),
            "avg_amt":     round(float(raw["TransactionAmt"].mean()), 2),
            "med_amt":     round(float(raw["TransactionAmt"].median()), 2),
            "imbalance":   round(float((1 - raw["isFraud"].mean()) / raw["isFraud"].mean()), 1),
        }
        for col, key in [("card6","by_card_type"),("card4","by_network"),
                          ("DeviceType","by_device"),("ProductCD","by_product")]:
            if col in raw.columns:
                g = raw.groupby(col)["isFraud"].agg(["mean","count"])
                facts[key] = {
                    str(k): {"pct": round(float(v["mean"])*100, 2), "n": int(v["count"])}
                    for k, v in g.iterrows()
                }
        if "P_emaildomain" in raw.columns:
            g = raw.groupby("P_emaildomain")["isFraud"].agg(["mean","count"])
            g = g[g["count"] > 500].sort_values("mean", ascending=False).head(8)
            facts["by_email"] = {
                str(k): {"pct": round(float(v["mean"])*100, 2), "n": int(v["count"])}
                for k, v in g.iterrows()
            }
        raw["_bucket"] = pd.cut(raw["TransactionAmt"],
            bins=[0,50,100,500,1000,50000],
            labels=["₹0-50","₹51-100","₹101-500","₹501-1k","₹1k+"])
        g = raw.groupby("_bucket", observed=True)["isFraud"].agg(["mean","count"])
        facts["by_amount"] = {
            str(k): {"pct": round(float(v["mean"])*100, 2), "n": int(v["count"])}
            for k, v in g.iterrows()
        }
        facts_cache = facts
        logger.info("Dataset facts cached.")
    except Exception as e:
        logger.error(f"Facts cache failed: {e}")
        facts_cache = {}

    yield
    logger.info("PAISA API shutting down.")


app = FastAPI(
    title="PAISA API",
    description="Payment AI for Smart Authentication",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    tx_type:    str = "CARD"
    amount:     float = 499.0
    card_type:  str = "debit"
    network:    str = "visa"
    email:      str = "gmail.com"
    device:     str = "desktop"
    hour:       int = 14
    card_id:    int = 2755

class ChatRequest(BaseModel):
    message:  str
    history:  list = []

class TransactionResponse(BaseModel):
    fraud_probability:     float
    iso_anomaly_score:     float
    fraud_decision:        str
    fraud_risk_level:      str
    recommended_action:    str
    routed_gateway:        str
    routing_reason:        str
    expected_success_rate: float
    processing_time_ms:    float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "pipeline_loaded": pipeline_instance is not None and pipeline_instance._loaded,
        "facts_loaded":    facts_cache is not None,
    }


@app.get("/api/metrics")
def get_metrics():
    return {
        "metrics":        metrics_cache or {},
        "impact":         impact_cache  or {},
        "drift":          drift_cache   or {},
        "router_state":   router_state_cache or {},
    }


@app.get("/api/facts")
def get_facts():
    return facts_cache or {}


@app.post("/api/transaction/score")
def score_transaction(req: TransactionRequest):
    if pipeline_instance is None or not pipeline_instance._loaded:
        raise HTTPException(503, "Pipeline not ready")

    tx = {
        "TransactionID":  "API_001",
        "TransactionDT":  req.hour * 3600 + 9_000_000,
        "TransactionAmt": req.amount,
        "ProductCD":      "W",
        "card1":          req.card_id,
        "card4":          req.network,
        "card6":          req.card_type,
        "P_emaildomain":  req.email,
        "DeviceType":     req.device,
    }

    try:
        decision = pipeline_instance.process(tx, req.tx_type)
        return decision.to_dict()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/analyst/chat")
def chat(req: ChatRequest):
    import re
    import requests as req_lib

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(503, "GROQ_API_KEY not configured")

    facts   = facts_cache   or {}
    metrics = metrics_cache or {}
    impact  = impact_cache  or {}
    router  = router_state_cache or {}

    # Intent detection
    q = req.message.lower()
    if any(w in q for w in ["fraud rate","pattern","which","highest","email","card","device","amount","dataset","what is","explain"]):
        intent = "data"
    elif any(w in q for w in ["model","auc","accuracy","shap","xgboost","smote","calibrat","feature"]):
        intent = "model"
    elif any(w in q for w in ["route","gateway","razorpay","payu","stripe","ccavenue","routing","upi","netbanking"]):
        intent = "routing"
    elif any(w in q for w in ["if","what if","threshold","lower","raise","simulate"]):
        intent = "whatif"
    else:
        intent = "general"

    system = f"""You are PAISA, a fraud analyst AI for a payment intelligence system trained on the IEEE-CIS dataset.

RULES:
- Only use numbers from this context. Never invent statistics.
- Explain in plain English. Assume the user may not know fintech.
- Be concise: 3-5 sentences for simple questions, up to 8 for complex.
- If unsure, say so honestly.

DATASET (IEEE-CIS, 590k real transactions):
- Total: {facts.get('total', 590540):,} transactions
- Fraud: {facts.get('fraud_count', 20663):,} ({facts.get('fraud_pct', 3.5)}%)
- Legitimate: {facts.get('legit_count', 569877):,}
- Imbalance: {facts.get('imbalance', 27.6)}:1
- Avg amount: ₹{facts.get('avg_amt', 135)} | Median: ₹{facts.get('med_amt', 68)}

MODEL (chronological holdout):
- AUC-ROC: {metrics.get('test_auc', 0.8763):.4f}
- F1 (fraud): {metrics.get('test_f1', 0.462):.4f}
- Precision: {metrics.get('test_precision', 0.591):.4f}
- Recall: {metrics.get('test_recall', 0.379):.4f}
- Threshold: {metrics.get('threshold', 0.41):.2f}

BUSINESS IMPACT:
- Fraud caught: {impact.get('fraud_caught', 965):,}
- Value prevented: ₹{impact.get('fraud_value_prevented_inr', 2316000):,}
- Net benefit: ₹{impact.get('net_benefit_inr', 1570000):,}
- ROI: {impact.get('roi_percent', 21.2)}%
- Novel fraud caught by Isolation Forest: 718 cases
- Routing gain vs random: +10.8%
"""

    if intent == "data":
        system += "\nFRAUD PATTERNS:\n"
        for key, label in [("by_card_type","Card type"),("by_network","Card network"),
                            ("by_device","Device"),("by_product","Product"),
                            ("by_email","Email domain"),("by_amount","Amount bucket")]:
            if key in facts:
                system += f"{label}: {json.dumps({k: v['pct'] for k,v in facts[key].items()})}\n"

    if intent == "routing":
        system += "\nROUTING (learned success rates after 10k simulated transactions):\n"
        for tx_type, gws in router.items():
            system += f"{tx_type}: " + ", ".join(
                f"{g}={v['posterior_mean']:.3f}" for g,v in gws.items()) + "\n"

    if intent == "model":
        system += "\nMODEL DETAILS:\n- XGBoost + SMOTE + isotonic calibration\n- Two-stage: XGBoost (known fraud) + Isolation Forest (novel fraud)\n- SHAP explainability\n- Chronological 70/15/15 split (no leakage)\n"

    if intent == "whatif":
        system += f"\nTHRESHOLD SENSITIVITY:\n- Current: {metrics.get('threshold', 0.41):.2f}\n- Lower = more fraud caught, more legitimate blocked\n- Business optimal ~0.35, F1 optimal ~0.31\n"

    messages = [{"role": "system", "content": system}]
    for m in req.history[-8:]:
        messages.append({"role": m.get("role","user"), "content": m.get("content","")})
    messages.append({"role": "user", "content": req.message})

    try:
        resp = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages,
                  "max_tokens": 500, "temperature": 0.3},
            timeout=15,
        )
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(502, f"LLM error: {str(e)[:100]}")

    # Self-healing verification
    verified = True
    corrections = []
    for m in re.finditer(r'(\d+\.?\d*)\s*%\s*(?:fraud)', response_text.lower()):
        claimed = float(m.group(1))
        actual  = facts.get("fraud_pct", 3.5)
        if abs(claimed - actual) > 1.5:
            corrections.append(f"Fraud rate is {actual}%, not {claimed}%.")
            verified = False

    if not verified and corrections:
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content":
            f"Correction: {' '.join(corrections)} Regenerate with correct numbers."})
        try:
            resp2 = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": messages,
                      "max_tokens": 500, "temperature": 0.3},
                timeout=15,
            )
            resp2.raise_for_status()
            response_text = resp2.json()["choices"][0]["message"]["content"].strip()
            verified = True
        except Exception:
            pass

    # Generate follow-up suggestions
    suggestions = []
    try:
        sug_resp = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "Generate exactly 3 short follow-up questions (one per line, no numbering) about a payment fraud detection system based on this answer."},
                    {"role": "user", "content": response_text[:300]},
                ],
                "max_tokens": 100, "temperature": 0.3,
            },
            timeout=10,
        )
        sug_resp.raise_for_status()
        raw_sug = sug_resp.json()["choices"][0]["message"]["content"].strip()
        suggestions = [s.strip().strip("-").strip() for s in raw_sug.split("\n") if s.strip()][:3]
    except Exception:
        suggestions = [
            "Which card type has the highest fraud rate?",
            "How does Thompson Sampling routing work?",
            "What is the net benefit in rupees?",
        ]

    # Determine if chart should be rendered
    chart_type = None
    q = req.message.lower()
    if any(w in q for w in ["card type","credit","debit"]):
        chart_type = "by_card_type"
    elif any(w in q for w in ["network","visa","mastercard","discover"]):
        chart_type = "by_network"
    elif any(w in q for w in ["device","mobile","desktop"]):
        chart_type = "by_device"
    elif any(w in q for w in ["email","domain"]):
        chart_type = "by_email"
    elif any(w in q for w in ["amount","₹","spend"]):
        chart_type = "by_amount"
    elif any(w in q for w in ["gateway","routing","route","upi","netbank","wallet"]):
        chart_type = "gateway"

    chart_data = None
    if chart_type and chart_type != "gateway" and chart_type in facts:
        data = facts[chart_type]
        chart_data = {
            "type":   "bar",
            "labels": list(data.keys()),
            "values": [data[k]["pct"] for k in data],
            "title":  f"Fraud Rate — {chart_type.replace('by_','').replace('_',' ').title()} (%)",
        }
    elif chart_type == "gateway":
        tx = "UPI"
        for t in ["CARD","NETBANKING","WALLET","UPI"]:
            if t.lower() in q: tx = t; break
        gd = router.get(tx, {})
        if gd:
            chart_data = {
                "type":       "gateway",
                "tx_type":    tx,
                "labels":     list(gd.keys()),
                "values":     [gd[g]["posterior_mean"]*100 for g in gd],
                "true_rates": [gd[g]["true_rate"]*100 for g in gd],
                "title":      f"Gateway Success Rate — {tx} (%)",
            }

    return {
        "response":    response_text,
        "verified":    verified,
        "suggestions": suggestions,
        "chart":       chart_data,
        "intent":      intent,
    }