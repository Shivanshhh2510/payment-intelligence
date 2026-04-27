# PAISA — Payment AI for Smart Authentication

> **A production-grade payment intelligence system** that solves two problems Razorpay tackles in production every day: real-time fraud detection and smart payment gateway routing.

Built on 590,540 real transactions from the IEEE-CIS Fraud Detection dataset. Every design decision is defensible in a senior engineering interview.

---

## What This System Does

Every online payment triggers two decisions in milliseconds:

1. **Is this transaction fraud?** — PAISA scores it using a two-stage ML pipeline (XGBoost + Isolation Forest) with calibrated probabilities and SHAP explainability
2. **Which gateway should process it?** — A Thompson Sampling bandit that has learned optimal routing across Razorpay, PayU, Stripe, and CCAvenue — adapting in real time, including during gateway outages

This is exactly what [Razorpay Optimizer](https://razorpay.com/blog/product/razorpay-optimizer/) does in production. PAISA is a serious engineering attempt to understand and replicate that architecture.

---

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC (temporal holdout) | **0.8763** |
| F1 Score (fraud class) | 0.4620 |
| Decision Threshold | 0.41 |
| Net Benefit vs No-Model | **₹15.7L** on 88k transactions |
| Fraud Value Prevented | ₹23.16L |
| Routing Gain vs Random | **+10.8%** success rate |
| Novel Fraud Caught by IsoForest | **718 cases** XGBoost missed |
| Pipeline Latency | ~43ms end-to-end |
| Training Dataset | 590,540 IEEE-CIS transactions |

---

## Architecture

```
Raw Transaction (JSON)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                   Feature Store                        │
│   367 features · Velocity encoding · Cyclic time       │
│   Card aggregations · Training-serving skew solved     │
└───────────────────────┬───────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌──────────────────┐       ┌─────────────────────┐
│ Isolation Forest │       │  XGBoost Classifier  │
│  (Unsupervised)  │       │  + SMOTE + Isotonic  │
│  Novel fraud     │       │  Calibration + SHAP  │
│  detection       │       │  Known patterns      │
└────────┬─────────┘       └──────────┬──────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
          ┌──────────────────┐
          │  Two-Stage       │
          │  Decision Engine │
          │                  │
          │  CONFIRMED_FRAUD │ ← Both flag it
          │  NOVEL_ANOMALY   │ ← Only IsoForest
          │  KNOWN_FRAUD     │ ← Only XGBoost
          │  LEGITIMATE      │ ← Neither flags
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Thompson        │
          │  Sampling Router │
          │                  │
          │  Fraud-context   │
          │  aware · Adapts  │
          │  to outages      │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  PAISA Analyst   │
          │  (Self-Verifying │
          │   AI Layer)      │
          └──────────────────┘
```

---

## Key Engineering Decisions

### 1. Chronological Train/Test Split (No Data Leakage)
```
Train (70%)  │  Val (15%)  │  Test (15%)
─────────────┼─────────────┼─────────────
TransactionDT order — never random
```
Random splits leak future fraud patterns into training. Every number in this project was computed on transactions the model never saw during training — simulating real production conditions.

### 2. Two-Stage Detection
XGBoost alone is a supervised model — it only catches fraud patterns it has seen before. Isolation Forest is unsupervised — it flags statistically unusual transactions regardless of whether that pattern exists in labeled data.

Result: **718 additional fraud cases caught** that XGBoost missed entirely.

### 3. Isotonic Probability Calibration
Raw XGBoost scores are not probabilities. A score of 0.7 ≠ 70% chance of fraud. Isotonic regression (proven superior to Platt scaling for tree models) maps raw scores to true calibrated probabilities.

Brier Score improvement: `0.0262 → 0.0243`

### 4. SMOTE Applied Inside CV Folds Only
SMOTE was applied after splitting, never before — preventing synthetic fraud samples from appearing in both train and validation sets (a common leakage mistake in class-imbalance handling).

### 5. Thompson Sampling — Beta-Bernoulli Conjugate
Each (gateway, transaction_type) pair maintains a Beta distribution. Posterior update is O(1). After 10,000 simulated transactions, the bandit converged to true optimal gateways:

| Transaction Type | Optimal Gateway | Learned SR | True SR |
|-----------------|-----------------|------------|---------|
| UPI | Razorpay | 94.2% | 94.0% |
| CARD | Stripe | 94.4% | 95.0% |
| NETBANKING | CCAvenue | 91.9% | 92.0% |
| WALLET | Razorpay | 91.2% | 91.0% |

### 6. Non-Stationary Simulation (Gateway Outage)
The bandit simulation includes a Razorpay UPI outage (steps 4000–6000, success rate drops to 45%). Thompson Sampling detects and adapts. Round Robin suffers the full impact. This is the real-world scenario.

### 7. Feature Store (Training-Serving Skew Solved)
A dedicated `FeatureStore` class loads once and transforms raw transaction dicts into model-ready feature vectors at inference time — using identical transformations to training. Training-serving skew is one of the most common production ML failures.

### 8. Business Impact in ₹ (Not Just F1)
Cost parameters sourced from RBI data:
- Chargeback cost: ₹2,400
- Lost revenue (blocked legit tx): ₹1,200
- Manual review cost: ₹200

At the optimal threshold (0.41), the system produces **₹15.7L net benefit** on 88,581 test transactions vs a no-model baseline.

---

## Platform Performance:

<img width="1568" height="759" alt="image" src="https://github.com/user-attachments/assets/e4ec3740-315d-4195-8b50-02374d92db6b" />

<img width="1568" height="476" alt="image" src="https://github.com/user-attachments/assets/f440cf4d-30e9-41a4-9519-2ae298142777" />

<img width="1568" height="518" alt="image" src="https://github.com/user-attachments/assets/450c3b35-81d3-4f09-8470-9c0b73952c93" />

<img width="1568" height="316" alt="image" src="https://github.com/user-attachments/assets/64923150-2fe9-4df5-84c7-7b500011f06c" />







## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15 · TypeScript · Tailwind CSS · Recharts |
| Backend API | FastAPI · Python · Uvicorn |
| Fraud ML | XGBoost · Isolation Forest · SMOTE · SHAP · Isotonic Regression |
| Routing | Thompson Sampling (NumPy) · LinUCB |
| MLOps | MLflow · Model Registry · Staging/Production workflow |
| Data | Pandas · NumPy · PyArrow |
| Explainability | SHAP · Calibration curves · PR curves |
| AI Analyst | Groq (Llama 3.3 70B) · Self-healing verification |

---

## Project Structure

```
payment_intelligence/
├── api/
│   └── main.py                    ← FastAPI — all ML engines as REST endpoints
├── frontend/                      ← Next.js 15 + TypeScript
│   ├── app/
│   │   ├── page.tsx               ← Main layout — Transaction + Analyst tabs
│   │   └── globals.css
│   ├── components/
│   │   ├── Sidebar.tsx            ← Live metrics, model health
│   │   ├── TransactionPanel.tsx   ← Transaction form + decision output
│   │   ├── AnalystPanel.tsx       ← AI chat interface
│   │   ├── DecisionCard.tsx       ← Risk display + routing result
│   │   ├── InlineChart.tsx        ← Recharts bar/pie charts
│   │   └── ChatMessage.tsx        ← Message bubbles + verified badge
│   └── lib/
│       ├── api.ts                 ← All FastAPI calls
│       └── types.ts               ← Shared TypeScript types
├── src/
│   ├── fraud_engine/
│   │   ├── features.py            ← Feature engineering pipeline
│   │   ├── train.py               ← XGBoost + SMOTE + MLflow
│   │   ├── anomaly.py             ← Isolation Forest two-stage
│   │   ├── calibrate.py           ← Isotonic calibration + business impact
│   │   ├── drift_monitor.py       ← PSI + concept drift + retraining rec
│   │   ├── feature_store.py       ← Real-time inference feature computation
│   │   ├── velocity.py            ← Card velocity encoding
│   │   ├── explain.py             ← SHAP plots
│   │   ├── registry.py            ← MLflow model registry
│   │   └── predict.py             ← Inference class
│   ├── routing_engine/
│   │   └── bandit.py              ← Thompson Sampling + LinUCB + outage sim
│   ├── pipeline.py                ← Unified fraud → routing pipeline
│   └── utils/
│       └── eda.py                 ← Data loading + EDA
├── data/
│   ├── raw/                       ← IEEE-CIS CSVs (not committed)
│   └── processed/                 ← Parquet files, SHAP plots
├── mlflow_artifacts/              ← Models, calibrator, metrics JSON
└── requirements.txt
```

---

## Setup & Running

### Prerequisites
- Python 3.10+
- Node.js 18+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- IEEE-CIS dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

### 1. Clone and set up Python environment
```bash
git clone https://github.com/yourusername/payment-intelligence
cd payment_intelligence
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Place dataset
```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

### 3. Run the ML pipeline (one time)
```bash
python src/utils/eda.py
python src/fraud_engine/features.py
python src/fraud_engine/train.py
python src/fraud_engine/explain.py
python src/fraud_engine/anomaly.py
python src/fraud_engine/calibrate.py
python src/fraud_engine/drift_monitor.py
python src/routing_engine/bandit.py
python src/fraud_engine/registry.py --promote
```

### 4. Set up environment variables
```bash
# payment_intelligence/.env
GROQ_API_KEY=gsk_your_key_here
```

### 5. Start the FastAPI backend
```bash
uvicorn api.main:app --reload --port 8001
```

### 6. Start the Next.js frontend
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Pipeline status |
| GET | `/api/metrics` | Model metrics + business impact |
| GET | `/api/facts` | Pre-computed dataset statistics |
| POST | `/api/transaction/score` | Score a transaction through full pipeline |
| POST | `/api/analyst/chat` | Self-verifying AI analyst chat |

### Example: Score a transaction
```bash
curl -X POST http://localhost:8001/api/transaction/score \
  -H "Content-Type: application/json" \
  -d '{
    "tx_type": "UPI",
    "amount": 4999,
    "card_type": "debit",
    "network": "visa",
    "email": "gmail.com",
    "device": "mobile",
    "hour": 2,
    "card_id": 9999
  }'
```

```json
{
  "fraud_probability": 0.674,
  "iso_anomaly_score": 0.0,
  "fraud_decision": "KNOWN_FRAUD",
  "fraud_risk_level": "HIGH",
  "recommended_action": "REVIEW",
  "routed_gateway": "Razorpay",
  "routing_reason": "Fraud-risk override (HIGH) → strongest-auth gateway",
  "expected_success_rate": 0.859,
  "processing_time_ms": 43.2
}
```

---

## Dataset

**IEEE-CIS Fraud Detection** — Kaggle Competition Dataset
Provided by Vesta Corporation (real-world e-commerce transactions)

- 590,540 transactions · 394 features
- 3.5% fraud rate (20,663 fraud · 569,877 legitimate)
- 27:1 class imbalance
- Features: transaction amount, card details, email domains, device info, C/D/V/id columns

---

## Model Performance Details

### Why AUC-ROC dropped from ~0.94 to 0.8763
The first run used random CV splits — inflated by temporal leakage. After switching to chronological splits (train on past, test on future), AUC dropped to 0.8763. **This is the honest number.** It reflects real production performance where the model only ever sees historical data.

### Threshold Selection
Default threshold of 0.5 is wrong for imbalanced fraud data. The optimal threshold (0.41) was selected by maximizing F1 on the validation set — not the test set — to prevent threshold overfitting.

### Why Precision-Recall > ROC for Fraud
With 3.5% fraud rate, a model predicting all legitimate achieves 96.5% accuracy. ROC is misleading. Precision-Recall curve reveals the real tradeoff — AP score: 0.4627.

---

## Drift Monitoring

The system monitors three drift types across time-ordered batches:

| Drift Type | Method | Trigger |
|-----------|--------|---------|
| Covariate drift | PSI per feature | PSI > 0.20 |
| Prior probability shift | Fraud rate over time | >30% change |
| Concept drift | AUC degradation per batch | >0.05 drop |

Current status: **5 features with PSI > 0.20** · Fraud rate shifted +31.7% · Retraining recommended.

---

## References

- [Razorpay Smart Routing Blog](https://razorpay.com/blog/product/razorpay-optimizer/)
- [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection)
- [Thompson Sampling — Russo et al. 2018](https://arxiv.org/abs/1209.3352)
- [Probability Calibration — Niculescu-Mizil & Caruana 2005](https://dl.acm.org/doi/10.1145/1102351.1102430)
- [SHAP — Lundberg & Lee 2017](https://arxiv.org/abs/1705.07874)

---

## Author

**Shivansh Mishra**
Final Year B.Tech CSE (Data Science) · Bennett University
[LinkedIn](https://www.linkedin.com/in/shivansh-mishra-855a0925a/) · [GitHub](https://github.com/Shivanshhh2510)

---

*PAISA is a portfolio project demonstrating production ML engineering principles applied to a real fintech problem. It is not affiliated with Razorpay.*
