"""
src/routing_engine/bandit.py
─────────────────────────────
Dual bandit implementation:
  1. Thompson Sampling  — Beta-Bernoulli, context-free
  2. LinUCB             — Contextual bandit, uses transaction features

Key additions vs v1:
  - Cumulative regret tracking (academic rigour)
  - Non-stationary simulation (gateway outage mid-run)
  - Head-to-head comparison: Round Robin vs Thompson vs LinUCB
  - Confidence intervals on posterior estimates
  - Context features for LinUCB: [log_amt, hour_sin, hour_cos, is_high_value]
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum
from scipy.stats import beta as beta_dist

ARTIFACTS_DIR = "mlflow_artifacts"
PLOTS_DIR     = "data/processed/routing_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ── transaction types ─────────────────────────────────────────────────────────
class TxType(str, Enum):
    UPI        = "UPI"
    CARD       = "CARD"
    NETBANKING = "NETBANKING"
    WALLET     = "WALLET"


# ── gateway success rate matrix ───────────────────────────────────────────────
GATEWAY_TRUE_RATES: Dict[str, Dict[TxType, float]] = {
    "Razorpay": {TxType.UPI: 0.94, TxType.CARD: 0.87, TxType.NETBANKING: 0.83, TxType.WALLET: 0.91},
    "PayU":     {TxType.UPI: 0.78, TxType.CARD: 0.91, TxType.NETBANKING: 0.88, TxType.WALLET: 0.72},
    "Stripe":   {TxType.UPI: 0.65, TxType.CARD: 0.95, TxType.NETBANKING: 0.71, TxType.WALLET: 0.82},
    "CCAvenue": {TxType.UPI: 0.81, TxType.CARD: 0.79, TxType.NETBANKING: 0.92, TxType.WALLET: 0.88},
}

GATEWAYS = list(GATEWAY_TRUE_RATES.keys())

# Outage scenario: Razorpay UPI drops to 0.45 between step 4000-6000
OUTAGE_SCENARIO = {
    "gateway": "Razorpay",
    "tx_type": TxType.UPI,
    "start":   4000,
    "end":     6000,
    "degraded_rate": 0.45,
}


# ── Thompson Sampling ─────────────────────────────────────────────────────────
@dataclass
class BetaArm:
    name:    str
    tx_type: TxType
    alpha:   float = 1.0
    beta_:   float = 1.0

    @property
    def true_rate(self) -> float:
        return GATEWAY_TRUE_RATES[self.name][self.tx_type]

    def sample(self) -> float:
        return float(np.random.beta(self.alpha, self.beta_))

    def update(self, success: bool) -> None:
        if success: self.alpha += 1
        else:       self.beta_ += 1

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_)

    @property
    def credible_interval_95(self) -> tuple:
        lo = beta_dist.ppf(0.025, self.alpha, self.beta_)
        hi = beta_dist.ppf(0.975, self.alpha, self.beta_)
        return round(float(lo), 4), round(float(hi), 4)

    @property
    def n_trials(self) -> int:
        return int(self.alpha + self.beta_ - 2)


class ThompsonRouter:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.arms: Dict[tuple, BetaArm] = {
            (gw, tx): BetaArm(name=gw, tx_type=tx)
            for gw in GATEWAYS for tx in TxType
        }
        self.history: List[dict] = []

    def route(self, tx_type: TxType) -> str:
        return max(GATEWAYS, key=lambda gw: self.arms[(gw, tx_type)].sample())

    def simulate_transaction(self, tx_type: TxType, step: int,
                              true_rates_override: dict = None) -> dict:
        chosen   = self.route(tx_type)
        true_p   = (true_rates_override or {}).get((chosen, tx_type),
                    GATEWAY_TRUE_RATES[chosen][tx_type])
        success  = np.random.rand() < true_p
        self.arms[(chosen, tx_type)].update(success)

        # Best possible rate at this step (for regret)
        best_rate = max(
            (true_rates_override or {}).get((gw, tx_type), GATEWAY_TRUE_RATES[gw][tx_type])
            for gw in GATEWAYS
        )
        regret = best_rate - true_p

        record = {
            "step": step, "tx_type": tx_type.value, "gateway": chosen,
            "success": int(success), "true_p": true_p, "regret": regret,
        }
        self.history.append(record)
        return record

    def get_state(self) -> dict:
        state = {}
        for tx in TxType:
            state[tx.value] = {
                gw: {
                    "alpha":            self.arms[(gw, tx)].alpha,
                    "beta":             self.arms[(gw, tx)].beta_,
                    "posterior_mean":   round(self.arms[(gw, tx)].posterior_mean, 4),
                    "credible_interval_95": self.arms[(gw, tx)].credible_interval_95,
                    "n_trials":         self.arms[(gw, tx)].n_trials,
                    "true_rate":        GATEWAY_TRUE_RATES[gw][tx],
                }
                for gw in GATEWAYS
            }
        return state

    def summary(self) -> pd.DataFrame:
        rows = []
        for (gw, tx), arm in self.arms.items():
            lo, hi = arm.credible_interval_95
            rows.append({
                "Gateway": gw, "TxType": tx.value,
                "True Rate": arm.true_rate,
                "Posterior Mean": round(arm.posterior_mean, 4),
                "95% CI Lower": lo, "95% CI Upper": hi,
                "Trials": arm.n_trials,
                "Error": round(abs(arm.posterior_mean - arm.true_rate), 4),
            })
        return pd.DataFrame(rows).sort_values(["TxType", "True Rate"], ascending=[True, False])


# ── LinUCB Contextual Bandit ───────────────────────────────────────────────────
class LinUCBRouter:
    """
    LinUCB (Linear Upper Confidence Bound) contextual bandit.
    Each arm maintains a linear model: reward = theta^T * context
    Context features: [log_amt, hour_sin, hour_cos, is_high_value, bias]

    alpha parameter controls exploration-exploitation tradeoff.
    Higher alpha → more exploration.
    """

    def __init__(self, n_features: int = 5, alpha: float = 0.5, seed: int = 42):
        np.random.seed(seed)
        self.alpha      = alpha
        self.d          = n_features
        self.history:   List[dict] = []

        # Per (gateway, tx_type) arm parameters
        self.A: Dict[tuple, np.ndarray] = {}   # d×d identity matrix
        self.b: Dict[tuple, np.ndarray] = {}   # d-dim zero vector

        for gw in GATEWAYS:
            for tx in TxType:
                key       = (gw, tx)
                self.A[key] = np.eye(self.d)
                self.b[key] = np.zeros(self.d)

    def _get_context(self, tx_type: TxType, amt: float, hour: float) -> np.ndarray:
        """Build context vector for this transaction."""
        return np.array([
            np.log1p(amt) / 10.0,           # normalised log amount
            np.sin(2 * np.pi * hour / 24),   # hour sin
            np.cos(2 * np.pi * hour / 24),   # hour cos
            float(amt > 500),                # is high value
            1.0,                             # bias term
        ])

    def _ucb_score(self, key: tuple, context: np.ndarray) -> float:
        A_inv  = np.linalg.inv(self.A[key])
        theta  = A_inv @ self.b[key]
        exploit = float(theta @ context)
        explore = float(self.alpha * np.sqrt(context @ A_inv @ context))
        return exploit + explore

    def route(self, tx_type: TxType, amt: float = 100.0, hour: float = 12.0) -> str:
        context = self._get_context(tx_type, amt, hour)
        scores  = {gw: self._ucb_score((gw, tx_type), context) for gw in GATEWAYS}
        return max(scores, key=scores.__getitem__)

    def update(self, gateway: str, tx_type: TxType,
               amt: float, hour: float, success: bool) -> None:
        key     = (gateway, tx_type)
        context = self._get_context(tx_type, amt, hour)
        self.A[key] += np.outer(context, context)
        self.b[key] += int(success) * context

    def simulate_transaction(self, tx_type: TxType, step: int,
                              amt: float = None, hour: float = None,
                              true_rates_override: dict = None) -> dict:
        amt  = amt  or float(np.random.lognormal(4.5, 1.0))
        hour = hour or float(np.random.uniform(0, 24))

        chosen  = self.route(tx_type, amt, hour)
        true_p  = (true_rates_override or {}).get((chosen, tx_type),
                   GATEWAY_TRUE_RATES[chosen][tx_type])
        success = np.random.rand() < true_p
        self.update(chosen, tx_type, amt, hour, success)

        best_rate = max(
            (true_rates_override or {}).get((gw, tx_type), GATEWAY_TRUE_RATES[gw][tx_type])
            for gw in GATEWAYS
        )
        regret = best_rate - true_p

        record = {
            "step": step, "tx_type": tx_type.value, "gateway": chosen,
            "success": int(success), "true_p": true_p, "regret": regret,
            "amt": amt, "hour": hour,
        }
        self.history.append(record)
        return record


# ── Round Robin baseline ───────────────────────────────────────────────────────
class RoundRobinRouter:
    def __init__(self):
        self._idx = 0

    def route(self, tx_type: TxType) -> str:
        gw = GATEWAYS[self._idx % len(GATEWAYS)]
        self._idx += 1
        return gw


# ── Simulation ────────────────────────────────────────────────────────────────
def run_simulation(n_steps: int = 10_000,
                   seed: int = 42,
                   with_outage: bool = True) -> tuple:
    np.random.seed(seed)
    tx_types = list(TxType)

    ts  = ThompsonRouter(seed=seed)
    lcb = LinUCBRouter(alpha=0.5, seed=seed)
    rr  = RoundRobinRouter()

    rr_history  = []

    print(f"Running simulation: {n_steps:,} transactions "
          f"{'(with gateway outage)' if with_outage else ''} …")

    for step in range(1, n_steps + 1):
        tx_type = tx_types[np.random.randint(len(tx_types))]
        amt     = float(np.random.lognormal(4.5, 1.0))
        hour    = float(np.random.uniform(0, 24))

        # Build true rate override for outage scenario
        override = {}
        if with_outage and OUTAGE_SCENARIO["start"] <= step <= OUTAGE_SCENARIO["end"]:
            gw = OUTAGE_SCENARIO["gateway"]
            tx = OUTAGE_SCENARIO["tx_type"]
            override[(gw, tx)] = OUTAGE_SCENARIO["degraded_rate"]

        ts.simulate_transaction(tx_type, step, override)
        lcb.simulate_transaction(tx_type, step, amt, hour, override)

        # Round Robin
        gw_rr   = rr.route(tx_type)
        true_p  = override.get((gw_rr, tx_type), GATEWAY_TRUE_RATES[gw_rr][tx_type])
        success = int(np.random.rand() < true_p)
        best_p  = max(override.get((gw, tx_type), GATEWAY_TRUE_RATES[gw][tx_type]) for gw in GATEWAYS)
        rr_history.append({
            "step": step, "tx_type": tx_type.value, "gateway": gw_rr,
            "success": success, "true_p": true_p, "regret": best_p - true_p,
        })

    ts_df  = pd.DataFrame(ts.history)
    lcb_df = pd.DataFrame(lcb.history)
    rr_df  = pd.DataFrame(rr_history)

    print(f"  Thompson Sampling SR:  {ts_df['success'].mean()*100:.2f}%")
    print(f"  LinUCB SR:             {lcb_df['success'].mean()*100:.2f}%")
    print(f"  Round Robin SR:        {rr_df['success'].mean()*100:.2f}%")
    print(f"  TS gain vs RR:         +{(ts_df['success'].mean()-rr_df['success'].mean())/rr_df['success'].mean()*100:.1f}%")
    print(f"  LinUCB gain vs RR:     +{(lcb_df['success'].mean()-rr_df['success'].mean())/rr_df['success'].mean()*100:.1f}%")

    return ts, lcb, ts_df, lcb_df, rr_df


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_cumulative_regret(ts_df, lcb_df, rr_df, outage: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(ts_df["step"],  ts_df["regret"].cumsum(),
            color="steelblue", linewidth=2, label="Thompson Sampling")
    ax.plot(lcb_df["step"], lcb_df["regret"].cumsum(),
            color="seagreen",  linewidth=2, label="LinUCB (Contextual)")
    ax.plot(rr_df["step"],  rr_df["regret"].cumsum(),
            color="crimson",   linewidth=2, linestyle="--", label="Round Robin")

    if outage:
        ax.axvspan(OUTAGE_SCENARIO["start"], OUTAGE_SCENARIO["end"],
                   alpha=0.15, color="orange", label="Gateway outage (Razorpay UPI)")

    ax.set_xlabel("Transaction #")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret — Thompson Sampling vs LinUCB vs Round Robin",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cumulative_regret.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_success_rate(ts_df, lcb_df, rr_df, window: int = 200, outage: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(ts_df["step"],  ts_df["success"].rolling(window).mean(),
            color="steelblue", linewidth=2,   label="Thompson Sampling")
    ax.plot(lcb_df["step"], lcb_df["success"].rolling(window).mean(),
            color="seagreen",  linewidth=2,   label="LinUCB (Contextual)")
    ax.plot(rr_df["step"],  rr_df["success"].rolling(window).mean(),
            color="crimson",   linewidth=2,   linestyle="--", label="Round Robin")

    if outage:
        ax.axvspan(OUTAGE_SCENARIO["start"], OUTAGE_SCENARIO["end"],
                   alpha=0.15, color="orange", label="Gateway outage")

    ax.set_xlabel("Transaction #")
    ax.set_ylabel(f"Rolling Success Rate (window={window})")
    ax.set_title("Routing Success Rate — All Algorithms", fontweight="bold")
    ax.set_ylim(0.6, 1.0)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "success_rate_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_confidence_intervals(ts_router: ThompsonRouter) -> None:
    """Plot 95% credible intervals for each gateway per tx type."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    for ax, tx in zip(axes, TxType):
        gws, means, lows, highs, true_rates = [], [], [], [], []
        for gw in GATEWAYS:
            arm = ts_router.arms[(gw, tx)]
            lo, hi = arm.credible_interval_95
            gws.append(gw)
            means.append(arm.posterior_mean)
            lows.append(arm.posterior_mean - lo)
            highs.append(hi - arm.posterior_mean)
            true_rates.append(arm.true_rate)

        x = range(len(gws))
        ax.bar(x, means, color=colors[:len(gws)], alpha=0.7, label="Posterior mean")
        ax.errorbar(x, means, yerr=[lows, highs], fmt="none",
                    color="black", capsize=5, linewidth=2, label="95% CI")
        ax.scatter(x, true_rates, color="black", marker="x", s=80,
                   zorder=5, label="True rate")
        ax.set_xticks(list(x))
        ax.set_xticklabels(gws, rotation=20, fontsize=8)
        ax.set_title(tx.value, fontweight="bold")
        ax.set_ylim(0, 1.1)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    plt.suptitle("Thompson Sampling: 95% Credible Intervals vs True Rates",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "confidence_intervals.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_outage_recovery(ts_df, lcb_df, rr_df) -> None:
    """Zoom into outage window to show adaptation speed."""
    start = OUTAGE_SCENARIO["start"] - 500
    end   = OUTAGE_SCENARIO["end"]   + 500

    fig, ax = plt.subplots(figsize=(11, 5))
    window = 50

    # Filter to UPI only for clarity
    ts_upi  = ts_df[ts_df["tx_type"]  == "UPI"]
    lcb_upi = lcb_df[lcb_df["tx_type"] == "UPI"]
    rr_upi  = rr_df[rr_df["tx_type"]  == "UPI"]

    ax.plot(ts_upi["step"],  ts_upi["success"].rolling(window).mean(),
            color="steelblue", linewidth=2, label="Thompson Sampling")
    ax.plot(lcb_upi["step"], lcb_upi["success"].rolling(window).mean(),
            color="seagreen",  linewidth=2, label="LinUCB")
    ax.plot(rr_upi["step"],  rr_upi["success"].rolling(window).mean(),
            color="crimson",   linewidth=2, linestyle="--", label="Round Robin")

    ax.axvspan(OUTAGE_SCENARIO["start"], OUTAGE_SCENARIO["end"],
               alpha=0.2, color="orange")
    ax.axvline(OUTAGE_SCENARIO["start"], color="orange", linestyle=":",
               linewidth=2, label="Outage start/end")
    ax.axvline(OUTAGE_SCENARIO["end"],   color="orange", linestyle=":", linewidth=2)

    ax.set_xlim(start, end)
    ax.set_xlabel("Transaction #")
    ax.set_ylabel("Rolling UPI Success Rate")
    ax.set_title("Gateway Outage Adaptation — Razorpay UPI drops to 45%",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "outage_recovery.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


if __name__ == "__main__":
    ts_router, lcb_router, ts_df, lcb_df, rr_df = run_simulation(
        n_steps=10_000, seed=42, with_outage=True
    )

    print("\n── Thompson Sampling Summary ─────────────────────────────────")
    print(ts_router.summary().to_string(index=False))

    print("\nGenerating plots …")
    plot_cumulative_regret(ts_df, lcb_df, rr_df)
    plot_success_rate(ts_df, lcb_df, rr_df)
    plot_confidence_intervals(ts_router)
    plot_outage_recovery(ts_df, lcb_df, rr_df)

    # Save state
    state_path = os.path.join(ARTIFACTS_DIR, "router_state.json")
    with open(state_path, "w") as f:
        json.dump(ts_router.get_state(), f, indent=2)

    # Save simulation results
    ts_df.to_parquet(os.path.join(ARTIFACTS_DIR,  "ts_simulation.parquet"),  index=False)
    lcb_df.to_parquet(os.path.join(ARTIFACTS_DIR, "lcb_simulation.parquet"), index=False)
    rr_df.to_parquet(os.path.join(ARTIFACTS_DIR,  "rr_simulation.parquet"),  index=False)

    print(f"\n  State saved → {state_path}")
    print("\n✓  bandit.py complete.")