"use client";
import { TransactionDecision } from "@/types";

interface SidebarProps {
  metrics: { test_auc: number; test_f1: number; threshold: number } | null;
  impact: { net_benefit_inr: number } | null;
  drift: { recommendation: string } | null;
  lastDecision: TransactionDecision | null;
}

function riskColor(risk: string): string {
  if (risk === "LOW") return "var(--green)";
  if (risk === "MEDIUM") return "var(--orange)";
  if (risk === "HIGH") return "var(--red)";
  return "#e11d48";
}

function Row(p: { label: string; value: string; mono?: boolean; valueColor?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
      <span style={{ fontSize: "11px", color: "var(--text-3)" }}>{p.label}</span>
      <span style={{ fontSize: "11px", color: p.valueColor || "var(--text-2)", fontFamily: p.mono ? "monospace" : "inherit", fontWeight: 500 }}>{p.value}</span>
    </div>
  );
}

export default function Sidebar(p: SidebarProps) {
  const healthy = p.drift?.recommendation === "MONITOR";
  const d = p.lastDecision;
  const m = p.metrics;
  const imp = p.impact;
  return (
    <aside style={{ width: "220px", minWidth: "220px", height: "100vh", background: "var(--surface)", borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ padding: "24px 20px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-1)" }}>PAISA</div>
        <div style={{ fontSize: "10px", color: "var(--text-3)", marginTop: "4px", lineHeight: 1.5 }}>Payment AI for Smart Authentication</div>
      </div>
      <div style={{ padding: "20px 20px 0 20px" }}>
        <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "12px" }}>System</div>
        <Row label="Model" value="v1 Production" />
        <Row label="Dataset" value="590k txns" />
        <Row label="Health" value={healthy ? "Stable" : "Retrain"} valueColor={healthy ? "var(--green)" : "var(--red)"} />
      </div>
      <div style={{ margin: "20px", borderTop: "1px solid var(--border)" }} />
      <div style={{ padding: "0 20px" }}>
        <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "12px" }}>Performance</div>
        <Row label="AUC-ROC" value={m ? m.test_auc.toFixed(4) : "-"} mono />
        <Row label="F1 Score" value={m ? m.test_f1.toFixed(4) : "-"} mono />
        <Row label="Threshold" value={m ? m.threshold.toFixed(2) : "-"} mono />
        <Row label="Net Benefit" value={imp ? "Rs" + (imp.net_benefit_inr / 100000).toFixed(1) + "L" : "-"} mono />
        <Row label="Routing Gain" value="+10.8%" mono valueColor="var(--green)" />
      </div>
      <div style={{ margin: "20px", borderTop: "1px solid var(--border)" }} />
      {d && (
        <div style={{ padding: "0 20px" }}>
          <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "12px" }}>Last Decision</div>
          <Row label="Risk" value={d.fraud_risk_level} valueColor={riskColor(d.fraud_risk_level)} />
          <Row label="Action" value={d.recommended_action} />
          <Row label="Gateway" value={d.routed_gateway} />
          <Row label="Fraud" value={(d.fraud_probability * 100).toFixed(1) + "%"} mono />
        </div>
      )}
      <div style={{ flex: 1 }} />
      <div style={{ padding: "20px", borderTop: "1px solid var(--border)" }}>
        <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "10px" }}>References</div>
        <a href="https://razorpay.com/" target="_blank" rel="noreferrer" style={{ display: "block", fontSize: "11px", color: "var(--text-3)", textDecoration: "none", marginBottom: "6px" }}>Razorpay Optimizer</a>
        <a href="https://www.kaggle.com/c/ieee-fraud-detection" target="_blank" rel="noreferrer" style={{ display: "block", fontSize: "11px", color: "var(--text-3)", textDecoration: "none", marginBottom: "6px" }}>IEEE-CIS Dataset</a>
        <a href="https://arxiv.org/abs/1209.3352" target="_blank" rel="noreferrer" style={{ display: "block", fontSize: "11px", color: "var(--text-3)", textDecoration: "none" }}>Thompson Sampling</a>
      </div>
    </aside>
  );
}
