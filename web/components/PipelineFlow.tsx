"use client";
import { TransactionDecision } from "@/types";

export default function PipelineFlow({ decision: d }: { decision: TransactionDecision }) {
  const steps = [
    {
      label: "Feature Store",
      detail: "367 features",
      color: "#6366f1",
      ok: true,
    },
    {
      label: "Isolation Forest",
      detail: `Anomaly: ${d.iso_anomaly_score.toFixed(3)}`,
      color: d.iso_anomaly_score > 0.5 ? "#f59e0b" : "#10b981",
      ok: d.iso_anomaly_score <= 0.5,
    },
    {
      label: "XGBoost",
      detail: `${(d.fraud_probability * 100).toFixed(1)}% fraud`,
      color: d.fraud_probability > 0.41 ? "#f43f5e" : "#10b981",
      ok: d.fraud_probability <= 0.41,
    },
    {
      label: "Two-Stage",
      detail: d.fraud_decision.replace("_"," "),
      color: d.fraud_decision === "LEGITIMATE" ? "#10b981" : "#f59e0b",
      ok: d.fraud_decision === "LEGITIMATE",
    },
    {
      label: "Smart Router",
      detail: d.routed_gateway,
      color: "#6366f1",
      ok: true,
    },
  ];

  return (
    <div style={{ marginTop: "14px" }}>
      <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "10px", fontWeight: 600 }}>
        Decision Pipeline
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "0" }}>
        {steps.map((step, i) => (
          <div key={step.label} style={{ display: "flex", alignItems: "center", flex: 1, minWidth: 0 }}>
            <div style={{
              background: "var(--surface)",
              border: `1px solid ${step.color}33`,
              borderTop: `2px solid ${step.color}`,
              borderRadius: "8px",
              padding: "8px 10px",
              flex: 1,
              minWidth: 0,
              transition: "all 0.15s",
            }}>
              <div style={{ fontSize: "10px", color: step.color, fontWeight: 600, marginBottom: "2px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {step.label}
              </div>
              <div style={{ fontSize: "10px", color: "var(--text-3)", fontFamily: "var(--mono)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {step.detail}
              </div>
            </div>
            {i < steps.length - 1 && (
              <div style={{ color: "var(--text-3)", fontSize: "12px", padding: "0 3px", flexShrink: 0 }}>
                →
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}