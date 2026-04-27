"use client";
import { TransactionDecision } from "@/types";

const RISK_COLOR: Record<string, string> = {
  LOW: "#10b981", MEDIUM: "#f59e0b", HIGH: "#f43f5e", CRITICAL: "#e11d48",
};
const RISK_BG: Record<string, string> = {
  LOW: "rgba(16,185,129,0.06)", MEDIUM: "rgba(245,158,11,0.06)",
  HIGH: "rgba(244,63,94,0.06)", CRITICAL: "rgba(225,29,72,0.06)",
};
const ACTION_STYLE: Record<string, { bg: string; color: string; border: string }> = {
  ALLOW:  { bg: "rgba(16,185,129,0.1)",  color: "#10b981", border: "rgba(16,185,129,0.3)" },
  REVIEW: { bg: "rgba(245,158,11,0.1)",  color: "#f59e0b", border: "rgba(245,158,11,0.3)" },
  BLOCK:  { bg: "rgba(244,63,94,0.1)",   color: "#f43f5e", border: "rgba(244,63,94,0.3)" },
};
const DC: Record<string, string> = {
  LEGITIMATE: "#10b981", KNOWN_FRAUD: "#f59e0b",
  NOVEL_ANOMALY: "#f59e0b", CONFIRMED_FRAUD: "#f43f5e",
};

export default function DecisionCard({ decision: d }: { decision: TransactionDecision }) {
  const rc  = RISK_COLOR[d.fraud_risk_level]  || "#94a3b8";
  const rbg = RISK_BG[d.fraud_risk_level]     || "transparent";
  const as  = ACTION_STYLE[d.recommended_action] || ACTION_STYLE.REVIEW;

  return (
    <div style={{
      background: "var(--surface)",
      border: `1px solid var(--border)`,
      borderTop: `3px solid ${rc}`,
      borderRadius: "var(--radius)",
      padding: "22px",
      position: "relative",
      overflow: "hidden",
    }}>

      {/* Subtle glow background */}
      <div style={{
        position: "absolute", inset: 0,
        background: rbg,
        pointerEvents: "none",
      }} />

      {/* Top row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "18px", position: "relative" }}>
        <div>
          <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "6px", fontWeight: 600 }}>
            Fraud Risk
          </div>
          <div style={{ fontSize: "36px", fontWeight: 800, color: rc, lineHeight: 1, letterSpacing: "-1px" }}>
            {d.fraud_risk_level}
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-3)", marginTop: "6px", fontFamily: "var(--mono)" }}>
            {(d.fraud_probability * 100).toFixed(2)}% probability
          </div>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "8px", fontWeight: 600 }}>
            Action
          </div>
          <span style={{
            background: as.bg,
            color: as.color,
            border: `1px solid ${as.border}`,
            borderRadius: "20px",
            padding: "5px 16px",
            fontSize: "12px",
            fontWeight: 700,
            letterSpacing: "0.5px",
            textTransform: "uppercase",
          }}>
            {d.recommended_action}
          </span>
        </div>
      </div>

      <div style={{ borderTop: "1px solid var(--border)", marginBottom: "18px", position: "relative" }} />

      {/* Stats */}
      <div style={{ display: "flex", gap: "20px", flexWrap: "wrap", position: "relative" }}>
        <Stat label="Decision" value={d.fraud_decision.replace(/_/g," ")} color={DC[d.fraud_decision]} />
        <Stat label="Gateway" value={d.routed_gateway} color="var(--accent)" />
        <Stat label="Success Rate" value={`${(d.expected_success_rate * 100).toFixed(1)}%`} color="var(--green)" />
        <Stat label="Anomaly" value={d.iso_anomaly_score.toFixed(4)} mono />
        <Stat label="Latency" value={`${d.processing_time_ms.toFixed(0)}ms`} mono />
      </div>

      {/* Routing reason */}
      <div style={{
        marginTop: "16px",
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        borderRadius: "8px",
        padding: "10px 14px",
        fontSize: "12px",
        color: "var(--text-2)",
        lineHeight: 1.6,
        position: "relative",
      }}>
        <span style={{ color: "var(--text-3)", marginRight: "6px", fontWeight: 600 }}>Routing:</span>
        {d.routing_reason}
      </div>
    </div>
  );
}

function Stat({ label, value, color, mono }: {
  label: string; value: string; color?: string; mono?: boolean;
}) {
  return (
    <div>
      <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "4px", fontWeight: 600 }}>
        {label}
      </div>
      <div style={{
        fontSize: "13px", fontWeight: 600,
        color: color || "var(--text-1)",
        fontFamily: mono ? "var(--mono)" : "inherit",
      }}>
        {value}
      </div>
    </div>
  );
}