"use client";
import { useState } from "react";
import { scoreTransaction } from "@/lib/api";
import { TransactionDecision } from "@/types";
import DecisionCard from "./DecisionCard";
import PipelineFlow from "./PipelineFlow";

interface Props {
  onDecision: (d: TransactionDecision) => void;
}

const HOURS = Array.from({ length: 24 }, (_, i) => ({
  value: i,
  label: i === 0 ? "12 AM (Midnight)" : i < 12 ? `${i} AM` : i === 12 ? "12 PM (Noon)" : `${i - 12} PM`,
}));

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize: "10px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "6px", fontWeight: 600 }}>
        {label}
      </div>
      {children}
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  width: "100%",
  background: "var(--bg)",
  border: "1px solid var(--border-2)",
  borderRadius: "8px",
  padding: "9px 12px",
  color: "var(--text-1)",
  fontSize: "13px",
  outline: "none",
  cursor: "pointer",
  fontFamily: "inherit",
  transition: "border-color 0.15s",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  background: "var(--bg)",
  border: "1px solid var(--border-2)",
  borderRadius: "8px",
  padding: "9px 12px",
  color: "var(--text-1)",
  fontSize: "13px",
  outline: "none",
  fontFamily: "var(--mono)",
  transition: "border-color 0.15s",
};

const DC: Record<string, string> = {
  LEGITIMATE: "#10b981", CONFIRMED_FRAUD: "#f43f5e",
  NOVEL_ANOMALY: "#f59e0b", KNOWN_FRAUD: "#f59e0b",
};
const AC: Record<string, string> = {
  ALLOW: "#10b981", REVIEW: "#f59e0b", BLOCK: "#f43f5e",
};

export default function TransactionPanel({ onDecision }: Props) {
  const [form, setForm] = useState({
    tx_type: "UPI", amount: 499, card_type: "debit",
    network: "visa", email: "gmail.com", device: "desktop",
    hour: 14, card_id: 2755,
  });
  const [decision, setDecision] = useState<TransactionDecision | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [history, setHistory]   = useState<TransactionDecision[]>([]);

  const process = async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await scoreTransaction(form);
      setDecision(d);
      onDecision(d);
      setHistory(prev => [d, ...prev].slice(0, 8));
    } catch (e: any) {
      setError(e.message || "Failed to process transaction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ height: "100%", overflowY: "auto", padding: "28px 32px" }}>

      {/* Header */}
      <div style={{ marginBottom: "28px", animation: "fadeUp 0.3s ease" }}>
        <h1 style={{ fontSize: "20px", fontWeight: 700, color: "var(--text-1)", letterSpacing: "-0.3px" }}>
          Transaction Intelligence
        </h1>
        <p style={{ fontSize: "13px", color: "var(--text-3)", marginTop: "4px" }}>
          Real-time fraud scoring · Smart gateway routing · Two-stage detection
        </p>
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", alignItems: "start", animation: "fadeUp 0.3s ease 0.05s both" }}>

        {/* Form */}
        <div style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: "24px",
        }}>
          <div style={{ fontSize: "11px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "20px", fontWeight: 600 }}>
            Transaction Details
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
            <Field label="Type">
              <select value={form.tx_type} onChange={e => setForm(f => ({ ...f, tx_type: e.target.value }))} style={selectStyle}>
                {["UPI","CARD","NETBANKING","WALLET"].map(o => <option key={o}>{o}</option>)}
              </select>
            </Field>
            <Field label="Amount (₹)">
              <input type="number" value={form.amount} min={1} max={100000}
                onChange={e => setForm(f => ({ ...f, amount: Number(e.target.value) }))}
                style={inputStyle} />
            </Field>
            <Field label="Card Type">
              <select value={form.card_type} onChange={e => setForm(f => ({ ...f, card_type: e.target.value }))} style={selectStyle}>
                {["debit","credit"].map(o => <option key={o}>{o}</option>)}
              </select>
            </Field>
            <Field label="Card Network">
              <select value={form.network} onChange={e => setForm(f => ({ ...f, network: e.target.value }))} style={selectStyle}>
                {["visa","mastercard","discover","american express"].map(o => <option key={o}>{o}</option>)}
              </select>
            </Field>
            <Field label="Email Domain">
              <select value={form.email} onChange={e => setForm(f => ({ ...f, email: e.target.value }))} style={selectStyle}>
                {["gmail.com","yahoo.com","hotmail.com","outlook.com","mail.com"].map(o => <option key={o}>{o}</option>)}
              </select>
            </Field>
            <Field label="Device">
              <select value={form.device} onChange={e => setForm(f => ({ ...f, device: e.target.value }))} style={selectStyle}>
                {["desktop","mobile"].map(o => <option key={o}>{o}</option>)}
              </select>
            </Field>
            <Field label="Hour of Day">
              <select value={String(form.hour)}
                onChange={e => setForm(f => ({ ...f, hour: parseInt(e.target.value) }))}
                style={selectStyle}>
                {HOURS.map(h => <option key={h.value} value={String(h.value)}>{h.label}</option>)}
              </select>
            </Field>
            <Field label="Card ID">
              <input type="number" value={form.card_id} min={1} max={20000}
                onChange={e => setForm(f => ({ ...f, card_id: Number(e.target.value) }))}
                style={inputStyle} />
            </Field>
          </div>

          <button
            onClick={process}
            disabled={loading}
            style={{
              marginTop: "20px",
              width: "100%",
              background: loading
                ? "var(--border)"
                : "linear-gradient(135deg, #4f46e5, #6366f1)",
              color: "white",
              border: "none",
              borderRadius: "8px",
              padding: "12px",
              fontSize: "13px",
              fontWeight: 600,
              cursor: loading ? "not-allowed" : "pointer",
              transition: "all 0.15s",
              fontFamily: "inherit",
              letterSpacing: "0.3px",
              boxShadow: loading ? "none" : "0 4px 16px rgba(99,102,241,0.3)",
            }}
          >
            {loading ? (
              <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ animation: "spin 1s linear infinite" }}>
                  <path d="M12 2a10 10 0 110 20" />
                </svg>
                Analyzing…
              </span>
            ) : "Analyze Transaction →"}
          </button>

          {error && (
            <div style={{ marginTop: "12px", fontSize: "12px", color: "var(--red)", padding: "10px 14px", background: "rgba(244,63,94,0.08)", borderRadius: "8px", border: "1px solid rgba(244,63,94,0.2)" }}>
              {error}
            </div>
          )}
        </div>

        {/* Decision */}
        <div>
          {decision ? (
            <div style={{ animation: "fadeUp 0.3s ease" }}>
              <DecisionCard decision={decision} />
              <PipelineFlow decision={decision} />
            </div>
          ) : (
            <div style={{
              height: "340px",
              background: "var(--surface)",
              border: "1px dashed var(--border-2)",
              borderRadius: "var(--radius)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--text-3)",
              gap: "10px",
            }}>
              <div style={{ fontSize: "32px", opacity: 0.4 }}>⚡</div>
              <div style={{ fontSize: "13px" }}>Decision appears here</div>
              <div style={{ fontSize: "11px", opacity: 0.6 }}>Fill the form and click Analyze</div>
            </div>
          )}
        </div>
      </div>

      {/* History */}
      {history.length > 0 && (
        <div style={{ marginTop: "32px", animation: "fadeUp 0.3s ease" }}>
          <div style={{ fontSize: "11px", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "12px", fontWeight: 600 }}>
            Recent Transactions
          </div>
          <div style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            overflow: "hidden",
          }}>
            <div style={{
              display: "grid",
              gridTemplateColumns: "70px 90px 1fr 80px 120px 90px",
              padding: "10px 18px",
              borderBottom: "1px solid var(--border)",
              fontSize: "10px",
              color: "var(--text-3)",
              textTransform: "uppercase",
              letterSpacing: "1px",
              fontWeight: 600,
            }}>
              {["Type","Amount","Decision","Action","Gateway","Fraud %"].map(h => <span key={h}>{h}</span>)}
            </div>
            {history.map((row, i) => (
              <div key={i} style={{
                display: "grid",
                gridTemplateColumns: "70px 90px 1fr 80px 120px 90px",
                padding: "11px 18px",
                borderBottom: i < history.length - 1 ? "1px solid var(--border)" : "none",
                fontSize: "12px",
                alignItems: "center",
                transition: "background 0.15s",
              }}
                onMouseEnter={e => (e.currentTarget as HTMLDivElement).style.background = "var(--surface-2)"}
                onMouseLeave={e => (e.currentTarget as HTMLDivElement).style.background = "transparent"}
              >
                <span style={{ color: "var(--text-2)", fontWeight: 500 }}>{row.fraud_risk_level === "LOW" ? "✓" : "⚠"} {row.routed_gateway.slice(0,3)}</span>
                <span style={{ color: "var(--text-2)", fontFamily: "var(--mono)" }}>₹{form.amount.toLocaleString()}</span>
                <span style={{ color: DC[row.fraud_decision] || "var(--text-2)", fontSize: "11px", fontWeight: 500 }}>{row.fraud_decision}</span>
                <span style={{ color: AC[row.recommended_action] || "var(--text-2)", fontWeight: 700, fontSize: "11px" }}>{row.recommended_action}</span>
                <span style={{ color: "var(--accent)", fontWeight: 500 }}>{row.routed_gateway}</span>
                <span style={{ color: "var(--text-2)", fontFamily: "var(--mono)" }}>{(row.fraud_probability * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}