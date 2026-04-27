"use client";
import { useState, useEffect, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import TransactionPanel from "@/components/TransactionPanel";
import AnalystPanel from "@/components/AnalystPanel";
import { getMetrics } from "@/lib/api";
import { TransactionDecision } from "@/types";

export default function Home() {
  const [tab, setTab]           = useState<"transaction" | "analyst">("transaction");
  const [metricsData, setMetrics] = useState<any>(null);
  const [lastDecision, setLastDecision] = useState<TransactionDecision | null>(null);
  const [toast, setToast]       = useState<{ msg: string; type: string } | null>(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {});
  }, []);

  const showToast = (msg: string, type = "info") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const metrics = metricsData?.metrics || null;
  const impact  = metricsData?.impact  || null;
  const drift   = metricsData?.drift   || null;

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: "var(--bg)" }}>

      <Sidebar
        metrics={metrics}
        impact={impact}
        drift={drift}
        lastDecision={lastDecision}
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>

        {/* Tab bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "2px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
          padding: "0 28px",
          flexShrink: 0,
          height: "48px",
        }}>
          {([
            { key: "transaction", label: "⚡ Transaction", desc: "Score & route" },
            { key: "analyst",     label: "🤖 PAISA Analyst", desc: "Ask anything" },
          ] as const).map(t => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              style={{
                background: "none",
                border: "none",
                borderBottom: tab === t.key
                  ? "2px solid var(--accent)"
                  : "2px solid transparent",
                color: tab === t.key ? "var(--text-1)" : "var(--text-3)",
                padding: "0 16px",
                height: "48px",
                fontSize: "13px",
                fontWeight: tab === t.key ? 600 : 400,
                cursor: "pointer",
                transition: "all 0.15s",
                fontFamily: "inherit",
                marginBottom: "-1px",
                letterSpacing: "0.2px",
              }}
            >
              {t.label}
            </button>
          ))}

          {/* Live indicator */}
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px" }}>
            <div style={{
              width: "6px", height: "6px", borderRadius: "50%",
              background: "var(--green)",
              boxShadow: "0 0 8px rgba(16,185,129,0.6)",
              animation: "pulse-dot 2s infinite",
            }} />
            <span style={{ fontSize: "11px", color: "var(--text-3)", letterSpacing: "0.5px" }}>
              LIVE
            </span>
          </div>
        </div>

        {/* Panel */}
        <div style={{ flex: 1, overflow: "hidden" }}>
          <div style={{ display: tab === "transaction" ? "block" : "none", height: "100%" }}>
            <TransactionPanel
              onDecision={d => { setLastDecision(d); showToast("Transaction scored", "success"); }}
            />
          </div>
          <div style={{ display: tab === "analyst" ? "block" : "none", height: "100%" }}>
            <AnalystPanel lastDecision={lastDecision} />
          </div>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.msg}
        </div>
      )}
    </div>
  );
}