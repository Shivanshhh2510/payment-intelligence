"use client";
import { ChatMessage as ChatMsg } from "@/types";
import InlineChart from "./InlineChart";

export default function ChatMessage({ message: msg }: { message: ChatMsg }) {
  if (msg.role === "user") {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "16px" }}>
        <div style={{
          background: "var(--surface-2)",
          border: "1px solid var(--border-2)",
          borderRadius: "14px 14px 4px 14px",
          padding: "10px 16px",
          maxWidth: "72%",
          fontSize: "13px",
          color: "var(--text-1)",
          lineHeight: 1.6,
        }}>
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", gap: "12px", marginBottom: "20px", alignItems: "flex-start", animation: "fadeUp 0.25s ease" }}>
      <div style={{
        width: "28px", height: "28px", minWidth: "28px",
        background: "linear-gradient(135deg, #4f46e5, #6366f1)",
        borderRadius: "50%",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: "11px", fontWeight: 700, color: "white",
        marginTop: "2px",
      }}>
        P
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "4px 14px 14px 14px",
          padding: "14px 18px",
          fontSize: "13px",
          color: "var(--text-1)",
          lineHeight: 1.75,
          whiteSpace: "pre-wrap",
        }}>
          {msg.content}
        </div>

        {msg.chart && (
          <div style={{ marginTop: "10px" }}>
            <InlineChart chart={msg.chart} />
          </div>
        )}

        <div style={{
          fontSize: "10px",
          color: msg.verified ? "var(--green)" : "var(--text-3)",
          marginTop: "6px",
          paddingLeft: "2px",
          display: "flex", alignItems: "center", gap: "4px",
        }}>
          {msg.verified ? (
            <><span>✓</span><span>verified against dataset</span></>
          ) : (
            <><span>⚡</span><span>response generated</span></>
          )}
        </div>
      </div>
    </div>
  );
}