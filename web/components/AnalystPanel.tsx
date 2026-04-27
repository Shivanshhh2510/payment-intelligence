"use client";
import { useState, useRef, useEffect } from "react";
import { chatWithAnalyst } from "@/lib/api";
import { ChatMessage as ChatMsg, TransactionDecision } from "@/types";
import ChatMessage from "./ChatMessage";

interface Props {
  lastDecision: TransactionDecision | null;
}

const INITIAL_SUGGESTIONS = [
  "What is this dataset and what does PAISA solve?",
  "Which card type has the highest fraud rate?",
  "How does smart payment routing work?",
];

export default function AnalystPanel({ lastDecision }: Props) {
  const [messages, setMessages]       = useState<ChatMsg[]>([]);
  const [history, setHistory]         = useState<{ role: string; content: string }[]>([]);
  const [suggestions, setSuggestions] = useState(INITIAL_SUGGESTIONS);
  const [input, setInput]             = useState("");
  const [loading, setLoading]         = useState(false);
  const bottomRef                     = useRef<HTMLDivElement>(null);
  const textareaRef                   = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "24px";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  const send = async (text: string) => {
    if (!text.trim() || loading) return;
    const userMsg: ChatMsg = { role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setHistory(prev => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const data = await chatWithAnalyst(text, history);
      const aiMsg: ChatMsg = {
        role: "assistant",
        content: data.response,
        verified: data.verified,
        chart: data.chart || null,
      };
      setMessages(prev => [...prev, aiMsg]);
      setHistory(prev => [...prev, { role: "assistant", content: data.response }]);
      if (data.suggestions?.length) setSuggestions(data.suggestions.slice(0, 3));
    } catch (e: any) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Unable to reach the AI service. Please check your GROQ_API_KEY.",
        verified: false,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>

      {/* Header */}
      <div style={{ padding: "28px 32px 0 32px", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "12px", marginBottom: "6px" }}>
          <h1 style={{ fontSize: "20px", fontWeight: 700, color: "var(--text-1)", letterSpacing: "-0.3px" }}>
            PAISA Analyst
          </h1>
          <span style={{
            fontSize: "10px", color: "var(--green)",
            background: "rgba(16,185,129,0.1)",
            border: "1px solid rgba(16,185,129,0.25)",
            padding: "2px 10px", borderRadius: "20px",
            fontWeight: 600, letterSpacing: "0.5px",
          }}>
            SELF-VERIFYING
          </span>
        </div>
        <p style={{ fontSize: "13px", color: "var(--text-3)", marginBottom: "16px" }}>
          Ask anything about fraud patterns, routing decisions, or the model.
        </p>

        {/* Last tx context */}
        {lastDecision && messages.length === 0 && (
          <div style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderLeft: "3px solid var(--accent)",
            borderRadius: "8px",
            padding: "10px 16px",
            marginBottom: "16px",
            fontSize: "12px",
            color: "var(--text-3)",
            animation: "fadeUp 0.3s ease",
          }}>
            <span style={{ color: "var(--accent)", fontWeight: 600 }}>Last transaction loaded</span>
            {" — "}
            <span style={{ color: lastDecision.fraud_risk_level === "LOW" ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
              {lastDecision.fraud_risk_level} risk
            </span>
            {" · "}{lastDecision.fraud_decision}{" · "}
            Routed to {lastDecision.routed_gateway}. Ask me why.
          </div>
        )}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 32px 16px 32px" }}>

        {messages.length === 0 && (
          <div style={{
            height: "160px",
            display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center",
            color: "var(--text-3)", gap: "8px",
            animation: "fadeIn 0.4s ease",
          }}>
            <div style={{ fontSize: "32px", opacity: 0.3 }}>🤖</div>
            <div style={{ fontSize: "13px" }}>Click a suggestion or ask anything</div>
          </div>
        )}

        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}

        {loading && (
          <div style={{ display: "flex", gap: "12px", marginBottom: "16px", alignItems: "flex-start", animation: "fadeUp 0.2s ease" }}>
            <div style={{
              width: "28px", height: "28px", minWidth: "28px",
              background: "linear-gradient(135deg, #4f46e5, #6366f1)",
              borderRadius: "50%",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "11px", fontWeight: 700, color: "white",
              animation: "glow 2s ease infinite",
            }}>P</div>
            <div style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "4px 14px 14px 14px",
              padding: "14px 18px",
              display: "flex", gap: "5px", alignItems: "center",
            }}>
              <div className="typing-dot" />
              <div className="typing-dot" />
              <div className="typing-dot" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Suggestions */}
      <div style={{ padding: "0 32px 10px 32px", flexShrink: 0 }}>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          {suggestions.map((s, i) => (
            <button
              key={i}
              onClick={() => send(s)}
              disabled={loading}
              className="chip"
              style={{ opacity: loading ? 0.5 : 1 }}
            >
              <span style={{ color: "var(--accent)", fontSize: "10px" }}>✦</span>
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div style={{
        padding: "10px 32px 24px 32px",
        borderTop: "1px solid var(--border)",
        flexShrink: 0,
      }}>
        <div style={{
          display: "flex",
          gap: "10px",
          alignItems: "flex-end",
          background: "var(--surface)",
          border: "1px solid var(--border-2)",
          borderRadius: "12px",
          padding: "10px 14px",
          transition: "border-color 0.15s",
        }}
          onFocusCapture={e => (e.currentTarget as HTMLDivElement).style.borderColor = "var(--accent)"}
          onBlurCapture={e => (e.currentTarget as HTMLDivElement).style.borderColor = "var(--border-2)"}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about fraud patterns, routing, model performance…"
            rows={1}
            disabled={loading}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              color: "var(--text-1)",
              fontSize: "13px",
              resize: "none",
              lineHeight: 1.6,
              fontFamily: "Inter, sans-serif",
              opacity: loading ? 0.6 : 1,
            }}
          />
          <button
            onClick={() => send(input)}
            disabled={loading || !input.trim()}
            style={{
              background: input.trim() && !loading
                ? "linear-gradient(135deg, #4f46e5, #6366f1)"
                : "var(--border)",
              color: "white",
              border: "none",
              borderRadius: "8px",
              width: "34px", height: "34px",
              fontSize: "16px",
              cursor: input.trim() && !loading ? "pointer" : "not-allowed",
              transition: "all 0.15s",
              flexShrink: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: input.trim() && !loading ? "0 2px 8px rgba(99,102,241,0.4)" : "none",
            }}
          >
            →
          </button>
        </div>
      </div>
    </div>
  );
}