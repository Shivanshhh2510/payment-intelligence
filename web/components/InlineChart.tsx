"use client";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { ChartData } from "@/types";

const PALETTE = ["#6366f1","#10b981","#f59e0b","#f43f5e","#8b5cf6","#06b6d4","#ec4899","#14b8a6"];

export default function InlineChart({ chart }: { chart: ChartData }) {
  const data = chart.labels.map((label, i) => ({
    name: label,
    value: parseFloat(chart.values[i].toFixed(2)),
    tr: chart.true_rates ? parseFloat(chart.true_rates[i].toFixed(2)) : undefined,
  }));

  const max = Math.max(...chart.values);

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius)",
      padding: "16px",
    }}>
      <div style={{ fontSize: "11px", color: "var(--text-3)", marginBottom: "14px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px" }}>
        {chart.title}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
          <XAxis
            dataKey="name"
            tick={{ fontSize: 10, fill: "#4a6080" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#4a6080" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: "8px",
              fontSize: "12px",
              color: "var(--text-1)",
            }}
            formatter={(v: any) => [`${v}%`, chart.type === "gateway" ? "Success Rate" : "Fraud Rate"]}
            labelStyle={{ color: "var(--text-2)", marginBottom: "4px" }}
            cursor={{ fill: "rgba(99,102,241,0.06)" }}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.value === max ? "#f43f5e" : PALETTE[i % PALETTE.length]}
                opacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}