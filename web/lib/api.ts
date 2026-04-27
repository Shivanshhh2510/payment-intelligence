const BASE = "http://localhost:8001";

export async function scoreTransaction(req: {
  tx_type: string; amount: number; card_type: string;
  network: string; email: string; device: string;
  hour: number; card_id: number;
}) {
  const res = await fetch(`${BASE}/api/transaction/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getMetrics() {
  const res = await fetch(`${BASE}/api/metrics`);
  if (!res.ok) throw new Error("Failed to load metrics");
  return res.json();
}

export async function getFacts() {
  const res = await fetch(`${BASE}/api/facts`);
  if (!res.ok) throw new Error("Failed to load facts");
  return res.json();
}

export async function chatWithAnalyst(message: string, history: { role: string; content: string }[]) {
  const res = await fetch(`${BASE}/api/analyst/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}