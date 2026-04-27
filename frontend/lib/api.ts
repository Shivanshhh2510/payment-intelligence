import type {
  TransactionInput,
  TransactionScore,
  MetricsResponse,
  ChatRequest,
  ChatResponse,
  Fact,
} from './types'

const API_BASE = 'http://localhost:8001'

export async function scoreTransaction(input: TransactionInput): Promise<TransactionScore> {
  const response = await fetch(`${API_BASE}/api/transaction/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })

  if (!response.ok) {
    throw new Error(`Failed to score transaction: ${response.statusText}`)
  }

  return response.json()
}

export async function getMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${API_BASE}/api/metrics`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch metrics: ${response.statusText}`)
  }

  return response.json()
}

export async function getFacts(): Promise<Fact[]> {
  const response = await fetch(`${API_BASE}/api/facts`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch facts: ${response.statusText}`)
  }

  return response.json()
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/analyst/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`Failed to send chat message: ${response.statusText}`)
  }

  return response.json()
}
