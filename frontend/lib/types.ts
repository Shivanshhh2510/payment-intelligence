export type TransactionType = 'UPI' | 'CARD' | 'NETBANKING' | 'WALLET'
export type CardType = 'debit' | 'credit'
export type CardNetwork = 'visa' | 'mastercard' | 'discover' | 'american express'
export type EmailDomain = 'gmail.com' | 'yahoo.com' | 'hotmail.com' | 'outlook.com' | 'mail.com'
export type DeviceType = 'desktop' | 'mobile'

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
export type ActionType = 'ALLOW' | 'REVIEW' | 'BLOCK'
export type DecisionType = 'CONFIRMED_FRAUD' | 'NOVEL_ANOMALY' | 'KNOWN_FRAUD' | 'LEGITIMATE'

export interface TransactionInput {
  tx_type: TransactionType
  amount: number
  card_type: CardType
  network: CardNetwork
  email: EmailDomain
  device: DeviceType
  hour: number
  card_id: number
}

export interface TransactionScore {
  fraud_probability: number
  iso_anomaly_score: number
  fraud_decision: DecisionType
  fraud_risk_level: RiskLevel
  recommended_action: ActionType
  routed_gateway: string
  routing_reason: string
  expected_success_rate: number
  processing_time_ms: number
  // legacy fields for DecisionCard compatibility
  risk_level?: RiskLevel
  action?: ActionType
  route_to?: string
  expected_sr?: number
  latency?: number
  gateway?: string
  explanation?: string
}

export interface TransactionRecord {
  id: string
  tx_type: TransactionType
  amount: number
  decision: DecisionType
  action: ActionType
  gateway: string
  fraud_percent: number
  timestamp: Date
}

export interface Metrics {
  test_auc?: number
  test_f1?: number
  test_precision?: number
  test_recall?: number
  threshold?: number
  net_benefit_inr?: number
  auc_roc?: number
  net_benefit?: number
  routing_improvement?: number
}

export interface Impact {
  savings: number
  blocked_fraud: number
}

export interface Drift {
  retrain_recommended: boolean
  drift_score?: number
  recommendation?: string
}

export interface RouterState {
  active_gateway?: string
  thompson_alpha?: number
  thompson_beta?: number
}

export interface MetricsResponse {
  metrics: Metrics
  impact: Impact
  drift: Drift
  router_state: RouterState
}

export interface Fact {
  id: string
  text: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  chart_data?: ChartData[]
  verified?: boolean
  suggestions?: string[]
}

export interface ChartData {
  name: string
  value: number
}

export interface ChatRequest {
  message: string
  history: Array<{ role: string; content: string }>
}

export interface ChatResponse {
  response: string
  verified?: boolean
  suggestions?: string[]
  chart?: {
    type: string
    labels: string[]
    values: number[]
    title: string
    tx_type?: string
    true_rates?: number[]
  } | null
  intent?: string
}