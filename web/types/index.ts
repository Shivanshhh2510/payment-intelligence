export interface TransactionRequest {
  tx_type:   string;
  amount:    number;
  card_type: string;
  network:   string;
  email:     string;
  device:    string;
  hour:      number;
  card_id:   number;
}

export interface TransactionDecision {
  transaction_id:        string;
  fraud_probability:     number;
  iso_anomaly_score:     number;
  fraud_decision:        string;
  fraud_risk_level:      string;
  recommended_action:    string;
  routed_gateway:        string;
  routing_reason:        string;
  expected_success_rate: number;
  processing_time_ms:    number;
}

export interface ChatMessage {
  role:     "user" | "assistant";
  content:  string;
  verified?: boolean;
  chart?:   ChartData | null;
}

export interface ChartData {
  type:        string;
  labels:      string[];
  values:      number[];
  title:       string;
  tx_type?:    string;
  true_rates?: number[];
}

export interface Metrics {
  test_auc:       number;
  test_f1:        number;
  test_precision: number;
  test_recall:    number;
  threshold:      number;
}

export interface Impact {
  fraud_caught:                number;
  fraud_missed:                number;
  legitimate_blocked:          number;
  fraud_value_prevented_inr:   number;
  net_benefit_inr:             number;
  roi_percent:                 number;
}

export interface DriftData {
  recommendation:        string;
  n_high_psi_features:   number;
  auc_degradation:       number;
  fraud_rate_shift_pct:  number;
  reason:                string;
}