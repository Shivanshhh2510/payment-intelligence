'use client'

import { useState } from 'react'
import { Zap, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { DecisionCard } from './DecisionCard'
import { scoreTransaction } from '@/lib/api'
import type {
  TransactionType,
  CardType,
  CardNetwork,
  EmailDomain,
  DeviceType,
  TransactionScore,
  TransactionRecord,
  DecisionType,
  ActionType,
  RiskLevel,
} from '@/lib/types'

interface TransactionPanelProps {
  transactions: TransactionRecord[]
  onNewTransaction: (record: TransactionRecord) => void
  onAskAnalyst?: (message: string) => void
}

const TX_TYPES: TransactionType[] = ['UPI', 'CARD', 'NETBANKING', 'WALLET']
const CARD_NETWORKS: CardNetwork[] = ['visa', 'mastercard', 'discover', 'american express']
const EMAIL_DOMAINS: EmailDomain[] = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'mail.com']
const HOURS = Array.from({ length: 24 }, (_, i) => i)

export function TransactionPanel({ transactions, onNewTransaction, onAskAnalyst }: TransactionPanelProps) {
  const [txType, setTxType] = useState<TransactionType>('UPI')
  const [amount, setAmount] = useState<string>('5000')
  const [cardType, setCardType] = useState<CardType>('debit')
  const [network, setNetwork] = useState<CardNetwork>('visa')
  const [email, setEmail] = useState<EmailDomain>('gmail.com')
  const [device, setDevice] = useState<DeviceType>('mobile')
  const [hour, setHour] = useState<number>(14)
  const [isLoading, setIsLoading] = useState(false)
  const [score, setScore] = useState<TransactionScore | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await scoreTransaction({
        tx_type: txType,
        amount: parseFloat(amount) || 0,
        card_type: cardType,
        network,
        email,
        device,
        hour,
        card_id: 2755,
      })

      // Normalize to legacy fields for DecisionCard
      const normalized: TransactionScore = {
        ...result,
        risk_level: result.fraud_risk_level,
        action: result.recommended_action,
        route_to: result.routed_gateway,
        expected_sr: result.expected_success_rate,
        latency: Math.round(result.processing_time_ms),
        gateway: result.routed_gateway,
        explanation: result.routing_reason,
      }
      setScore(normalized)

      const newRecord: TransactionRecord = {
        id: `tx_${Date.now()}`,
        tx_type: txType,
        amount: parseFloat(amount) || 0,
        decision: result.fraud_decision,
        action: result.recommended_action,
        gateway: result.routed_gateway,
        fraud_percent: result.fraud_probability * 100,
        timestamp: new Date(),
      }
      onNewTransaction(newRecord)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze transaction')
      const amtNum = parseFloat(amount) || 0
      const riskLevel: RiskLevel = amtNum > 50000 ? 'HIGH' : amtNum > 10000 ? 'MEDIUM' : 'LOW'
      const mockScore: TransactionScore = {
        fraud_probability: amtNum > 50000 ? 0.72 : amtNum > 10000 ? 0.35 : 0.08,
        iso_anomaly_score: 0.0,
        fraud_decision: riskLevel === 'LOW' ? 'LEGITIMATE' : riskLevel === 'MEDIUM' ? 'NOVEL_ANOMALY' : 'CONFIRMED_FRAUD',
        fraud_risk_level: riskLevel,
        recommended_action: amtNum > 50000 ? 'REVIEW' : 'ALLOW',
        routed_gateway: 'Razorpay',
        routing_reason: `Transaction analyzed based on amount ₹${amount}, ${txType} payment via ${device} at ${hour}:00 hours.`,
        expected_success_rate: 0.94,
        processing_time_ms: 120,
        risk_level: riskLevel,
        action: amtNum > 50000 ? 'REVIEW' : 'ALLOW',
        route_to: 'Razorpay',
        expected_sr: 0.94,
        latency: 120,
        gateway: 'Razorpay',
        explanation: `Transaction analyzed based on amount ₹${amount}, ${txType} payment via ${device} at ${hour}:00 hours.`,
      }
      setScore(mockScore)

      const newRecord: TransactionRecord = {
        id: `tx_${Date.now()}`,
        tx_type: txType,
        amount: amtNum,
        decision: mockScore.fraud_decision,
        action: mockScore.recommended_action,
        gateway: mockScore.routed_gateway,
        fraud_percent: mockScore.fraud_probability * 100,
        timestamp: new Date(),
      }
      onNewTransaction(newRecord)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-white mb-1">Transaction Intelligence</h1>
        <p className="text-sm text-white/50">Real-time fraud scoring + smart payment routing</p>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Input Form */}
        <div className="lg:col-span-2 space-y-5">
          {/* Transaction Type Pills */}
          <div>
            <label className="block text-xs text-white/50 mb-2">Transaction Type</label>
            <div className="flex flex-wrap gap-2">
              {TX_TYPES.map((type) => (
                <button
                  key={type}
                  onClick={() => setTxType(type)}
                  className={cn(
                    'px-4 py-2 rounded-full text-xs font-medium transition-all',
                    txType === type
                      ? 'bg-indigo-500 text-white'
                      : 'bg-white/[0.05] text-white/60 hover:bg-white/[0.08] hover:text-white'
                  )}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Amount */}
          <div>
            <label className="block text-xs text-white/50 mb-2">Amount in ₹</label>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-full px-4 py-3 text-2xl font-semibold bg-white/[0.03] border border-white/[0.08] rounded-lg text-white placeholder-white/30 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/50"
              placeholder="0"
            />
          </div>

          {/* Card Type Toggle */}
          <div>
            <label className="block text-xs text-white/50 mb-2">Card Type</label>
            <div className="flex gap-2">
              {(['debit', 'credit'] as CardType[]).map((type) => (
                <button
                  key={type}
                  onClick={() => setCardType(type)}
                  className={cn(
                    'flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all capitalize',
                    cardType === type
                      ? 'bg-white/[0.1] text-white border border-white/[0.15]'
                      : 'bg-white/[0.03] text-white/50 border border-white/[0.06] hover:bg-white/[0.05]'
                  )}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Card Network */}
          <div>
            <label className="block text-xs text-white/50 mb-2">Card Network</label>
            <select
              value={network}
              onChange={(e) => setNetwork(e.target.value as CardNetwork)}
              className="w-full px-4 py-2.5 bg-white/[0.03] border border-white/[0.08] rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500/50 appearance-none cursor-pointer"
            >
              {CARD_NETWORKS.map((n) => (
                <option key={n} value={n} className="bg-zinc-900 capitalize">
                  {n.charAt(0).toUpperCase() + n.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Email Domain */}
          <div>
            <label className="block text-xs text-white/50 mb-2">Email Domain</label>
            <select
              value={email}
              onChange={(e) => setEmail(e.target.value as EmailDomain)}
              className="w-full px-4 py-2.5 bg-white/[0.03] border border-white/[0.08] rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500/50 appearance-none cursor-pointer"
            >
              {EMAIL_DOMAINS.map((e) => (
                <option key={e} value={e} className="bg-zinc-900">
                  {e}
                </option>
              ))}
            </select>
          </div>

          {/* Device Toggle */}
          <div>
            <label className="block text-xs text-white/50 mb-2">Device</label>
            <div className="flex gap-2">
              {(['desktop', 'mobile'] as DeviceType[]).map((d) => (
                <button
                  key={d}
                  onClick={() => setDevice(d)}
                  className={cn(
                    'flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all capitalize',
                    device === d
                      ? 'bg-white/[0.1] text-white border border-white/[0.15]'
                      : 'bg-white/[0.03] text-white/50 border border-white/[0.06] hover:bg-white/[0.05]'
                  )}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Transaction Hour */}
          <div>
            <label className="block text-xs text-white/50 mb-2">Transaction Hour</label>
            <select
              value={hour}
              onChange={(e) => setHour(parseInt(e.target.value))}
              className="w-full px-4 py-2.5 bg-white/[0.03] border border-white/[0.08] rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500/50 appearance-none cursor-pointer"
            >
              {HOURS.map((h) => (
                <option key={h} value={h} className="bg-zinc-900">
                  {h.toString().padStart(2, '0')}:00
                </option>
              ))}
            </select>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="w-full px-6 py-3.5 bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-colors flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                Analyze Transaction
                <span className="text-lg">→</span>
              </>
            )}
          </button>

          {error && (
            <div className="mt-2 space-y-1">
              <p className="text-xs text-amber-400">Using simulated response — start FastAPI with:</p>
              <code className="text-xs text-white/40 bg-white/[0.03] px-2 py-1 rounded block">
                uvicorn api.main:app --reload --port 8001
              </code>
            </div>
          )}
        </div>

        {/* Decision Output */}
        <div className="lg:col-span-3">
          {score ? (
            <div className="space-y-3">
              <DecisionCard score={score} />
              {onAskAnalyst && (
                <button
                  onClick={() => onAskAnalyst(
                    `I just processed a ${txType} transaction of ₹${amount}. The decision was ${score.risk_level} risk with ${(score.fraud_probability * 100).toFixed(1)}% fraud probability, routed to ${score.gateway}. Can you explain this decision?`
                  )}
                  className="w-full px-4 py-2.5 rounded-lg border border-indigo-500/30 text-indigo-400 text-sm hover:bg-indigo-500/10 transition-colors"
                >
                  🤖 Ask PAISA Analyst to explain this decision →
                </button>
              )}
            </div>
          ) : (
            <div className="h-full min-h-[300px] rounded-xl border-2 border-dashed border-white/[0.08] flex flex-col items-center justify-center">
              <Zap className="w-10 h-10 text-white/20 mb-3" />
              <p className="text-sm text-white/40">Decision appears here</p>
            </div>
          )}
        </div>
      </div>

      {/* Recent Transactions Table */}
      {transactions.length > 0 && (
        <div className="mt-8">
          <h2 className="text-lg font-medium text-white mb-4">Recent Transactions</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs text-white/40 uppercase tracking-wider">
                  <th className="pb-3 font-medium">Type</th>
                  <th className="pb-3 font-medium">Amount</th>
                  <th className="pb-3 font-medium">Decision</th>
                  <th className="pb-3 font-medium">Action</th>
                  <th className="pb-3 font-medium">Gateway</th>
                  <th className="pb-3 font-medium">Fraud %</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((tx, idx) => (
                  <tr
                    key={tx.id}
                    className={cn(
                      'text-sm',
                      idx % 2 === 0 ? 'bg-white/[0.02]' : ''
                    )}
                  >
                    <td className="py-3 px-2 rounded-l-lg">
                      <span className="px-2 py-1 rounded bg-white/[0.05] text-white/70 text-xs">
                        {tx.tx_type}
                      </span>
                    </td>
                    <td className="py-3 text-white/80">₹{tx.amount.toLocaleString()}</td>
                    <td className="py-3">
                      <span
                        className={cn(
                          'text-xs font-medium',
                          tx.decision === 'CONFIRMED_FRAUD' && 'text-red-400',
                          tx.decision === 'NOVEL_ANOMALY' && 'text-amber-400',
                          tx.decision === 'LEGITIMATE' && 'text-emerald-400'
                        )}
                      >
                        {tx.decision}
                      </span>
                    </td>
                    <td className="py-3">
                      <ActionBadge action={tx.action} />
                    </td>
                    <td className="py-3">
                      <span className="px-2 py-1 rounded-full bg-violet-500/20 text-violet-300 text-xs">
                        {tx.gateway}
                      </span>
                    </td>
                    <td className="py-3 pr-2 rounded-r-lg text-white/60">
                      {tx.fraud_percent.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function ActionBadge({ action }: { action: ActionType }) {
  const colorClass = {
    ALLOW: 'bg-emerald-500/20 text-emerald-400',
    REVIEW: 'bg-amber-500/20 text-amber-400',
    BLOCK: 'bg-red-500/20 text-red-400',
  }[action]

  return (
    <span className={cn('px-2 py-1 rounded text-xs font-medium', colorClass)}>
      {action}
    </span>
  )
}
