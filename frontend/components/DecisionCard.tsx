'use client'

import { cn } from '@/lib/utils'
import type { TransactionScore } from '@/lib/types'

interface DecisionCardProps {
  score: TransactionScore
}

export function DecisionCard({ score }: DecisionCardProps) {
  const riskColorClass = {
    LOW: 'text-emerald-400',
    MEDIUM: 'text-amber-400',
    HIGH: 'text-orange-500',
    CRITICAL: 'text-red-500',
  }[score.risk_level]

  const actionColorClass = {
    ALLOW: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    REVIEW: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    BLOCK: 'bg-red-500/20 text-red-400 border-red-500/30',
  }[score.action]

  return (
    <div className="animate-in fade-in slide-in-from-bottom-2 duration-300 p-5 rounded-xl bg-white/[0.02] border border-white/[0.08]">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-white/40 mb-1">Fraud Risk</p>
          <p className={cn('text-3xl font-semibold', riskColorClass)}>{score.risk_level}</p>
          <p className="text-sm text-white/50 mt-1">
            {(score.fraud_probability * 100).toFixed(1)}% probability
          </p>
        </div>
        <span
          className={cn(
            'px-3 py-1.5 rounded-full text-xs font-medium border',
            actionColorClass
          )}
        >
          {score.action}
        </span>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <MetricItem label="Route To" value={score.route_to} />
        <MetricItem label="Expected SR" value={`${(score.expected_sr * 100).toFixed(1)}%`} />
        <MetricItem label="Latency" value={`${score.latency}ms`} />
      </div>

      {/* Gateway Badge */}
      <div className="mb-4">
        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-violet-500/20 text-violet-300 border border-violet-500/30">
          {score.gateway}
        </span>
      </div>

      {/* Explanation */}
      <div className="pt-4 border-t border-white/[0.06]">
        <p className="text-sm text-white/60 leading-relaxed">{score.explanation}</p>
      </div>
    </div>
  )
}

function MetricItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-lg bg-white/[0.03]">
      <p className="text-[10px] uppercase tracking-wider text-white/40 mb-1">{label}</p>
      <p className="text-sm font-medium text-white">{value}</p>
    </div>
  )
}
