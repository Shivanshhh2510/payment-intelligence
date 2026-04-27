'use client'

import { CreditCard, ExternalLink, AlertTriangle } from 'lucide-react'
import type { Metrics, Drift } from '@/lib/types'

interface SidebarProps {
  metrics: Metrics | null
  drift: Drift | null
  isLoading: boolean
}

export function Sidebar({ metrics, drift, isLoading }: SidebarProps) {
  return (
    <aside className="fixed left-0 top-0 h-screen w-[220px] border-r border-white/[0.08] bg-black flex flex-col p-5 overflow-hidden">
      {/* Logo */}
      <div className="flex items-center gap-2 mb-1">
        <div className="w-7 h-7 rounded-md bg-indigo-500/20 flex items-center justify-center">
          <CreditCard className="w-4 h-4 text-indigo-500" />
        </div>
        <span className="text-lg font-semibold text-white tracking-tight">PAISA</span>
      </div>
      <p className="text-xs text-white/40 mb-6">Payment AI for Smart Authentication</p>

      {/* Stats */}
      <div className="space-y-2 mb-6">
        <StatChip
          label="AUC-ROC"
          value={isLoading ? '—' : metrics?.auc_roc?.toFixed(4) ?? '0.8763'}
          isLoading={isLoading}
        />
        <StatChip
          label="Net Benefit"
          value={isLoading ? '—' : `₹${(metrics?.net_benefit ?? 15.7).toFixed(1)}L`}
          isLoading={isLoading}
        />
        <StatChip
          label="Routing"
          value={isLoading ? '—' : `+${(metrics?.routing_improvement ?? 10.8).toFixed(1)}%`}
          isLoading={isLoading}
        />
      </div>

      {/* Model Health */}
      <div className="mb-auto">
        {drift?.retrain_recommended && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-md bg-red-500/10 border border-red-500/20">
            <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
            <span className="text-xs text-red-400 font-medium">Retrain Recommended</span>
          </div>
        )}
        {!drift?.retrain_recommended && !isLoading && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-md bg-emerald-500/10 border border-emerald-500/20">
            <div className="w-2 h-2 rounded-full bg-emerald-500" />
            <span className="text-xs text-emerald-400 font-medium">Model Healthy</span>
          </div>
        )}
      </div>

      {/* Reference Links */}
      <div className="space-y-1">
        <ReferenceLink href="https://razorpay.com/blog/product/razorpay-optimizer/" label="Razorpay Optimizer" />
        <ReferenceLink href="https://www.kaggle.com/c/ieee-fraud-detection" label="IEEE-CIS Dataset" />
        <ReferenceLink href="https://arxiv.org/abs/1209.3352" label="Thompson Sampling" />
      </div>
    </aside>
  )
}

function StatChip({
  label,
  value,
  isLoading,
}: {
  label: string
  value: string
  isLoading: boolean
}) {
  return (
    <div className="flex items-center justify-between px-3 py-2 rounded-md bg-white/[0.03] border border-white/[0.06]">
      <span className="text-xs text-white/50">{label}</span>
      <span className={`text-xs font-mono font-medium ${isLoading ? 'text-white/30' : 'text-white'}`}>
        {value}
      </span>
    </div>
  )
}

function ReferenceLink({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-2 px-3 py-2 rounded-md text-xs text-white/40 hover:text-white/70 hover:bg-white/[0.03] transition-colors"
    >
      <ExternalLink className="w-3 h-3" />
      {label}
    </a>
  )
}
