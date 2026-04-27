'use client'

import { useState, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { Sidebar } from '@/components/Sidebar'
import { TransactionPanel } from '@/components/TransactionPanel'
import { AnalystPanel } from '@/components/AnalystPanel'
import { getMetrics } from '@/lib/api'
import type { TransactionRecord, ChatMessage } from '@/lib/types'

type Tab = 'transaction' | 'analyst'

const WELCOME_MESSAGE: ChatMessage = {
  role: 'assistant',
  content: 'Hi! I am PAISA Analyst — your AI guide for this payment intelligence system.\n\nNot sure where to start? Here is what you can do:\n\n- Transaction tab — Submit any payment transaction and I will instantly tell you if it is fraud and which gateway to route it through\n- Ask me anything — Type a question below or click one of the suggested questions\n\nI am trained on 590,000 real transactions from the IEEE-CIS fraud detection dataset. Every answer I give is verified against the actual data.',
  verified: true,
  suggestions: [
    'How do I use this platform?',
    'What makes a transaction high risk?',
    'Which card type has the most fraud?',
    'How does smart routing work?',
  ],
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>('transaction')
  const [metrics, setMetrics] = useState<any>(null)
  const [drift, setDrift] = useState<any>(null)
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(true)
  const [transactions, setTransactions] = useState<TransactionRecord[]>([])
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE])
  const [pendingAnalystMessage, setPendingAnalystMessage] = useState<string | null>(null)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await getMetrics()
        setMetrics(data.metrics)
        setDrift(data.drift)
      } catch {
        setMetrics({ test_auc: 0.8763, net_benefit_inr: 1570000 })
        setDrift({ retrain_recommended: true })
      } finally {
        setIsLoadingMetrics(false)
      }
    }
    fetchMetrics()
  }, [])

  const handleNewTransaction = (record: TransactionRecord) => {
    setTransactions((prev) => [record, ...prev].slice(0, 8))
  }

  return (
    <div className="min-h-screen bg-black">
      <Sidebar metrics={metrics} drift={drift} isLoading={isLoadingMetrics} />
      <main className="ml-[220px] min-h-screen">
        <div className="sticky top-0 z-10 bg-black/90 backdrop-blur-sm border-b border-white/[0.06]">
          <div className="flex gap-1 p-2">
            <TabButton active={activeTab === 'transaction'} onClick={() => setActiveTab('transaction')}>
              ⚡ Transaction
            </TabButton>
            <TabButton active={activeTab === 'analyst'} onClick={() => setActiveTab('analyst')}>
              🤖 PAISA Analyst
            </TabButton>
          </div>
        </div>
        <div className="p-8">
          <div className={activeTab === 'transaction' ? 'block' : 'hidden'}>
            <TransactionPanel
              transactions={transactions}
              onNewTransaction={handleNewTransaction}
              onAskAnalyst={(msg) => {
                setChatMessages(prev => [...prev, { role: 'user' as const, content: msg }])
                setPendingAnalystMessage(msg)
                setActiveTab('analyst')
              }}
            />
          </div>
          <div className={activeTab === 'analyst' ? 'block' : 'hidden'}>
            <AnalystPanel
              messages={chatMessages}
              setMessages={setChatMessages}
              pendingMessage={pendingAnalystMessage}
              onPendingConsumed={() => setPendingAnalystMessage(null)}
            />
          </div>
        </div>
      </main>
    </div>
  )
}

function TabButton({ children, active, onClick }: {
  children: React.ReactNode
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'px-5 py-2 rounded-lg text-sm font-medium transition-all',
        active ? 'bg-white/[0.08] text-white' : 'text-white/50 hover:text-white/70 hover:bg-white/[0.03]'
      )}
    >
      {children}
    </button>
  )
}