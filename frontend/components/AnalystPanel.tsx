'use client'

import { useState, useRef, useEffect, Dispatch, SetStateAction } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { ChatMessage as ChatMessageComponent } from './ChatMessage'
import { sendChatMessage } from '@/lib/api'
import type { ChatMessage } from '@/lib/types'

interface AnalystPanelProps {
  messages: ChatMessage[]
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>
  pendingMessage?: string | null
  onPendingConsumed?: () => void
}

export function AnalystPanel({ messages, setMessages, pendingMessage, onPendingConsumed }: AnalystPanelProps) {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (pendingMessage && !isLoading) {
      onPendingConsumed?.()
      // Don't add user message here — already added in page.tsx
      // Just trigger the AI response
      const sendPending = async () => {
        setIsLoading(true)
        try {
          const history = messages
            .filter((_, i) => i < messages.length - 1)
            .map((m) => ({ role: m.role, content: m.content }))
          const response = await sendChatMessage({ message: pendingMessage, history })
          let chartData = undefined
          if (response.chart?.labels && response.chart?.values) {
            chartData = {
              title: response.chart.title || '',
              items: response.chart.labels.map((name: string, i: number) => ({
                name,
                value: Number(response.chart!.values[i].toFixed(2)),
              })),
            }
          }
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: response.response,
            chart_data: chartData,
            verified: response.verified ?? true,
            suggestions: response.suggestions,
          }])
        } catch {
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: 'Unable to reach the API. Make sure FastAPI is running: `uvicorn api.main:app --reload --port 8001`',
            verified: false,
          }])
        } finally {
          setIsLoading(false)
        }
      }
      sendPending()
    }
  }, [pendingMessage]) // eslint-disable-line

  const handleSend = async (messageText?: string, skipAddingUserMsg?: boolean) => {
    const text = messageText || input.trim()
    if (!text || isLoading) return

    if (!skipAddingUserMsg) {
      setMessages((prev) => [...prev, { role: 'user' as const, content: text }])
    }
    setInput('')
    setIsLoading(true)

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }))
      const response = await sendChatMessage({ message: text, history })

      // Build chart_data from response.chart
      let chartData: import('@/lib/types').ChartPayload | undefined = undefined
      if (response.chart?.labels && response.chart?.values) {
        chartData = {
          title: response.chart.title || '',
          items: response.chart.labels.map((name: string, i: number) => ({
            name,
            value: Number(response.chart!.values[i].toFixed(2)),
          })),
        }
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        chart_data: chartData,
        verified: response.verified ?? true,
        suggestions: response.suggestions,
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (e) {
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: 'Unable to reach the API. Make sure the FastAPI backend is running on port 8001 with `uvicorn api.main:app --reload --port 8001`',
        verified: false,
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Get suggestions from last assistant message
  const lastAssistantMsg = [...messages].reverse().find(m => m.role === 'assistant')
  const currentSuggestions = lastAssistantMsg?.suggestions || [
    'How do I use this platform?',
    'Which card type has the most fraud?',
    'How does routing work?',
  ]

  return (
    <div className="flex flex-col h-[calc(100vh-140px)]">
      <div className="mb-4">
        <h1 className="text-2xl font-semibold text-white mb-1">PAISA Analyst</h1>
        <p className="text-sm text-white/50">
          Ask anything about the fraud data, model, or routing decisions • Self-verifying AI
        </p>
      </div>

      <div className="flex-1 overflow-y-auto pr-2 space-y-2">
        {messages.map((message, idx) => (
          <ChatMessageComponent key={idx} message={message} onSuggestionClick={handleSend} />
        ))}
        {isLoading && (
          <div className="flex gap-3 mb-4">
            <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center flex-shrink-0">
              <span className="text-xs font-semibold text-white">P</span>
            </div>
            <div className="px-4 py-3 rounded-2xl rounded-bl-md bg-white/[0.03] border border-white/[0.06] flex items-center gap-2">
              <Loader2 className="w-4 h-4 text-white/40 animate-spin" />
              <span className="text-xs text-white/30">Analyzing and verifying...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggestion chips — from last AI message */}
      <div className="flex flex-wrap gap-2 py-3">
        {currentSuggestions.slice(0, 3).map((chip) => (
          <button
            key={chip}
            onClick={() => handleSend(chip)}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-full text-xs text-white/50 bg-white/[0.03] border border-white/[0.06] hover:border-indigo-500/50 hover:text-white/70 transition-colors disabled:opacity-50 text-left"
            style={{ maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            title={chip}
          >
            {chip}
          </button>
        ))}
      </div>

      <div className="flex gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about fraud patterns, model performance, routing decisions..."
          className="flex-1 px-4 py-3 bg-white/[0.03] border border-white/[0.08] rounded-xl text-white text-sm placeholder-white/30 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/50"
        />
        <button
          onClick={() => handleSend()}
          disabled={isLoading || !input.trim()}
          className="px-5 py-3 bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-white transition-colors"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}