'use client'

import { Check, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { InlineChart } from './InlineChart'
import type { ChatMessage as ChatMessageType } from '@/lib/types'

interface ChatMessageProps {
  message: ChatMessageType
  onSuggestionClick?: (text: string) => void
}

function formatContent(content: string) {
  return content
    .split('\n')
    .map((line) => {
      return line
        .replace(/\*\*(.*?)\*\*/g, '<strong style="color:rgba(255,255,255,0.95);font-weight:600">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
    })
    .join('<br/>')
}

export function ChatMessage({ message, onSuggestionClick }: ChatMessageProps) {
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex gap-3 mb-4', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center flex-shrink-0 mt-1">
          <span className="text-xs font-semibold text-white">P</span>
        </div>
      )}
      <div className={cn('max-w-[80%]', isUser ? 'order-1' : 'order-2')}>
        <div
          className={cn(
            'px-4 py-3 rounded-2xl text-sm leading-relaxed',
            isUser
              ? 'bg-indigo-500/20 text-white rounded-br-md border border-indigo-500/30'
              : 'bg-white/[0.03] text-white/90 border border-white/[0.06] rounded-bl-md'
          )}
        >
          {/* Render content with basic markdown */}
          <div
            className="whitespace-pre-wrap"
            dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
          />

          {message.chart_data?.items?.length ? (
            <InlineChart data={message.chart_data} />
          ) : null}
        </div>

        {/* Verified badge */}
        {!isUser && (
          <div className="flex items-center gap-1 mt-1.5 ml-1">
            {message.verified ? (
              <>
                <Check className="w-3 h-3 text-emerald-400" />
                <span className="text-[10px] text-emerald-400">verified against dataset</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3 h-3 text-amber-400" />
                <span className="text-[10px] text-amber-400">unverified — check API connection</span>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}