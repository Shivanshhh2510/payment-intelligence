'use client'

import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie,
} from 'recharts'
import type { ChartPayload } from '@/lib/types'

interface InlineChartProps {
  data: ChartPayload
}

const COLORS = ['#6366f1','#10b981','#f59e0b','#f43f5e','#8b5cf6','#06b6d4','#ec4899','#14b8a6']

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: '#0a0f1e',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '8px',
        padding: '8px 12px',
        fontSize: '12px',
      }}>
        <div style={{ color: 'rgba(255,255,255,0.5)', marginBottom: '2px' }}>{label}</div>
        <div style={{ fontWeight: 600, color: '#6366f1' }}>{payload[0].value}%</div>
      </div>
    )
  }
  return null
}

export function InlineChart({ data }: InlineChartProps) {
  if (!data?.items?.length) return null

  const max = Math.max(...data.items.map(d => d.value))
  const usePie = data.items.length <= 4

  return (
    <div style={{
      marginTop: '14px',
      background: 'rgba(0,0,0,0.3)',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: '10px',
      padding: '16px',
    }}>
      {data.title && (
        <div style={{
          fontSize: '10px',
          color: 'rgba(255,255,255,0.35)',
          textTransform: 'uppercase',
          letterSpacing: '0.8px',
          marginBottom: '14px',
          fontWeight: 600,
        }}>
          {data.title}
        </div>
      )}

      {usePie ? (
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={data.items}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={70}
              label={({ name, value }) => `${name}: ${value}%`}
              labelLine={true}
            >
              {data.items.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} opacity={0.85} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: '#0a0f1e',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '8px',
                fontSize: '12px',
                color: 'white',
              }}
              formatter={(v: any) => [`${v}%`]}
            />
          </PieChart>
        </ResponsiveContainer>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data.items} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
            <XAxis
              dataKey="name"
              tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.35)' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.35)' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
              domain={[0, Math.ceil(max * 1.25)]}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(99,102,241,0.06)' }} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.items.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.value === max ? '#f43f5e' : COLORS[i % COLORS.length]}
                  opacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}