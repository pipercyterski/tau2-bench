import React, { useState } from 'react'

// Minimum event count for a rate to be displayed. Below this the cell shows
// "—" with a tooltip, since rates over a handful of events are noise.
const MIN_N = 10

const SELECTIVITY_PARTS = [
  { key: 'selectivity_backchannel', label: 'Backchannel (S_BC)', countKey: 'backchannel_total' },
  { key: 'selectivity_vocal_tic', label: 'Vocal tic (S_VT)', countKey: 'vocal_tic_total' },
  { key: 'selectivity_non_directed', label: 'Non-directed (S_ND)', countKey: 'non_directed_total' },
]

// Main-table column definitions. `better` drives the default sort direction
// and the ↓/↑ marker in the header.
const COLUMNS = [
  {
    key: 'response_latency_mean',
    label: 'Response Latency',
    sub: 'L_R',
    unit: 's',
    better: 'lower',
    countKey: 'response_total',
    desc: 'Mean seconds from the end of a user turn to the start of the agent response. Lower is better.',
  },
  {
    key: 'yield_latency_mean',
    label: 'Yield Latency',
    sub: 'L_Y',
    unit: 's',
    better: 'lower',
    countKey: 'yield_total',
    desc: 'Mean seconds for the agent to stop talking after the user interrupts. Lower is better.',
  },
  {
    key: 'response_rate',
    label: 'Responsiveness',
    sub: 'R_R',
    unit: '%',
    better: 'higher',
    countKey: 'response_total',
    desc: 'Fraction of user turns that received an agent response before the user had to speak again. Higher is better.',
  },
  {
    key: 'agent_interruption_rate',
    label: 'Interrupts',
    sub: 'I_A',
    unit: '%',
    better: 'lower',
    countKey: 'response_total',
    desc: 'How often the agent starts talking over the user, per user turn. Lower is better.',
  },
  {
    key: 'selectivity',
    label: 'Selectivity',
    sub: 'S',
    unit: '%',
    better: 'higher',
    desc: 'How well the agent ignores audio not directed at it: mean of backchannel (S_BC), vocal tic (S_VT), and non-directed speech (S_ND) correct-rates. Higher is better.',
  },
]

// Rows shown in the expanded per-domain breakdown.
const DETAIL_METRICS = [
  { key: 'response_latency_mean', label: 'L_R Response latency', unit: 's', countKey: 'response_total' },
  { key: 'yield_latency_mean', label: 'L_Y Yield latency', unit: 's', countKey: 'yield_total' },
  { key: 'response_rate', label: 'R_R Response rate', unit: '%', countKey: 'response_total' },
  { key: 'yield_rate', label: 'R_Y Yield rate', unit: '%', countKey: 'yield_total' },
  { key: 'agent_interruption_rate', label: 'I_A Agent interrupts', unit: '%', countKey: 'response_total' },
  { key: 'selectivity_backchannel', label: 'S_BC Backchannel', unit: '%', countKey: 'backchannel_total' },
  { key: 'selectivity_vocal_tic', label: 'S_VT Vocal tic', unit: '%', countKey: 'vocal_tic_total' },
  { key: 'selectivity_non_directed', label: 'S_ND Non-directed', unit: '%', countKey: 'non_directed_total' },
]

const getPanel = (interactionMetrics, domainKey) => {
  if (!interactionMetrics) return null
  return domainKey === 'overall'
    ? interactionMetrics.overall || null
    : interactionMetrics.domains?.[domainKey] || null
}

const getCount = (panel, countKey) => {
  const n = panel?.counts?.[countKey]
  return n === null || n === undefined ? null : n
}

// Returns { value, lowN, n } for a metric on a panel, applying the min-n rule.
const getMetric = (panel, column) => {
  if (!panel) return { value: null, lowN: false, n: null }

  if (column.key === 'selectivity') {
    // Mean of the selectivity components that pass the min-n rule.
    const parts = SELECTIVITY_PARTS.map((part) => ({
      value: panel[part.key],
      n: getCount(panel, part.countKey),
    })).filter((p) => p.value !== null && p.value !== undefined)
    const usable = parts.filter((p) => p.n === null || p.n >= MIN_N)
    if (usable.length === 0) {
      return { value: null, lowN: parts.length > 0, n: null }
    }
    return {
      value: usable.reduce((s, p) => s + p.value, 0) / usable.length,
      lowN: false,
      n: null,
    }
  }

  const value = panel[column.key]
  if (value === null || value === undefined) return { value: null, lowN: false, n: null }
  const n = column.countKey ? getCount(panel, column.countKey) : null
  if (n !== null && n < MIN_N) return { value: null, lowN: true, n }
  return { value, lowN: false, n }
}

const formatValue = (value, unit) => {
  if (value === null || value === undefined) return '—'
  return unit === 's' ? `${value.toFixed(2)}s` : `${(value * 100).toFixed(1)}%`
}

const InteractionQualityTable = ({ models, domain, domains, onModelClick }) => {
  const [sortKey, setSortKey] = useState('response_latency_mean')
  const [sortAscending, setSortAscending] = useState(true)
  const [expandedRows, setExpandedRows] = useState(new Set())

  const sortColumn = COLUMNS.find((c) => c.key === sortKey) || COLUMNS[0]

  const handleSort = (column) => {
    if (column.key === sortKey) {
      setSortAscending(!sortAscending)
    } else {
      setSortKey(column.key)
      setSortAscending(column.better === 'lower')
    }
  }

  const toggleExpand = (key) => {
    setExpandedRows((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const rows = models
    .map((model) => {
      const panel = getPanel(model.data.interactionMetrics, domain)
      return { ...model, panel }
    })
    .sort((a, b) => {
      const aMetric = getMetric(a.panel, sortColumn).value
      const bMetric = getMetric(b.panel, sortColumn).value
      // Models without metrics sort last regardless of direction
      if (aMetric === null && bMetric === null) return 0
      if (aMetric === null) return 1
      if (bMetric === null) return -1
      return sortAscending ? aMetric - bMetric : bMetric - aMetric
    })

  if (rows.length === 0) return null

  return (
    <div className="interaction-quality-section">
      <div className="interaction-quality-header">
        <h3 className="interaction-quality-title">🎛️ Interaction Quality</h3>
        <p className="interaction-quality-subtitle">
          Conversational dynamics measured from the same full-duplex trajectories:
          how fast, how responsive, and how selective each model is on the open
          audio channel. Independent of task success above.{' '}
          <a
            href="https://github.com/sierra-research/tau2-bench/blob/main/docs/interaction-metrics.md"
            target="_blank"
            rel="noopener noreferrer"
          >
            Metric definitions →
          </a>
        </p>
      </div>
      <div className="metrics-table-container">
        <table className="reliability-table interaction-quality-table">
          <thead>
            <tr>
              <th>Model</th>
              {COLUMNS.map((column) => (
                <th
                  key={column.key}
                  className={`iq-metric-header ${sortKey === column.key ? 'iq-sorted' : ''}`}
                  onClick={() => handleSort(column)}
                  title={column.desc}
                >
                  <span className="iq-header-label">{column.label}</span>
                  <span className="iq-header-sub">
                    {column.sub} {column.better === 'lower' ? '↓' : '↑'}
                    {sortKey === column.key && (
                      <span className="iq-sort-arrow">{sortAscending ? '▲' : '▼'}</span>
                    )}
                  </span>
                </th>
              ))}
              <th className="expand-header"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((model) => {
              const isExpanded = expandedRows.has(model.key)
              return (
                <React.Fragment key={model.key}>
                  <tr className={`model-row ${isExpanded ? 'expanded' : ''}`}>
                    <td className="model-info">
                      <div className="model-name">
                        {model.displayName}
                        {model.data.reasoningEffort && (
                          <span className="iq-reasoning">{model.data.reasoningEffort}</span>
                        )}
                        {model.data.submissionType === 'custom' && (
                          <span
                            className="iq-custom-badge"
                            title="Custom submission — evaluated with a non-default scaffold or user simulator; not directly comparable to standard rows"
                          >
                            custom
                          </span>
                        )}
                      </div>
                    </td>
                    {COLUMNS.map((column) => {
                      const { value, lowN, n } = getMetric(model.panel, column)
                      if (value === null) {
                        return (
                          <td key={column.key} className="metric-cell">
                            <span
                              className="iq-no-data"
                              title={
                                lowN
                                  ? `Not shown: fewer than ${MIN_N} events${n !== null ? ` (n=${n})` : ''}`
                                  : 'Interaction metrics not available for this submission'
                              }
                            >
                              —
                            </span>
                          </td>
                        )
                      }
                      return (
                        <td key={column.key} className="metric-cell iq-value-cell">
                          {formatValue(value, column.unit)}
                        </td>
                      )
                    })}
                    <td className="expand-cell" onClick={() => toggleExpand(model.key)}>
                      <span className={`expand-caret ${isExpanded ? 'open' : ''}`}>▶</span>
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr className="domain-detail-row">
                      <td colSpan={COLUMNS.length + 2} className="domain-detail-cell">
                        {model.panel || model.data.interactionMetrics ? (
                          <div className="iq-detail-container">
                            <table className="iq-detail-table">
                              <thead>
                                <tr>
                                  <th>Metric</th>
                                  {domains.map((d) => (
                                    <th key={d.key}>{d.label}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {DETAIL_METRICS.map((metric) => (
                                  <tr key={metric.key}>
                                    <td className="iq-detail-metric">{metric.label}</td>
                                    {domains.map((d) => {
                                      const panel = getPanel(model.data.interactionMetrics, d.key)
                                      const value = panel?.[metric.key]
                                      const n = getCount(panel, metric.countKey)
                                      if (value === null || value === undefined) {
                                        return (
                                          <td key={d.key}>
                                            <span className="iq-no-data">—</span>
                                          </td>
                                        )
                                      }
                                      return (
                                        <td key={d.key} className={n !== null && n < MIN_N ? 'iq-low-n' : ''}>
                                          {formatValue(value, metric.unit)}
                                          {n !== null && (
                                            <span
                                              className="iq-count"
                                              title={n < MIN_N ? `Low event count — treat with caution` : `${n} events`}
                                            >
                                              {' '}n={n}
                                            </span>
                                          )}
                                        </td>
                                      )
                                    })}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            <div className="iq-detail-footnote">
                              Rates over fewer than {MIN_N} events are hidden from the summary
                              table above; they are shown here with their event counts.
                            </div>
                          </div>
                        ) : (
                          <div className="iq-detail-empty">
                            Interaction metrics have not been computed for this submission yet.
                          </div>
                        )}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default InteractionQualityTable
