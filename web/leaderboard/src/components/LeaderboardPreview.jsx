import { useState, useEffect } from 'react'
import './LeaderboardPreview.css'

const MEDAL_EMOJI = ['🥇', '🥈', '🥉']

const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`

const NO_CACHE = { cache: 'no-cache' }

const CORE_DOMAINS = ['retail', 'airline', 'telecom']

// Average pass^1 across the three core domains; null unless all three exist.
const corePass1 = (results) => {
  const values = CORE_DOMAINS.map((d) => results[d]?.pass_1)
  if (values.some((v) => v == null)) return null
  return values.reduce((s, v) => s + v, 0) / values.length
}

// One preview card per leaderboard bucket: Core (τ²-bench), Knowledge
// (τ-knowledge), and Voice (τ-voice), each showing its Overall top 3.
// TODO(voice-banking): when banking is supported in voice mode, its scores
// belong in the Knowledge bucket (as a text/voice split), not in Voice.
function LeaderboardPreview({ onViewFullLeaderboard }) {
  const [coreTop3, setCoreTop3] = useState([])
  const [knowledgeTop3, setKnowledgeTop3] = useState([])
  const [voiceTop3, setVoiceTop3] = useState([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadPreviewData()
  }, [])

  const loadPreviewData = async () => {
    try {
      const manifestResponse = await fetch(`${SUBMISSIONS_BASE}/manifest.json`, NO_CACHE)
      if (!manifestResponse.ok) return
      const manifest = await manifestResponse.json()

      const textDirs = [...(manifest.submissions || []), ...(manifest.legacy_submissions || [])]
      const voiceDirs = manifest.voice_submissions || []

      // Load text submissions once; they feed both the Core and Knowledge cards.
      const coreModels = []
      const knowledgeModels = []
      for (const dir of textDirs) {
        try {
          const res = await fetch(`${SUBMISSIONS_BASE}/${dir}/submission.json`, NO_CACHE)
          if (!res.ok) continue
          const sub = await res.json()

          // Only include standard submissions
          if (sub.submission_type && sub.submission_type !== 'standard') continue

          const core = corePass1(sub.results)
          if (core != null) {
            coreModels.push({
              name: sub.model_name,
              org: sub.model_organization,
              score: core,
            })
          }

          const banking = sub.results.banking_knowledge?.pass_1
          if (banking != null) {
            knowledgeModels.push({
              name: sub.model_name,
              org: sub.model_organization,
              score: banking,
            })
          }
        } catch { /* skip */ }
      }

      coreModels.sort((a, b) => b.score - a.score)
      setCoreTop3(coreModels.slice(0, 3))
      knowledgeModels.sort((a, b) => b.score - a.score)
      setKnowledgeTop3(knowledgeModels.slice(0, 3))

      // Load voice submissions and compute overall pass^1.
      const voiceModels = []
      for (const dir of voiceDirs) {
        try {
          const res = await fetch(`${SUBMISSIONS_BASE}/${dir}/submission.json`, NO_CACHE)
          if (!res.ok) continue
          const sub = await res.json()

          if (sub.submission_type && sub.submission_type !== 'standard') continue

          const values = CORE_DOMAINS
            .map(d => sub.results[d]?.pass_1)
            .filter(v => v != null)

          if (values.length > 0) {
            voiceModels.push({
              name: sub.model_name,
              org: sub.voice_config?.provider || sub.model_organization,
              score: values.reduce((s, v) => s + v, 0) / values.length,
            })
          }
        } catch { /* skip */ }
      }

      voiceModels.sort((a, b) => b.score - a.score)
      setVoiceTop3(voiceModels.slice(0, 3))
    } catch (err) {
      console.warn('Failed to load leaderboard preview:', err)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="leaderboard-preview">
        <div className="preview-loading">Loading leaderboard...</div>
      </div>
    )
  }

  // Newest tracks first, matching the leaderboard toggle order.
  const cards = [
    { badge: 'τ-knowledge', badgeClass: 'knowledge', mode: 'Text', domains: 'Banking', models: knowledgeTop3 },
    { badge: 'τ-voice', badgeClass: 'voice', mode: 'Voice', domains: 'Retail · Airline · Telecom', models: voiceTop3 },
    { badge: 'τ²-bench', badgeClass: 'core', mode: 'Text', domains: 'Retail · Airline · Telecom', models: coreTop3 },
  ]

  return (
    <div className="leaderboard-preview">
      <div className="preview-tables">
        {cards.map((card) => (
          <div className="preview-table-wrapper" key={card.badge}>
            <h3 className="preview-table-title">
              <span className={`preview-mode-badge ${card.badgeClass}`}>{card.badge}</span>
              <span className="preview-table-subtitle">
                <span className="preview-mode">{card.mode}</span>
                <span className="preview-subtitle-divider" aria-hidden="true" />
                {card.domains}
              </span>
            </h3>
            <table className="preview-table">
              <thead>
                <tr>
                  <th className="preview-rank-col">#</th>
                  <th className="preview-model-col">Model</th>
                  <th className="preview-score-col">Pass^1</th>
                </tr>
              </thead>
              <tbody>
                {card.models.map((model, i) => (
                  <tr key={i} className="preview-row">
                    <td className="preview-rank">{MEDAL_EMOJI[i]}</td>
                    <td className="preview-model">
                      <span className="preview-model-name">{model.name}</span>
                      <span className="preview-model-org">{model.org}</span>
                    </td>
                    <td className="preview-score">{model.score.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="preview-more">⋯</div>
          </div>
        ))}
      </div>
      <button className="preview-cta" onClick={onViewFullLeaderboard}>
        View Full Leaderboard →
      </button>
    </div>
  )
}

export default LeaderboardPreview
