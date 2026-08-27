import React, { useState } from 'react';
import axios from 'axios';

/**
 * QuickAnalysisPanel Component
 * 
 * Provides FAST structural analysis alternative to full MD simulation
 * - Helix propensity (Chou-Fasman)
 * - Amphipathic moment
 * - Hydrophobic face identification
 * 
 * Typical execution time: <1 second vs 20+ minutes for MD
 */

const QuickAnalysisPanel = ({ candidates, onAnalysisComplete, apiBase }) => {
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const runQuickAnalysis = async () => {
    setAnalyzing(true);
    setError(null);
    setResults(null);

    try {
      const response = await axios.post(`${apiBase}/quick-analysis`, {
        candidates: candidates
      });

      setResults(response.data);
      
      // Pass results back to parent
      if (onAnalysisComplete) {
        onAnalysisComplete(response.data);
      }

    } catch (err) {
      console.error('Quick analysis error:', err);
      setError(err.response?.data?.error || 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const getHelixCategory = (propensity) => {
    if (propensity > 1.15) return { text: 'Strong Helix Former', class: 'strong' };
    if (propensity > 1.05) return { text: 'Moderate Helix Former', class: 'moderate' };
    if (propensity > 0.95) return { text: 'Neutral', class: 'neutral' };
    return { text: 'Helix Breaker', class: 'weak' };
  };

  const getAmphipathicCategory = (moment) => {
    if (moment > 0.5) return { text: 'Strongly Amphipathic', class: 'strong' };
    if (moment > 0.3) return { text: 'Moderately Amphipathic', class: 'moderate' };
    if (moment > 0.15) return { text: 'Weakly Amphipathic', class: 'weak' };
    return { text: 'Non-Amphipathic', class: 'none' };
  };

  return (
    <div className="quick-analysis-panel">
      <div className="panel-header">
        <h3>⚡ Quick Structural Analysis</h3>
        <p className="panel-description">
          Fast alternative to MD simulation. Analyzes helix propensity and amphipathic character.
          <br />
          <strong>Time: ~1 second</strong> (vs 20+ minutes for full MD)
        </p>
      </div>

      <button 
        className="run-quick-analysis-btn"
        onClick={runQuickAnalysis}
        disabled={analyzing || !candidates || candidates.length === 0}
      >
        {analyzing ? '⚙️ Analyzing...' : '⚡ Run Quick Analysis'}
      </button>

      {error && (
        <div className="error-message">
          <span>❌ {error}</span>
        </div>
      )}

      {results && (
        <div className="quick-results">
          <div className="results-summary">
            <h4>📊 Summary</h4>
            <div className="summary-grid">
              <div className="summary-stat">
                <span className="stat-label">Analyzed:</span>
                <span className="stat-value">{results.summary.total_analyzed} candidates</span>
              </div>
              <div className="summary-stat">
                <span className="stat-label">Avg Helix Score:</span>
                <span className="stat-value">{(results.summary.avg_helix_score * 100).toFixed(1)}%</span>
              </div>
              <div className="summary-stat">
                <span className="stat-label">Avg Amphipathic:</span>
                <span className="stat-value">{(results.summary.avg_amphipathic_score * 100).toFixed(1)}%</span>
              </div>
              <div className="summary-stat">
                <span className="stat-label">Favorable:</span>
                <span className="stat-value">{results.summary.favorable_count} / {results.summary.total_analyzed}</span>
              </div>
            </div>
          </div>

          <div className="candidate-results">
            <h4>🎯 Top Candidates</h4>
            {results.candidates.slice(0, 10).map((candidate, idx) => {
              const analysis = candidate.quick_structural_analysis;
              const helix = analysis.helix_analysis;
              const amphipathic = analysis.amphipathic_analysis;
              
              const helixCat = getHelixCategory(helix.avg_propensity);
              const amphipathicCat = getAmphipathicCategory(amphipathic.normalized_moment);

              return (
                <div key={idx} className={`quick-candidate-card ${analysis.is_favorable_for_nes ? 'favorable' : ''}`}>
                  <div className="candidate-header">
                    <span className="candidate-rank">#{idx + 1}</span>
                    <code className="candidate-sequence">{candidate.sequence}</code>
                    <span className="candidate-position">
                      Residues {candidate.start}-{candidate.end}
                    </span>
                  </div>

                  <div className="candidate-metrics">
                    <div className="metric-row">
                      <span className="metric-label">Combined Score:</span>
                      <div className="score-bar">
                        <div 
                          className="score-fill"
                          style={{ width: `${analysis.combined_score * 100}%` }}
                        />
                      </div>
                      <span className="metric-value">{(analysis.combined_score * 100).toFixed(1)}%</span>
                    </div>

                    <div className="metric-row">
                      <span className="metric-label">Helix Propensity:</span>
                      <span className={`category-badge ${helixCat.class}`}>
                        {helixCat.text}
                      </span>
                      <span className="metric-value">{helix.avg_propensity.toFixed(3)}</span>
                    </div>

                    <div className="metric-row">
                      <span className="metric-label">Amphipathic:</span>
                      <span className={`category-badge ${amphipathicCat.class}`}>
                        {amphipathicCat.text}
                      </span>
                      <span className="metric-value">{amphipathic.normalized_moment.toFixed(3)}</span>
                    </div>

                    <div className="metric-row">
                      <span className="metric-label">Hydrophobic Face:</span>
                      <code className="hydrophobic-face">{amphipathic.hydrophobic_face_sequence}</code>
                    </div>

                    <div className="metric-row">
                      <span className="metric-label">Face Contrast:</span>
                      <span className="metric-value">{amphipathic.face_contrast.toFixed(2)}</span>
                      <span className="metric-unit">(hydrophobic - hydrophilic)</span>
                    </div>
                  </div>

                  {analysis.is_favorable_for_nes && (
                    <div className="favorable-badge">
                      ✅ Favorable NES Structure
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <style jsx>{`
        .quick-analysis-panel {
          background: var(--card-bg);
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .panel-header h3 {
          margin: 0 0 8px 0;
          color: var(--text-primary);
          font-size: 1.4em;
        }

        .panel-description {
          margin: 0 0 16px 0;
          color: var(--text-secondary);
          font-size: 0.95em;
          line-height: 1.5;
        }

        .run-quick-analysis-btn {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 8px;
          font-size: 1.05em;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
          width: 100%;
        }

        .run-quick-analysis-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .run-quick-analysis-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .error-message {
          margin-top: 16px;
          padding: 12px;
          background: #fee;
          border-left: 4px solid #e00;
          border-radius: 4px;
          color: #c00;
        }

        .quick-results {
          margin-top: 24px;
        }

        .results-summary {
          background: var(--bg-secondary);
          padding: 16px;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .results-summary h4 {
          margin: 0 0 12px 0;
          color: var(--text-primary);
        }

        .summary-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 12px;
        }

        .summary-stat {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .stat-label {
          font-size: 0.85em;
          color: var(--text-secondary);
          font-weight: 500;
        }

        .stat-value {
          font-size: 1.2em;
          color: var(--text-primary);
          font-weight: 700;
        }

        .candidate-results h4 {
          margin: 0 0 16px 0;
          color: var(--text-primary);
        }

        .quick-candidate-card {
          background: var(--bg-secondary);
          border: 2px solid transparent;
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 12px;
          transition: all 0.3s ease;
        }

        .quick-candidate-card.favorable {
          border-color: #4caf50;
          background: linear-gradient(to right, rgba(76, 175, 80, 0.05), transparent);
        }

        .quick-candidate-card:hover {
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
          transform: translateY(-2px);
        }

        .candidate-header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 12px;
          flex-wrap: wrap;
        }

        .candidate-rank {
          background: var(--accent-color);
          color: white;
          padding: 4px 10px;
          border-radius: 12px;
          font-weight: 700;
          font-size: 0.9em;
        }

        .candidate-sequence {
          font-family: 'Courier New', monospace;
          background: rgba(0,0,0,0.05);
          padding: 6px 12px;
          border-radius: 4px;
          font-size: 0.95em;
          font-weight: 600;
        }

        .candidate-position {
          color: var(--text-secondary);
          font-size: 0.9em;
        }

        .candidate-metrics {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .metric-row {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        .metric-label {
          font-weight: 600;
          color: var(--text-secondary);
          min-width: 140px;
          font-size: 0.9em;
        }

        .metric-value {
          color: var(--text-primary);
          font-weight: 700;
        }

        .metric-unit {
          color: var(--text-secondary);
          font-size: 0.85em;
          font-style: italic;
        }

        .score-bar {
          flex: 1;
          height: 8px;
          background: rgba(0,0,0,0.1);
          border-radius: 4px;
          overflow: hidden;
          max-width: 200px;
        }

        .score-fill {
          height: 100%;
          background: linear-gradient(90deg, #667eea, #764ba2);
          transition: width 0.5s ease;
        }

        .category-badge {
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 0.85em;
          font-weight: 600;
        }

        .category-badge.strong {
          background: #4caf50;
          color: white;
        }

        .category-badge.moderate {
          background: #2196f3;
          color: white;
        }

        .category-badge.neutral {
          background: #ff9800;
          color: white;
        }

        .category-badge.weak {
          background: #9e9e9e;
          color: white;
        }

        .category-badge.none {
          background: #757575;
          color: white;
        }

        .hydrophobic-face {
          font-family: 'Courier New', monospace;
          background: rgba(255, 165, 0, 0.2);
          padding: 4px 8px;
          border-radius: 4px;
          font-weight: 600;
        }

        .favorable-badge {
          margin-top: 12px;
          padding: 8px;
          background: rgba(76, 175, 80, 0.1);
          border-left: 4px solid #4caf50;
          border-radius: 4px;
          color: #2e7d32;
          font-weight: 600;
        }
      `}</style>
    </div>
  );
};

export default QuickAnalysisPanel;
