/**
 * Enhanced MD Refinement Panel - Two CRM1 Analysis Modes
 * 1. CRM1/NES Screening (fast, comprehensive)
 * 2. CRM1 MD Docking (slow, accurate molecular dynamics)
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';

const MDRefinementPanel = ({ structureData, API_BASE, theme }) => {
  const [mdAvailable, setMdAvailable] = useState(false);
  const [mdJobId, setMdJobId] = useState(null);
  const [mdStatus, setMdStatus] = useState(null);
  const [mdResults, setMdResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState(null);

  // Check if MD is available from initial analysis
  useEffect(() => {
    if (structureData?.md_refinement_available) {
      setMdAvailable(true);
    }
  }, [structureData]);

  // Poll job status when job is running
  useEffect(() => {
    if (!mdJobId || !mdStatus || mdStatus.status === 'completed' || mdStatus.status === 'failed') {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/md_status/${mdJobId}`);
        setMdStatus(response.data);

        if (response.data.status === 'completed') {
          // Fetch results
          const resultsResponse = await axios.get(`${API_BASE}/md_result/${mdJobId}`);
          setMdResults(resultsResponse.data);
          setLoading(false);
        } else if (response.data.status === 'failed') {
          setLoading(false);
        }
      } catch (error) {
        console.error('Error polling MD status:', error);
      }
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(pollInterval);
  }, [mdJobId, mdStatus, API_BASE]);

  const handleCRM1Screening = async () => {
    if (!structureData?.candidates || structureData.candidates.length === 0) {
      alert('No NES candidates to analyze');
      return;
    }

    setLoading(true);
    setSelectedMode('screening');
    setMdJobId(null);
    setMdStatus(null);
    setMdResults(null);

    try {
      const response = await axios.post(`${API_BASE}/crm1_screening`, {
        model_id: structureData.model_id,
        pdb_content: structureData.pdb_content,
        candidates: structureData.candidates.slice(0, 20) // Top 20 for screening
      });

      setMdResults(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error running CRM1 screening:', error);
      alert('Failed to run CRM1 screening');
      setLoading(false);
    }
  };

  const handleCRM1MD = async () => {
    if (!structureData?.candidates || structureData.candidates.length === 0) {
      alert('No NES candidates to analyze');
      return;
    }

    setLoading(true);
    setSelectedMode('md_docking');
    setMdResults(null);

    try {
      const response = await axios.post(`${API_BASE}/crm1_md_docking`, {
        model_id: structureData.model_id,
        pdb_content: structureData.pdb_content,
        candidates: structureData.candidates.slice(0, 5) // Top 5 for MD
      });

      setMdJobId(response.data.job_id);
      setMdStatus(response.data);
    } catch (error) {
      console.error('Error submitting CRM1 MD job:', error);
      alert('Failed to submit CRM1 MD docking job');
      setLoading(false);
    }
  };

  if (!mdAvailable) {
    return null;
  }

  return (
    <div style={{
      background: theme === 'dark' ? '#1a1a2e' : '#ffffff',
      borderRadius: '12px',
      padding: '24px',
      marginTop: '20px',
      border: `1px solid ${theme === 'dark' ? '#2d2d44' : '#e5e7eb'}`
    }}>
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{
          color: theme === 'dark' ? '#e0e0e0' : '#1f2937',
          fontSize: '18px',
          fontWeight: '600',
          marginBottom: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <span style={{ fontSize: '24px' }}>🧬</span>
          CRM1/NES Binding Analysis
        </h3>
        <p style={{
          color: theme === 'dark' ? '#9ca3af' : '#6b7280',
          fontSize: '14px',
          margin: 0
        }}>
          Validate NES motifs through CRM1 interaction analysis
        </p>
      </div>

      {/* Two Analysis Buttons */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '16px',
        marginBottom: '20px'
      }}>
        {/* CRM1/NES Screening Button */}
        <button
          onClick={handleCRM1Screening}
          disabled={loading}
          style={{
            background: loading && selectedMode === 'screening' 
              ? 'linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%)'
              : 'linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '16px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading && selectedMode !== 'screening' ? 0.5 : 1,
            transition: 'all 0.3s',
            fontWeight: '600',
            fontSize: '14px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '8px',
            boxShadow: '0 4px 6px rgba(139, 92, 246, 0.3)'
          }}
        >
          <div style={{ fontSize: '32px' }}>⚡</div>
          <div>CRM1/NES Screening</div>
          <div style={{ fontSize: '11px', opacity: 0.9 }}>
            Fast • Comprehensive • ~30 sec
          </div>
          <div style={{ fontSize: '10px', opacity: 0.8, textAlign: 'center', marginTop: '4px' }}>
            Consensus patterns, ML scoring, disorder analysis, SASA, pocket detection, flexibility
          </div>
        </button>

        {/* CRM1 MD Docking Button */}
        <button
          onClick={handleCRM1MD}
          disabled={loading}
          style={{
            background: loading && selectedMode === 'md_docking'
              ? 'linear-gradient(135deg, #db2777 0%, #ec4899 100%)'
              : 'linear-gradient(135deg, #ec4899 0%, #f472b6 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '16px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading && selectedMode !== 'md_docking' ? 0.5 : 1,
            transition: 'all 0.3s',
            fontWeight: '600',
            fontSize: '14px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '8px',
            boxShadow: '0 4px 6px rgba(236, 72, 153, 0.3)'
          }}
        >
          <div style={{ fontSize: '32px' }}>🎯</div>
          <div>CRM1 MD Docking</div>
          <div style={{ fontSize: '11px', opacity: 0.9 }}>
            Accurate • Slow • 30-60 min
          </div>
          <div style={{ fontSize: '10px', opacity: 0.8, textAlign: 'center', marginTop: '4px' }}>
            Full OpenMM simulation with α-helix formation & hydrophobic groove binding
          </div>
        </button>
      </div>

      {/* Job Status Display */}
      {loading && selectedMode === 'md_docking' && mdStatus && (
        <div style={{
          background: theme === 'dark' ? '#2d2d44' : '#f3f4f6',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '16px'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '12px'
          }}>
            <div style={{
              color: theme === 'dark' ? '#e0e0e0' : '#1f2937',
              fontSize: '14px',
              fontWeight: '600'
            }}>
              {mdStatus.status === 'queued' && '⏳ Queued...'}
              {mdStatus.status === 'running' && '🧬 Running MD Simulation...'}
              {mdStatus.status === 'completed' && '✅ Completed'}
              {mdStatus.status === 'failed' && '❌ Failed'}
            </div>
            {mdStatus.progress !== undefined && (
              <div style={{
                color: theme === 'dark' ? '#9ca3af' : '#6b7280',
                fontSize: '13px'
              }}>
                {mdStatus.progress}%
              </div>
            )}
          </div>

          {/* Progress Bar */}
          {mdStatus.progress !== undefined && (
            <div style={{
              background: theme === 'dark' ? '#1a1a2e' : '#e5e7eb',
              borderRadius: '4px',
              height: '8px',
              overflow: 'hidden'
            }}>
              <div style={{
                background: 'linear-gradient(90deg, #ec4899, #f472b6)',
                height: '100%',
                width: `${mdStatus.progress}%`,
                transition: 'width 0.3s'
              }} />
            </div>
          )}

          {/* Queue Info */}
          {mdStatus.status === 'queued' && (
            <div style={{
              marginTop: '12px',
              color: theme === 'dark' ? '#9ca3af' : '#6b7280',
              fontSize: '12px'
            }}>
              <div>Job ID: {mdJobId}</div>
              <div>Estimated time: 30-60 minutes</div>
              <div style={{ marginTop: '8px', fontStyle: 'italic' }}>
                Simulating CRM1-NES binding with α-helix formation analysis...
              </div>
            </div>
          )}

          {/* Running Info */}
          {mdStatus.status === 'running' && (
            <div style={{
              marginTop: '12px',
              color: theme === 'dark' ? '#9ca3af' : '#6b7280',
              fontSize: '12px'
            }}>
              <div>Analyzing top {mdStatus.num_candidates || 5} NES candidates</div>
              <div>Duration: {mdStatus.duration_ns || 5.0} ns per candidate</div>
              <div style={{ marginTop: '8px', fontStyle: 'italic' }}>
                Testing hydrophobic anchor binding to CRM1 groove Cys528...
              </div>
            </div>
          )}
        </div>
      )}

      {/* Loading Spinner for Screening */}
      {loading && selectedMode === 'screening' && (
        <div style={{
          background: theme === 'dark' ? '#2d2d44' : '#f3f4f6',
          borderRadius: '8px',
          padding: '24px',
          textAlign: 'center',
          marginBottom: '16px'
        }}>
          <div style={{ fontSize: '32px', marginBottom: '12px' }}>⚡</div>
          <div style={{
            color: theme === 'dark' ? '#e0e0e0' : '#1f2937',
            fontSize: '14px',
            fontWeight: '600',
            marginBottom: '8px'
          }}>
            Running CRM1/NES Screening...
          </div>
          <div style={{
            color: theme === 'dark' ? '#9ca3af' : '#6b7280',
            fontSize: '12px'
          }}>
            Analyzing consensus sequences, ML predictions, disorder, SASA, pockets, and flexibility
          </div>
        </div>
      )}

      {/* Results Display */}
      {mdResults && (
        <div style={{
          background: theme === 'dark' ? '#2d2d44' : '#f3f4f6',
          borderRadius: '8px',
          padding: '20px'
        }}>
          <h4 style={{
            color: theme === 'dark' ? '#e0e0e0' : '#1f2937',
            fontSize: '16px',
            fontWeight: '600',
            marginBottom: '16px'
          }}>
            {selectedMode === 'screening' ? '⚡ CRM1/NES Screening Results' : '🎯 CRM1 MD Docking Results'}
          </h4>

          {/* Summary Stats */}
          {mdResults.summary && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: '12px',
              marginBottom: '20px'
            }}>
              {selectedMode === 'screening' ? (
                <>
                  <StatCard
                    label="Total Analyzed"
                    value={mdResults.summary.total_analyzed || 0}
                    theme={theme}
                  />
                  <StatCard
                    label="High Confidence"
                    value={mdResults.summary.high_confidence || 0}
                    theme={theme}
                  />
                  <StatCard
                    label="Avg ML Score"
                    value={(mdResults.summary.avg_ml_score || 0).toFixed(3)}
                    theme={theme}
                  />
                  <StatCard
                    label="Pocket Matches"
                    value={mdResults.summary.pocket_matches || 0}
                    theme={theme}
                  />
                </>
              ) : (
                <>
                  <StatCard
                    label="Total Simulated"
                    value={mdResults.summary.total_refined || 0}
                    theme={theme}
                  />
                  <StatCard
                    label="Strong Binders"
                    value={mdResults.summary.strong_binders || 0}
                    theme={theme}
                  />
                  <StatCard
                    label="Avg Binding Score"
                    value={(mdResults.summary.avg_binding_score || 0).toFixed(2)}
                    theme={theme}
                  />
                  <StatCard
                    label="α-Helix Formers"
                    value={mdResults.summary.helix_formers || 0}
                    theme={theme}
                  />
                </>
              )}
            </div>
          )}

          {/* Top Candidates Table */}
          {mdResults.enhanced_candidates && mdResults.enhanced_candidates.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <div style={{
                color: theme === 'dark' ? '#e0e0e0' : '#1f2937',
                fontSize: '14px',
                fontWeight: '600',
                marginBottom: '12px'
              }}>
                Top Candidates:
              </div>
              <div style={{
                maxHeight: '400px',
                overflowY: 'auto'
              }}>
                {mdResults.enhanced_candidates.slice(0, 10).map((candidate, idx) => (
                  <CandidateCard
                    key={idx}
                    candidate={candidate}
                    rank={idx + 1}
                    mode={selectedMode}
                    theme={theme}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Helper Components
const StatCard = ({ label, value, theme }) => (
  <div style={{
    background: theme === 'dark' ? '#1a1a2e' : '#ffffff',
    borderRadius: '8px',
    padding: '12px',
    border: `1px solid ${theme === 'dark' ? '#2d2d44' : '#e5e7eb'}`
  }}>
    <div style={{
      color: theme === 'dark' ? '#9ca3af' : '#6b7280',
      fontSize: '11px',
      marginBottom: '4px'
    }}>
      {label}
    </div>
    <div style={{
      color: theme === 'dark' ? '#e0e0e0' : '#1f2937',
      fontSize: '18px',
      fontWeight: '700'
    }}>
      {value}
    </div>
  </div>
);

const CandidateCard = ({ candidate, rank, mode, theme }) => {
  const getScoreColor = (score) => {
    if (score >= 0.8) return '#10b981';
    if (score >= 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const mainScore = mode === 'screening' 
    ? candidate.ml_score 
    : candidate.md_metrics?.binding_score || 0;

  return (
    <div style={{
      background: theme === 'dark' ? '#1a1a2e' : '#ffffff',
      borderRadius: '8px',
      padding: '12px',
      marginBottom: '8px',
      border: `1px solid ${theme === 'dark' ? '#2d2d44' : '#e5e7eb'}`,
      display: 'flex',
      alignItems: 'center',
      gap: '12px'
    }}>
      <div style={{
        background: getScoreColor(mainScore),
        color: 'white',
        borderRadius: '6px',
        padding: '4px 8px',
        fontSize: '12px',
        fontWeight: '700',
        minWidth: '30px',
        textAlign: 'center'
      }}>
        #{rank}
      </div>

      <div style={{ flex: 1 }}>
        <div style={{
          color: theme === 'dark' ? '#e0e0e0' : '#1f2937',
          fontSize: '13px',
          fontWeight: '600',
          fontFamily: 'monospace',
          marginBottom: '4px'
        }}>
          {candidate.sequence}
        </div>
        <div style={{
          color: theme === 'dark' ? '#9ca3af' : '#6b7280',
          fontSize: '11px',
          display: 'flex',
          gap: '12px'
        }}>
          <span>Position: {candidate.start_pos}-{candidate.end_pos}</span>
          {mode === 'screening' ? (
            <>
              <span>ML: {(candidate.ml_score || 0).toFixed(3)}</span>
              <span>Pattern: {candidate.pattern_type}</span>
              {candidate.in_pocket && <span>🎯 In Pocket</span>}
            </>
          ) : (
            <>
              <span>Binding: {(candidate.md_metrics?.binding_score || 0).toFixed(2)}</span>
              <span>α-Helix: {((candidate.md_metrics?.helix_content || 0) * 100).toFixed(0)}%</span>
              <span>{candidate.md_metrics?.binding_category}</span>
            </>
          )}
        </div>
      </div>

      <div style={{
        background: getScoreColor(mainScore),
        color: 'white',
        borderRadius: '6px',
        padding: '6px 12px',
        fontSize: '14px',
        fontWeight: '700'
      }}>
        {mainScore.toFixed(3)}
      </div>
    </div>
  );
};

export default MDRefinementPanel;
