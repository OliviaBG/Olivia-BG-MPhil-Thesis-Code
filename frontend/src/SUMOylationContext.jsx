import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

/**
 * SUMOylationContext Component
 * 
 * Displays SUMOylation-related warnings and context for NES candidates
 * Helps interpret why a NES might score lower due to SUMOylation effects
 * UPDATED: Lower threshold (0.3 instead of 0.4) for better detection
 */

const SUMOylationContext = ({ sequence, nesCandidate, apiBase }) => {
  const [sumoContext, setSumoContext] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeSumoylation = useCallback(async () => {
    try {
      setLoading(true);
      
      const response = await axios.post(`${apiBase}/sumoylation/predict`, {
        sequence: sequence,
        nes_candidates: [nesCandidate],
        min_score: 0.3  // LOWERED from 0.4 to detect more sites
      });

      const nesKey = `${nesCandidate.start}_${nesCandidate.end}_${nesCandidate.sequence}`;
      const annotation = response.data.nes_annotations?.[nesKey];

      if (annotation && annotation.has_relevant_sumo) {
        setSumoContext({
          impact: annotation.interpretation.sumo_impact,
          warning: annotation.interpretation.warnings.join(' '),
          sites_on_nes: annotation.sites_on_nes,
          sites_adjacent: annotation.sites_adjacent,
          sites_near: annotation.sites_near
        });
      } else {
        setSumoContext(null);
      }
    } catch (error) {
      console.error('SUMOylation analysis error:', error);
      setSumoContext(null);
    } finally {
      setLoading(false);
    }
  }, [sequence, nesCandidate, apiBase]);

  useEffect(() => {
    if (sequence && nesCandidate) {
      analyzeSumoylation();
    }
  }, [sequence, nesCandidate, analyzeSumoylation]);

  if (loading) {
    return (
      <div className="sumoylation-context" style={{ 
        background: '#f0f0f0',
        padding: '12px',
        borderRadius: '8px',
        margin: '12px 0'
      }}>
        <p style={{ margin: 0, fontStyle: 'italic', color: '#666' }}>
          Checking for SUMOylation sites...
        </p>
      </div>
    );
  }

  if (!sumoContext || sumoContext.impact === 'none') {
    return null;
  }

  const getImpactStyle = (impact) => {
    switch (impact) {
      case 'direct_masking':
        return {
          bgColor: '#fff3cd',
          borderColor: '#f39c12',
          iconColor: '#f39c12',
          icon: '⚠️'
        };
      case 'local_perturbation':
        return {
          bgColor: '#cff4fc',
          borderColor: '#0dcaf0',
          iconColor: '#0dcaf0',
          icon: 'ℹ️'
        };
      case 'possible_indirect':
        return {
          bgColor: '#e7f3ff',
          borderColor: '#2196f3',
          iconColor: '#2196f3',
          icon: '💡'
        };
      default:
        return {
          bgColor: '#f0f0f0',
          borderColor: '#999',
          iconColor: '#999',
          icon: 'ℹ️'
        };
    }
  };

  const style = getImpactStyle(sumoContext.impact);

  return (
    <div className="sumoylation-context" style={{ 
      background: style.bgColor,
      borderLeft: `4px solid ${style.borderColor}`,
      padding: '16px',
      borderRadius: '8px',
      margin: '12px 0'
    }}>
      <div className="context-header" style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '12px'
      }}>
        <span className="context-icon" style={{ 
          color: style.iconColor,
          fontSize: '1.5em'
        }}>
          {style.icon}
        </span>
        <h4 className="context-title" style={{
          margin: 0,
          fontSize: '1.1em',
          color: '#333'
        }}>SUMOylation Context</h4>
      </div>

      {sumoContext.warning && (
        <div className="context-warning" style={{
          padding: '12px',
          background: 'rgba(0,0,0,0.05)',
          borderRadius: '6px',
          marginBottom: '12px',
          lineHeight: '1.6'
        }}>
          {sumoContext.warning}
        </div>
      )}

      {/* Show counts of sites */}
      <div className="sumo-site-counts" style={{
        marginBottom: '12px',
        fontSize: '0.9em',
        color: '#555'
      }}>
        <div>• Sites ON NES: {sumoContext.sites_on_nes?.length || 0}</div>
        <div>• Sites ADJACENT (±5 aa): {sumoContext.sites_adjacent?.length || 0}</div>
        <div>• Sites NEAR (±10 aa): {sumoContext.sites_near?.length || 0}</div>
      </div>

      <div className="context-explanation" style={{
        marginBottom: '12px',
        lineHeight: '1.6',
        color: '#555'
      }}>
        <strong style={{ color: '#333' }}>What this means:</strong> SUMOylation generally promotes nuclear retention 
        by masking NESs or altering protein interactions. If this NES scores lower than expected, 
        SUMOylation may be reducing its accessibility or function. This sequence may still be a 
        functional NES in the un-SUMOylated state.
      </div>

      <div className="context-caveat" style={{
        fontSize: '0.9em',
        color: '#666',
        fontStyle: 'italic',
        paddingTop: '8px',
        borderTop: '1px solid rgba(0,0,0,0.1)'
      }}>
        <em>Note: The effect of SUMOylation on nuclear export is protein- and context-dependent. 
        These predictions should be validated experimentally.</em>
      </div>
    </div>
  );
};

export default SUMOylationContext;
