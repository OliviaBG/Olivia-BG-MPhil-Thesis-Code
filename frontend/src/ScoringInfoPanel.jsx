import React from 'react';

/**
 * ScoringInfoPanel Component
 * 
 * Displays accurate scoring breakdown with correct weightings
 * Shows how the NES score is calculated
 */

const ScoringInfoPanel = () => {
  const scoringComponents = [
    {
      name: 'ML Prediction',
      weight: 25,
      description: 'Trained on 27 validated NES sequences',
      details: 'Uses SVM with PSSM scoring, flanking region analysis, and disorder propensity'
    },
    {
      name: 'CRM1 Pocket Binding',
      weight: 30,
      description: 'Hydrophobic groove compatibility around Cys528',
      details: 'fpocket analysis of hydrophobic groove compatibility around Cys528 using real CRM1 structure template'
    },
    {
      name: 'Hydrophobicity',
      weight: 15,
      description: 'NES requires 4-5 hydrophobic anchors (L, I, V, F, M)',
      details: 'Evaluates presence and spacing of hydrophobic residues'
    },
    {
      name: 'Accessibility (RSA)',
      weight: 10,
      description: 'Surface exposure required for CRM1 accessibility',
      details: 'Consensus relative solvent accessibility (FreeSASA + Biopython, Tien et al. 2013 residue-specific normalization) indicates true burial vs. exposure'
    },
    {
      name: 'Flexibility',
      weight: 10,
      description: 'Higher pLDDT variance indicates flexibility',
      details: 'Beneficial for NES as it allows conformational adaptation'
    },
    {
      name: 'Disorder',
      weight: 10,
      description: 'IUPred2A-predicted disorder (falls back to a pLDDT/composition heuristic if unavailable), plus UniProt-annotated disorder',
      details: 'NES sequences are often in disordered regions. IUPred2A is sequence-based, so unlike pLDDT it works the same for AlphaFold and experimental structures.'
    },
    {
      name: 'ANCHOR2',
      weight: null,
      description: 'Predicted disordered binding regions (from the same IUPred2A call as Disorder)',
      details: 'A NES is itself a short linear motif that folds up on engaging CRM1’s groove -- exactly what ANCHOR2 is designed to flag. Applied as a small bonus on top of the weighted score above, not one of the named percentages (it was never a feature the trained ML model saw, so there’s no data-driven weight to assign it yet).'
    }
  ];

  const totalWeight = scoringComponents.reduce((sum, c) => sum + c.weight, 0);

  return (
    <div className="scoring-info-panel">
      <div className="panel-header">
        <h3>📊 About this Enhanced Analysis</h3>
      </div>

      <div className="scoring-components">
        {scoringComponents.map((component, idx) => (
          <div key={idx} className="scoring-component">
            <div className="component-header">
              <div className="component-name-weight">
                <span className="component-name">{component.name}</span>
                <span className="component-weight">{component.weight != null ? `${component.weight}%` : 'bonus'}</span>
              </div>
              <div className="weight-bar">
                <div
                  className="weight-fill"
                  style={{
                    width: `${((component.weight || 0) / totalWeight) * 100}%`,
                    background: `hsl(${200 + idx * 30}, 70%, 50%)`
                  }}
                />
              </div>
            </div>
            <p className="component-description">{component.description}</p>
            <p className="component-details">{component.details}</p>
          </div>
        ))}
      </div>

      <div className="pattern-info">
        <h4>📐 Consensus Pattern</h4>
        <div className="pattern-formula">
          <code>Φ-X<sub>(2-3)</sub>-Φ-X<sub>(2-3)</sub>-Φ-X-Φ</code>
        </div>
        <p className="pattern-note">
          where <strong>Φ = hydrophobic</strong> (L, I, V, F, M)
        </p>
      </div>

      <div className="scoring-note">
        <p>
          <strong>Scoring:</strong> Balanced weighting emphasizing CRM1 pocket compatibility 
          and structural features over ML alone
        </p>
      </div>

      <style jsx>{`
        .scoring-info-panel {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
        }

        .panel-header h3 {
          margin: 0 0 20px 0;
          font-size: 1.4em;
        }

        .scoring-components {
          display: flex;
          flex-direction: column;
          gap: 16px;
          margin-bottom: 24px;
        }

        .scoring-component {
          background: rgba(255,255,255,0.1);
          backdrop-filter: blur(10px);
          padding: 16px;
          border-radius: 8px;
          border: 1px solid rgba(255,255,255,0.2);
        }

        .component-header {
          margin-bottom: 8px;
        }

        .component-name-weight {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
        }

        .component-name {
          font-weight: 700;
          font-size: 1.05em;
        }

        .component-weight {
          background: rgba(255,255,255,0.25);
          padding: 2px 10px;
          border-radius: 12px;
          font-weight: 700;
          font-size: 0.9em;
        }

        .weight-bar {
          height: 6px;
          background: rgba(255,255,255,0.2);
          border-radius: 3px;
          overflow: hidden;
        }

        .weight-fill {
          height: 100%;
          transition: width 0.5s ease;
        }

        .component-description {
          margin: 8px 0 4px 0;
          font-size: 0.95em;
          opacity: 0.95;
        }

        .component-details {
          margin: 0;
          font-size: 0.85em;
          opacity: 0.8;
          font-style: italic;
        }

        .pattern-info {
          background: rgba(255,255,255,0.1);
          backdrop-filter: blur(10px);
          padding: 16px;
          border-radius: 8px;
          border: 1px solid rgba(255,255,255,0.2);
          margin-bottom: 16px;
        }

        .pattern-info h4 {
          margin: 0 0 12px 0;
          font-size: 1.1em;
        }

        .pattern-formula {
          background: rgba(0,0,0,0.2);
          padding: 12px;
          border-radius: 6px;
          text-align: center;
          margin-bottom: 8px;
        }

        .pattern-formula code {
          font-family: 'Courier New', monospace;
          font-size: 1.2em;
          font-weight: 700;
        }

        .pattern-note {
          margin: 0;
          font-size: 0.9em;
          opacity: 0.9;
          text-align: center;
        }

        .scoring-note {
          padding: 12px;
          background: rgba(255,255,255,0.1);
          border-radius: 6px;
          font-size: 0.9em;
        }

        .scoring-note p {
          margin: 0;
        }
      `}</style>
    </div>
  );
};

export default ScoringInfoPanel;
