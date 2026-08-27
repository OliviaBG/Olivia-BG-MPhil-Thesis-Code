import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';
import QuickAnalysisPanel from './QuickAnalysisPanel';
import SUMOylationContext from './SUMOylationContext';


const API_BASE = 'http://localhost:5000/api';

// ---------------------------------------------------------------------------
// CIDER 2x2 panel + combined RSA panel, ported from
// prototype_cider_rsa_panels.py (the standalone script these designs were
// prototyped/reviewed in first). Cambridge colours here must match
// thesis_plot_style.py's CAM dict so the live app's plots look the same as
// the thesis figures.
// ---------------------------------------------------------------------------
const CAM = {
  darkBlue: '#133844',
  blue: '#8EE8D8',
  lightBlue: '#D1F9F1',
  warmBlue: '#00BDB6',   // "Cambridge teal"
  crest: '#FD8153',      // "Cambridge orange"
  darkCrest: '#DD3025',  // "Cambridge red"
  slate3: '#546072',
};

// Diverging colourscale for Plotly (dark_blue -> white -> crest), matching
// thesis_plot_style.py's CAM_DIVERGING matplotlib colormap used for the
// same consensus_z colouring in the thesis figures.
const CAM_DIVERGING_SCALE = [
  [0, CAM.darkBlue],
  [0.5, '#FFFFFF'],
  [1, CAM.crest],
];

// Plotly can't colour one continuous line differently above/below a
// threshold -- the standard workaround is two traces, each holding 0
// wherever the real value is on the OTHER side of zero, so they visually
// meet exactly at every zero-crossing. Used for NCPR (the one CIDER profile
// with a biologically meaningful zero: net charge).
function splitAboveBelowZero(y) {
  const above = y.map(v => (v >= 0 ? v : 0));
  const below = y.map(v => (v < 0 ? v : 0));
  return { above, below };
}

// 2x2 CIDER panel (NCPR / hydropathy / FCR / complexity). `profile` is a
// cider_profile object from the backend (positions, linear_ncpr,
// linear_hydropathy, linear_fcr, linear_complexity, cider_computed,
// nes_start_idx_in_profile, nes_end_idx_in_profile -- the index fields keep
// the NES-side name even when rendering an NLS candidate, so this component
// works for both without needing target-specific prop names).
function CiderPanel({ profile, filenamePrefix }) {
  if (!profile || !profile.cider_computed) {
    return <p>localCIDER isn't installed on the server, so these graphs aren't available.</p>;
  }
  const shadeShape = {
    type: 'rect', xref: 'x', yref: 'paper',
    x0: profile.positions[profile.nes_start_idx_in_profile] ?? profile.positions[0],
    x1: profile.positions[profile.nes_end_idx_in_profile] ?? profile.positions[profile.positions.length - 1],
    y0: 0, y1: 1,
    fillcolor: 'rgba(16, 185, 129, 0.12)',
    line: { width: 0 },
  };
  const zeroLine = {
    type: 'line', xref: 'x', yref: 'y',
    x0: profile.positions[0], x1: profile.positions[profile.positions.length - 1],
    y0: 0, y1: 0,
    line: { color: CAM.slate3, width: 1, dash: 'dot' },
  };
  const { above, below } = splitAboveBelowZero(profile.linear_ncpr);

  const smallLayout = (title, extraShapes) => ({
    title: { text: title, font: { size: 12 } },
    xaxis: { title: 'Residue position' },
    yaxis: { title },
    height: 240,
    margin: { t: 32, b: 32, l: 48, r: 16 },
    showlegend: false,
    shapes: extraShapes || [shadeShape],
  });

  const singleLinePlot = (key, y, title, color) => (
    <div className="cider-plot" key={key}>
      <Plot
        data={[{ x: profile.positions, y, type: 'scatter', mode: 'lines',
                 line: { color, width: 2 }, name: title }]}
        layout={smallLayout(title)}
        config={{ toImageButtonOptions: { format: 'png', filename: `${filenamePrefix}_${key}`, scale: 4 }, displaylogo: false }}
        style={{ width: '100%' }}
      />
    </div>
  );

  return (
    <>
      <p className="cider-instruction">
        Real localCIDER profiles (shaded region = the predicted candidate itself, plus its
        +/-20 residue flanking context). NCPR is coloured teal above zero / orange below
        zero. Click the camera icon on any plot for a high-res PNG.
      </p>
      <div className="cider-grid-2x2">
        <div className="cider-plot" key="ncpr">
          <Plot
            data={[
              { x: profile.positions, y: above, type: 'scatter', mode: 'lines',
                line: { color: CAM.warmBlue, width: 2 }, name: 'NCPR (+)' },
              { x: profile.positions, y: below, type: 'scatter', mode: 'lines',
                line: { color: CAM.crest, width: 2 }, name: 'NCPR (-)' },
            ]}
            layout={smallLayout('Linear NCPR', [shadeShape, zeroLine])}
            config={{ toImageButtonOptions: { format: 'png', filename: `${filenamePrefix}_ncpr`, scale: 4 }, displaylogo: false }}
            style={{ width: '100%' }}
          />
        </div>
        {singleLinePlot('hydropathy', profile.linear_hydropathy, 'Linear hydropathy', CAM.darkBlue)}
        {singleLinePlot('fcr', profile.linear_fcr, 'Linear FCR', CAM.darkBlue)}
        {singleLinePlot('complexity', profile.linear_complexity, 'Linear complexity', CAM.darkBlue)}
      </div>
    </>
  );
}

// Combined RSA panel: consensus_rsa as the main line, +/- agreement_sd as a
// shaded ribbon, consensus_z (exposure relative to the rest of THIS
// protein) colouring the line's markers via the Cambridge diverging scale.
// Replaces the old 3-separate-plot RSA dropdown (consensus_rsa / consensus_z
// / agreement_sd each as their own full-width line graph).
function RsaPanel({ profile, filenamePrefix }) {
  if (!profile || !profile.rsa_computed) {
    return <p>Real RSA calculation wasn't available for this structure (fell back to a
      neutral placeholder), so a per-residue profile can't be shown.</p>;
  }
  const { positions, consensus_rsa, consensus_z, agreement_sd,
          nes_start_idx_in_profile, nes_end_idx_in_profile } = profile;

  const shadeShape = {
    type: 'rect', xref: 'x', yref: 'paper',
    x0: positions[nes_start_idx_in_profile] ?? positions[0],
    x1: positions[nes_end_idx_in_profile] ?? positions[positions.length - 1],
    y0: 0, y1: 1,
    fillcolor: 'rgba(16, 185, 129, 0.12)',
    line: { width: 0 },
  };
  const buriedLine = {
    type: 'line', xref: 'x', yref: 'y',
    x0: positions[0], x1: positions[positions.length - 1],
    y0: 0.25, y1: 0.25,
    line: { color: CAM.darkCrest, width: 1, dash: 'dot' },
  };

  const upper = consensus_rsa.map((v, i) => (v ?? 0) + (agreement_sd[i] ?? 0));
  const lower = consensus_rsa.map((v, i) => (v ?? 0) - (agreement_sd[i] ?? 0));
  const zVals = (consensus_z || []).filter(v => v !== null && v !== undefined);
  const zMax = Math.max(1e-6, ...zVals.map(v => Math.abs(v)));

  return (
    <>
      <p className="cider-instruction">
        Real per-residue RSA (Tien et al. 2013-normalized, 3-method consensus) over this
        candidate (shaded) plus its +/-20 residue flanking context. Ribbon = consensus RSA
        +/- cross-method agreement SD (wider band = the 3 methods disagree more here --
        treat with caution). Point colour = consensus z-score (how exposed each residue is
        relative to the rest of THIS protein). Dotted red line = consensus_accessibility.py's
        buried threshold (RSA&lt;0.25). Click the camera icon for a high-res PNG.
      </p>
      <div className="cider-plot">
        <Plot
          data={[
            { x: positions, y: upper, type: 'scatter', mode: 'lines',
              line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
            { x: positions, y: lower, type: 'scatter', mode: 'lines',
              fill: 'tonexty', fillcolor: 'rgba(142, 232, 216, 0.55)',
              line: { width: 0 }, name: 'RSA +/- agreement SD', hoverinfo: 'skip' },
            { x: positions, y: consensus_rsa, type: 'scatter', mode: 'lines+markers',
              line: { color: CAM.slate3, width: 1.2 },
              marker: {
                size: 6, color: consensus_z, colorscale: CAM_DIVERGING_SCALE,
                cmin: -zMax, cmax: zMax,
                colorbar: { title: { text: 'Consensus z-score', font: { size: 10 } }, tickfont: { size: 9 }, thickness: 14 },
                line: { color: CAM.slate3, width: 0.5 },
              },
              name: 'Consensus RSA',
            },
          ]}
          layout={{
            title: { text: 'Consensus RSA (ribbon = agreement SD, colour = z-score)', font: { size: 12 } },
            xaxis: { title: 'Residue position' },
            yaxis: { title: 'Consensus RSA' },
            height: 340,
            margin: { t: 40, b: 40 },
            shapes: [shadeShape, buriedLine],
            showlegend: false,
          }}
          config={{ toImageButtonOptions: { format: 'png', filename: `${filenamePrefix}_rsa`, scale: 4 }, displaylogo: false }}
          style={{ width: '100%' }}
        />
      </div>
    </>
  );
}

function App() {
  // State Management
  const [searchQuery, setSearchQuery] = useState('');
  const [organism, setOrganism] = useState('all');
  // Which structure source to fetch: 'alphafold' (predicted), 'experimental'
  // (real, elucidated structures from the PDB), or 'both'
  const [structureSource, setStructureSource] = useState('alphafold');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedProtein, setSelectedProtein] = useState(null);
  const [structures, setStructures] = useState([]);
  const [structureData, setStructureData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Analysis state
  const [colourMode, setColourMode] = useState('plddt');
  const [regionStart, setRegionStart] = useState('');
  const [regionEnd, setRegionEnd] = useState('');
  const [regionAnalysis, setRegionAnalysis] = useState(null);
  const [selectedResidues, setSelectedResidues] = useState([]);
  const [showSequence, setShowSequence] = useState(false);
  // Which NES prediction card(s) have their auto-generated CIDER graph
  // dropdown expanded -- keyed by index in the sorted results list.
  const [expandedNesCider, setExpandedNesCider] = useState({});
  // Same, for the RSA consensus/z-score/agreement-SD dropdown (rsa_profile,
  // see calculate_sasa(return_stats=True) in app.py -- ported from the
  // standalone consensus_accessibility.py script).
  const [expandedNesRsa, setExpandedNesRsa] = useState({});
  // Same two, for NLS candidate cards (addition -- NLS previously
  // had no CIDER/RSA dropdowns at all).
  const [expandedNlsCider, setExpandedNlsCider] = useState({});
  const [expandedNlsRsa, setExpandedNlsRsa] = useState({});
  // Region Analysis panel's CIDER dropdown -- a single boolean (not indexed
  // by idx like the candidate cards above) since there's only ever one
  // Region Analysis result showing at a time. this used to
  // render unconditionally whenever regionAnalysis was set; now gated
  // behind a dropdown like everywhere else.
  const [expandedRegionCider, setExpandedRegionCider] = useState(false);
  const [sequenceSelection, setSequenceSelection] = useState({ start: null, end: null, selecting: false });
  
  // MD state
  const [mdJobStatus, setMdJobStatus] = useState(null);
  const [mdResults, setMdResults] = useState(null);
  const [showMdResults, setShowMdResults] = useState(false);
  const [selectedMdCandidateIdx, setSelectedMdCandidateIdx] = useState(0);
  // 'single' = simulate just the chosen candidate, 'top10' = simulate the
  // top 10 candidates by combined_score (dropdown selection is ignored then)
  const [mdCandidateMode, setMdCandidateMode] = useState('single');

  // New state for enhancements
  const [quickAnalysisMode, setQuickAnalysisMode] = useState(false);

  // Theme
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    document.body.className = savedTheme;
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    document.body.className = newTheme;
  };

  // ===========================================================================
  // PROTEIN SEARCH
  // ===========================================================================
  
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError(null);
    setSearchResults([]);
    setSelectedProtein(null);
    setStructures([]);

    try {
      const response = await axios.get(`${API_BASE}/search`, {
        params: {
          query: searchQuery,
          organism: organism
        }
      });

      setSearchResults(response.data);
      
      if (response.data.length === 0) {
        setError('No proteins found. Try a different search term or organism.');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Search failed. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  // ===========================================================================
  // GET STRUCTURES FOR SELECTED PROTEIN
  // ===========================================================================
  
  const handleGetStructures = async (protein) => {
    setSelectedProtein(protein);
    setLoading(true);
    setError(null);
    setStructures([]);
    setStructureData(null);

    try {
      const response = await axios.get(`${API_BASE}/models/${protein.id}`, {
        params: {
          source: structureSource
        }
      });

      if (response.data && response.data.length > 0) {
        setStructures(response.data);
        setSearchResults([]);
      } else {
        setError('No structures available for this protein with the selected source');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to load structures');
      console.error('Structure loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  // ===========================================================================
  // LOAD AND ANALYZE SELECTED STRUCTURE
  // ===========================================================================
  
  const handleLoadStructure = async (structure) => {
    setLoading(true);
    setError(null);
    setStructureData(null);
    setRegionAnalysis(null);
    setSelectedResidues([]);
    setShowSequence(false);

    try {
      // Extract model_id - try different possible fields from API response
      const modelId = structure.model_id || structure.entryId || structure.id;
      
      if (!modelId) {
        throw new Error('No model ID found in structure data');
      }
      
      console.log('Loading structure with ID:', modelId);
      const response = await axios.get(`${API_BASE}/structure/${modelId}`, {
        params: {
          // Needed so the backend can look up UniProt domain annotations for
          // experimental (PDB-XXXX) structures, whose IDs don't encode it
          uniprot_id: selectedProtein?.id
        }
      });
      setStructureData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Structure analysis failed');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  // ===========================================================================
  // REGION SELECTION & ANALYSIS
  // ===========================================================================
  
  const handleRegionSelect = () => {
    if (!regionStart || !regionEnd) {
      setError('Please enter both start and end positions');
      return;
    }

    const start = parseInt(regionStart);
    const end = parseInt(regionEnd);

    if (start < 1 || end > (structureData.sequence?.length || 0) || start > end) {
      setError('Invalid region selection');
      return;
    }

    const selectedRange = [];
    for (let i = start - 1; i < end; i++) {
      selectedRange.push(i);
    }
    setSelectedResidues(selectedRange);
    setError(null);
  };

  const handleUnselectRegion = () => {
    setSelectedResidues([]);
    setRegionStart('');
    setRegionEnd('');
    setRegionAnalysis(null);
    setShowSequence(false);
    setSequenceSelection({ start: null, end: null, selecting: false });
  };

  const handleSequenceClick = (residueIndex) => {
    if (!sequenceSelection.selecting) {
      return;
    }

    if (sequenceSelection.start === null) {
      setSequenceSelection({ ...sequenceSelection, start: residueIndex });
    } else if (sequenceSelection.end === null) {
      const start = Math.min(sequenceSelection.start, residueIndex);
      const end = Math.max(sequenceSelection.start, residueIndex);
      setRegionStart((start + 1).toString());
      setRegionEnd((end + 1).toString());
      setSequenceSelection({ start: null, end: null, selecting: false });
      setShowSequence(false);
      
      // Auto-highlight the region
      const selectedRange = [];
      for (let i = start; i <= end; i++) {
        selectedRange.push(i);
      }
      setSelectedResidues(selectedRange);
    }
  };

  const toggleSequenceHighlight = () => {
    if (!showSequence) {
      setShowSequence(true);
      setSequenceSelection({ start: null, end: null, selecting: true });
    } else {
      setShowSequence(false);
      setSequenceSelection({ start: null, end: null, selecting: false });
    }
  };

  const handleRegionAnalysis = async () => {
    if (!regionStart || !regionEnd) {
      setError('Please select a region first');
      return;
    }

    const start = parseInt(regionStart);
    const end = parseInt(regionEnd);

    if (start < 1 || end > (structureData.sequence?.length || 0) || start > end) {
      setError('Invalid region selection');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE}/analyze_region`, {
        model_id: structureData.model_id,
        start: start,
        end: end,
        sasa: structureData.sasa || [],
        charges: structureData.charges || [],
        disorder: structureData.disorder || [],
        hydrophobicity: structureData.hydrophobicity || [],
        sequence: structureData.sequence || ''
      });

      setRegionAnalysis(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Region analysis failed');
      console.error('Region analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  // ===========================================================================
  // CRM1/NES BINDING ANALYSIS
  // ===========================================================================
  
  const handleCRM1Analysis = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE}/unified_crm1_nes/${structureData.model_id}`);
      
      setStructureData({
        ...structureData,
        crm1_binding_regions: response.data.nes_motifs,
        crm1_summary: response.data.summary,
        crm1_scores: response.data.crm1_scores,
        pocket_scores: response.data.pocket_scores,
        pockets: response.data.pockets
      });

      // Auto-switch to NES color mode
      setColourMode('nes');
    } catch (err) {
      setError(err.response?.data?.error || 'CRM1 analysis failed');
      console.error('CRM1 analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Whole-structure NLS scan -- the structural analog of the CRM1/NES
  // button above (same request/response pattern, feeding into the same
  // structureData + colour-mode machinery), rather than a free-text
  // single-peptide prediction. Unlike CRM1/NES there's no fpocket/MD step
  // (no single dominant cargo receptor pocket for classical NLS import --
  // see NLS_predictor_landscape_and_novelty.md), so this just sends the
  // sequence + per-residue pLDDT/SASA the structure view already loaded.
  const handleNLSPredict = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE}/nls_scan`, {
        model_id: structureData.model_id,
        sequence: structureData.sequence,
        plddt: structureData.plddt,
        sasa: structureData.sasa,
        // consensus_z/agreement_sd now come back from /api/structure's
        // initial load (see get_structure()) -- sending them through lets
        // the backend build a real rsa_profile per NLS candidate without a
        // second structure download.
        consensus_z: structureData.consensus_z,
        agreement_sd: structureData.agreement_sd,
      });

      setStructureData({
        ...structureData,
        nls_binding_regions: response.data.nls_binding_regions,
        nls_scores: response.data.nls_scores,
        nls_summary: response.data.summary,
      });

      // Auto-switch to NLS color mode
      setColourMode('nls');
    } catch (err) {
      setError(err.response?.data?.error || 'NLS prediction failed');
      console.error('NLS prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // ===========================================================================
  // MD SIMULATION HANDLERS
  // ===========================================================================
  
  // Same sort order used for both the candidate picker dropdown and the
  // "Predicted NES Motifs" list below, so the index a user picks in the
  // dropdown always points at the candidate they think it does.
  const getSortedMdCandidates = () => {
    if (!structureData || !structureData.crm1_binding_regions) return [];
    return [...structureData.crm1_binding_regions].sort((a, b) => b.combined_score - a.combined_score);
  };

  const handleRunMD = async (duration_ns = 10) => {
    if (!structureData || !structureData.crm1_binding_regions || structureData.crm1_binding_regions.length === 0) {
      setError('Please run CRM1 analysis first');
      return;
    }

    const sortedCandidates = getSortedMdCandidates();

    let candidatesToRun;
    if (mdCandidateMode === 'top10') {
      candidatesToRun = sortedCandidates.slice(0, 10);
      if (candidatesToRun.length === 0) {
        setError('No NES candidates available to simulate');
        return;
      }
    } else {
      const selectedCandidate = sortedCandidates[selectedMdCandidateIdx];
      if (!selectedCandidate) {
        setError('Please select a NES candidate to simulate');
        return;
      }
      candidatesToRun = [selectedCandidate];
    }

    setLoading(true);
    setError(null);
    setMdJobStatus(null);
    setMdResults(null);

    try {
      const response = await axios.post(`${API_BASE}/md_docking`, {
        model_id: structureData.model_id,
        candidates: candidatesToRun,
        duration_ns: duration_ns
      });

      const jobId = response.data.job_id;
      setMdJobStatus({
        job_id: jobId,
        status: 'queued',
        num_candidates: response.data.num_candidates,
        duration_ns: response.data.duration_ns
      });

      // Poll for status
      pollMDStatus(jobId);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to start MD simulation');
      console.error('MD error:', err);
    } finally {
      setLoading(false);
    }
  };

  const pollMDStatus = async (jobId) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/md_job_status/${jobId}`);
        const status = response.data;

        setMdJobStatus(status);

        if (status.status === 'completed') {
          clearInterval(pollInterval);
          
          // Fetch results
          const resultResponse = await axios.get(`${API_BASE}/md_job_result/${jobId}`);
          setMdResults(resultResponse.data);
          setShowMdResults(true);
        } else if (status.status === 'failed') {
          clearInterval(pollInterval);
          setError(`MD simulation failed: ${status.error || 'Unknown error'}`);
        }
      } catch (err) {
        clearInterval(pollInterval);
        setError('Failed to check MD status');
        console.error('MD polling error:', err);
      }
    }, 2000); // Poll every 2 seconds
  };

  // ===========================================================================
  // 3D STRUCTURE VISUALIZATION
  // ===========================================================================
  
  const getStructurePlot = () => {
    if (!structureData || !structureData.coordinates) return null;

    const coords = structureData.coordinates;
    const { sequence, plddt, sasa, disorder, anchor2_binding, disorder_source,
            hydrophobicity, charge, domains, crm1_scores, bfactor } = structureData;

    if (!coords || coords.length === 0) return null;

    // Get color array based on selected mode
    let colors = [];
    let colorscale = [];
    let colorbarTitle = '';
    
    switch(colourMode) {
      case 'plddt':
        colors = plddt;
        colorscale = [
          [0, '#FF7D45'], [0.5, '#FFDB13'], [0.7, '#65CBF3'], [1, '#0053D6']
        ];
        colorbarTitle = structureData.confidence_metric === 'bfactor' ? 'B-factor' : 'pLDDT';
        break;
      case 'bfactor':
        colors = bfactor || plddt;
        colorscale = 'Viridis';
        colorbarTitle = 'B-factor';
        break;
      case 'sasa':
        colors = sasa;
        colorscale = 'Blues';
        colorbarTitle = 'Accessibility (RSA)';
        break;
      case 'disorder':
        colors = disorder;
        colorscale = 'Reds';
        // disorder_source: 'iupred2a' (real predictor) or 'structural_heuristic'
        // (pLDDT/composition fallback -- used when IUPred2A couldn't be
        // reached or this protein has no UniProt mapping).
        colorbarTitle = disorder_source === 'iupred2a' ? 'Disorder (IUPred2A)' : 'Disorder (heuristic)';
        break;
      case 'anchor2':
        // ANCHOR2: predicted disordered *binding* regions -- segments that
        // fold up on engaging a partner (e.g. a NES engaging CRM1's
        // groove). Only available when IUPred2A was reachable for this
        // protein (structureData.anchor2_binding is null otherwise).
        colors = anchor2_binding;
        colorscale = 'Purples';
        colorbarTitle = 'ANCHOR2 (Binding Region)';
        break;
      case 'hydrophobicity':
        colors = hydrophobicity;
        colorscale = 'RdYlGn_r';
        colorbarTitle = 'Hydrophobicity';
        break;
      case 'charge':
        colors = charge;
        colorscale = [[0, '#d32f2f'], [0.5, '#ffffff'], [1, '#1976d2']];
        colorbarTitle = 'Charge';
        break;
      case 'domain':
        // Use domain colors - these are already color strings like '#FF6B6B'
        colors = coords.map((_, idx) => {
          const domain = domains?.find(d => 
            idx >= d.start - 1 && idx <= d.end - 1
          );
          return domain ? domain.color : '#cccccc';
        });
        colorscale = null;  // Don't use colorscale for categorical colors
        break;
      case 'nes': {
        // Build per-residue NES likelihood from predicted binding regions.
        const nesScores = new Array(coords.length).fill(0);
        if (structureData.crm1_binding_regions) {
          structureData.crm1_binding_regions.forEach(region => {
            const startIdx = region.start - 1;
            const endIdx = region.end;
            for (let i = startIdx; i < endIdx && i < nesScores.length; i++) {
              nesScores[i] = Math.max(nesScores[i], region.combined_score);
            }
          });
        }
        colors = nesScores;
        colorscale = [
          [0, '#ffffff'],
          [0.25, '#fff176'],
          [0.5, '#ffeb3b'],
          [0.75, '#ff9800'],
          [1, '#d32f2f']
        ];
        colorbarTitle = 'NES Likelihood';
        break;
      }
      case 'nls':
        // Backend already returns a per-residue array (same convention as
        // crm1_scores), computed from the non-overlapping predicted regions.
        colors = structureData.nls_scores || new Array(coords.length).fill(0);
        colorscale = [
          [0, '#ffffff'],
          [0.25, '#c8e6c9'],
          [0.5, '#66bb6a'],
          [0.75, '#2e7d32'],
          [1, '#1b5e20']
        ];
        colorbarTitle = 'NLS Likelihood';
        break;
      case 'linear_hydropathy':
        colors = structureData.linear_hydropathy;
        colorscale = 'RdYlGn_r';
        colorbarTitle = 'Linear Hydropathy (CIDER)';
        break;
      case 'linear_ncpr':
        colors = structureData.linear_ncpr;
        colorscale = [[0, '#d32f2f'], [0.5, '#ffffff'], [1, '#1976d2']];
        colorbarTitle = 'Linear NCPR (CIDER)';
        break;
      case 'linear_fcr':
        colors = structureData.linear_fcr;
        colorscale = 'Purples';
        colorbarTitle = 'Linear FCR (CIDER)';
        break;
      case 'pockets':
        colors = structureData.pocket_scores;
        colorscale = [
          [0, '#ffffff'],
          [0.3, '#b2dfdb'],
          [0.6, '#26a69a'],
          [1, '#00695c']
        ];
        colorbarTitle = 'CRM1 Pocket Compatibility';
        break;
      default:
        colors = plddt;
        colorscale = [
          [0, '#FF7D45'], [0.5, '#FFDB13'], [0.7, '#65CBF3'], [1, '#0053D6']
        ];
        colorbarTitle = structureData.confidence_metric === 'bfactor' ? 'B-factor' : 'pLDDT';
    }

    // Prepare ball and stick representation
    const traces = [];

    // 1. BALLS (spheres for CA atoms)
    // Apply magenta highlighting for selected residues in ALL color modes
    const markerColors = coords.map((_, idx) => {
      // If this residue is selected, override with magenta
      if (selectedResidues.includes(idx)) {
        return '#FF00FF';  // Magenta for selected
      }
      // Otherwise use the color from the current mode
      return colourMode === 'domain' ? colors[idx] : colors[idx];
    });

    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: coords.map(c => c[0]),
      y: coords.map(c => c[1]),
      z: coords.map(c => c[2]),
      marker: {
        size: 8,
        // Always keep the continuous colorscale/colorbar visible for
        // non-domain modes, even when residues are selected -- selection is
        // shown via the marker outline (below) instead of overriding the
        // fill color to flat magenta, which used to also force
        // showscale/colorbar off and made the scale bar disappear as soon
        // as you clicked any NES motif card or selected a region.
        color: colourMode === 'domain' ? markerColors : colors,
        colorscale: colourMode === 'domain' ? undefined : colorscale,
        showscale: colourMode !== 'domain',
        colorbar: colourMode !== 'domain' ? {
          title: colorbarTitle,
          thickness: 20,
          len: 0.7
        } : undefined,
        line: {
          color: selectedResidues.length > 0 ? 
            coords.map((_, idx) => selectedResidues.includes(idx) ? '#FF00FF' : 'rgba(0,0,0,0.1)') :
            'rgba(0,0,0,0.1)',
          width: selectedResidues.length > 0 ? 
            coords.map((_, idx) => selectedResidues.includes(idx) ? 3 : 0.5) :
            0.5
        }
      },
      hovertemplate: coords.map((c, idx) => 
        `Residue: ${idx + 1}<br>` +
        `${sequence?.[idx] || ''}<br>` +
        `${colorbarTitle}: ${typeof colors[idx] === 'number' ? colors[idx]?.toFixed(2) : 'N/A'}<br>` +
        `<extra></extra>`
      ),
      name: 'Residues'
    });

    // 2. STICKS (bonds between consecutive CA atoms)
    const bondX = [];
    const bondY = [];
    const bondZ = [];
    
    for (let i = 0; i < coords.length - 1; i++) {
      bondX.push(coords[i][0], coords[i+1][0], null);
      bondY.push(coords[i][1], coords[i+1][1], null);
      bondZ.push(coords[i][2], coords[i+1][2], null);
    }

    traces.push({
      type: 'scatter3d',
      mode: 'lines',
      x: bondX,
      y: bondY,
      z: bondZ,
      line: {
        color: '#666666',
        width: 4
      },
      hoverinfo: 'skip',
      showlegend: false,
      name: 'Bonds'
    });

    return {
      data: traces,
      layout: {
        scene: {
          xaxis: { visible: false, showgrid: false, zeroline: false },
          yaxis: { visible: false, showgrid: false, zeroline: false },
          zaxis: { visible: false, showgrid: false, zeroline: false },
          bgcolor: theme === 'dark' ? '#1e293b' : '#ffffff',
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 }
          }
        },
        paper_bgcolor: theme === 'dark' ? '#1e293b' : '#ffffff',
        plot_bgcolor: theme === 'dark' ? '#1e293b' : '#ffffff',
        margin: { l: 0, r: 0, t: 0, b: 0 },
        showlegend: false,
        hovermode: 'closest'
      },
      config: {
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['toImage']
      }
    };
  };

  // ===========================================================================
  // RENDER
  // ===========================================================================
  
  return (
    <div className="app">
      {/* Header */}
      <div className="app-header">
        <h1>🧬 AlphaFold NES/CRM1 Analyser</h1>
        <button onClick={toggleTheme} className="theme-toggle">
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* Search Section */}
      {!selectedProtein && !structureData && (
        <div className="search-section card">
          <h2>Search Proteins</h2>
          <div className="search-controls">
              <input
                type="text"
                placeholder="Enter protein name or UniProt ID (e.g., P04637, p53)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                className="search-input"
              />
              <select
                value={organism}
                onChange={(e) => setOrganism(e.target.value)}
                className="organism-select"
              >
                <option value="all">All Organisms</option>
                <option value="human">Human</option>
                <option value="mouse">Mouse</option>
                <option value="rat">Rat</option>
                <option value="yeast">Yeast</option>
                <option value="e.coli">E. coli</option>
              </select>
              <select
                value={structureSource}
                onChange={(e) => setStructureSource(e.target.value)}
                className="structure-source-select"
                title="Choose whether to show AlphaFold predictions, real experimentally solved structures from the PDB, or both"
              >
                <option value="alphafold">AlphaFold (predicted)</option>
                <option value="experimental">Experimental (PDB)</option>
                <option value="both">Both</option>
              </select>
              <button onClick={handleSearch} disabled={loading} className="search-btn">
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>
      )}

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="search-results">
          <h2>Search Results ({searchResults.length})</h2>
          {searchResults.map((protein, idx) => (
            <div key={idx} className="protein-card card">
              <div className="protein-info">
                <h3>{protein.name}</h3>
                <p className="protein-details">
                  <span>🆔 {protein.id}</span>
                  {protein.organism && <span>🧬 {protein.organism}</span>}
                  <span>📏 {protein.residues} residues</span>
                </p>
              </div>
              <button
                onClick={() => handleGetStructures(protein)}
                className="primary-btn"
              >
                View Structures →
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Structure Selection */}
      {structures.length > 0 && !structureData && (
        <div className="structures-section">
          <button onClick={() => { setStructures([]); setSelectedProtein(null); }} className="back-btn">
            ← Back to Search
          </button>
          
          <h2>Available Structures for {selectedProtein?.name}</h2>
          <div className="structures-grid">
            {structures.map((structure, idx) => {
              const modelId = structure.model_id || structure.entryId || structure.id;
              const isExperimental = structure.source === 'experimental';
              const numResidues = structure.numResidues || structure.length || 'N/A';
              const avgConfidence = structure.avgConfidence || structure.confidenceAvgLocalScore || 'N/A';

              return (
                <div key={idx} className="structure-card">
                  <div className="structure-header">
                    <h3>{modelId}</h3>
                    <span className="version-badge">
                      {isExperimental ? 'PDB' : `v${structure.latestVersion || structure.version || '?'}`}
                    </span>
                  </div>
                  <div className="structure-info">
                    {isExperimental ? (
                      <>
                        <p><strong>Method:</strong> {structure.method || 'Unknown'}</p>
                        <p><strong>Resolution:</strong> {structure.resolution ? `${structure.resolution} Å` : 'N/A'}</p>
                        <p><strong>Chains:</strong> {structure.chains || 'N/A'}</p>
                      </>
                    ) : (
                      <>
                        <p><strong>Residues:</strong> {numResidues}</p>
                        <p><strong>Avg Confidence:</strong> {typeof avgConfidence === 'number' ? avgConfidence.toFixed(2) : avgConfidence}</p>
                      </>
                    )}
                  </div>
                  <button
                    onClick={() => handleLoadStructure(structure)}
                    className="primary-btn"
                  >
                    Load Structure →
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Structure Visualization and Analysis */}
      {structureData && (
        <div className="structure-viewer">
          <button onClick={() => { setStructureData(null); setStructures([]); setSelectedProtein(null); }} className="back-btn">
            ← New Search
          </button>

          {/* Protein Info Card */}
          <div className="protein-info-card card">
            <h2>{structureData.model_id}</h2>
            <div className="info-grid">
              <div className="info-item">
                <span className="info-label">Model ID:</span>
                <span className="info-value">{structureData.model_id}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Sequence Length:</span>
                <span className="info-value">{structureData.sequence?.length || 0} residues</span>
              </div>
              <div className="info-item">
                <span className="info-label">
                  {structureData.confidence_metric === 'bfactor' ? 'Mean B-factor:' : 'Mean pLDDT:'}
                </span>
                <span className="info-value">{structureData.mean_plddt?.toFixed(2) || 'N/A'}</span>
              </div>
              {structureData.source === 'experimental' && (
                <div className="info-item">
                  <span className="info-label">Source:</span>
                  <span className="info-value">Experimental (PDB) — B-factor is real crystallographic/cryo-EM data, not a confidence score</span>
                </div>
              )}
            </div>
          </div>

          {/* Color Mode Selector */}
          <div className="color-controls card">
            <h3>Color Structure By:</h3>
            <div className="color-buttons">
              <button
                className={colourMode === 'plddt' ? 'active' : ''}
                onClick={() => setColourMode('plddt')}
              >
                {structureData.confidence_metric === 'bfactor' ? 'B-factor (confidence)' : 'pLDDT'}
              </button>
              <button 
                className={colourMode === 'bfactor' ? 'active' : ''} 
                onClick={() => setColourMode('bfactor')}
              >
                B-factor
              </button>
              <button
                className={colourMode === 'sasa' ? 'active' : ''}
                onClick={() => setColourMode('sasa')}
              >
                Accessibility
              </button>
              <button
                className={colourMode === 'disorder' ? 'active' : ''}
                onClick={() => setColourMode('disorder')}
                title={structureData?.disorder_source === 'iupred2a'
                  ? 'IUPred2A-predicted disorder'
                  : 'Structural heuristic disorder (IUPred2A unavailable for this protein)'}
              >
                Disorder
              </button>
              <button
                className={colourMode === 'anchor2' ? 'active' : ''}
                onClick={() => setColourMode('anchor2')}
                disabled={!structureData?.anchor2_binding}
                title="ANCHOR2: predicted disordered binding regions (from IUPred2A)"
              >
                ANCHOR2
              </button>
              <button
                className={colourMode === 'hydrophobicity' ? 'active' : ''}
                onClick={() => setColourMode('hydrophobicity')}
              >
                Hydrophobicity
              </button>
              <button 
                className={colourMode === 'charge' ? 'active' : ''} 
                onClick={() => setColourMode('charge')}
              >
                Charge
              </button>
              <button 
                className={colourMode === 'domain' ? 'active' : ''} 
                onClick={() => setColourMode('domain')}
              >
                Domains
              </button>
              <button
                className={colourMode === 'nes' ? 'active' : ''}
                onClick={() => setColourMode('nes')}
                disabled={!structureData.crm1_binding_regions}
                title={structureData.crm1_binding_regions ? 'Color by predicted NES likelihood' : 'Run CRM1 analysis first'}
              >
                🔬 NES Likelihood
              </button>
              <button
                className={colourMode === 'nls' ? 'active' : ''}
                onClick={() => setColourMode('nls')}
                disabled={!structureData.nls_binding_regions}
                title={structureData.nls_binding_regions ? 'Color by predicted NLS likelihood' : 'Run NLS prediction first'}
              >
                🧬 NLS Likelihood
              </button>
              <button
                className={colourMode === 'pockets' ? 'active' : ''}
                onClick={() => setColourMode('pockets')}
                disabled={!structureData.pocket_scores}
                title={structureData.pocket_scores ? 'Color by detected CRM1-compatible binding pockets' : 'Run CRM1 analysis first'}
              >
                🎯 CRM1 Pockets
              </button>
              <button
                className={colourMode === 'linear_hydropathy' ? 'active' : ''}
                onClick={() => setColourMode('linear_hydropathy')}
                disabled={!structureData.cider_computed}
                title={structureData.cider_computed ? 'Color by CIDER linear hydropathy' : 'localCIDER not available on the server'}
              >
                Linear Hydropathy
              </button>
              <button
                className={colourMode === 'linear_ncpr' ? 'active' : ''}
                onClick={() => setColourMode('linear_ncpr')}
                disabled={!structureData.cider_computed}
                title={structureData.cider_computed ? 'Color by CIDER linear net charge per residue' : 'localCIDER not available on the server'}
              >
                Linear NCPR
              </button>
              <button
                className={colourMode === 'linear_fcr' ? 'active' : ''}
                onClick={() => setColourMode('linear_fcr')}
                disabled={!structureData.cider_computed}
                title={structureData.cider_computed ? 'Color by CIDER linear fraction of charged residues' : 'localCIDER not available on the server'}
              >
                Linear FCR
              </button>
            </div>
          </div>

          {/* Domain Legend (only show when in domain mode) */}
          {colourMode === 'domain' && structureData.domains && structureData.domains.length > 0 && (
            <div className="domain-legend card">
              <h3>Domain Legend</h3>
              <div className="domain-items">
                {structureData.domains.map((domain, idx) => (
                  <div key={idx} className="domain-item">
                    <div style={{
                      width: '30px',
                      height: '20px',
                      backgroundColor: domain.color,
                      border: '1px solid #ccc',
                      borderRadius: '3px'
                    }}></div>
                    <span>{domain.description} ({domain.start}-{domain.end})</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Region Selection - REORGANIZED BUTTONS */}
          <div className="region-selector card">
            <h3>Select Sequence Region</h3>
            <div className="region-controls">
              <input
                type="number"
                placeholder="Start"
                value={regionStart}
                onChange={(e) => setRegionStart(e.target.value)}
                min="1"
                max={structureData.sequence?.length || 0}
              />
              <span>to</span>
              <input
                type="number"
                placeholder="End"
                value={regionEnd}
                onChange={(e) => setRegionEnd(e.target.value)}
                min="1"
                max={structureData.sequence?.length || 0}
              />
              <button onClick={handleRegionSelect} disabled={loading}>
                Highlight Region
              </button>
              <button onClick={toggleSequenceHighlight} disabled={loading} className="sequence-btn">
                {showSequence ? '✕ Close' : '📝 Highlight on Sequence'}
              </button>
              <button onClick={handleRegionAnalysis} disabled={loading} className="analyse-btn">
                Analyse
              </button>
              <button onClick={handleUnselectRegion} disabled={loading} className="unselect-btn">
                Unselect
              </button>
            </div>

            {/* Sequence Display */}
            {showSequence && structureData.sequence && (
              <div className="sequence-display">
                <p className="sequence-instruction">
                  {sequenceSelection.start === null ? 
                    'Click on the first residue to start selection' : 
                    'Click on the last residue to complete selection'}
                </p>
                <div className="sequence-viewer">
                  {structureData.sequence.split('').map((aa, idx) => (
                    <span
                      key={idx}
                      className={`residue ${
                        sequenceSelection.start === idx ? 'selected-start' : ''
                      } ${
                        selectedResidues.includes(idx) ? 'highlighted' : ''
                      }`}
                      onClick={() => handleSequenceClick(idx)}
                      title={`${aa}${idx + 1}`}
                    >
                      {aa}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* 3D Structure Viewer */}
          <div className="structure-plot card">
            {getStructurePlot() && (
              <Plot
                data={getStructurePlot().data}
                layout={getStructurePlot().layout}
                config={getStructurePlot().config}
                style={{ width: '100%', height: '600px' }}
              />
            )}
          </div>

          {/* Region Analysis Results */}
          {regionAnalysis && (
            <div className="region-analysis-results card">
              <h3>Region Analysis (Residues {regionStart}-{regionEnd})</h3>
              <div className="analysis-grid">
                <div className="analysis-item">
                  <span className="analysis-label">Average Accessibility (RSA):</span>
                  <span className="analysis-value">{regionAnalysis.avg_sasa}</span>
                </div>
                <div className="analysis-item">
                  <span className="analysis-label">Total Accessibility (sum RSA):</span>
                  <span className="analysis-value">{regionAnalysis.total_sasa}</span>
                </div>
                <div className="analysis-item">
                  <span className="analysis-label">Average Charge:</span>
                  <span className="analysis-value">{regionAnalysis.avg_charge}</span>
                </div>
                <div className="analysis-item">
                  <span className="analysis-label">Net Charge:</span>
                  <span className="analysis-value">{regionAnalysis.net_charge}</span>
                </div>
                <div className="analysis-item">
                  <span className="analysis-label">Disorder Score:</span>
                  <span className="analysis-value">{regionAnalysis.avg_disorder}</span>
                </div>
                <div className="analysis-item">
                  <span className="analysis-label">Hydrophobicity:</span>
                  <span className="analysis-value">{regionAnalysis.avg_hydrophobicity}</span>
                </div>
                <div className="analysis-item highlight">
                  <span className="analysis-label">PPI Likelihood:</span>
                  <span className="analysis-value">{regionAnalysis.ppi_likelihood}%</span>
                </div>
              </div>
              <div className="analysis-interpretation">
                <strong>Interpretation:</strong> {regionAnalysis.interpretation}
              </div>
            </div>
          )}

          {/* CIDER Linear Profiles (real localCIDER, not the NES predictor).
              was unconditionally expanded whenever regionAnalysis
              was set; now gated behind a dropdown like the candidate-card
              versions, and uses the same restyled 2x2 CiderPanel. */}
          {regionAnalysis && regionAnalysis.cider_profile && (
            <div className="cider-analysis-results card">
              <h3>CIDER Linear Profiles (Residues {regionStart}-{regionEnd})</h3>
              <button
                className="cider-dropdown-toggle"
                onClick={() => setExpandedRegionCider(prev => !prev)}
              >
                {expandedRegionCider ? '▲ Hide' : '▼ Show'} CIDER profile (hydropathy / NCPR / FCR / complexity)
              </button>
              {expandedRegionCider && (
                <div className="nes-cider-graphs">
                  <CiderPanel profile={regionAnalysis.cider_profile} filenamePrefix="region_analysis" />
                </div>
              )}
            </div>
          )}

          {/* CRM1/NES Analysis Button + NLS Structure Scan Button --
              same section, same styling, NLS directly below NES per request.
              NLS has no MD/fpocket step (no single dominant cargo receptor
              pocket for classical import -- see
              NLS_predictor_landscape_and_novelty.md), so its button just
              triggers the lighter scan_sequence() pass and unlocks the NLS
              colour mode + region list below, same UX shape as the NES button. */}
          <div className="crm1-section card">
            <button
              onClick={handleCRM1Analysis}
              disabled={loading}
              className="crm1-button primary-btn"
            >
              🔬 Predict CRM1/NES Binding Sites
            </button>
            <button
              onClick={handleNLSPredict}
              disabled={loading}
              className="crm1-button primary-btn"
              style={{ marginTop: '12px' }}
            >
              🧬 Predict NLS
            </button>
          </div>

          {/* Predicted NLS Regions -- structural analog of the "Predicted
              NES Motifs" list below, same card/list styling, just without
              the MD/pocket/SUMOylation panels that don't apply to NLS. */}
          {structureData.nls_summary && (
            <div className="nes-motifs-section card">
              <h3>Predicted NLS Regions</h3>
              <div className="summary-grid" style={{ marginBottom: '16px' }}>
                <div className="summary-item">
                  <span className="summary-value">{structureData.nls_summary.filtered_predictions}</span>
                  <span className="summary-label">NLS regions</span>
                </div>
                <div className="summary-item">
                  <span className="summary-value">{structureData.nls_summary.high_confidence}</span>
                  <span className="summary-label">High confidence</span>
                </div>
                <div className="summary-item">
                  <span className="summary-value">{structureData.nls_summary.medium_confidence}</span>
                  <span className="summary-label">Medium confidence</span>
                </div>
                <div className="summary-item">
                  <span className="summary-value">{structureData.nls_summary.low_confidence}</span>
                  <span className="summary-label">Low confidence</span>
                </div>
              </div>

              {structureData.nls_binding_regions && structureData.nls_binding_regions.length > 0 ? (
                <div className="nes-motifs">
                  {structureData.nls_binding_regions.map((region, idx) => {
                    const confidenceColor =
                      region.nls_probability > 0.75 ? '#10b981' :
                      region.nls_probability > 0.6 ? '#f59e0b' : '#6b7280';
                    const confidenceLabel =
                      region.nls_probability > 0.75 ? 'HIGH' :
                      region.nls_probability > 0.6 ? 'MEDIUM' : 'LOW';

                    return (
                      <div key={idx} className="nes-motif-card" style={{ borderLeftColor: confidenceColor, cursor: 'pointer' }}
                        onClick={() => {
                          setColourMode('nls');
                          const range = [];
                          for (let i = region.start - 1; i < region.end; i++) range.push(i);
                          setSelectedResidues(range);
                        }}
                      >
                        <div className="nes-motif-header">
                          <code className="nes-sequence">{region.sequence}</code>
                          <span className="confidence-badge" style={{ backgroundColor: confidenceColor }}>
                            {confidenceLabel}
                          </span>
                        </div>
                        <div className="nes-motif-info">
                          <span>Residues {region.start}-{region.end}</span>
                          <span>Length: {region.length} aa</span>
                          <span>Probability: {(region.nls_probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="nes-motif-info">
                          <span>Class: {region.predicted_class}</span>
                          <span>PSSM score: {region.pssm_score}</span>
                        </div>
                        <div className="nes-motif-info">
                          <span>Accessibility (RSA): {region.accessibility_rsa}</span>
                          <span>
                            Raw ML probability: {(region.raw_nls_probability * 100).toFixed(1)}%
                            {region.exposure_factor < 1 && (
                              <span className="warning"> (×{region.exposure_factor} exposure gate)</span>
                            )}
                          </span>
                        </div>

                        {/* Potential tripartite NLS note -- heuristic flag
                            only, does not affect nls_probability above. See
                            detect_extra_basic_cluster in nls_ml_predictor.py. */}
                        {region.potential_tripartite && region.tripartite_note && (
                          <p className="crm1-pocket-note">
                            ⚠️ {region.tripartite_note}
                          </p>
                        )}

                        {/* CIDER graphs dropdown -- same 2x2 panel as the
                            NES cards, auto-generated for this NLS region +
                            its flanking context. NLS previously
                            had no CIDER/RSA dropdowns at all. */}
                        <button
                          className="cider-dropdown-toggle"
                          onClick={(e) => {
                            e.stopPropagation();
                            setExpandedNlsCider(prev => ({ ...prev, [idx]: !prev[idx] }));
                          }}
                        >
                          {expandedNlsCider[idx] ? '▲ Hide' : '▼ Show'} CIDER profile (hydropathy / NCPR / FCR / complexity)
                        </button>

                        {expandedNlsCider[idx] && (
                          <div className="nes-cider-graphs" onClick={(e) => e.stopPropagation()}>
                            <CiderPanel profile={region.cider_profile} filenamePrefix={`nls_${region.start}`} />
                          </div>
                        )}

                        {/* RSA consensus/z-score/agreement-SD dropdown --
                            same combined panel as the NES cards. */}
                        <button
                          className="cider-dropdown-toggle"
                          onClick={(e) => {
                            e.stopPropagation();
                            setExpandedNlsRsa(prev => ({ ...prev, [idx]: !prev[idx] }));
                          }}
                        >
                          {expandedNlsRsa[idx] ? '▲ Hide' : '▼ Show'} RSA profile (consensus / z-score / SD)
                        </button>

                        {expandedNlsRsa[idx] && (
                          <div className="nes-cider-graphs" onClick={(e) => e.stopPropagation()}>
                            <RsaPanel profile={region.rsa_profile} filenamePrefix={`nls_${region.start}`} />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p>No NLS regions found above the confidence threshold.</p>
              )}
            </div>
          )}

          {/* CRM1/NES Results */}
          {structureData.crm1_binding_regions && structureData.crm1_binding_regions.length > 0 && (
            <div className="crm1-results">
              {/* Scoring Info Panel at top of results */}
              
              {/* Summary */}
              {structureData.crm1_summary && (
                <div className="crm1-summary card">
                  <h3>Analysis Summary</h3>
                  <div className="summary-grid">
                    <div className="summary-item">
                      <span className="summary-value">{structureData.crm1_summary.filtered_predictions}</span>
                      <span className="summary-label">NES motifs</span>
                    </div>
                    <div className="summary-item">
                      <span className="summary-value">{structureData.crm1_summary.high_confidence}</span>
                      <span className="summary-label">High confidence</span>
                    </div>
                    <div className="summary-item">
                      <span className="summary-value">{structureData.crm1_summary.medium_confidence}</span>
                      <span className="summary-label">Medium confidence</span>
                    </div>
                    <div className="summary-item">
                      <span className="summary-value">{structureData.crm1_summary.pockets_detected}</span>
                      <span className="summary-label">CRM1 pockets</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Analysis Mode Selector */}
              <div className="analysis-mode-selector">
                <button
                  className={`mode-btn ${quickAnalysisMode ? 'active' : ''}`}
                  onClick={() => setQuickAnalysisMode(true)}
                >
                  ⚡ Quick Analysis (~1 sec)
                </button>
                <button
                  className={`mode-btn ${!quickAnalysisMode ? 'active' : ''}`}
                  onClick={() => setQuickAnalysisMode(false)}
                >
                  🔬 Full MD Analysis
                </button>
              </div>

              {/* Quick Analysis Panel */}
              {quickAnalysisMode && (
                <QuickAnalysisPanel
                  candidates={structureData.crm1_binding_regions}
                  onAnalysisComplete={(results) => {
                    // Update predictions with quick analysis results
                    if (results.candidates) {
                      setStructureData({
                        ...structureData,
                        crm1_binding_regions: results.candidates
                      });
                    }
                  }}
                  apiBase={API_BASE}
                />
              )}

              {/* MD Docking Panel */}
              {!quickAnalysisMode && (
                <div className="md-docking-section card">
                  <h3>🧬 Advanced CRM1 Binding Analysis</h3>
                  <p className="md-description">
                    Run a molecular dynamics simulation to analyse CRM1 binding affinity.
                    This computational approach provides binding scores, contact analysis, and affinity estimates.
                    {mdCandidateMode === 'top10'
                      ? ' (Fixed at 10 ns each; simulates the top 10 candidates by score, ~3-4 hrs total.)'
                      : ' (Fixed at 10 ns, ~20 min for the selected candidate.)'}
                  </p>

                  <div className="md-controls">
                    <label className="md-candidate-label">
                      Candidates to simulate:
                    </label>
                    <div
                      className="md-mode-toggle"
                      role="group"
                      aria-label="MD candidate selection mode"
                      style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}
                    >
                      <button
                        type="button"
                        onClick={() => setMdCandidateMode('single')}
                        disabled={loading || (mdJobStatus && mdJobStatus.status === 'running')}
                        className={`mode-btn ${mdCandidateMode === 'single' ? 'active' : ''}`}
                      >
                        1 candidate
                      </button>
                      <button
                        type="button"
                        onClick={() => setMdCandidateMode('top10')}
                        disabled={loading || (mdJobStatus && mdJobStatus.status === 'running')}
                        className={`mode-btn ${mdCandidateMode === 'top10' ? 'active' : ''}`}
                      >
                        Top 10 candidates
                      </button>
                    </div>

                    {mdCandidateMode === 'single' && (
                      <>
                        <label htmlFor="md-candidate-select" className="md-candidate-label">
                          NES candidate to simulate:
                        </label>
                        <select
                          id="md-candidate-select"
                          className="md-candidate-select"
                          value={selectedMdCandidateIdx}
                          onChange={(e) => setSelectedMdCandidateIdx(Number(e.target.value))}
                          disabled={loading || (mdJobStatus && mdJobStatus.status === 'running')}
                        >
                          {getSortedMdCandidates().map((c, idx) => (
                            <option key={idx} value={idx}>
                              #{idx + 1} · {c.sequence} (res {c.start}-{c.end}, score {c.combined_score?.toFixed(3) ?? 'N/A'})
                            </option>
                          ))}
                        </select>
                      </>
                    )}

                    {mdCandidateMode === 'top10' && (
                      <p className="md-description" style={{ fontSize: '0.9em', opacity: 0.8 }}>
                        Will simulate the top {Math.min(10, getSortedMdCandidates().length)} candidates by combined score.
                      </p>
                    )}

                    <button
                      onClick={() => handleRunMD()}
                      disabled={loading || (mdJobStatus && mdJobStatus.status === 'running')}
                      className="md-run-btn primary-btn"
                    >
                      🚀 Run MD Simulation
                    </button>
                  </div>

                  {/* MD Job Status */}
                  {mdJobStatus && (
                    <div className="md-status-panel">
                      <h4>Simulation Status</h4>
                      
                      {mdJobStatus.status === 'running' && (
                        <div className="md-progress">
                          <div className="progress-bar">
                            <div 
                              className="progress-fill" 
                              style={{ width: `${mdJobStatus.progress || 0}%` }}
                            />
                          </div>
                          <p className="progress-text">
                            Processing {mdJobStatus.progress || 0}% - 
                            Simulating {mdJobStatus.num_candidates} NES candidates 
                            ({mdJobStatus.duration_ns} ns each)
                          </p>
                        </div>
                      )}
                      
                      {mdJobStatus.status === 'queued' && (
                        <p className="status-message">
                          ⏳ Job queued - Will process {mdJobStatus.num_candidates} candidates
                        </p>
                      )}
                      
                      {mdJobStatus.status === 'failed' && (
                        <p className="status-message error">
                          ❌ Simulation failed: {mdJobStatus.error}
                        </p>
                      )}
                      
                      {mdJobStatus.status === 'completed' && (
                        <p className="status-message success">
                          ✓ MD simulation completed successfully
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* NES Motifs - UPDATED WITH FLANKING INFO AND NES CLASS */}
              <div className="nes-motifs-section card">
                <h3>Predicted NES Motifs</h3>
                <div className="nes-motifs">
                  {structureData.crm1_binding_regions
                    .sort((a, b) => b.combined_score - a.combined_score)
                    .map((region, idx) => {
                      const confidenceColor = 
                        region.combined_score > 0.70 ? '#10b981' :
                        region.combined_score > 0.55 ? '#f59e0b' : '#6b7280';
                      const confidenceLabel =
                        region.combined_score > 0.70 ? 'HIGH' :
                        region.combined_score > 0.55 ? 'MEDIUM' : 'LOW';
                      
                      return (
                        <div key={idx} className="nes-motif-card" style={{ borderLeftColor: confidenceColor, cursor: 'pointer' }}
                          onClick={() => {
                            // Switch to NES colour mode so the heatmap is visible
                            setColourMode('nes');
                            // Highlight this motif's residues on the structure
                            const range = [];
                            for (let i = region.start - 1; i < region.end; i++) range.push(i);
                            setSelectedResidues(range);
                          }}
                        >
                          <div className="nes-motif-header">
                            <code className="nes-sequence">{region.sequence}</code>
                            <span className="confidence-badge" style={{ backgroundColor: confidenceColor }}>
                              {confidenceLabel}
                            </span>
                          </div>
                          <div className="nes-motif-info">
                            <span>Residues {region.start}-{region.end}</span>
                            <span>Length: {region.length} aa</span>
                            <span>Score: {region.combined_score?.toFixed(3) || 'N/A'}</span>
                          </div>

                          {/* Clear, standalone note when this NES overlaps a
                              real fpocket-detected CRM1-compatible pocket --
                              distinct from the small numeric compatibility
                              score buried in the components grid below. */}
                          {region.has_crm1_pocket && (
                            <p className="crm1-pocket-note">
                              🎯 Predicted CRM1 binding pocket
                            </p>
                          )}

                          {/* CIDER graphs dropdown -- auto-generated for this
                              exact predicted NES region + its flanking
                              context, no manual region selection needed */}
                          <button
                            className="cider-dropdown-toggle"
                            onClick={(e) => {
                              e.stopPropagation();
                              setExpandedNesCider(prev => ({ ...prev, [idx]: !prev[idx] }));
                            }}
                          >
                            {expandedNesCider[idx] ? '▲ Hide' : '▼ Show'} CIDER profile (hydropathy / NCPR / FCR / complexity)
                          </button>

                          {expandedNesCider[idx] && (
                            <div className="nes-cider-graphs" onClick={(e) => e.stopPropagation()}>
                              <CiderPanel profile={region.cider_profile} filenamePrefix={`nes_${region.start}`} />
                            </div>
                          )}

                          {/* RSA consensus/z-score/agreement-SD dropdown --
                              auto-generated for this exact predicted NES
                              region + its flanking context, same pattern as
                              the CIDER dropdown above. consensus_rsa is the
                              real Tien et al. 2013-normalized 3-method
                              accessibility (FreeSASA Lee-Richards + FreeSASA
                              Shrake-Rupley + Biopython Shrake-Rupley); z-score
                              flags residues unusually buried/exposed relative
                              to the rest of THIS protein; agreement_sd flags
                              where the 3 methods disagree (treat those
                              residues' RSA with caution). */}
                          <button
                            className="cider-dropdown-toggle"
                            onClick={(e) => {
                              e.stopPropagation();
                              setExpandedNesRsa(prev => ({ ...prev, [idx]: !prev[idx] }));
                            }}
                          >
                            {expandedNesRsa[idx] ? '▲ Hide' : '▼ Show'} RSA profile (consensus / z-score / SD)
                          </button>

                          {expandedNesRsa[idx] && (
                            <div className="nes-cider-graphs" onClick={(e) => e.stopPropagation()}>
                              <RsaPanel profile={region.rsa_profile} filenamePrefix={`nes_${region.start}`} />
                            </div>
                          )}

                          {region.components && (
                            <div className="nes-components">
                              <div className="component-item">
                                <strong>ML:</strong> {region.components.ml_probability?.toFixed(3) || 'N/A'}
                              </div>
                              
                              {/* NES Class badges */}
                              {region.components.nes_classes && region.components.nes_classes.length > 0 && (
                                <div className="component-item">
                                  <strong>Class:</strong> {region.components.nes_classes.map(cls => {
                                    const displayClass = cls.replace('class_', '');
                                    return <span key={cls} className="nes-class-badge">{displayClass}</span>;
                                  })}
                                </div>
                              )}
                              
                              {/* Spacer hydrophobicity warning */}
                              {region.components.spacer_hydrophobicity >= 0.4 && (
                                <div className="component-item warning">
                                  <strong>⚠️ Spacer HPR:</strong> {(region.components.spacer_hydrophobicity * 100).toFixed(0)}%
                                  <small> (High - possible TM region)</small>
                                </div>
                              )}
                              
                              <div className="component-item">
                                <strong>Hydro:</strong> {
                                  region.avg_hydrophobicity?.toFixed(3) || 
                                  region.components.hydrophobicity?.toFixed(3) || 
                                  'N/A'
                                }
                              </div>
                              <div className="component-item">
                                <strong>Accessibility:</strong> {region.components.surface_accessibility?.toFixed(3) || 'N/A'}
                              </div>
                              <div className="component-item">
                                <strong>Disorder:</strong> {region.components.disorder?.toFixed(3) || 'N/A'}
                                {region.components.disorder_source && (
                                  <span className="disorder-source-tag">
                                    {' '}({region.components.disorder_source === 'iupred2a' ? 'IUPred2A' : 'heuristic'})
                                  </span>
                                )}
                              </div>
                              {region.components.anchor2_binding != null && (
                                <div className="component-item">
                                  <strong>ANCHOR2 (binding region):</strong> {region.components.anchor2_binding.toFixed(3)}
                                </div>
                              )}
                              <div className="component-item">
                                <strong>Flexibility:</strong> {region.components.flexibility?.toFixed(3) || 'N/A'}
                              </div>
                              {region.components.pocket_compatibility > 0 && (
                                <div className="component-item highlight">
                                  <strong>✓ CRM1 Pocket:</strong> {region.components.pocket_compatibility?.toFixed(3)}
                                </div>
                              )}
                              
                              {/* UPDATED: Show flanking region analysis */}
                              {region.components.flanking_analysis && (
                                <div className="flanking-analysis">
                                  <details>
                                    <summary><strong>📊 Flanking Region</strong></summary>
                                    <div className="flanking-details">
                                      <div className="component-item">
                                        <strong>Accessibility:</strong> {region.components.flanking_analysis.sasa?.toFixed(3) || 'N/A'}
                                      </div>
                                      <div className="component-item">
                                        <strong>Disorder:</strong> {region.components.flanking_analysis.disorder?.toFixed(3) || 'N/A'}
                                      </div>
                                      <div className="component-item">
                                        <strong>N-flank HPR:</strong> {region.components.flanking_analysis.hpr?.toFixed(1) || 'N/A'}%
                                        {region.components.flanking_analysis.hpr > 60 && (
                                          <span className="warning"> (Too hydrophobic)</span>
                                        )}
                                      </div>
                                      <div className="component-item">
                                        <strong>C-flank Net Charge:</strong> {region.components.flanking_analysis.nc || 'N/A'}
                                      </div>
                                      <div className="component-item">
                                        <strong>Likelihood Adjustment:</strong> 
                                        <span className={region.components.flanking_analysis.combined_likelihood >= 1 ? 'success' : 'warning'}>
                                          {region.components.flanking_analysis.combined_likelihood?.toFixed(2) || 'N/A'}×
                                        </span>
                                      </div>
                                    </div>
                                  </details>
                                </div>
                              )}
                              
                              {region.components.uniprot_disorder && (
                                <div className="component-item highlight">
                                  ✓ In UniProt disordered region
                                </div>
                              )}
                            </div>
                          )}
                          
                          {/* SUMOylation Context */}
                          {structureData.sequence && (
                            <SUMOylationContext 
                              sequence={structureData.sequence}
                              nesCandidate={region}
                              apiBase={API_BASE}
                            />
                          )}
                        </div>
                      );
                    })}
                </div>
              </div>

              {/* Information Panel */}
              <div className="info-panel card">
                <h4>📊 About this Enhanced Analysis</h4>
                <ul>
                  <li><strong>ML Prediction:</strong> Uses SVM with PSSM scoring, flanking region analysis, and disorder propensity</li>
                  <li><strong>CRM1 Pocket Binding:</strong> fpocket analysis of hydrophobic groove compatibility around Cys528 using real CRM1 structure template</li>
                  <li><strong>Hydrophobicity:</strong> Evaluates presence and spacing of 4-5 hydrophobic anchors (L, I, V, F, M)</li>
                  <li><strong>Accessibility (RSA):</strong> Consensus relative solvent accessibility -- FreeSASA (Lee-Richards + Shrake-Rupley) and Biopython Shrake-Rupley, each normalized against residue-specific theoretical max surface area (Tien et al. 2013), not a flat divisor. 0 = fully buried, 1 = fully exposed. Required for CRM1 accessibility.</li>
                  <li><strong>Flexibility:</strong> Higher pLDDT variance indicates flexibility (beneficial for NES)</li>
                  <li><strong>Disorder:</strong> IUPred2A-predicted disorder when available (falls back to a pLDDT/composition heuristic otherwise -- see the tag next to each candidate's Disorder score), plus UniProt-annotated disorder regions</li>
                  <li><strong>ANCHOR2:</strong> Predicted disordered <em>binding</em> regions -- segments likely to fold up on engaging a partner, which is what a NES engaging CRM1's groove actually is</li>
                  <li><strong>Pattern:</strong> Φ-X₍₂₋₃₎-Φ-X₍₂₋₃₎-Φ-X-Φ (Φ = hydrophobic)</li>
                  <li><strong>Scoring:</strong> Balanced weighting emphasising CRM1 pocket compatibility and structural features over ML alone</li>
                </ul>
              </div>
            </div>
          )}

          {/* MD Results Display */}
          {showMdResults && mdResults && mdResults.enhanced_candidates && (
            <div className="md-results-section card">
              <div className="md-results-header">
                <h3>🧬 MD Simulation Results - CRM1 Binding Analysis</h3>
                <button onClick={() => setShowMdResults(false)} className="close-btn">✕</button>
              </div>
              
              {/* Summary */}
              {mdResults.summary && (
                <div className="md-summary-grid">
                  <div className="summary-item">
                    <span className="summary-label">Total Refined</span>
                    <span className="summary-value">{mdResults.summary.total_refined}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Strong Binders</span>
                    <span className="summary-value highlight">
                      {mdResults.summary.strong_binders || 0}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Moderate Binders</span>
                    <span className="summary-value">{mdResults.summary.moderate_binders || 0}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Avg Binding Score</span>
                    <span className="summary-value">
                      {(mdResults.summary.avg_binding_score || 0).toFixed(3)}
                    </span>
                  </div>
                </div>
              )}
              
              {/* Individual Candidates */}
              <div className="md-candidates-list">
                <h4>Detailed Binding Analysis</h4>
                
                {mdResults.enhanced_candidates
                  .sort((a, b) => 
                    (b.md_metrics?.binding_score || 0) - (a.md_metrics?.binding_score || 0)
                  )
                  .map((candidate, idx) => {
                    const metrics = candidate.md_metrics || {};
                    const bindingScore = metrics.binding_score || 0;
                    const category = metrics.binding_category || 'unknown';
                    
                    const scoreColor = 
                      bindingScore > 0.7 ? '#10b981' :
                      bindingScore > 0.4 ? '#f59e0b' : '#ef4444';
                    
                    const categoryLabel = 
                      category === 'strong_binder' ? '🟢 STRONG BINDER' :
                      category === 'moderate_binder' ? '🟡 MODERATE BINDER' :
                      category === 'weak_binder' ? '🔴 WEAK BINDER' : '⚪ PREDICTED';
                    
                    return (
                      <div key={idx} className="md-candidate-card" style={{ borderLeftColor: scoreColor }}>
                        <div className="candidate-header">
                          <div>
                            <h5>NES Candidate {idx + 1}</h5>
                            <p className="sequence-display">{candidate.sequence}</p>
                            <p className="position-info">
                              Position: {candidate.start}-{candidate.end}
                            </p>
                          </div>
                          <div className="binding-badge" style={{ backgroundColor: scoreColor }}>
                            {categoryLabel}
                          </div>
                        </div>
                        
                        <div className="metrics-grid">
                          <div className="metric-item">
                            <span className="metric-label">Binding Score:</span>
                            <span className="metric-value">{bindingScore.toFixed(3)}</span>
                          </div>
                          
                          {metrics.binding_affinity_kcal_mol && (
                            <div className="metric-item">
                              <span className="metric-label">Binding Affinity:</span>
                              <span className="metric-value">
                                {metrics.binding_affinity_kcal_mol.toFixed(1)} kcal/mol
                              </span>
                            </div>
                          )}
                          
                          {metrics.avg_groove_contacts !== undefined && (
                            <div className="metric-item">
                              <span className="metric-label">Groove Contacts:</span>
                              <span className="metric-value">
                                {metrics.avg_groove_contacts.toFixed(1)}
                              </span>
                            </div>
                          )}
                          
                          {metrics.avg_cys528_distance_nm !== undefined && (
                            <div className="metric-item">
                              <span className="metric-label">Distance to Cys528:</span>
                              <span className="metric-value">
                                {metrics.avg_cys528_distance_nm.toFixed(2)} nm
                              </span>
                            </div>
                          )}
                          
                          {metrics.avg_hydrophobic_contacts !== undefined && (
                            <div className="metric-item">
                              <span className="metric-label">Hydrophobic Contacts:</span>
                              <span className="metric-value">
                                {metrics.avg_hydrophobic_contacts.toFixed(1)}
                              </span>
                            </div>
                          )}
                        </div>
                        
                        {metrics.binding_likelihood && (
                          <div className="likelihood-section">
                            <strong>Assessment:</strong> {metrics.binding_likelihood}
                          </div>
                        )}
                        
                        {metrics.note && (
                          <div className="note-section">
                            <em>{metrics.note}</em>
                          </div>
                        )}
                      </div>
                    );
                  })}
              </div>
              
              {/* Scientific Interpretation */}
              <div className="interpretation-box">
                <h4>🔬 Scientific Interpretation</h4>
                <p>
                  These results show the predicted CRM1-NES binding based on molecular dynamics simulations.
                  The analysis considers:
                </p>
                <ul>
                  <li><strong>Hydrophobic groove binding:</strong> 4-5 anchor hydrophobic residues (L, I, V, F, M) 
                      binding in the CRM1 hydrophobic groove near Cys528</li>
                  <li><strong>Binding affinity:</strong> Estimated free energy of binding (more negative = stronger)</li>
                  <li><strong>Contact analysis:</strong> Number of stable contacts between NES and CRM1 groove</li>
                  <li><strong>Spatial positioning:</strong> Distance from NES centre of mass to Cys528</li>
                </ul>
                <p className="disclaimer">
                  <em>Note: These are computational predictions. Experimental validation (e.g., co-IP, 
                  fluorescence microscopy, mutagenesis) is recommended for candidates of interest.</em>
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Loading Overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Loading...</p>
        </div>
      )}
    </div>
  );
}

export default App;
