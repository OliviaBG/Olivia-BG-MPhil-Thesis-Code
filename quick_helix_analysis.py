"""
Quick Helix Propensity and Amphipathic Scoring Module
Provides fast structural analysis without MD simulation

This module is a QUICK, sequence-only alternative to MD simulations. It is meant
to run as a distinct triage step AFTER the ML/consensus-pattern NES predictor has
already produced candidates, and only when the user clicks "Run Quick Analysis"
in the UI (see QuickAnalysisPanel.jsx -> POST /api/quick-analysis in app.py).
It should stay independent of nes_ml_predictor_improved.py and of the OpenMM/
md_docking pipeline -- it is not meant to replace either, only to give a fast,
rough structural plausibility score on top of whatever candidates are handed to it.

It analyzes:
1. alpha-helix propensity using Chou-Fasman parameters
2. Amphipathic moment (hydrophobic face detection, assumes idealized helix)
3. Hydrophobic (Phi) residue spacing register, i.e. whether the candidate's
   L/I/V/F/M residues fall on the Phi-x(1,3)-Phi-x(1,3)-Phi-x(1,2)-Phi-style
   spacing that Kosugi et al. (2008, J Biol Chem) and Fung & Chook (2017,
   PNAS/eLife structural surveys) describe as the actual register CRM1's
   hydrophobic groove reads out.

   this used to be an independent, looser re-implementation
   (five hand-weighted "1a/1b/1c/2/3" class regexes, never checked against
   real labels). It's now the SAME PHI_REGISTER_RE pattern already used by
   md_refinement.NESMDRefiner._find_phi_register (MD anchor-registration)
   and nes_ml_predictor_improved.PSSM_ANCHOR_RE (candidate-generation PSSM
   anchoring) -- three independent call sites, one pattern, so they can't
   silently drift apart. This specific pattern was empirically validated
   this project on 34 REAL, labeled, held-out candidates (real AlphaFold
   structures, real MD docking via evaluate_anchor_occupancy_signal.py):
   87% of real positives had a matching register vs. 42% of hard negatives
   (coiled-coil/leucine-zipper decoys) -- the single largest, cheapest
   (sequence-only, no MD/structure needed) discriminative signal found
   anywhere in this project's evaluation work so far. See
   CRM1_NES_collated_report.docx / the session notes for the
   full evaluation.

Why spacing was added: overall helix propensity and a generic ~90-degree
hydrophobic-moment split can both look favorable for a window that does NOT
actually have hydrophobic residues in the right register, and can also both
look unfavorable for real NES motifs that CRM1 is documented to accept in
non-helical form. In particular, structural surveys of CRM1-NES complexes
(Fung & Chook 2017, eLife 6:e23961) found that although one turn of helix
contacting the central groove is the most conserved feature, NESs bind in at
least 5-6 distinct backbone conformations, including short helix + beta-strand
hybrids (e.g. PKI, Snurportin-1) and fully extended, proline-rich conformations
with no helix at all (e.g. HIV-1 Rev). A scorer that hard-requires both "helix
former" AND "amphipathic" categories will systematically reject legitimate
non-helical NES candidates. This version treats correct Phi spacing as the
primary (necessary) signal and helicity/amphipathicity as corroborating
(not mandatory) evidence, so extended-type candidates with a strong spacing
match are no longer auto-rejected.

UPDATE -- conformation_class corrected against real crystal ground
truth: an evaluation run (see eval_quick_helix_positives_only.py /
this project's notes) found that `conformation_class` mislabeled this
module's OWN worked example -- HIV-1 Rev, explicitly documented since Guttler
et al. 2010 (Nat Struct Mol Biol 17:1367-1376, PDB 3NBZ/3NC0) as binding in
an EXTENDED, proline-containing, non-helical conformation -- as
'amphipathic_helix'. Root cause, confirmed on the project's own real crystal
ground-truth set (crystal_sanity_check.py, CRM1_NES_collated_report.docx, ): raw Chou-Fasman avg_propensity does NOT separate confirmed-helical
from confirmed-extended real NES motifs at the short (~8-12 residue) window
this module scores -- PKI-alpha (3NBY, confirmed alpha-helical) averages
~1.15 (barely 'moderate_helix_former'), and HIV-1 Rev (3NBZ/3NC0, confirmed
EXTENDED) also lands at ~1.05 ('moderate_helix_former') -- a ~0.1 gap that is
not a reliable structural signal. 'moderate' helicity/amphipathicity is
therefore no longer treated as sufficient evidence of a true amphipathic-helix
binding mode (see `_CONFORMATION_CONFIDENCE_CATEGORIES` below); only 'strong'
categories, or a match against `KNOWN_NES_CRYSTAL_REFERENCES` (real,
citation-backed structures), can assert 'amphipathic_helix' with confidence.
This is also the conservative direction given the actual cost of being wrong:
CRM1_NES_collated_report.docx Section 3 found that trusting an
alpha-helix-shaped starting pose (`idealized_helix` in md_refinement.py) for a
candidate that is actually non-helical doesn't just underscore it -- it
actively mislocates the peptide (8-22 Angstrom RMSD from the real crystal
pose) and can report a real binder as near-zero/non-binding. Among this
project's five independently-solved real crystal ground-truth structures,
three (HIV-1 Rev x2 crystal forms, Snurportin1 x2 crystal forms) are
non-helical/atypical and only one (PKI-alpha) is confirmed canonical helical
-- so defaulting an ambiguous case toward 'extended_or_hybrid' rather than
'amphipathic_helix' is the literature-supported default, not just a more
cautious guess.

LITERATURE REFERENCES (see KNOWN_NES_CRYSTAL_REFERENCES for the exact
sequences/PDB IDs used as ground truth):
  - Guttler T, Madl T, Neumann P, Deichsel D, Corsini L, Monecke T, Ficner R,
    Sattler M, Gorlich D (2010) NES consensus redefined by structures of
    PKI-type and Rev-type nuclear export signals bound to CRM1.
    Nat Struct Mol Biol 17(11):1367-1376. [PKI-alpha/3NBY: helical;
    HIV-1 Rev/3NBZ+3NC0: extended]
  - Dong X, Biswas A, Suel KE, Jackson LK, Martinez R, Gu H, Chook YM (2009)
    Structural basis for leucine-rich nuclear export signal recognition by
    CRM1. Nature 458(7242):1136-1141. [Snurportin1/3GJX+3GB8: atypical hybrid]
  - Fung HYJ, Fu SC, Chook YM (2017) Nuclear export receptor CRM1 recognizes
    diverse conformations in nuclear export signals. eLife 6:e23961.
    [5-6 distinct backbone conformations survey; Paxillin/5UWH, SMAD4/5UWU,
    X11L2/5UWS]
  - Fung HYJ, Fu SC, Brautigam CA, Chook YM (2015) Structural determinants of
    nuclear export signal orientation in binding to exportin CRM1. eLife
    4:e10034. [hRio2/5DHF, CPEB4/5DIF: reversed N/C-terminal binding
    polarity, a third distinct binding mode]
  - Kosugi S, Hasebe M, Tomita M, Yanagawa H (2008) Nuclear export signal
    consensus sequences defined using a localization-based yeast selection
    system. J Biol Chem. [Phi-anchor spacing register]
  - Chou PY, Fasman GD (1978) Prediction of the secondary structure of
    proteins from their amino acid sequence. Adv Enzymol Relat Areas Mol
    Biol 47:45-148. [helix propensity scale]

Typical execution time: <1 second per NES candidate
"""

import re
import numpy as np
from typing import Dict, List, Tuple

# Chou-Fasman alpha-helix propensity parameters
# Based on Chou, PY & Fasman, GD (1978) Adv. Enzymol. 47:45-148
# Note: this is a generic globular-protein helix scale, not NES-specific.
# It measures overall helicity, not hydrophobic register -- use
# analyze_hydrophobic_spacing() below for the register-aware signal.
HELIX_PROPENSITY = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06,
    'X': 1.00  # Unknown
}

# Hydrophobicity scale (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0
}

# Real, citation-backed conformation ground truth from this
# project's own crystal_sanity_check.py reference set (exact sequences
# extracted from the real PDB structures, see that file's CRYSTAL_STRUCTURES
# dict). Checked BEFORE the sequence heuristic below -- an exact or
# substring match here is a literature-documented fact, not a guess, and
# overrides the heuristic's conformation_class/conformation_confidence.
# Only structures whose backbone conformation is explicitly stated in the
# cited paper are included; 5UWH/5UWU/5UWS (Fung et al. 2017) are used
# elsewhere in this project for their Phi-register spacing class but are NOT
# included here because their backbone-conformation category specifically
# isn't confirmed in this project's own notes -- don't assert it.
KNOWN_NES_CRYSTAL_REFERENCES = {
    'LALKLAGLDI': {
        'conformation_class': 'amphipathic_helix',
        'pdb': '3NBY', 'protein': 'PKI-alpha',
        'citation': 'Guttler et al. 2010, Nat Struct Mol Biol 17:1367-1376',
        'note': 'Classic class-1 alpha-helical NES.',
    },
    'LPPLERLTL': {
        'conformation_class': 'extended_or_hybrid',
        'pdb': '3NBZ', 'protein': 'HIV-1 Rev',
        'citation': 'Guttler et al. 2010, Nat Struct Mol Biol 17:1367-1376',
        'note': 'Extended, proline-containing, non-helical binding mode.',
    },
    'LPPLERLTLS': {
        'conformation_class': 'extended_or_hybrid',
        'pdb': '3NC0', 'protein': 'HIV-1 Rev (replicate crystal form)',
        'citation': 'Guttler et al. 2010, Nat Struct Mol Biol 17:1367-1376',
        'note': 'Extended, proline-containing, non-helical binding mode.',
    },
    'SQALASSFSVS': {
        'conformation_class': 'extended_or_hybrid',
        'pdb': '3GJX', 'protein': 'Snurportin1',
        'citation': 'Dong et al. 2009, Nature 458:1136-1141',
        'note': 'Atypical, weakly-hydrophobic short-helix/beta-strand hybrid.',
    },
    'LASSFSVS': {
        'conformation_class': 'extended_or_hybrid',
        'pdb': '3GB8', 'protein': 'Snurportin1 (replicate crystal form)',
        'citation': 'Dong et al. 2009, Nature 458:1136-1141',
        'note': 'Atypical, weakly-hydrophobic short-helix/beta-strand hybrid.',
    },
    'TEFNQAL': {
        'conformation_class': 'reverse_direction',
        'pdb': '5DHF', 'protein': 'hRio2',
        'citation': 'Fung, Fu & Chook 2015, eLife 4:e10034',
        'note': 'Reversed N/C-terminal binding polarity -- a third, distinct '
                'binding mode this module does not otherwise model.',
    },
    'MHSLESSL': {
        'conformation_class': 'reverse_direction',
        'pdb': '5DIF', 'protein': 'CPEB4',
        'citation': 'Fung, Fu & Chook 2015, eLife 4:e10034',
        'note': 'Reversed N/C-terminal binding polarity -- a third, distinct '
                'binding mode this module does not otherwise model.',
    },
}


def _check_literature_reference(sequence_upper: str):
    """Exact match, or (for a candidate window that only partially overlaps
    the crystal-defined core) substring match in either direction, against
    KNOWN_NES_CRYSTAL_REFERENCES. Returns the matching entry dict or None."""
    if sequence_upper in KNOWN_NES_CRYSTAL_REFERENCES:
        return KNOWN_NES_CRYSTAL_REFERENCES[sequence_upper]
    for ref_seq, entry in KNOWN_NES_CRYSTAL_REFERENCES.items():
        if ref_seq in sequence_upper or sequence_upper in ref_seq:
            return entry
    return None


# Canonical NES hydrophobic (Phi) anchor residues
PHI_RESIDUES = set('LIVFM')

# The SAME Phi-anchor register pattern as md_refinement.py's
# PHI_ANCHOR_VOCAB/PHI_REGISTER_RE/PHI0_LOOKBACK (which cites Guttler et al.
# 2010 for the P0-P4 sub-pocket convention) and nes_ml_predictor_improved.py's
# PSSM_ANCHOR_RE -- kept as its own copy here (not imported) so this module
# stays free of md_refinement.py's heavy OpenMM/PDBFixer import chain, per
# this file's own "should stay independent" design note above. If this
# pattern is ever retuned, update all three call sites together.
PHI_ANCHOR_VOCAB = set('LIVFM')
PHI_REGISTER_RE = re.compile(r'([LIVFM]).{1,3}([LIVFM]).{1,3}([LIVFM]).{1,2}([LIVFM])')
PHI0_LOOKBACK = 3  # Guttler 2010: Phi0 sits a few residues N-terminal of Phi1

# Relaxed 3-of-4 register match, mirroring md_refinement.py's
# _PHI_RELAXED_PATTERNS (see that module for the full rationale -- in
# short, Snurportin1's real NES-like segment SQALASSFSVS has L/F/V at the
# right spacing but only 3 of 4 anchors, so the strict regex misses it
# entirely; a 3-of-4 match is treated as real, lower-confidence signal
# rather than "no register at all"). Duplicated, not imported, same reason
# as PHI_REGISTER_RE above.
_PHI_RELAXED_PATTERNS = [
    ('P1', re.compile(r'(.).{1,3}([LIVFM]).{1,3}([LIVFM]).{1,2}([LIVFM])')),
    ('P2', re.compile(r'([LIVFM]).{1,3}(.).{1,3}([LIVFM]).{1,2}([LIVFM])')),
    ('P3', re.compile(r'([LIVFM]).{1,3}([LIVFM]).{1,3}(.).{1,2}([LIVFM])')),
    ('P4', re.compile(r'([LIVFM]).{1,3}([LIVFM]).{1,3}([LIVFM]).{1,2}(.)')),
]


def calculate_helix_propensity(sequence: str) -> Dict:
    """
    Calculate α-helix propensity using Chou-Fasman parameters

    Args:
        sequence: Amino acid sequence

    Returns:
        Dictionary with helix propensity metrics
    """
    sequence_upper = sequence.upper()

    # Calculate residue-wise helix propensity
    propensities = [HELIX_PROPENSITY.get(aa, 1.0) for aa in sequence_upper]

    # Calculate overall metrics
    avg_propensity = np.mean(propensities)
    max_propensity = np.max(propensities)
    min_propensity = np.min(propensities)

    # Identify helix-favorable regions (windows of 4+ residues with avg > 1.0)
    helix_regions = []
    window_size = 4

    for i in range(len(propensities) - window_size + 1):
        window = propensities[i:i + window_size]
        window_avg = np.mean(window)

        if window_avg > 1.03:  # Threshold from Chou-Fasman
            helix_regions.append({
                'start': i,
                'end': i + window_size,
                'propensity': window_avg,
                'sequence': sequence_upper[i:i + window_size]
            })

    # Merge overlapping regions
    merged_regions = []
    if helix_regions:
        current_region = helix_regions[0].copy()

        for region in helix_regions[1:]:
            if region['start'] <= current_region['end']:
                # Overlapping - extend current region
                current_region['end'] = max(current_region['end'], region['end'])
                current_region['sequence'] = sequence_upper[current_region['start']:current_region['end']]
                current_region['propensity'] = max(current_region['propensity'], region['propensity'])
            else:
                # Non-overlapping - save current and start new
                merged_regions.append(current_region)
                current_region = region.copy()

        merged_regions.append(current_region)

    # Calculate helix content (fraction of sequence in helix regions)
    helix_residues = sum(r['end'] - r['start'] for r in merged_regions)
    helix_content = helix_residues / len(sequence) if sequence else 0

    # Classify helix propensity
    if avg_propensity > 1.15:
        category = 'strong_helix_former'
    elif avg_propensity > 1.05:
        category = 'moderate_helix_former'
    elif avg_propensity > 0.95:
        category = 'neutral'
    else:
        category = 'helix_breaker'

    return {
        'avg_propensity': avg_propensity,
        'max_propensity': max_propensity,
        'min_propensity': min_propensity,
        'helix_content': helix_content,
        'helix_regions': merged_regions,
        'category': category,
        'per_residue_propensity': propensities
    }


def calculate_amphipathic_moment(sequence: str, angle: float = 100.0) -> Dict:
    """
    Calculate amphipathic moment and identify hydrophobic face

    The amphipathic moment quantifies the degree to which hydrophobic and
    hydrophilic residues are segregated on opposite faces of an α-helix.

    Args:
        sequence: Amino acid sequence
        angle: Angular rotation per residue (100° for α-helix, 180° for β-sheet)

    Returns:
        Dictionary with amphipathic moment and hydrophobic face residues
    """
    sequence_upper = sequence.upper()

    # Calculate hydrophobic moment vector
    # Each residue contributes a vector based on its position on the helix
    moment_x = 0.0
    moment_y = 0.0

    for i, aa in enumerate(sequence_upper):
        hydro = HYDROPHOBICITY.get(aa, 0.0)
        theta = np.radians(i * angle)

        moment_x += hydro * np.cos(theta)
        moment_y += hydro * np.sin(theta)

    # Calculate magnitude of amphipathic moment
    moment_magnitude = np.sqrt(moment_x**2 + moment_y**2)

    # Normalize by sequence length
    normalized_moment = moment_magnitude / len(sequence) if sequence else 0

    # Calculate angle of hydrophobic face
    hydrophobic_angle = np.degrees(np.arctan2(moment_y, moment_x))

    # Identify residues on hydrophobic face (±90° from peak hydrophobic angle)
    hydrophobic_face_residues = []
    hydrophilic_face_residues = []

    for i, aa in enumerate(sequence_upper):
        residue_angle = (i * angle) % 360

        # Calculate angular distance from hydrophobic peak
        angle_diff = abs(residue_angle - hydrophobic_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff <= 90:
            # On hydrophobic face
            hydrophobic_face_residues.append({
                'position': i,
                'residue': aa,
                'hydrophobicity': HYDROPHOBICITY.get(aa, 0.0)
            })
        else:
            # On hydrophilic face
            hydrophilic_face_residues.append({
                'position': i,
                'residue': aa,
                'hydrophobicity': HYDROPHOBICITY.get(aa, 0.0)
            })

    # Calculate average hydrophobicity of each face
    hydrophobic_face_avg = np.mean([r['hydrophobicity'] for r in hydrophobic_face_residues]) if hydrophobic_face_residues else 0
    hydrophilic_face_avg = np.mean([r['hydrophobicity'] for r in hydrophilic_face_residues]) if hydrophilic_face_residues else 0

    # Calculate face contrast (difference between faces)
    face_contrast = hydrophobic_face_avg - hydrophilic_face_avg

    # Classify amphipathicity
    if normalized_moment > 0.5 and face_contrast > 3.0:
        category = 'strongly_amphipathic'
    elif normalized_moment > 0.3 and face_contrast > 2.0:
        category = 'moderately_amphipathic'
    elif normalized_moment > 0.15:
        category = 'weakly_amphipathic'
    else:
        category = 'non_amphipathic'

    # Get hydrophobic face sequence
    hydrophobic_face_seq = ''.join([r['residue'] for r in sorted(hydrophobic_face_residues, key=lambda x: x['position'])])

    return {
        'moment_magnitude': moment_magnitude,
        'normalized_moment': normalized_moment,
        'hydrophobic_angle': hydrophobic_angle,
        'face_contrast': face_contrast,
        'category': category,
        'hydrophobic_face_residues': hydrophobic_face_residues,
        'hydrophilic_face_residues': hydrophilic_face_residues,
        'hydrophobic_face_sequence': hydrophobic_face_seq,
        'hydrophobic_face_avg': hydrophobic_face_avg,
        'hydrophilic_face_avg': hydrophilic_face_avg
    }


def find_phi_register(sequence: str) -> Dict[str, int]:
    """
    Locate the Phi-anchor register within `sequence` using the SAME pattern/
    convention as md_refinement.NESMDRefiner._find_phi_register -- see that
    method's docstring for the full rationale (rightmost-match-wins for
    P1-P4, P0 found independently by lookback). Duplicated here rather than
    imported so this module doesn't pull in md_refinement.py's OpenMM/
    PDBFixer dependency chain.

    Returns {'P0': idx or None, 'P1': idx, 'P2': idx, 'P3': idx, 'P4': idx,
    'register_match_type': 'full'|'partial'|'none'} (0-indexed positions
    within `sequence`). A strict all-4 match ('full') is preferred; if none
    exists, falls back to the best relaxed 3-of-4 match ('partial' -- one
    of P1-P4 will be None). 'none' means neither exists (all of P0-P4 are
    None). See md_refinement.py's _find_phi_register for the full
    rationale (same logic, duplicated per this module's design note).
    """
    seq = sequence.upper()
    last_match = None
    for m in PHI_REGISTER_RE.finditer(seq):
        last_match = m

    if last_match is not None:
        register = {
            'P1': last_match.start(1),
            'P2': last_match.start(2),
            'P3': last_match.start(3),
            'P4': last_match.start(4),
        }
        register['register_match_type'] = 'full'
    else:
        best = None  # (match_start, missing_label, match_obj)
        for missing_label, pattern in _PHI_RELAXED_PATTERNS:
            for m in pattern.finditer(seq):
                if best is None or m.start() >= best[0]:
                    best = (m.start(), missing_label, m)
        if best is None:
            return {'P0': None, 'P1': None, 'P2': None, 'P3': None, 'P4': None,
                     'register_match_type': 'none'}
        _, missing_label, m = best
        register = {}
        for i, label in enumerate(('P1', 'P2', 'P3', 'P4'), start=1):
            register[label] = None if label == missing_label else m.start(i)
        register['register_match_type'] = 'partial'

    phi1_pos = register.get('P1')
    phi0_pos = None
    if phi1_pos is not None:
        for i in range(max(0, phi1_pos - PHI0_LOOKBACK), phi1_pos):
            if seq[i] in PHI_ANCHOR_VOCAB:
                phi0_pos = i
                break
    register['P0'] = phi0_pos
    return register


def analyze_hydrophobic_spacing(sequence: str) -> Dict:
    """
    Check whether the candidate's Phi (L/I/V/F/M) residues fall on the
    Phi-anchor spacing register CRM1's hydrophobic groove reads out, using
    the SAME PHI_REGISTER_RE pattern as md_refinement.py's MD anchor-
    registration and nes_ml_predictor_improved.py's PSSM anchoring (see
    module docstring for the empirical validation: 87% real-
    positive vs. 42% real-negative match rate, n=34).

    This is the closest proxy this fast module has to "does this actually
    fit CRM1's groove," since the groove reads out discrete, regularly
    spaced Phi side chains rather than a smooth average hydrophobicity.

    Args:
        sequence: Amino acid sequence

    Returns:
        Dictionary with spacing score, the full P0-P4 register (phi_register),
        matched anchor positions, and category.
    """
    sequence_upper = sequence.upper()
    phi_positions_overall = [i for i, aa in enumerate(sequence_upper) if aa in PHI_RESIDUES]

    register = find_phi_register(sequence_upper)
    match_type = register.get('register_match_type', 'none')

    if match_type == 'full':
        best_score = 1.0
        anchor_positions = [register[p] for p in ('P0', 'P1', 'P2', 'P3', 'P4') if register[p] is not None]
        matched_class = 'phi_register_matched'
        matched_classes = ['phi_register_matched']
    elif match_type == 'partial':
        # 3-of-4 anchors hydrophobic at the right spacing, one
        # slot isn't -- the Snurportin1 case (SQALASSFSVS: L/F/V correctly
        # spaced, one slot filled by a non-Phi residue). Real signal, but
        # deliberately scored below a full match: 0.65 clears the
        # 'plausible_spacing' category (>=0.5) and, combined with any
        # supporting helix/amphipathic character, still clears
        # is_favorable_for_nes -- but it will never alone reach the 0.85
        # 'canonical_spacing' bar that a genuine 4-of-4 match gets.
        best_score = 0.65
        anchor_positions = [register[p] for p in ('P0', 'P1', 'P2', 'P3', 'P4') if register[p] is not None]
        matched_class = 'phi_register_partial_match'
        matched_classes = ['phi_register_partial_match']
    else:
        anchor_positions = []
        matched_class = None
        matched_classes = []
        best_score = 0.0
        # Partial credit: sequence has enough Phi residues to plausibly form a
        # register even if it doesn't match the validated pattern exactly
        # (e.g. it sits in a slightly longer/shorter candidate window than
        # the motif). Unchanged from the pre- behavior.
        if len(phi_positions_overall) >= 3:
            best_score = 0.25

    if best_score >= 0.85:
        category = 'canonical_spacing'
    elif best_score >= 0.5:
        category = 'plausible_spacing'
    elif best_score > 0.0:
        category = 'weak_spacing'
    else:
        category = 'no_spacing_match'

    return {
        'spacing_score': best_score,
        'matched_class': matched_class,
        'matched_classes': matched_classes,
        'anchor_positions': anchor_positions,
        'phi_positions': phi_positions_overall,
        'phi_register': register,
        'category': category
    }


def quick_structural_analysis(sequence: str) -> Dict:
    """
    Perform quick structural analysis combining helix propensity,
    amphipathicity, and Phi-spacing register.

    Args:
        sequence: Amino acid sequence

    Returns:
        Combined analysis with helix, amphipathic, and spacing metrics
    """
    sequence_upper = sequence.upper()
    helix_analysis = calculate_helix_propensity(sequence)
    amphipathic_analysis = calculate_amphipathic_moment(sequence)
    spacing_analysis = analyze_hydrophobic_spacing(sequence)

    # Calculate combined score
    # Spacing register is weighted highest since it's the closest proxy to
    # the actual CRM1-groove determinant; helix/amphipathic character are
    # supporting evidence for the (most common, but not universal) canonical
    # amphipathic-helix binding mode.
    helix_score = min(1.0, helix_analysis['avg_propensity'] / 1.2)  # Normalize to 0-1
    amphipathic_score = min(1.0, amphipathic_analysis['normalized_moment'] / 0.6)  # Normalize to 0-1
    spacing_score = spacing_analysis['spacing_score']  # Already 0-1

    combined_score = (0.45 * spacing_score) + (0.30 * helix_score) + (0.25 * amphipathic_score)

    # Determine if structure is favorable for NES.
    # Canonical case: correct-ish spacing plus at least some helical or
    # amphipathic character (covers the majority amphipathic-helix mode,
    # e.g. PKI/Snurportin-1-like NESs).
    # Non-canonical case: very strong spacing match on its own is still
    # accepted, since CRM1 is documented to bind extended/proline-rich,
    # non-helical NESs (e.g. HIV-1 Rev) that would otherwise be rejected
    # by a helix+amphipathic-only filter.
    # NOTE: this favorability gate deliberately still uses the permissive
    # 'moderate' bar -- eval_quick_helix_positives_only.py, real
    # 290-motif labeled set) confirmed this recall-oriented gate already
    # rescues 92% of non-helix-forming real NES motifs to favorable. The bug
    # fixed below is specifically in conformation_class (see
    # module docstring update), not in this favorability decision.
    has_supporting_secondary_structure = (
        helix_analysis['category'] in ['strong_helix_former', 'moderate_helix_former'] or
        amphipathic_analysis['category'] in ['strongly_amphipathic', 'moderately_amphipathic']
    )
    is_favorable = (
        spacing_analysis['spacing_score'] >= 0.85
        or (spacing_analysis['spacing_score'] >= 0.5 and has_supporting_secondary_structure)
    )

    # Conformation_class now checks real literature ground truth
    # FIRST (KNOWN_NES_CRYSTAL_REFERENCES), since this is a documented fact,
    # not an inference. Only when there's no match does it fall back to the
    # sequence heuristic -- which now requires STRONG (not merely moderate)
    # helical/amphipathic evidence before asserting 'amphipathic_helix', per
    # the module docstring's update: on this project's own real
    # crystal ground truth, confirmed-helical PKI-alpha (~1.15 avg
    # propensity) and confirmed-EXTENDED HIV-1 Rev (~1.05 avg propensity)
    # both land in the 'moderate_helix_former' bucket, so 'moderate' cannot
    # be trusted to mean "this candidate is actually helical" -- it means
    # "inconclusive by this metric." An inconclusive case now defaults to
    # 'extended_or_hybrid' rather than 'amphipathic_helix', matching this
    # project's own real crystal sample (3 of 5 independently-solved
    # ground-truth structures were non-helical/atypical) and the asymmetric
    # cost of being wrong (CRM1_NES_collated_report.docx Section 3: treating
    # a non-helical candidate as helical actively mislocates it in MD, 8-22A
    # RMSD from truth; the reverse mistake just means trying both starting
    # conformations, which the pipeline already does).
    lit_ref = _check_literature_reference(sequence_upper)
    if lit_ref is not None:
        conformation_class = lit_ref['conformation_class']
        conformation_source = 'literature_confirmed'
        conformation_reference = f"{lit_ref['protein']} / PDB {lit_ref['pdb']} ({lit_ref['citation']}): {lit_ref['note']}"
    else:
        conformation_source = 'sequence_heuristic'
        conformation_reference = None
        has_strong_secondary_structure = (
            helix_analysis['category'] == 'strong_helix_former'
            or amphipathic_analysis['category'] == 'strongly_amphipathic'
        )
        if spacing_analysis['spacing_score'] >= 0.85 and has_strong_secondary_structure:
            conformation_class = 'amphipathic_helix'  # canonical, most common mode
        elif spacing_analysis['spacing_score'] >= 0.5:
            # Includes the old 0.5-0.85-spacing-with-only-moderate-support
            # case, which used to be mislabeled 'amphipathic_helix' -- see
            # docstring update (this is the HIV-1 Rev-shaped bug).
            conformation_class = 'extended_or_hybrid'
        else:
            conformation_class = 'weak_candidate'

    return {
        'helix_analysis': helix_analysis,
        'amphipathic_analysis': amphipathic_analysis,
        'spacing_analysis': spacing_analysis,
        'combined_score': combined_score,
        'helix_score': helix_score,
        'amphipathic_score': amphipathic_score,
        'spacing_score': spacing_score,
        'conformation_class': conformation_class,
        'conformation_source': conformation_source,
        'conformation_reference': conformation_reference,
        'is_favorable_for_nes': is_favorable,
        'summary': {
            'helix_propensity': helix_analysis['avg_propensity'],
            'helix_content': helix_analysis['helix_content'],
            'amphipathic_moment': amphipathic_analysis['normalized_moment'],
            'face_contrast': amphipathic_analysis['face_contrast'],
            'hydrophobic_face': amphipathic_analysis['hydrophobic_face_sequence'],
            'spacing_class': spacing_analysis['matched_class'],
            'conformation_class': conformation_class,
            'conformation_source': conformation_source,
            'combined_score': combined_score
        }
    }


def batch_quick_analysis(candidates: List[Dict]) -> List[Dict]:
    """
    Perform quick structural analysis on multiple NES candidates

    Args:
        candidates: List of NES candidate dictionaries with 'sequence' key

    Returns:
        List of candidates with added structural analysis
    """
    enhanced_candidates = []

    for candidate in candidates:
        sequence = candidate.get('sequence', '')

        # Perform quick analysis
        analysis = quick_structural_analysis(sequence)

        # Add to candidate
        enhanced_candidate = candidate.copy()
        enhanced_candidate['quick_structural_analysis'] = analysis
        enhanced_candidate['quick_combined_score'] = analysis['combined_score']

        enhanced_candidates.append(enhanced_candidate)

    # Sort by combined score
    enhanced_candidates.sort(key=lambda x: x['quick_combined_score'], reverse=True)

    return enhanced_candidates


if __name__ == '__main__':
    # Test the quick analysis
    print("Quick Structural Analysis Test")
    print("=" * 70)

    test_sequences = {
        'PKI-alpha (3NBY, Guttler 2010 -- confirmed alpha-helical)': 'LALKLAGLDI',
        'HIV-1 Rev (3NBZ core, Guttler 2010 -- confirmed EXTENDED)': 'LPPLERLTL',
        'HIV-1 Rev (NESbase window containing the 3NBZ core)': 'LQLPPLERLTLD',
        'Snurportin1 (3GJX, Dong 2009 -- confirmed atypical hybrid)': 'SQALASSFSVS',
    }

    for label, test_seq in test_sequences.items():
        analysis = quick_structural_analysis(test_seq)

        print(f"\n--- {label} ---")
        print(f"Sequence: {test_seq}")
        print(f"Helix Propensity: {analysis['helix_analysis']['avg_propensity']:.3f} ({analysis['helix_analysis']['category']})")
        print(f"Amphipathic Moment: {analysis['amphipathic_analysis']['normalized_moment']:.3f} ({analysis['amphipathic_analysis']['category']})")
        print(f"Spacing: {analysis['spacing_analysis']['spacing_score']:.2f} (class {analysis['spacing_analysis']['matched_class']}, {analysis['spacing_analysis']['category']})")
        print(f"Conformation Class: {analysis['conformation_class']}  (source: {analysis['conformation_source']})")
        if analysis['conformation_reference']:
            print(f"  -> {analysis['conformation_reference']}")
        print(f"Combined Score: {analysis['combined_score']:.3f}")
        print(f"Favorable for NES: {analysis['is_favorable_for_nes']}")
