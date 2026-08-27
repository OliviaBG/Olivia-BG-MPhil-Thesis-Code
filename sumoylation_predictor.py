"""
SUMOylation Site Predictor for NES Analysis
Based on consensus motif detection and spatial relationship to NES sequences

Key Features:
1. Detects canonical ψKxE/D motifs (where ψ = I,V,L,M,F)
2. Detects extended NDSM (with downstream acidic clusters)
3. Detects inverted motifs (E/D)xKψ
4. Assesses spatial relationship to NES sequences (ON, ADJACENT, NEAR)
5. Provides context-aware interpretation for NES scoring

References:
- SUMOgo: Chang et al. 2018 (Sci Rep)
- NDSM motif: Yang et al. 2006 (EMBO J)
- PSSM-Sumo: Khan et al. 2024 (BMC Bioinformatics)

Context-dependent effects of SUMOylation on nuclear export:
- Generally enriches proteins in the nucleus
- Stabilizes interactions with nuclear scaffolds
- Often masks NESs
- Effect is protein- and context-dependent
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np

# SUMOylation consensus motif amino acids
HYDROPHOBIC_SUMO = set('IVLMFP')  # ψ in consensus motif ψKxE/D
ACIDIC = set('DE')  # Acidic residues
BASIC = set('RK')  # Basic residues

# Enhanced hydrophobicity scale (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0
}


class SUMOylationPredictor:
    """
    Predicts SUMOylation sites using consensus motif detection
    Focuses on sites relevant to nuclear export signals
    """

    def __init__(self):
        # Consensus: ψKxE/D where ψ is hydrophobic (I,V,L,M,F,P)
        self.consensus_pattern = re.compile(r'[IVLMFP]K.{1}[DE]')

        # Inverted consensus: (E/D)xKψ
        self.inverted_pattern = re.compile(r'[DE].{1}K[IVLMFP]')

        # Weak consensus (less hydrophobic ψ): ψKxE/D where ψ = A,T,S
        self.weak_consensus_pattern = re.compile(r'[ATS]K.{1}[DE]')

    def predict_sumo_sites(self, sequence: str, min_score: float = 0.4) -> List[Dict]:
        """
        Predict SUMOylation sites in a protein sequence

        Args:
            sequence: Protein sequence
            min_score: Minimum score threshold (0-1)

        Returns:
            List of predicted SUMOylation sites with scores and details
        """
        sites = []
        sequence_upper = sequence.upper()

        # 1. Find strong consensus motifs (ψKxE/D)
        for match in self.consensus_pattern.finditer(sequence_upper):
            k_pos = match.start() + 1  # K position (0-indexed)
            motif = match.group()

            score, details = self._score_sumo_site(
                sequence_upper, k_pos, motif, motif_type='consensus'
            )

            if score >= min_score:
                sites.append({
                    'lysine_position': k_pos,
                    'motif': motif,
                    'type': 'consensus',
                    'score': score,
                    'confidence': self._get_confidence_level(score),
                    'details': details
                })

        # 2. Find inverted motifs (E/D)xKψ
        for match in self.inverted_pattern.finditer(sequence_upper):
            k_pos = match.start() + 2  # K position in inverted motif
            motif = match.group()

            score, details = self._score_sumo_site(
                sequence_upper, k_pos, motif, motif_type='inverted'
            )

            if score >= min_score:
                sites.append({
                    'lysine_position': k_pos,
                    'motif': motif,
                    'type': 'inverted',
                    'score': score,
                    'confidence': self._get_confidence_level(score),
                    'details': details
                })

        # 3. Find weak consensus motifs
        for match in self.weak_consensus_pattern.finditer(sequence_upper):
            k_pos = match.start() + 1
            motif = match.group()

            # Skip if already found as strong consensus
            if any(site['lysine_position'] == k_pos for site in sites):
                continue

            score, details = self._score_sumo_site(
                sequence_upper, k_pos, motif, motif_type='weak_consensus'
            )

            if score >= min_score:
                sites.append({
                    'lysine_position': k_pos,
                    'motif': motif,
                    'type': 'weak_consensus',
                    'score': score,
                    'confidence': self._get_confidence_level(score),
                    'details': details
                })

        # 4. Find non-consensus lysines with favorable context
        all_k_positions = [i for i, aa in enumerate(sequence_upper) if aa == 'K']
        consensus_k = {site['lysine_position'] for site in sites}

        for k_pos in all_k_positions:
            if k_pos not in consensus_k:
                score, details = self._score_non_consensus_lysine(sequence_upper, k_pos)

                if score >= min_score:
                    # Extract local context
                    start = max(0, k_pos - 2)
                    end = min(len(sequence_upper), k_pos + 3)
                    motif = sequence_upper[start:end]

                    sites.append({
                        'lysine_position': k_pos,
                        'motif': motif,
                        'type': 'non_consensus',
                        'score': score,
                        'confidence': self._get_confidence_level(score),
                        'details': details
                    })

        # Sort by score
        sites.sort(key=lambda x: x['score'], reverse=True)

        return sites

    def _score_sumo_site(self, sequence: str, k_pos: int, motif: str,
                        motif_type: str) -> Tuple[float, Dict]:
        """
        Score a SUMOylation site based on multiple criteria

        Returns:
            score: Confidence score (0-1)
            details: Dictionary with scoring breakdown
        """
        score_components = {}

        # 1. Base score for motif type
        base_scores = {
            'consensus': 0.75,       # Strong consensus ψKxE/D
            'inverted': 0.55,        # Inverted (E/D)xKψ
            'weak_consensus': 0.50   # Weak consensus with A/T/S
        }
        base_score = base_scores.get(motif_type, 0.3)
        score_components['base_score'] = base_score

        # 2. NDSM: Check for acidic cluster downstream (Yang et al. 2006)
        downstream_start = k_pos + 4
        downstream_end = min(len(sequence), k_pos + 14)
        downstream_seq = sequence[downstream_start:downstream_end]

        acidic_count = sum(1 for aa in downstream_seq if aa in ACIDIC)
        acidic_ratio = acidic_count / len(downstream_seq) if downstream_seq else 0

        # NDSM bonus
        ndsm_bonus = 0.0
        if acidic_ratio > 0.4:  # >40% acidic = strong NDSM
            ndsm_bonus = 0.20
        elif acidic_ratio > 0.3:  # >30% acidic = moderate NDSM
            ndsm_bonus = 0.15
        elif acidic_ratio > 0.2:  # >20% acidic = weak NDSM
            ndsm_bonus = 0.10

        score_components['ndsm_bonus'] = ndsm_bonus
        score_components['downstream_acidic_ratio'] = acidic_ratio

        # 3. Check upstream hydrophobicity (ψ quality)
        if k_pos > 0:
            upstream_aa = sequence[k_pos - 1]
            upstream_hydro = HYDROPHOBICITY.get(upstream_aa, 0)

            # Bonus for strong hydrophobics
            hydro_bonus = 0.0
            if upstream_hydro > 3.0:  # I, L, V, F
                hydro_bonus = 0.10
            elif upstream_hydro > 1.0:  # M, P, C, A
                hydro_bonus = 0.05

            score_components['upstream_hydrophobicity'] = upstream_hydro
            score_components['hydro_bonus'] = hydro_bonus
        else:
            hydro_bonus = 0.0
            score_components['upstream_hydrophobicity'] = 0.0
            score_components['hydro_bonus'] = 0.0

        # 4. Check for proximal basic residues (inhibitory)
        # SUMOylation is inhibited by nearby positive charges
        upstream_basic = 0
        if k_pos >= 5:
            upstream_region = sequence[k_pos-5:k_pos]
            upstream_basic = sum(1 for aa in upstream_region if aa in BASIC)

        basic_penalty = -0.05 * upstream_basic  # -0.05 per basic residue
        score_components['basic_penalty'] = basic_penalty

        # 5. Check surface accessibility context (disordered regions favor SUMOylation)
        # We approximate this by looking for disorder-promoting amino acids
        disorder_promoting = set('PQSAGDEKNR')

        context_start = max(0, k_pos - 5)
        context_end = min(len(sequence), k_pos + 6)
        context_seq = sequence[context_start:context_end]

        disorder_ratio = sum(1 for aa in context_seq if aa in disorder_promoting) / len(context_seq)

        disorder_bonus = 0.0
        if disorder_ratio > 0.7:
            disorder_bonus = 0.10
        elif disorder_ratio > 0.5:
            disorder_bonus = 0.05

        score_components['disorder_bonus'] = disorder_bonus
        score_components['disorder_ratio'] = disorder_ratio

        # Calculate final score (capped at 1.0)
        final_score = min(1.0, base_score + ndsm_bonus + hydro_bonus + basic_penalty + disorder_bonus)
        score_components['final_score'] = final_score

        return final_score, score_components

    def _score_non_consensus_lysine(self, sequence: str, k_pos: int) -> Tuple[float, Dict]:
        """
        Score a lysine that doesn't match consensus motif
        Based on favorable surrounding context

        Returns:
            score: Confidence score (0-1)
            details: Dictionary with scoring breakdown
        """
        score_components = {}
        base_score = 0.3  # Lower base for non-consensus

        # Check for nearby acidic residues (within ±3)
        context_start = max(0, k_pos - 3)
        context_end = min(len(sequence), k_pos + 4)
        context = sequence[context_start:context_end]

        acidic_nearby = sum(1 for aa in context if aa in ACIDIC)
        hydrophobic_nearby = sum(1 for aa in context if aa in HYDROPHOBIC_SUMO)

        # Need at least one acidic and one hydrophobic
        if acidic_nearby == 0 or hydrophobic_nearby == 0:
            return 0.0, {'reason': 'insufficient_context'}

        # Bonus for multiple acidic residues
        acidic_bonus = min(0.15, acidic_nearby * 0.05)

        # Bonus for hydrophobic residues
        hydro_bonus = min(0.10, hydrophobic_nearby * 0.03)

        # Check downstream acidic cluster (NDSM-like)
        downstream_start = k_pos + 1
        downstream_end = min(len(sequence), k_pos + 10)
        downstream = sequence[downstream_start:downstream_end]

        downstream_acidic = sum(1 for aa in downstream if aa in ACIDIC)
        downstream_bonus = min(0.15, downstream_acidic * 0.03)

        final_score = min(0.85, base_score + acidic_bonus + hydro_bonus + downstream_bonus)

        score_components.update({
            'base_score': base_score,
            'acidic_nearby': acidic_nearby,
            'hydrophobic_nearby': hydrophobic_nearby,
            'acidic_bonus': acidic_bonus,
            'hydro_bonus': hydro_bonus,
            'downstream_acidic': downstream_acidic,
            'downstream_bonus': downstream_bonus,
            'final_score': final_score
        })

        return final_score, score_components

    def _get_confidence_level(self, score: float) -> str:
        """Convert score to confidence level"""
        if score >= 0.75:
            return 'high'
        elif score >= 0.55:
            return 'medium'
        elif score >= 0.40:
            return 'low'
        else:
            return 'very_low'

    def analyze_sumo_nes_relationship(self, sequence: str, nes_start: int, nes_end: int,
                                     sumo_sites: List[Dict] = None) -> Dict:
        """
        Analyze spatial relationship between SUMOylation sites and a specific NES

        Args:
            sequence: Full protein sequence
            nes_start: Start position of NES (0-indexed)
            nes_end: End position of NES (0-indexed)
            sumo_sites: Pre-computed SUMOylation sites (or will compute)

        Returns:
            Dictionary with:
            - sites_on_nes: SUMOylation sites within the NES
            - sites_adjacent: Sites within 5 residues of NES
            - sites_near: Sites within 10 residues of NES
            - interpretation: Context-dependent interpretation for scoring
        """
        if sumo_sites is None:
            sumo_sites = self.predict_sumo_sites(sequence)

        # Categorize sites by proximity to NES
        sites_on_nes = []
        sites_adjacent = []  # Within 5 residues
        sites_near = []      # Within 6-10 residues

        for site in sumo_sites:
            k_pos = site['lysine_position']

            # Check if lysine is within NES
            if nes_start <= k_pos < nes_end:
                sites_on_nes.append(site)
            # Check if adjacent (within 5 residues)
            elif (nes_start - 5 <= k_pos < nes_start) or (nes_end <= k_pos < nes_end + 5):
                sites_adjacent.append(site)
            # Check if near (within 10 residues)
            elif (nes_start - 10 <= k_pos < nes_start) or (nes_end <= k_pos < nes_end + 10):
                sites_near.append(site)

        # Generate interpretation
        interpretation = self._generate_nes_interpretation(
            sites_on_nes, sites_adjacent, sites_near, nes_start, nes_end
        )

        return {
            'sites_on_nes': sites_on_nes,
            'sites_adjacent': sites_adjacent,
            'sites_near': sites_near,
            'has_relevant_sumo': len(sites_on_nes) + len(sites_adjacent) > 0,
            'interpretation': interpretation
        }

    def _generate_nes_interpretation(self, sites_on, sites_adjacent, sites_near,
                                    nes_start, nes_end) -> Dict:
        """
        Generate context-dependent interpretation for NES scoring

        Returns:
            Dictionary with warning messages and scoring implications
        """
        warnings = []
        scoring_notes = []
        sumo_impact = 'none'

        # Check for SUMOylation ON the NES
        if sites_on:
            high_conf_on = [s for s in sites_on if s['confidence'] in ['high', 'medium']]

            if high_conf_on:
                sumo_impact = 'direct_masking'
                warnings.append(
                    f"Warning: SUMOylation site detected within NES (K{high_conf_on[0]['lysine_position'] + 1}). "
                    "This may directly block CRM1 binding and inhibit nuclear export."
                )
                scoring_notes.append(
                    "If this NES scores low due to poor SASA or structural issues, SUMOylation "
                    "may be the cause. The sequence could still be a functional NES when un-SUMOylated."
                )

        # Check for ADJACENT SUMOylation
        if sites_adjacent:
            high_conf_adj = [s for s in sites_adjacent if s['confidence'] in ['high', 'medium']]

            if high_conf_adj:
                if sumo_impact == 'none':
                    sumo_impact = 'local_perturbation'

                k_positions = [s['lysine_position'] + 1 for s in high_conf_adj]  # 1-indexed
                warnings.append(
                    f"Warning: SUMOylation site(s) adjacent to NES at K{', K'.join(map(str, k_positions))}. "
                    "These may alter local structure or electrostatics affecting NES accessibility."
                )
                scoring_notes.append(
                    "Adjacent SUMOylation can affect NES function through: "
                    "(1) Local conformational changes, "
                    "(2) Altered electrostatics, or "
                    "(3) Steric hindrance of CRM1 binding."
                )

        # Check for NEAR SUMOylation
        if sites_near and sumo_impact == 'none':
            high_conf_near = [s for s in sites_near if s['confidence'] == 'high']

            if high_conf_near:
                sumo_impact = 'possible_indirect'
                k_positions = [s['lysine_position'] + 1 for s in high_conf_near]
                warnings.append(
                    f"SUMOylation site(s) near NES at K{', K'.join(map(str, k_positions))}. "
                    "May indirectly affect NES through conformational changes."
                )

        # Generate summary message
        if sumo_impact != 'none':
            summary = (
                "**SUMOylation Context:** SUMOylation generally promotes nuclear retention by "
                "masking NESs or altering protein interactions. If this NES scores lower than expected, "
                "SUMOylation may be reducing its accessibility or function. Consider that this sequence "
                "may be a functional NES in the un-SUMOylated state."
            )
        else:
            summary = "No significant SUMOylation sites detected near this NES."

        return {
            'warnings': warnings,
            'scoring_notes': scoring_notes,
            'sumo_impact': sumo_impact,
            'summary': summary,
            'context_note': (
                "Note: The effect of SUMOylation on nuclear export is protein- and context-dependent. "
                "These predictions should be validated experimentally."
            )
        }


# Quick test function
if __name__ == '__main__':
    predictor = SUMOylationPredictor()

    # Test sequence with known SUMOylation motif
    test_seq = "MLLPIKDELVKKMAGENLVEFLLQKKIK"

    sites = predictor.predict_sumo_sites(test_seq)

    print("SUMOylation Site Predictions:")
    print("=" * 70)
    for site in sites:
        print(f"\nPosition K{site['lysine_position'] + 1}: {site['motif']}")
        print(f"  Type: {site['type']}")
        print(f"  Score: {site['score']:.3f} ({site['confidence']} confidence)")
        print(f"  Details: {site['details']}")

    # Test NES relationship
    print("\n\nNES-SUMO Relationship Analysis:")
    print("=" * 70)
    nes_start = 10
    nes_end = 20
    analysis = predictor.analyze_sumo_nes_relationship(test_seq, nes_start, nes_end, sites)

    print(f"\nNES region: {test_seq[nes_start:nes_end]} (pos {nes_start}-{nes_end})")
    print(f"Sites ON NES: {len(analysis['sites_on_nes'])}")
    print(f"Sites ADJACENT: {len(analysis['sites_adjacent'])}")
    print(f"Sites NEAR: {len(analysis['sites_near'])}")
    print(f"\nInterpretation:")
    print(f"  Impact: {analysis['interpretation']['sumo_impact']}")
    print(f"  Summary: {analysis['interpretation']['summary']}")
    for warning in analysis['interpretation']['warnings']:
        print(f"  {warning}")
