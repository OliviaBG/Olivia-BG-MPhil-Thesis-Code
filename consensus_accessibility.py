#!/usr/bin/env python3
"""
consensus_accessibility.py
============================================================
Standalone consensus solvent-accessibility scorer.

Why this exists
----------------
app.py / nes_ml_predictor_improved.py / nls_ml_predictor.py all compute a
single-algorithm SASA value (Biopython's ShrakeRupley, sometimes FreeSASA
Shrake-Rupley in the training pipelines) and then normalize it with a FLAT
constant that ignores residue identity:

    sasa_score = min(1.0, avg_sasa / 100)     # app.py line ~1651, 2272
    sasa_norm  = mean(sasa_values) / 100.0    # nes/nls_ml_predictor*.py

A flat divisor (100, or 120, or 150 depending which code path you're in --
these disagree with each other too) is NOT the same thing as relative
solvent accessibility (RSA). Raw SASA in Ų scales with the *size* of the
residue's side chain, not just how buried it is. Compare theoretical max
ASA (Tien et al. 2013):

    Gly  104 Ų      Leu  201 Ų      Trp  285 Ų      Arg  274 Ų

A Leucine that is genuinely 35% exposed (RSA=0.35) has a raw SASA of
~70 Ų. Divide that by 100 and you get sasa_score=0.70 -- reported as
"highly exposed" when it's actually mostly buried. NES motifs are
Leu/Ile/Val/Phe-rich and NLS motifs are Lys/Arg-rich -- i.e. exactly the
residue classes with the largest max-ASA values -- so this flat-divisor
bug systematically inflates the apparent accessibility of the residues
these motifs are built from. That is almost certainly what you're seeing
as "buried sites scoring high."

What this script does instead
------------------------------
1. Computes per-residue raw SASA three independent ways:
     - FreeSASA, Lee & Richards algorithm
     - FreeSASA, Shrake-Rupley algorithm
     - Biopython ShrakeRupley (the exact method app.py already uses)
2. Converts each to RSA using Tien et al. 2013 THEORETICAL max-ASA values
   (residue-specific, not a flat constant).
3. Combines the three RSA estimates into a consensus per residue:
     - consensus_rsa   = mean of the 3 RSA values (0-1, interpretable)
     - consensus_z     = mean of each method's z-score (RSA standardized
                          against that method's own distribution across the
                          chain, then averaged) -- flags residues that are
                          unusually buried/exposed *relative to the rest of
                          this protein*, independent of any one method's
                          absolute scale
     - agreement_sd    = stdev across the 3 methods (low = trustworthy,
                          high = methods disagree, treat with caution)
4. For comparison, also reports what the CURRENT app-style flat /100 score
   would say for the same residue, so you can see exactly where it
   diverges from the properly normalized consensus.

This file does not modify app.py, nes_ml_predictor_improved.py, or
nls_ml_predictor.py. It's a standalone diagnostic / drop-in feature
source you can wire in yourself.

Usage
-----
    python consensus_accessibility.py path/to/structure.pdb [chain_id]

    from consensus_accessibility import consensus_accessibility
    rows = consensus_accessibility("structure.pdb", chain_id="A")
"""

import sys
import tempfile
import statistics
from collections import namedtuple

import freesasa
from Bio.PDB import PDBParser, ShrakeRupley

# ----------------------------------------------------------------------
# Tien et al. 2013, theoretical MaxASA (Ų) -- PLoS ONE 8(11):e80635
# Recommended over the older Rose/Miller empirical scales because it's a
# true upper bound (systematic conformational enumeration), not just the
# max observed in a survey of real structures.
# ----------------------------------------------------------------------
MAX_ASA_TIEN2013 = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}

ResidueAccessibility = namedtuple('ResidueAccessibility', [
    'chain', 'resnum', 'resname',
    'raw_sasa_freesasa_lr', 'raw_sasa_freesasa_sr', 'raw_sasa_biopython_sr',
    'rsa_freesasa_lr', 'rsa_freesasa_sr', 'rsa_biopython_sr',
    'consensus_rsa', 'consensus_z', 'agreement_sd',
    'current_app_score',   # what app.py's flat /100 would report today
])


def _raw_sasa_freesasa(pdb_path, algorithm):
    """Per-residue raw SASA via FreeSASA, keyed by (chain, resnum)."""
    struct = freesasa.Structure(pdb_path)
    params = freesasa.Parameters({'algorithm': algorithm})
    result = freesasa.calc(struct, params)
    residue_areas = result.residueAreas()
    out = {}
    for chain_id, residues in residue_areas.items():
        for resnum_str, area in residues.items():
            try:
                resnum = int(resnum_str)
            except ValueError:
                resnum = resnum_str
            out[(chain_id, resnum)] = area.total
    return out


def _raw_sasa_biopython(pdb_path):
    """Per-residue raw SASA via Bio.PDB ShrakeRupley -- matches
    app.py's calculate_sasa() exactly, so the comparison is apples-to-apples."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    out = {}
    resnames = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    continue  # skip heteroatoms/water
                if not hasattr(residue, 'sasa'):
                    continue
                key = (chain.id, residue.id[1])
                out[key] = residue.sasa
                resnames[key] = residue.resname
        break  # first model only
    return out, resnames


def _zscores(values):
    """Standard z-score of a list; returns all zeros if stdev is 0 or n<2."""
    if len(values) < 2:
        return [0.0] * len(values)
    mean = statistics.mean(values)
    sd = statistics.pstdev(values)
    if sd == 0:
        return [0.0] * len(values)
    return [(v - mean) / sd for v in values]


def consensus_accessibility(pdb_path, chain_id=None):
    """Compute consensus per-residue accessibility for a PDB file.

    Returns a list of ResidueAccessibility namedtuples, one per standard
    amino acid residue found (optionally filtered to one chain).
    """
    lr_raw = _raw_sasa_freesasa(pdb_path, freesasa.LeeRichards)
    sr_raw = _raw_sasa_freesasa(pdb_path, freesasa.ShrakeRupley)
    bio_raw, resnames = _raw_sasa_biopython(pdb_path)

    keys = [k for k in bio_raw.keys() if resnames[k] in MAX_ASA_TIEN2013]
    if chain_id is not None:
        keys = [k for k in keys if k[0] == chain_id]
    keys.sort(key=lambda k: (k[0], k[1]))

    # per-method RSA lists (in key order) so we can z-score each method
    # against its own distribution across this chain
    rsa_lr_list, rsa_sr_list, rsa_bio_list = [], [], []
    for k in keys:
        resname = resnames[k]
        max_asa = MAX_ASA_TIEN2013[resname]
        rsa_lr_list.append(min(lr_raw.get(k, 0.0) / max_asa, 1.5))
        rsa_sr_list.append(min(sr_raw.get(k, 0.0) / max_asa, 1.5))
        rsa_bio_list.append(min(bio_raw.get(k, 0.0) / max_asa, 1.5))

    z_lr = _zscores(rsa_lr_list)
    z_sr = _zscores(rsa_sr_list)
    z_bio = _zscores(rsa_bio_list)

    rows = []
    for i, k in enumerate(keys):
        chain, resnum = k
        resname = resnames[k]
        raw_lr = lr_raw.get(k, 0.0)
        raw_sr = sr_raw.get(k, 0.0)
        raw_bio = bio_raw.get(k, 0.0)
        rsa_lr, rsa_sr, rsa_bio = rsa_lr_list[i], rsa_sr_list[i], rsa_bio_list[i]

        consensus_rsa = statistics.mean([rsa_lr, rsa_sr, rsa_bio])
        consensus_z = statistics.mean([z_lr[i], z_sr[i], z_bio[i]])
        agreement_sd = statistics.pstdev([rsa_lr, rsa_sr, rsa_bio])

        # what the live app would currently report for this residue
        # (app.py: min(1.0, avg_sasa / 100), using its own biopython SASA)
        current_app_score = min(1.0, raw_bio / 100.0)

        rows.append(ResidueAccessibility(
            chain=chain, resnum=resnum, resname=resname,
            raw_sasa_freesasa_lr=raw_lr, raw_sasa_freesasa_sr=raw_sr,
            raw_sasa_biopython_sr=raw_bio,
            rsa_freesasa_lr=rsa_lr, rsa_freesasa_sr=rsa_sr,
            rsa_biopython_sr=rsa_bio,
            consensus_rsa=consensus_rsa, consensus_z=consensus_z,
            agreement_sd=agreement_sd, current_app_score=current_app_score,
        ))
    return rows


def find_mismatches(rows, buried_rsa_thresh=0.25, inflated_score_thresh=0.5):
    """Residues the consensus calls buried (low RSA) that the CURRENT
    app-style flat-normalization would still score as accessible.
    This is the exact failure mode behind 'buried sites scoring high.'"""
    return [
        r for r in rows
        if r.consensus_rsa < buried_rsa_thresh
        and r.current_app_score > inflated_score_thresh
    ]


def _print_report(pdb_path, chain_id=None, top_n=25):
    rows = consensus_accessibility(pdb_path, chain_id=chain_id)
    print(f"{'Chain':<6}{'Res#':<7}{'Name':<6}{'MaxASA':<8}"
          f"{'RSA_LR':<8}{'RSA_SR':<8}{'RSA_bio':<9}"
          f"{'Consens.':<10}{'Z':<8}{'AgreeSD':<9}{'AppScore(now)':<14}")
    for r in rows[:top_n]:
        print(f"{r.chain:<6}{r.resnum:<7}{r.resname:<6}"
              f"{MAX_ASA_TIEN2013[r.resname]:<8.0f}"
              f"{r.rsa_freesasa_lr:<8.3f}{r.rsa_freesasa_sr:<8.3f}"
              f"{r.rsa_biopython_sr:<9.3f}{r.consensus_rsa:<10.3f}"
              f"{r.consensus_z:<8.2f}{r.agreement_sd:<9.3f}"
              f"{r.current_app_score:<14.3f}")

    mismatches = find_mismatches(rows)
    print(f"\n{len(mismatches)} residue(s) the consensus calls BURIED "
          f"(RSA<0.25) that the app's CURRENT flat-normalized score would "
          f"still call accessible (>0.5):\n")
    for r in sorted(mismatches, key=lambda r: r.current_app_score - r.consensus_rsa,
                     reverse=True)[:20]:
        print(f"  {r.chain} {r.resname}{r.resnum:<6} "
              f"consensus_RSA={r.consensus_rsa:.3f}  "
              f"current_app_score={r.current_app_score:.3f}  "
              f"(gap={r.current_app_score - r.consensus_rsa:+.3f})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pdb_path = sys.argv[1]
    chain_id = sys.argv[2] if len(sys.argv) > 2 else None
    _print_report(pdb_path, chain_id=chain_id)
