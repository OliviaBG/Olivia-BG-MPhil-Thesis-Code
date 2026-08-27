#!/usr/bin/env python3
"""
idealized_helix_vs_crystal_check.py
============================================================
WHY THIS EXISTS: refine_nes_candidates evaluates real (non-crystal)
candidates using TWO starting-pose hypotheses -- 'native' (AlphaFold's
isolated-protein prediction) and 'idealized_helix' (a literature-informed
canonical alpha helix, see _build_idealized_helix_pdb) -- and reports both,
since neither's real bound conformation is actually known. Prior runs this
session (anchor_occupancy_dual_hypothesis_v2.json n=43,
anchor_occupancy_sidechain_relax_v1.json n=24) compared these two hypotheses
against EACH OTHER, which only tells you which one this pipeline currently
prefers -- not which one is actually closer to how these peptides really
bind.

Now that three real crystal ground-truth structures exist
(crystal_sanity_check.py / CRYSTAL_STRUCTURES), we can ask the sharper
question directly: if you did NOT have the real crystal structure and had
to guess a starting pose via idealized_helix, how close would you actually
get to the truth?

WHAT IT DOES: for each of the three real crystal structures, takes the
REAL, experimentally-verified peptide sequence (same ones
crystal_sanity_check.py uses) and docks it via
starting_conformation='idealized_helix' (i.e. build a generic idealized
alpha helix for this sequence, then run it through the SAME Kabsch
sub-pocket-registration placement, relaxation, minimization, equilibration,
and 2ns production protocol every real candidate goes through -- NOT the
'crystal' shortcut that uses the real coordinates directly). After
production finishes, saves the peptide's final converged position
(save_final_peptide_pdb_path) and computes backbone (N/CA/C/O) RMSD against
the real crystal peptide's own coordinates -- both already in the same
coordinate frame, since both are referenced against the same CRM1+RanGTP
structure extracted from the same source PDB.

READING THE OUTPUT: a low RMSD (roughly <3-4 Angstrom backbone, a common
rule-of-thumb threshold for "same binding mode" in docking validation
literature) means idealized_helix, despite starting from a generic guess,
converges close to the real experimentally-solved pose -- meaning it's a
reasonable proxy for real candidates where no crystal structure exists. A
large RMSD (potentially with a comparable or even a "good" score anyway)
would mean the scoring can be fooled by the WRONG pose, which is a much
more serious problem than anything found so far this project, since it
would undercut the real-candidate runs' idealized_helix hypothesis
entirely, not just narrow a specificity gap. Also reports anchor_occupancy/
raw_binding_score for direct comparison against the real crystal pose's own
numbers (crystal_sanity_check_results_v3.json).

USAGE (run on the pod, alongside crystal_sanity_check.py -- requires
OpenMM + the crm1_reference/ files for all three structures):
    python3 idealized_helix_vs_crystal_check.py --duration-ns 2.0
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

from md_refinement import NESMDRefiner
from crystal_sanity_check import CRYSTAL_STRUCTURES

THIS_DIR = Path(__file__).resolve().parent

BACKBONE_ATOMS = ('N', 'CA', 'C', 'O')


def load_backbone_coords(pdb_path, chain_id=None):
    """Ordered list of (resnum_or_index, atom_name, xyz) for backbone atoms,
    in residue sequence order. If chain_id given, restricts to that chain
    (needed for the real crystal peptide files, which may retain their
    original chain letter); otherwise takes all chains (the saved
    final-pose files from _run_crm1_docking are peptide-only already, no
    chain filtering needed there)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('x', str(pdb_path))
    model = structure[0]
    coords_by_residue = []
    chains = [model[chain_id]] if chain_id else list(model)
    for chain in chains:
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            atoms = {}
            for atom_name in BACKBONE_ATOMS:
                if atom_name in residue:
                    atoms[atom_name] = residue[atom_name].get_coord()
            if atoms:
                coords_by_residue.append(atoms)
    return coords_by_residue


def compute_backbone_rmsd(residues_a, residues_b):
    """Positional (no re-superposition) RMSD over backbone atoms common to
    each aligned residue pair, matched by SEQUENCE ORDER (not residue
    number, since the idealized_helix build and the real crystal file use
    different numbering conventions). Deliberately NOT doing a best-fit
    Kabsch superposition first -- both structures are already independently
    placed in the same CRM1-referenced coordinate frame, so a raw RMSD
    tells us whether idealized_helix's docking converges to the actual
    right LOCATION and orientation, not just the right local backbone
    shape. Returns (rmsd_angstrom, n_atoms_compared, n_residues_compared)."""
    n = min(len(residues_a), len(residues_b))
    if n == 0:
        return None, 0, 0
    sq_diffs = []
    n_residues_compared = 0
    for i in range(n):
        ra, rb = residues_a[i], residues_b[i]
        common = set(ra) & set(rb)
        if not common:
            continue
        n_residues_compared += 1
        for atom_name in common:
            diff = ra[atom_name] - rb[atom_name]
            sq_diffs.append(float(np.dot(diff, diff)))
    if not sq_diffs:
        return None, 0, 0
    rmsd = float(np.sqrt(np.mean(sq_diffs)))
    return rmsd, len(sq_diffs), n_residues_compared


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration-ns', type=float, default=2.0)
    ap.add_argument('--structures', default='3NBY,3GJX,3NBZ')
    ap.add_argument('--out', default='idealized_helix_vs_crystal_results.json')
    args = ap.parse_args()

    struct_ids = [s.strip() for s in args.structures.split(',') if s.strip()]

    all_results = {}

    for sid in struct_ids:
        cfg = CRYSTAL_STRUCTURES[sid]
        if not cfg['crm1_pdb'].exists() or not cfg['peptide_pdb'].exists():
            print(f"Missing reference files for {sid}, skipping")
            continue

        print("\n" + "#" * 70)
        print(f"# {sid}: {cfg['label']}")
        print(f"# idealized_helix('{cfg['sequence']}') vs real crystal coordinates")
        print("#" * 70)

        refiner = NESMDRefiner(crm1_pdb_path=str(cfg['crm1_pdb']))
        real_backbone = load_backbone_coords(cfg['peptide_pdb'])
        print(f"  Real crystal peptide: {len(real_backbone)} residues with backbone atoms")

        all_results[sid] = {}

        for relax in (False, True):
            candidate = {
                'sequence': cfg['sequence'],
                'start': cfg['residue_range'][0],
                'end': cfg['residue_range'][1],
                'full_sequence': None,
                'combined_score': 0.5,
            }
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
                final_pose_path = tmp.name

            print("\n" + "=" * 70)
            print(f"{sid} / idealized_helix, relax_sidechains={relax}")
            print("=" * 70)
            result = refiner._run_crm1_docking(
                pdb_content='',  # unused for idealized_helix branch beyond a harmless tmp-file write
                candidate=candidate,
                duration_ns=args.duration_ns,
                starting_conformation='idealized_helix',
                scramble_registration=False,
                relax_sidechains=relax,
                save_final_peptide_pdb_path=final_pose_path,
            )
            metrics = result.get('md_metrics', {}) or {}

            rmsd_angstrom, n_atoms, n_res = None, 0, 0
            if Path(final_pose_path).exists() and Path(final_pose_path).stat().st_size > 0:
                idealized_backbone = load_backbone_coords(final_pose_path)
                rmsd_angstrom, n_atoms, n_res = compute_backbone_rmsd(real_backbone, idealized_backbone)

            print(f"  anchor_occupancy_score: {metrics.get('anchor_occupancy_score')}")
            print(f"  raw_binding_score:      {metrics.get('raw_binding_score')}")
            print(f"  Backbone RMSD vs real crystal pose: "
                  f"{f'{rmsd_angstrom:.2f} Angstrom' if rmsd_angstrom is not None else 'N/A (no final pose saved)'} "
                  f"({n_atoms} atoms / {n_res} residues compared)")

            all_results[sid][f"idealized_helix_{'relaxed' if relax else 'norelax'}"] = {
                'anchor_occupancy_score': metrics.get('anchor_occupancy_score'),
                'raw_binding_score': metrics.get('raw_binding_score'),
                'binding_score': metrics.get('binding_score'),
                'rmsd_vs_crystal_angstrom': rmsd_angstrom,
                'n_backbone_atoms_compared': n_atoms,
                'n_residues_compared': n_res,
            }

        Path(args.out).write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n(checkpoint saved to {args.out} after {sid})")

    print("\n" + "#" * 70)
    print("# SUMMARY")
    print("#" * 70)
    for sid in struct_ids:
        if sid not in all_results:
            continue
        print(f"\n{sid}:")
        for relax in (False, True):
            key = f"idealized_helix_{'relaxed' if relax else 'norelax'}"
            r = all_results[sid].get(key, {})
            rmsd = r.get('rmsd_vs_crystal_angstrom')
            print(f"  relax_sidechains={relax}: RMSD={f'{rmsd:.2f}A' if rmsd is not None else 'N/A'}  "
                  f"anchor_occupancy={r.get('anchor_occupancy_score')}  "
                  f"raw_binding_score={r.get('raw_binding_score')}")

    print(f"\nFull results saved to {args.out}")
    print("\nCompare the RMSD values above against crystal_sanity_check_results_v3.json's 'correct' scores "
          "for the same structures/relax conditions -- a low RMSD alongside a comparable score means "
          "idealized_helix is a trustworthy proxy for real candidates; a high RMSD alongside a similar or "
          "high score would mean the scoring doesn't actually distinguish the right pose from a merely "
          "plausible-looking one.")


if __name__ == '__main__':
    main()
