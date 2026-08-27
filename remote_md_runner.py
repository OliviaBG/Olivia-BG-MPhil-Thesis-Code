"""
Remote MD Runner

This file does NOT run on your Windows/WSL machine - it is meant to be
copied onto the remote compute machine (alongside a copy of md_refinement.py
and CRM1.pdb) and invoked there over SSH by remote_md_dispatch.py.

Usage on the remote machine:
    python remote_md_runner.py <input.json> <output.json>

input.json (written by remote_md_dispatch.py) contains:
    {
        "pdb_content": "<full PDB file text>",
        "candidate": { ...single NES candidate dict... },
        "duration_ns": 10.0
    }

Writes output.json containing the single enhanced candidate dict (with
md_enhanced_score / md_metrics attached), same shape NESMDRefiner produces
locally.

Requires on the remote machine: Python 3, OpenMM, PDBFixer, numpy.
See "MD files/REMOTE_MD_SETUP.md" for setup instructions.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from md_refinement import NESMDRefiner


def main():
    if len(sys.argv) != 3:
        print("Usage: python remote_md_runner.py <input.json> <output.json>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path) as f:
        job = json.load(f)

    # CRM1.pdb is expected next to this script by default; override with
    # CRM1_PDB_PATH if you keep it elsewhere on the remote machine.
    default_crm1_path = str(Path(__file__).parent / 'CRM1.pdb')
    crm1_pdb_path = os.environ.get('CRM1_PDB_PATH', default_crm1_path)
    if not os.path.exists(crm1_pdb_path):
        print(f"  no CRM1 structure found at {crm1_pdb_path} - "
              f"will fall back to helix-only scoring")
        crm1_pdb_path = None

    refiner = NESMDRefiner(crm1_pdb_path=crm1_pdb_path)

    enhanced = refiner.refine_nes_candidates(
        job['pdb_content'],
        [job['candidate']],
        job['duration_ns']
    )

    result = enhanced[0] if enhanced else job['candidate']

    with open(output_path, 'w') as f:
        json.dump(result, f)

    print(f"  wrote result to {output_path}")


if __name__ == '__main__':
    main()
