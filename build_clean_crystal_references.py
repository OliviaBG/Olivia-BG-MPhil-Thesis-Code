#!/usr/bin/env python3
"""
build_clean_crystal_references.py
============================================================
Root cause (found while reading extract_crystal_references.py's
own ChainSelect class, lines 558-565): every CRM1_Ran_<PDBID>.pdb this
project has ever generated keeps "CRM1 plus every other non-peptide
chain (Ran, RanBP1, whatever else is in the asymmetric unit) rather than
guessing which specific extra letter is 'Ran'" -- a DELIBERATE design
choice, reasoned at the time as harmless because "the docking pipeline
only actually reads CRM1's groove residues, so this is context, not a
functional requirement." That assumption is false: md_refinement.py's
_truncate_to_groove_shell() truncates PURELY BY DISTANCE to Cys528,
with no chain-identity filter at all, so any "context" chain sitting
near the groove CAN end up inside the actual simulated environment.

This is a systemic gap, not specific to the crm1.pdb file already fixed
this project -- it affects every crystal reference where the asymmetric
unit contains more than exactly {CRM1, Ran}. Confirmed by direct chain-
count inspection this project:
  - 3NBY: only 2 chains in the raw deposition (CRM1 + Ran) -- clean by
    luck of the source structure, not by design filtering.
  - 3NBZ: 5 chains (2x CRM1, 2x Ran, 1x a 293-residue chain containing
    the native Rev-NES sequence itself) -- kept per the ChainSelect
    design above. Directly verified (separately, via the actual docked
    output poses) that neither run this project's groove-shell happened
    to include material from the extra chains -- but that was a
    geometric coincidence of where this file's Cys528 happens to sit,
    not something the extraction script guaranteed.
  - 3NC0: 5 chains (2x CRM1, 2x Ran, 1x scaffold/cargo) -- SAME risk as
    3NBZ, never verified against an actual docked pose.
  - 3GJX: 5 chains (2x CRM1, 2x Ran, 1x likely a second real Snurportin1
    copy per this script's own docstring) -- SAME risk, never verified.

THE FIX: reuse find_crm1_chain() (already-validated 19-residue groove-
numbering fingerprint, handles multiple CRM1 copies correctly -- see its
own docstring fix for exactly this multi-copy scenario) to
identify the SAME CRM1 chain extract_crystal_references.py already used
for peptide-window identification, so the new file stays consistent with
the existing NES_peptide_<PDBID>_chain<X>.pdb outputs. Ran is identified
separately by its own sequence fingerprint (the canonical Walker A/
P-loop motif GDGGTGK, present in Ran's G-domain, absent from CRM1 and
Snurportin1) rather than by elimination, so it's found affirmatively, not
just "whatever's left." Every other chain -- duplicate CRM1 copies,
peptide/scaffold chains, anything else -- is explicitly excluded.

USAGE:
    python3 build_clean_crystal_references.py            # 3NC0 + 3GJX (the two flagged this project)
    python3 build_clean_crystal_references.py --all       # every structure, as a full regression check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Bio.PDB import PDBParser, PDBIO, Select  # noqa: E402
from extract_crystal_references import (  # noqa: E402
    CRM1_REF_DIR, find_crm1_chain, insert_ter_at_chain_breaks,
)

RAN_FINGERPRINT = "GDGGTGK"  # Walker A / P-loop motif, canonical in Ran's G-domain
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

ALL_STRUCTURES = ['3GJX', '3NBZ', '3NC0', '3GB8', '5UWH', '5UWU', '5UWS', '5DHF', '5DIF', '3NBY']


def chain_sequence(chain):
    return ''.join(THREE_TO_ONE.get(res.resname, '') for res in chain if res.id[0] == ' ')


def find_ran_chain(model, exclude_chain_ids):
    for chain in model:
        if chain.id in exclude_chain_ids:
            continue
        seq = chain_sequence(chain)
        if RAN_FINGERPRINT in seq:
            return chain
    return None


def build_clean_reference(pdb_id):
    raw_path = CRM1_REF_DIR / f'{pdb_id}_original.pdb'
    if not raw_path.exists():
        print(f"{pdb_id}: no raw source file at {raw_path} -- skipping")
        return None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(raw_path))
    model = structure[0]

    crm1_chain, offset = find_crm1_chain(model)
    if crm1_chain is None:
        print(f"{pdb_id}: find_crm1_chain() found no match -- skipping")
        return None

    ran_chain = find_ran_chain(model, exclude_chain_ids={crm1_chain.id})
    if ran_chain is None:
        print(f"{pdb_id}: no Ran-fingerprint chain found (checked all chains except "
              f"{crm1_chain.id}) -- writing CRM1-only reference, flag for manual review")

    keep_ids = {crm1_chain.id}
    if ran_chain is not None:
        keep_ids.add(ran_chain.id)

    class CleanSelect(Select):
        def accept_chain(self, chain):
            return chain.id in keep_ids

    io = PDBIO()
    io.set_structure(structure)
    out_path = CRM1_REF_DIR / f'CRM1_Ran_{pdb_id}_v2clean.pdb'
    io.save(str(out_path), CleanSelect())
    insert_ter_at_chain_breaks(out_path)

    # Verify: re-read what actually got written
    counts = {}
    for line in out_path.read_text().splitlines():
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            ch = line[21]
            counts[ch] = counts.get(ch, 0) + 1

    print(f"{pdb_id}: CRM1 chain={crm1_chain.id} (offset={offset}), "
          f"Ran chain={ran_chain.id if ran_chain else 'NOT FOUND'} "
          f"-> wrote {out_path.name}, chains: {counts}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='Process every structure, not just 3NC0/3GJX')
    args = ap.parse_args()

    targets = ALL_STRUCTURES if args.all else ['3NC0', '3GJX']
    print(f"Building clean references for: {targets}\n")
    for pdb_id in targets:
        build_clean_reference(pdb_id)


if __name__ == '__main__':
    main()
