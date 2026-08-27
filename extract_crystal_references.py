#!/usr/bin/env python3
"""
extract_crystal_references.py
============================================================
Builds a second and third real, experimentally-solved ground-truth
CRM1-NES(-like) reference, alongside the existing PKI-NES/3NBY one (see
crystal_sanity_check.py) -- for the same purpose: testing whether this
pipeline's scoring recognizes a peptide known by direct crystallographic
evidence to be correctly bound, now across THREE independent real
structures instead of one.

  - 3GJX: Dong et al. 2009, Nature 458:1136-1141 -- CRM1-Snurportin1-RanGTP.
    Snurportin1 isn't a classical short NES peptide; it's the protein whose
    OWN N-terminal region was the first NES shown to dock into this same
    five-pocket groove (that paper's title: "Structural basis for leucine-
    rich nuclear export signal recognition by CRM1"). Chain B/E here is
    real, full-length Snurportin1, not a grafted chimera.
  - 3NBZ: Guttler et al. 2010, Nat Struct Mol Biol 17:1367-1376 -- the SAME
    paper as 3NBY (PKI-NES), but for HIV-1 Rev's NES instead. Notable: the
    paper reports Rev-NES binds in an EXTENDED conformation with a critical
    proline, NOT the alpha-helical mode PKI-NES uses -- a genuinely
    different, non-helical real binding mode, useful as a tougher test of
    whatever this pipeline assumes about NES conformation. Chain B/E here
    is (based on the same construct-design pattern 3NBY used) very likely
    a Snurportin1-scaffold chimera carrying the grafted Rev-NES, but this
    script does NOT assume that -- see the geometric identification below.

WHY GEOMETRIC IDENTIFICATION, NOT SEQUENCE LOOKUP FROM MEMORY: for 3NBY
(PKI-NES), the actual bound peptide sequence was directly verified against
the file's own SEQRES/ATOM records earlier in this project (found to exactly
match LALKLAGLDI, the literature PKI-NES sequence) rather than assumed.
This script applies the same "check the real structure, don't assume"
policy to 3GJX and 3NBZ, but automates it: instead of needing to already
know which chain B residues are the NES, it finds them empirically -- any
chain B (Snurportin1 or Rev-NES-chimera) residue with a CA atom within
GROOVE_CONTACT_CUTOFF_ANGSTROM of any CRM1 (chain A) groove-lining residue
CA (the same CRM1_GROOVE_LINING_RESIDUES_1INDEXED list md_refinement.py's
own sub-pocket calibration already uses and has verified matches this
project's CRM1 numbering convention) is "in the groove" -- with no residue
numbers assumed in advance. The resulting contiguous segment is the real,
experimentally-observed NES(-like) peptide for that structure.

OUTPUT (per structure X in {3GJX, 3NBZ}):
  crm1_reference/CRM1_Ran_X.pdb           -- chains A+C (CRM1+RanGTP) only
  crm1_reference/NES_peptide_X_chainB.pdb -- the identified groove-contacting
                                              chain B segment, real coordinates

USAGE (run on a machine/pod with real internet access -- RCSB fetches, same
requirement as the rest of this project's structure-fetching scripts):
    python3 extract_crystal_references.py
"""

import re
import sys
import urllib.request
from pathlib import Path

try:
    from Bio.PDB import PDBParser, PDBIO, Select
except ImportError:
    print("Biopython not installed -- pip install biopython --break-system-packages")
    sys.exit(1)

THIS_DIR = Path(__file__).resolve().parent
CRM1_REF_DIR = THIS_DIR / 'crm1_reference'

# Same list md_refinement.py's CRM1_GROOVE_LINING_RESIDUES_1INDEXED uses --
# copied here (not imported) so this script has no dependency on OpenMM/
# PDBFixer being installed, just Biopython + requests, since its only job
# is producing the reference PDB files md_refinement.py will later consume.
CRM1_GROOVE_LINING_RESIDUES_1INDEXED = [514, 518, 521, 525, 528, 534, 537, 538,
                                         541, 544, 545, 554, 558, 561, 564, 565,
                                         568, 572, 575]
CRM1_GROOVE_LINING_RESIDUE_NAMES = {
    514: 'LYS', 518: 'VAL', 521: 'ILE', 525: 'LEU', 528: 'CYS', 534: 'LYS',
    537: 'LYS', 538: 'ALA', 541: 'ALA', 544: 'ILE', 545: 'MET', 554: 'PHE',
    558: 'HIS', 561: 'PHE', 564: 'THR', 565: 'VAL', 568: 'LYS', 572: 'PHE',
    575: 'GLU',
}

GROOVE_CONTACT_CUTOFF_ANGSTROM = 6.0  # CA-CA; generous enough to catch a
                                       # real bound peptide's full span, not
                                       # just its closest anchor residues

# Same pattern md_refinement.py's PHI_REGISTER_RE uses -- copied here (not
# imported) for the same "no OpenMM/PDBFixer dependency" reason
# CRM1_GROOVE_LINING_RESIDUES_1INDEXED above already documents. Added
# Used by find_groove_contacting_segment's window-expansion
# check below, NOT to do any register classification here (that stays
# md_refinement.py's job) -- only to test "does this candidate window's
# sequence contain a real, complete 4-anchor spacing pattern" as a guide
# for recovering anchors the plain distance-window trims off.
PHI_REGISTER_RE = re.compile(r'([LIVFM]).{1,3}([LIVFM]).{1,3}([LIVFM]).{1,2}([LIVFM])')

AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}


def fetch_pdb(pdb_id, dest_path):
    if dest_path.exists():
        print(f"  {dest_path.name} already present locally, skipping download")
        return
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  fetching {url} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode('utf-8')
    dest_path.write_text(text)
    print(f"  saved {dest_path.name} ({len(text):,} bytes)")


def verify_groove_numbering(chain_a, verbose=True):
    """Same check performed manually for 3NBY earlier in this project --
    confirms this structure's CRM1 chain uses the same residue numbering
    convention md_refinement.py's calibration already assumes, BEFORE
    trusting anything extracted from it. Returns True if all 19 match."""
    ok = True
    for resnum, expected in CRM1_GROOVE_LINING_RESIDUE_NAMES.items():
        residue = None
        for res in chain_a:
            if res.id[0] == ' ' and res.id[1] == resnum:
                residue = res
                break
        if residue is None:
            if verbose:
                print(f"    Warning: residue {resnum} not found in chain {chain_a.id}")
            ok = False
            continue
        actual = residue.resname
        match = 'OK' if actual == expected else 'MISMATCH'
        if actual != expected:
            ok = False
        if verbose:
            print(f"    {resnum}: expected {expected}, found {actual}  [{match}]")
    return ok


def _find_offset_for_chain(chain):
    """ : confirmed necessary in practice -- 5UWH/5UWU/5UWS (Fung
    et al. 2017 depositions) have a chain that's clearly CRM1-sized
    (1247-1416 residues, ~1071 real protein residues once HETATMs are
    excluded, matching full-length human CRM1/XPO1) but which does NOT
    match this project's groove-lining residue numbers at all -- i.e. it's
    the same protein, but numbered differently in this deposition batch
    (common when a different construct, tag, or numbering scheme was used
    at deposition time). Rather than assume offset 0, search for an
    integer offset O such that chain-residue (our_convention_resnum + O)
    has the expected residue type at all 19 groove-lining positions.
    Candidate offsets are narrowed first to residues typed LYS (matching
    position 514, our numbering's first anchor) before the full 19-way
    check, so this stays cheap even over a >1000-residue chain. Returns
    the offset (int, 0 if this chain already matches this project's
    numbering exactly) or None if no consistent offset exists anywhere in
    this chain (i.e. it's very likely not CRM1 at all, or has a genuinely
    different sequence)."""
    resnum_to_name = {res.id[1]: res.resname for res in chain if res.id[0] == ' '}
    if not resnum_to_name:
        return None

    def offset_matches(offset):
        for resnum, expected in CRM1_GROOVE_LINING_RESIDUE_NAMES.items():
            if resnum_to_name.get(resnum + offset) != expected:
                return False
        return True

    if offset_matches(0):
        return 0

    candidate_offsets = {resnum - 514 for resnum, name in resnum_to_name.items() if name == 'LYS'}
    for offset in candidate_offsets:
        if offset_matches(offset):
            return offset
    return None


def find_crm1_chain(model):
    """ : chain lettering for 'which chain is CRM1' is NOT
    consistent across deposition batches -- confirmed directly when 5UWH/
    5UWU/5UWS (Fung et al. 2017 depositions) turned out to use chain C for
    CRM1, not chain A like the Guttler/Dong-era structures this script was
    originally written against. Rather than hardcode a letter, silently
    test EVERY chain against the 19-residue groove-numbering fingerprint
    (allowing for a possible constant numbering offset -- see
    _find_offset_for_chain) and return whichever one actually matches.
    Returns (chain, offset) for the matching chain, or (None, None) if no
    chain in the model matches under any offset (a genuine sequence/
    identity mismatch, not just a wrong guess about the letter or
    numbering)."""
    for chain in model:
        offset = _find_offset_for_chain(chain)
        if offset is not None:
            return chain, offset
    return None, None


def renumber_chain(chain, offset):
    """Shift every residue in `chain` by -offset so it matches this
    project's CRM1 numbering convention (i.e. undoes the offset
    _find_offset_for_chain detected: this_structure_resnum - offset =
    our_convention_resnum). Mutates in place. Only called when offset != 0
    -- everything downstream of this (verify_groove_numbering,
    find_groove_contacting_segment, and md_refinement.py's own groove-
    residue lookups once this gets written out) assumes the standard
    514-575 numbering, so this has to happen before any of that runs, not
    just be tracked as metadata."""
    for res in chain:
        hetflag, resseq, icode = res.id
        res.id = (hetflag, resseq - offset, icode)


def insert_ter_at_chain_breaks(pdb_path):
    """
    root-caused the "No template found for residue N (ILE) ...
    missing 1 C atom ... is the chain missing a terminal capping group?"
    OpenMM error seen on 5DHF/5DIF. Confirmed directly (reproduced locally
    with the real pdbfixer/openmm packages, not guessed from the error
    text alone): those two structures' CRM1 chain has two genuine internal
    breaks (residues 366-402 deleted on purpose in the engineered ScCRM1*
    construct per the paper's own Methods, plus a ~19-residue disordered
    loop near their V441D mutation) -- but this project's PDB files have
    NO SEQRES records (Biopython's PDBIO never writes them), and
    PDBFixer's findMissingResidues() relies on SEQRES-vs-ATOM comparison
    to find gaps at all. Verified directly: fixer.findMissingResidues()
    returns an EMPTY dict for these files despite the two real gaps --
    it's not that it "can't tell a real gap from ours" (the concern the
    md_refinement.py fix from earlier in this project addressed), it doesn't
    see either gap in the first place. With no SEQRES and no TER record at
    the break (Bio.PDB's PDBIO only writes TER at genuine chain-object
    boundaries, not at internal numbering gaps within the SAME chain
    object), OpenMM's topology builder treats both fragments as one
    unbroken chain and tries to bond the last residue before the gap to
    the first residue after it -- which are ~40 Angstrom apart, not a
    real peptide bond, producing exactly the observed template-matching
    failure. This is NOT specific to the 3NBY/3GJX/3NBZ-family structures
    used earlier in this project -- it just happens that none of them had an
    internal numbering gap large enough to hit this failure mode; small
    gaps would silently mis-bond too, just less catastrophically.

    Fix: post-process the already-written PDB text and insert an explicit
    TER record (matching Bio.PDB's own TER formatting) at every point
    where a chain's residue numbers are non-consecutive, regardless of gap
    size -- since no bridging ever actually happens for these SEQRES-less
    files anyway (confirmed above), there's no legitimate "leave it alone,
    PDBFixer will fill it in" case being overridden here; every such gap
    was already silently mis-bonded before this fix, just not always
    egregiously enough to error out.
    """
    lines = Path(pdb_path).read_text().splitlines(keepends=True)
    out_lines = []
    prev_key = None  # (chain_id, resnum) of the last ATOM/HETATM residue seen
    prev_atom_line = None
    n_inserted = 0
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            chain_id = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                out_lines.append(line)
                continue
            key = (chain_id, resnum)
            if (prev_key is not None and prev_key[0] == chain_id
                    and resnum != prev_key[1] and resnum != prev_key[1] + 1):
                # Non-consecutive residue number within the same chain --
                # a real break this file doesn't otherwise mark. Insert a
                # TER using the previous residue's own atom line for the
                # serial/resname/chain/resnum fields, same format Bio.PDB
                # itself uses (see CRM1_Ran_*.pdb's existing end-of-chain
                # TER records).
                prev_resname = prev_atom_line[17:20]
                ter_serial = n_inserted + 1  # renumbered properly below anyway
                ter_line = (f"TER   {ter_serial:>5}      {prev_resname:>3} "
                            f"{prev_key[0]}{prev_key[1]:>4}"
                            + " " * 53 + "\n")
                out_lines.append(ter_line)
                n_inserted += 1
            prev_key = key
            prev_atom_line = line
        out_lines.append(line)
    if n_inserted:
        Path(pdb_path).write_text(''.join(out_lines))
        print(f"  Inserted {n_inserted} TER record(s) at internal chain break(s) "
              f"not otherwise marked in {Path(pdb_path).name} (fixes OpenMM "
              f"mis-bonding across genuine gaps)")
    return n_inserted


def find_peptide_chain(model, crm1_chain):
    """ : companion to find_crm1_chain -- once CRM1 is located
    (regardless of its letter), the bound NES/cargo peptide chain is
    likewise not guaranteed to be 'B'. Try every OTHER chain in the model
    through find_groove_contacting_segment() and keep whichever gives the
    LOWEST average groove distance -- the real bound peptide should sit
    much closer to the groove than Ran/RanBP1 or any other chain present,
    so this is robust rather than assuming a letter. Returns
    (chain_object, result_tuple_from_find_groove_contacting_segment) for
    the winning chain, or (None, None) if nothing plausible was found in
    any candidate chain.

     Confirmed real bug, caught while re-auditing 3NC0: many
    of these asymmetric units contain a SECOND copy of CRM1 itself (e.g.
    3NC0 has chain A -- the one find_crm1_chain identified -- AND chain D,
    both full ~1041-residue CRM1 copies). Nothing here was excluding other
    CRM1 copies from peptide-chain candidacy, and a crystal-packing
    contact between the two unrelated CRM1 copies can score a BETTER
    (lower) average groove-distance than the real, biologically-bound NES
    region in the correct cargo chain -- which is exactly what happened
    for 3NC0 (silently selected a meaningless region of chain D over the
    real 'LPPLERLTLS' Rev-NES region in chain B). Fix: reuse
    _find_offset_for_chain (the same 19-residue groove-numbering
    fingerprint find_crm1_chain itself uses) on every candidate BEFORE
    scoring it -- if a candidate ALSO matches that fingerprint, it's
    another CRM1 copy, not real cargo, and gets skipped outright."""
    best_chain, best_result = None, None
    for chain in model:
        if chain.id == crm1_chain.id:
            continue
        n_residues = sum(1 for _ in chain)
        if n_residues < 5:
            continue  # too short to plausibly contain a 7-14 residue NES window
        if _find_offset_for_chain(chain) is not None:
            continue  # another CRM1 copy in the asymmetric unit, not cargo
        try:
            result = find_groove_contacting_segment(crm1_chain, chain)
        except RuntimeError:
            continue
        if result is None:
            continue
        avg_dist = result[3]
        if best_result is None or avg_dist < best_result[3]:
            best_chain, best_result = chain, result
    return best_chain, best_result


def build_chain_b_distance_profile(chain_a, chain_b):
    """For every chain B residue with a CA atom, its minimum CA-CA distance
    (Angstrom) to any CRM1 groove-lining residue's CA. Sorted by residue
    number. This is the raw evidence everything else below is derived from
    -- kept separate so it can be dumped/inspected on its own."""
    groove_ca_coords = []
    for resnum in CRM1_GROOVE_LINING_RESIDUES_1INDEXED:
        for res in chain_a:
            if res.id[0] == ' ' and res.id[1] == resnum and 'CA' in res:
                groove_ca_coords.append(res['CA'].get_coord())
                break

    if not groove_ca_coords:
        raise RuntimeError("No groove-lining CA atoms found in chain A -- can't identify contacts")

    import numpy as np
    groove_arr = np.array(groove_ca_coords)

    profile = []
    for res in chain_b:
        if res.id[0] != ' ' or 'CA' not in res:
            continue
        ca = res['CA'].get_coord()
        dist = float(np.linalg.norm(groove_arr - ca, axis=1).min())
        profile.append((res.id[1], res.resname, dist))

    profile.sort(key=lambda p: p[0])
    return profile


def find_groove_contacting_segment(chain_a, chain_b, window_sizes=range(7, 15), max_numbering_gap=2,
                                    max_expansion=6, max_total_window=22):
    """Finds the real bound peptide span WITHOUT relying on a flat distance
    cutoff (an earlier version of this script did that and fragmented a
    genuine contiguous stretch into isolated single-residue hits, e.g.
    3GJX/3NBZ both returned exactly ONE residue -- clearly wrong for a real
    bound segment). Instead: build the full per-residue min-distance-to-
    groove profile, then slide windows of plausible NES length (7-14
    residues, matching the range of literature NES motifs including PKI's
    verified 10-residue span) over it and pick whichever window has the
    LOWEST average distance to the groove, i.e. the stretch that as a whole
    sits closest to the binding site -- this is robust to a few individual
    residues within a real bound peptide sitting a bit further from any
    single groove CA (loop/flanking positions) than the core anchor
    residues do.

     That plain "minimize average distance" rule turned out to
    have a real, confirmed blind spot: a genuine anchor residue just
    outside the tightest-average window can get silently trimmed off when
    an intervening LINKER residue (expected to sit farther from the groove
    -- that's what a linker is) drags the window's average up enough that
    a narrower window scores better on paper. Caught directly against
    real, published, structurally-confirmed data: Snurportin1's own paper
    (Dong et al. 2009, referenced in Guttler et al. 2010's Fig 1C) shows a
    real Phi0/Phi1 anchor pair (Met1, Leu4) that this algorithm's plain
    distance window trimmed off this project's 3GJX extraction in favor of
    a window starting 4 residues later with a marginally better average --
    even though Met1 (7.35 A) and Leu4 (7.22 A) were BOTH closer to the
    groove than several residues the chosen window already included
    anyway (e.g. Ser10 at 9.47 A -- a real, unavoidable linker residue,
    not evidence Met1/Leu4 don't belong). An audit of all 10 structures in
    this project's ground truth found the same class of near-miss in 9 of
    10 (only 3NBY/PKI, extracted by an earlier version of this script, was
    unaffected).

    Fix: after finding the plain distance-optimal base window, if its own
    sequence does NOT contain a full 4-anchor Phi-register match
    (PHI_REGISTER_RE, same strict pattern md_refinement.py's
    _find_phi_register starts with), try expanding the window by up to
    max_expansion residues on either side (capped at max_total_window
    total) and see whether any expansion's sequence DOES complete a full
    match. Among expansions that do, keep the SMALLEST one (by total
    residues added) -- the goal is recovering real anchors that got
    trimmed, not maximizing sequence length for its own sake. If no
    expansion helps (or the base window already has a full match), the
    base window is kept exactly as before.

    Returns (start_resnum, end_resnum, sequence, avg_distance, full_profile).
    """
    profile = build_chain_b_distance_profile(chain_a, chain_b)
    if not profile:
        return None

    prof_by_num = {resnum: (resname, dist) for resnum, resname, dist in profile}

    best = None
    for w in window_sizes:
        for i in range(len(profile) - w + 1):
            window = profile[i:i + w]
            resnums = [p[0] for p in window]
            span = resnums[-1] - resnums[0]
            if span > (w - 1) + max_numbering_gap:
                continue  # numbering isn't contiguous enough to be one real peptide stretch
            avg_dist = sum(p[2] for p in window) / w
            if best is None or avg_dist < best[0]:
                best = (avg_dist, window)

    if best is None:
        return None

    avg_dist, window = best
    start, end = window[0][0], window[-1][0]

    def seq_for(lo, hi):
        """None if the range isn't fully contiguous in the real structure
        (a genuine gap -- never bridge that with a synthesized sequence)."""
        letters = []
        for n in range(lo, hi + 1):
            if n not in prof_by_num:
                return None
            letters.append(AA3TO1.get(prof_by_num[n][0], 'X'))
        return ''.join(letters)

    seq = seq_for(start, end)
    if seq is None:
        # Base window itself has a small (<= max_numbering_gap) internal
        # gap, which the window-selection step above explicitly allows --
        # fall back to the original behavior of just skipping the missing
        # residue(s) rather than refusing to build a sequence at all. No
        # expansion attempt in this case (seq_for's contiguity requirement
        # would reject every expansion candidate too, for the same reason).
        seq = ''
        for res in chain_b:
            if res.id[0] == ' ' and start <= res.id[1] <= end:
                seq += AA3TO1.get(res.resname, 'X')

    if seq and not PHI_REGISTER_RE.search(seq):
        best_expansion = None  # (residues_added, new_start, new_end, new_seq)
        for left_ext in range(0, max_expansion + 1):
            for right_ext in range(0, max_expansion + 1):
                if left_ext == 0 and right_ext == 0:
                    continue
                new_start, new_end = start - left_ext, end + right_ext
                if (new_end - new_start + 1) > max_total_window:
                    continue
                candidate_seq = seq_for(new_start, new_end)
                if candidate_seq is None:
                    continue
                if PHI_REGISTER_RE.search(candidate_seq):
                    added = left_ext + right_ext
                    if best_expansion is None or added < best_expansion[0]:
                        best_expansion = (added, new_start, new_end, candidate_seq)
        if best_expansion is not None:
            added, new_start, new_end, candidate_seq = best_expansion
            print(f"    -> Distance-optimal window {start}-{end} ({seq}) has no full 4-anchor "
                  f"Phi-register match; expanding by {added} residue(s) to {new_start}-{new_end} "
                  f"({candidate_seq}) recovers one (see docstring addendum)")
            start, end, seq = new_start, new_end, candidate_seq
            avg_dist = sum(prof_by_num[n][1] for n in range(start, end + 1)) / (end - start + 1)

    return start, end, seq, avg_dist, profile


def process_structure(pdb_id, label):
    print(f"\n{'='*70}\n{pdb_id} ({label})\n{'='*70}")
    raw_path = CRM1_REF_DIR / f'{pdb_id}_original.pdb'
    fetch_pdb(pdb_id, raw_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(raw_path))
    model = structure[0]

    # These next three (5UWH, 5UWU, 5UWS) are from a different
    # deposition batch (Fung et al. 2017) than 3NBY/3GJX/3NBZ/3NC0/3GB8 (all
    # Guttler/Dong-era depositions this script was originally built against)
    # -- chain lettering across different depositors/years is NOT guaranteed
    # to follow the same A=CRM1/B=cargo/C=Ran convention. Print every chain
    # up front so a mismatch is visible immediately rather than silently
    # grabbing the wrong chain and reporting a plausible-looking but wrong
    # "peptide".
    print("  Chains present in this structure:")
    for chain in model:
        n_res = sum(1 for _ in chain)
        first_resnames = [r.resname for r in list(chain)[:3]]
        print(f"    chain {chain.id}: {n_res} residues (starts: {first_resnames})")

    # Auto-detect CRM1 and the peptide chain by content rather
    # than assuming letters A/B -- confirmed necessary in practice: 5UWH/
    # 5UWU/5UWS (Fung et al. 2017 depositions) use chain C for CRM1 and
    # chain D for the bound peptide, not A/B like the earlier Guttler/Dong-
    # era structures this script was originally written against.
    print("  Locating CRM1 chain by groove-numbering fingerprint (not assuming a letter or offset):")
    chain_a, offset = find_crm1_chain(model)
    if chain_a is None:
        print(f"  Warning: No chain in {pdb_id} matches this project's CRM1 groove-numbering "
              f"convention under any consistent offset -- DO NOT trust this structure's "
              f"extraction without manual review; skipping further processing for it.")
        return None
    if offset == 0:
        print(f"  -> chain {chain_a.id} matches (all 19 groove-lining residues confirmed, no offset needed)")
    else:
        print(f"  -> chain {chain_a.id} matches with a numbering offset of {offset:+d} "
              f"(this structure's residue N = our convention's residue N-{offset}) -- "
              f"renumbering chain {chain_a.id} to this project's convention before continuing")
        renumber_chain(chain_a, offset)

    print("  Locating bound peptide chain by closest average distance to the groove:")
    chain_b, result = find_peptide_chain(model, chain_a)
    if chain_b is None or result is None:
        print(f"  Warning: Couldn't find any plausible bound-peptide chain for {pdb_id}")
        return None
    print(f"  -> chain {chain_b.id} ({sum(1 for _ in chain_b)} residues) selected")

    start, end, seq, avg_dist, profile = result
    print(f"  Best-fit groove-proximal chain {chain_b.id} segment: residues {start}-{end} = {seq} "
          f"(avg CA-CA distance to nearest groove residue: {avg_dist:.2f} Angstrom)")

    # Diagnostic: print the 15 lowest-distance individual residues, and the
    # full profile immediately surrounding the chosen window, so a human can
    # sanity-check this pick before it's trusted for the MD sanity check.
    by_dist = sorted(profile, key=lambda p: p[2])[:15]
    print(f"  15 closest-to-groove chain {chain_b.id} residues (resnum, resname, min_dist_A): "
          f"{[(r, n, round(d, 2)) for r, n, d in by_dist]}")
    context_lo, context_hi = max(start - 5, profile[0][0]), min(end + 5, profile[-1][0])
    context_slice = [p for p in profile if context_lo <= p[0] <= context_hi]
    print(f"  Full profile around chosen window (residues {context_lo}-{context_hi}): "
          f"{[(r, n, round(d, 2)) for r, n, d in context_slice]}")

    peptide_chain_id = chain_b.id

    class ChainSelect(Select):
        # Keep CRM1 plus every other non-peptide chain (Ran, RanBP1,
        # whatever else is in the asymmetric unit) rather than guessing
        # which specific extra letter is "Ran" -- the docking pipeline
        # only actually reads CRM1's groove residues, so this is context,
        # not a functional requirement, and safer than a letter guess.
        def accept_chain(self, chain):
            return chain.id != peptide_chain_id

    io = PDBIO()
    io.set_structure(structure)
    crm1_out = CRM1_REF_DIR / f'CRM1_Ran_{pdb_id}.pdb'
    io.save(str(crm1_out), ChainSelect())
    print(f"  wrote {crm1_out.name}")
    insert_ter_at_chain_breaks(crm1_out)

    class PeptideSelect(Select):
        def accept_chain(self, chain):
            return chain.id == peptide_chain_id
        def accept_residue(self, residue):
            return residue.id[0] == ' ' and start <= residue.id[1] <= end

    io2 = PDBIO()
    io2.set_structure(structure)
    peptide_out = CRM1_REF_DIR / f'NES_peptide_{pdb_id}_chain{peptide_chain_id}.pdb'
    io2.save(str(peptide_out), PeptideSelect())
    print(f"  wrote {peptide_out.name}")

    return {'pdb_id': pdb_id, 'label': label, 'start': start, 'end': end, 'sequence': seq,
            'crm1_chain': chain_a.id, 'peptide_chain': peptide_chain_id, 'numbering_offset': offset}


def main():
    CRM1_REF_DIR.mkdir(exist_ok=True)
    results = []
    # Added 3NC0 and 3GB8 -- NOT new independent NES sequences,
    # but independently-solved, separate crystal FORMS of the same two
    # complexes already covered by 3NBZ (Rev-NES) and 3GJX (Snurportin1)
    # respectively. Purpose: a reproducibility check, not new coverage --
    # does the geometric segment-identification recover the same sequence
    # in a second, independently-determined crystal of the same complex,
    # and does idealized_helix's finding (works okay for helical PKI-NES,
    # fails badly for these two non-canonical/extended cases) hold up in
    # an independent structure rather than being a fluke of one crystal
    # form? verify_groove_numbering below will loudly flag (not silently
    # misuse) any case where these structures don't share 3NBY/3GJX/3NBZ's
    # chain-lettering convention (chain A=CRM1, B=cargo/NES, C=Ran).
    # Added 5UWH, 5UWU, 5UWS -- three real CRM1-Ran-RanBP1-NES
    # structures from Fung et al. 2017 eLife 6:e23961 (a later, larger
    # structural survey than the Guttler/Dong-era structures above), chosen
    # for independence and diversity rather than convenience:
    #   - 5UWH: Paxillin NES -- another real, distinct cargo (not a
    #     Snurportin1-scaffold chimera like the Rev-NES structures).
    #   - 5UWU: SMAD4 NES -- likewise a distinct real cargo.
    #   - 5UWS: X11L2 NES -- the paper reports this one uses a DIFFERENT
    #     5-anchor spacing pattern (Phi0-XX-Phi1-XXX-Phi2-XX-Phi3-XXX-Phi4)
    #     than PHI_REGISTER_RE assumes, i.e. it's expected to likely come
    #     back 'none' or a low-quality 'partial' match under this project's
    #     current register model -- deliberately included as a stress test
    #     of that assumption, not because it's expected to pass.
    # None of these three are the CRM1(K579A) mutant complexes (5UWT/5UWW)
    # -- skipped those since the point mutation is an extra confound on top
    # of everything else being tested here.
    # Added 5DHF/5DIF -- hRio2NES and CPEB4NES (Fung, Fu, Chook
    # 2015, eLife 4:e10034). NOT another instance of the two binding modes
    # already covered (alpha-helical PKI-type, extended-proline Rev-type)
    # -- these bind the CRM1 groove in the OPPOSITE polarity ("minus"
    # direction, N/C-termini reversed relative to every other structure in
    # this set) while still using the same P0-P4 pockets. Genuinely a third
    # binding mode, not more data on the first two.
    # CAUTION: these depositions use an ENGINEERED S. cerevisiae CRM1
    # construct ("CRM1*": ScCRM1 1-1058, Delta377-413, V441D, groove region
    # 537DLTVK541 mutated to GLCEQ to mimic human CRM1 and open the groove
    # for RanBP1-bound crystallization) -- NOT wild-type human CRM1 like
    # every other structure here. find_crm1_chain()'s exact 19-residue
    # fingerprint (with offset search) will only match if ScCRM1's groove
    # is truly identical to human at those positions; if not, this will
    # cleanly report "no match" per its existing design, not silently
    # extract something wrong.
    for pdb_id, label in [('3GJX', 'Snurportin1 (Dong et al. 2009)'),
                           ('3NBZ', 'HIV-1 Rev NES (Guttler et al. 2010, crystal I)'),
                           ('3NC0', 'HIV-1 Rev NES (Guttler et al. 2010, crystal II -- replicate of 3NBZ)'),
                           ('3GB8', 'Snurportin1 (Dong et al. 2009, alternate crystal form -- replicate of 3GJX)'),
                           ('5UWH', 'Paxillin NES (Fung et al. 2017)'),
                           ('5UWU', 'SMAD4 NES (Fung et al. 2017)'),
                           ('5UWS', 'X11L2 NES (Fung et al. 2017, novel class 4 spacing pattern)'),
                           ('5DHF', 'hRio2 NES (Fung, Fu, Chook 2015 -- minus/reverse-direction binding mode, engineered ScCRM1*)'),
                           ('5DIF', 'CPEB4 NES (Fung, Fu, Chook 2015 -- minus/reverse-direction binding mode, engineered ScCRM1*)')]:
        r = process_structure(pdb_id, label)
        if r:
            results.append(r)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        offset_note = f", numbering offset {r['numbering_offset']:+d} (renumbered)" if r['numbering_offset'] else ""
        print(f"{r['pdb_id']} ({r['label']}): CRM1=chain{r['crm1_chain']}{offset_note}, "
              f"peptide=chain{r['peptide_chain']} residues {r['start']}-{r['end']} = {r['sequence']}")
    if len(results) < 2:
        print("\nWarning: Not all structures processed successfully -- check the warnings above "
              "before using whatever DID succeed.")


if __name__ == '__main__':
    main()
