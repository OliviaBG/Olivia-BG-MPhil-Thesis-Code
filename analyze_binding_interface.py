#!/usr/bin/env python3
"""
analyze_binding_interface.py
============================================================
Built for the 6-panel real-vs-simulated NES figure (HIV Rev, PKI, ACK1
rank 1). Two problems needed solving before any figure could trust its own
labels:

1. RESIDUE RELABELING. The simulated complex PDBs (produced by
   md_refinement.py's _run_crm1_docking) go through _truncate_to_groove_shell(),
   which strips CRM1 down to a shell around the peptide and, in doing so,
   loses the original residue numbering (confirmed directly --
   e.g. true Cys528 shows up as "chain G resi 61" in one output file). This
   script re-derives the TRUE CRM1 residue number for every CRM1 residue
   near the peptide by spatial nearest-CA matching against a reference file
   that still has correct numbering (crm1_reference/CRM1_Ran_only.pdb or
   the structure-matched crystal reference) -- CRM1 is heavily restrained
   during production (1000 kJ/mol/nm^2 positional restraint), so this is a
   reliable correspondence, not a guess.

2. HYDROGEN BONDS. Crystal structures (3NBY/3NBZ as deposited) have no
   resolved hydrogen atoms -- standard for X-ray at this resolution -- so
   H-bonds there are reported via the standard heavy-atom proxy (N/O...N/O
   distance < 3.5 A). The simulated complexes DO have explicit hydrogens
   (added by PDBFixer for the MD run), so those are scored with real D-H...A
   geometry (H...acceptor < 2.5 A, D-H...A angle > 120 degrees) -- a
   stricter, more defensible criterion, used whenever hydrogens are present.

USAGE:
    python3 analyze_binding_interface.py
(edit the PANELS list below to add/remove structures)
"""
import math
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

CONTACT_CUTOFF_A = 4.5       # heavy-atom cutoff to call a CRM1 residue "at the interface"
HBOND_HEAVY_CUTOFF_A = 3.5   # N/O...N/O proxy cutoff (no-hydrogen structures)
HBOND_H_CUTOFF_A = 2.5       # H...acceptor cutoff (hydrogen-containing structures)
HBOND_ANGLE_MIN_DEG = 120.0  # D-H...A angle minimum
CA_MATCH_MAX_A = 2.5         # max allowed CA-CA distance to accept a relabeling match


def parse_pdb(path):
    atoms = []
    for line in open(path):
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        try:
            atoms.append({
                'chain': line[21],
                'resnum': int(line[22:26]),
                'resname': line[17:20].strip(),
                'atomname': line[12:16].strip(),
                'element': line[76:78].strip() or line[12:14].strip().rstrip('0123456789'),
                'x': float(line[30:38]), 'y': float(line[38:46]), 'z': float(line[46:54]),
            })
        except ValueError:
            continue
    return atoms


def residues(atoms, chains=None):
    from collections import OrderedDict
    res = OrderedDict()
    for a in atoms:
        if chains and a['chain'] not in chains:
            continue
        res.setdefault((a['chain'], a['resnum']), []).append(a)
    return res


def get_ca(atom_list):
    for a in atom_list:
        if a['atomname'] == 'CA':
            return a
    return None


def dist(a, b):
    return math.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2 + (a['z'] - b['z']) ** 2)


def angle_deg(p1, p2, p3):
    """Angle at p2, formed by p1-p2-p3, in degrees."""
    v1 = (p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z'])
    v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z'])
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(a * a for a in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def build_relabel_map(ref_path, ref_chain, target_atoms, ca_match_max_a=CA_MATCH_MAX_A):
    """Returns {(target_chain, target_resnum): true_resnum} for every
    target residue whose CA lands within ca_match_max_a of some reference
    (correctly-numbered) CA in ref_chain."""
    ref_atoms = parse_pdb(ref_path)
    ref_res = residues(ref_atoms, chains={ref_chain})
    target_res = residues(target_atoms)
    target_cas = [(key, get_ca(alist)) for key, alist in target_res.items() if get_ca(alist)]

    mapping = {}
    for (rc, rn), alist in ref_res.items():
        rca = get_ca(alist)
        if not rca:
            continue
        best_key, best_d = None, 1e9
        for key, tca in target_cas:
            d = dist(rca, tca)
            if d < best_d:
                best_d, best_key = d, key
        if best_key and best_d <= ca_match_max_a:
            mapping[best_key] = rn
    return mapping


def find_hbonds(pep_atoms, crm1_atoms):
    """Peptide<->CRM1 H-bonds. Auto-detects whether hydrogens are present;
    uses proper D-H...A geometry if so, else the N/O...N/O heavy-atom proxy."""
    has_h = any(a['element'].upper() == 'H' or a['atomname'].startswith('H') for a in pep_atoms + crm1_atoms)
    donor_acceptor_elements = {'N', 'O'}

    bonds = []
    if not has_h:
        pep_da = [a for a in pep_atoms if a['element'].upper() in donor_acceptor_elements]
        crm1_da = [a for a in crm1_atoms if a['element'].upper() in donor_acceptor_elements]
        for pa in pep_da:
            for ca_ in crm1_da:
                d = dist(pa, ca_)
                if d <= HBOND_HEAVY_CUTOFF_A:
                    bonds.append({
                        'pep_res': (pa['chain'], pa['resnum'], pa['resname']),
                        'pep_atom': pa['atomname'],
                        'crm1_res': (ca_['chain'], ca_['resnum'], ca_['resname']),
                        'crm1_atom': ca_['atomname'],
                        'distance_a': round(d, 2),
                        'method': 'heavy-atom proxy (no H in structure)',
                    })
        return bonds

    # Hydrogen-aware path: build donor(heavy)-H pairs by within-residue proximity (<1.2 A)
    def build_donor_h_pairs(atom_list):
        by_res = residues(atom_list)
        pairs = []
        for key, alist in by_res.items():
            heavies = [a for a in alist if a['element'].upper() in donor_acceptor_elements]
            hs = [a for a in alist if a['element'].upper() == 'H']
            for h in hs:
                nearest_heavy, nearest_d = None, 1e9
                for hv in heavies:
                    d = dist(h, hv)
                    if d < nearest_d:
                        nearest_d, nearest_heavy = d, hv
                if nearest_heavy and nearest_d <= 1.2:
                    pairs.append((nearest_heavy, h))
        return pairs

    pep_donor_h = build_donor_h_pairs(pep_atoms)
    crm1_donor_h = build_donor_h_pairs(crm1_atoms)
    pep_acceptors = [a for a in pep_atoms if a['element'].upper() in donor_acceptor_elements]
    crm1_acceptors = [a for a in crm1_atoms if a['element'].upper() in donor_acceptor_elements]

    def score_direction(donor_h_pairs, acceptors, donor_is_peptide):
        found = []
        for donor, h in donor_h_pairs:
            for acc in acceptors:
                if donor['chain'] == acc['chain'] and donor['resnum'] == acc['resnum']:
                    continue  # skip same-residue
                dha = dist(h, acc)
                if dha > HBOND_H_CUTOFF_A:
                    continue
                ang = angle_deg(donor, h, acc)
                if ang < HBOND_ANGLE_MIN_DEG:
                    continue
                found.append({
                    'pep_res': (donor['chain'], donor['resnum'], donor['resname']) if donor_is_peptide else (acc['chain'], acc['resnum'], acc['resname']),
                    'pep_atom': donor['atomname'] if donor_is_peptide else acc['atomname'],
                    'crm1_res': (acc['chain'], acc['resnum'], acc['resname']) if donor_is_peptide else (donor['chain'], donor['resnum'], donor['resname']),
                    'crm1_atom': acc['atomname'] if donor_is_peptide else donor['atomname'],
                    'distance_a': round(dha, 2),
                    'angle_deg': round(ang, 1),
                    'donor': 'peptide' if donor_is_peptide else 'CRM1',
                    'method': 'D-H...A geometry',
                })
        return found

    bonds += score_direction(pep_donor_h, crm1_acceptors, donor_is_peptide=True)
    bonds += score_direction(crm1_donor_h, pep_acceptors, donor_is_peptide=False)
    return bonds


def analyze_panel(label, target_path, pep_chains, ref_crm1_path, ref_crm1_chain,
                   native_peptide_offset, crm1_already_correctly_numbered=False):
    print(f"\n{'=' * 100}\n{label}\n{'=' * 100}")
    target_atoms = parse_pdb(target_path)
    pep_atoms = [a for a in target_atoms if a['chain'] in pep_chains]
    crm1_atoms = [a for a in target_atoms if a['chain'] not in pep_chains]

    if crm1_already_correctly_numbered:
        relabel = None
    else:
        relabel = build_relabel_map(ref_crm1_path, ref_crm1_chain, crm1_atoms)

    # Find CRM1 residues at the interface (any heavy atom within CONTACT_CUTOFF_A of any peptide atom)
    crm1_res = residues(crm1_atoms)
    pep_res = residues(pep_atoms)
    contact_resnums = set()
    for key, alist in crm1_res.items():
        for a in alist:
            if a['element'].upper() == 'H':
                continue
            for pkey, palist in pep_res.items():
                for pa in palist:
                    if pa['element'].upper() == 'H':
                        continue
                    if dist(a, pa) <= CONTACT_CUTOFF_A:
                        contact_resnums.add(key)
                        break
                else:
                    continue
                break

    print(f"CRM1 groove residues within {CONTACT_CUTOFF_A} A of the peptide "
          f"({len(contact_resnums)} residues):")
    rows = []
    for key in sorted(contact_resnums, key=lambda k: k[1]):
        resname = crm1_res[key][0]['resname']
        if crm1_already_correctly_numbered:
            true_num = key[1]
            note = ''
        else:
            true_num = relabel.get(key)
            note = '' if true_num is not None else '  (no confident relabel match)'
        rows.append((key, resname, true_num, note))
        label_str = f"{resname}{true_num}" if true_num is not None else "???"
        print(f"  file: chain {key[0]} resi {key[1]:>4} {resname}   -> LABEL AS: {label_str}{note}")

    print(f"\nPeptide residues (native numbering, offset {native_peptide_offset:+d} from file's local numbering):")
    for key in sorted(pep_res.keys(), key=lambda k: k[1]):
        resname = pep_res[key][0]['resname']
        native_num = key[1] + native_peptide_offset
        print(f"  file: chain {key[0]} resi {key[1]:>3} {resname}   -> LABEL AS: {resname}{native_num}")

    print(f"\nHydrogen bonds (peptide <-> CRM1):")
    bonds = find_hbonds(pep_atoms, crm1_atoms)
    if not bonds:
        print("  none found at current thresholds")
    for b in bonds:
        pep_resname, pep_resnum = b['pep_res'][2], b['pep_res'][1]
        pep_native = pep_resnum + native_peptide_offset
        crm1_resname, crm1_filenum = b['crm1_res'][2], b['crm1_res'][1]
        crm1_key = (b['crm1_res'][0], b['crm1_res'][1])
        if crm1_already_correctly_numbered:
            crm1_true = crm1_filenum
        else:
            crm1_true = relabel.get(crm1_key, '?') if relabel else '?'
        extra = f" angle={b['angle_deg']}deg" if 'angle_deg' in b else ""
        print(f"  {pep_resname}{pep_native}.{b['pep_atom']}  <-->  {crm1_resname}{crm1_true}.{b['crm1_atom']}"
              f"   dist={b['distance_a']} A{extra}   [{b['method']}]")

    return {'contact_residues': rows, 'hbonds': bonds}


PANELS = [
    dict(label="1. HIV Rev -- REAL (3NBZ crystal)",
         target_path=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBZ_v2clean.pdb"),
         pep_chains=set(),  # filled below by merging with peptide file
         ref_crm1_path=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBZ_v2clean.pdb"),
         ref_crm1_chain='A',
         native_peptide_offset=69,  # local 6-14 -> native 75-83
         crm1_already_correctly_numbered=True,
         peptide_path=str(THIS_DIR / "crm1_reference" / "NES_peptide_3NBZ_chainB.pdb")),
    dict(label="2. HIV Rev -- SIMULATED (extended, 50ns)",
         target_path=str(THIS_DIR / "REV_NES_3NBZ_extended_50ns_complex.pdb"),
         pep_chains={'A'},
         ref_crm1_path=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBZ_v2clean.pdb"),
         ref_crm1_chain='A',
         native_peptide_offset=74,  # local 1-9 -> native 75-83
         crm1_already_correctly_numbered=False),
    dict(label="3. PKI -- REAL (3NBY crystal)",
         target_path=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBY_v2clean.pdb"),
         pep_chains=set(),
         ref_crm1_path=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBY_v2clean.pdb"),
         ref_crm1_chain='A',
         native_peptide_offset=33,  # local 4-13 -> native 37-46
         crm1_already_correctly_numbered=True,
         peptide_path=str(THIS_DIR / "crm1_reference" / "PKI_NES_peptide_3NBY_chainB_4-13.pdb")),
    dict(label="4. PKI -- SIMULATED (idealized helix, 50ns)",
         target_path=str(THIS_DIR / "PKI_NES_3NBY_idealized_helix_50ns_complex.pdb"),
         pep_chains={'A'},
         ref_crm1_path=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBY_v2clean.pdb"),
         ref_crm1_chain='A',
         native_peptide_offset=36,  # local 1-10 -> native 37-46
         crm1_already_correctly_numbered=False),
    dict(label="5/6. ACK1 rank 1 -- SIMULATED (continued trajectory, best-anchor-frame)",
         target_path=str(THIS_DIR / "ack1_rank1_idealized_helix_continued_20ns_best_anchor_frame_complex.pdb"),
         pep_chains={'A'},
         ref_crm1_path=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_only.pdb"),
         ref_crm1_chain='A',
         native_peptide_offset=477,  # local 1-10 -> native 478-487
         crm1_already_correctly_numbered=False),
]


def main():
    results = {}
    for p in PANELS:
        target_path = p['target_path']
        pep_chains = p['pep_chains']
        # For the two REAL crystal panels, merge in the separate peptide file
        # first (they were never in the same file -- see earlier discussion).
        if not pep_chains and 'peptide_path' in p:
            crm1_atoms = parse_pdb(target_path)
            pep_atoms_raw = parse_pdb(p['peptide_path'])
            combined_path = str(THIS_DIR / f"_tmp_combined_{Path(target_path).stem}.pdb")
            with open(target_path) as f1, open(p['peptide_path']) as f2, open(combined_path, 'w') as out:
                out.writelines(l for l in f1 if l.startswith(('ATOM', 'HETATM')))
                out.writelines(l for l in f2 if l.startswith(('ATOM', 'HETATM')))
            target_path = combined_path
            pep_chains = {a['chain'] for a in pep_atoms_raw}

        res = analyze_panel(
            p['label'], target_path, pep_chains, p['ref_crm1_path'], p['ref_crm1_chain'],
            p['native_peptide_offset'], p.get('crm1_already_correctly_numbered', False),
        )
        results[p['label']] = res

    print(f"\n{'=' * 100}\nDONE -- {len(results)} panels analyzed\n{'=' * 100}")


if __name__ == '__main__':
    main()
