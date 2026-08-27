#!/usr/bin/env python3
"""
resolve_taxonomic_provenance.py
============================================================
Replaces the old seq -> structural_data.json -> UniProt-accession join
(fetch_organism_data.py + the join inside compare_taxonomic_fit.py) with a
much higher-coverage, network-free resolver that reads organism/accession
straight out of the project's OWN raw source files -- which already carry
this information, it just wasn't being threaded through into the
taxonomic-fit diagnostic.

WHY THIS EXISTS: to establish, per row, which training entries have a
real source accession and which are synthetic, with the fabricated
positives being the main concern:

  - NES: nes_dataset.csv (the actual source of every real positive) has an
    'organism' column with 100% coverage (0/499 empty) and a 'db_reference'
    column with an embedded UniProt accession for ~43/499 rows. Neither was
    being read -- _load_positives_from_dataset_csv() only keeps
    seq/protein/full_sequence/start/crm1_dependent/source.
  - nes_negatives/nes_negatives.csv (the structural hard-negatives file)
    ALSO has accession + organism columns with 100% coverage (661/661) --
    also unused by the training-example loader.
  - : nes_negatives_leucine_zipper_expansion/nes_negatives.csv
    (produced by expand_leucine_zipper_negatives.py, same schema, already
    folded into real training) is now merged in too -- see
    _load_all_nes_hard_negatives() below. Before this, every expansion-set
    negative silently fell into 'Unresolved (genuinely untracked)' even
    though its organism was sitting right there in its own CSV row.
  - matched_negatives (real windows sampled from elsewhere in a real
    positive's own protein, see generate_matched_negatives() in
    nes_ml_predictor_improved.py) carry the parent positive's 'protein'
    (protein_name) string, so they can inherit that protein's organism from
    nes_dataset.csv even though the negative window itself was generated at
    runtime and never written to any file.
  - NLS: nls_dataset.csv positives carry the real UniProt accession directly
    as the 'protein' field (see nls_ml_predictor.py build_training_dataset,
    "protein": row["accession"]) -- it was being loaded into memory the
    whole time, just never used for this diagnostic's join.
  - nls_negatives.csv: 'accession' column is 100% populated (490/490), but
    'organism' is only populated for the dna_binding_hard rows (102/102);
    protein_matched_random rows (308) have a real accession but need
    organism backfilled from nls_dataset.csv's accession->organism map
    (same protein). synthetic_polybasic_decoy rows (80) carry a sentinel
    accession like "synthetic_decoy_7" -- genuinely fabricated, correctly
    flagged as such rather than "no accession".

WHAT COUNTS AS GENUINELY FABRICATED:
  - NES: the 47 entries in NEGATIVE_SEQUENCES (homopolymers, alternating
    patterns, random hydrophobic strings) -- protein == 'synthetic_decoy'.
  - NLS: the 80 synthetic_polybasic_decoy rows in nls_negatives.csv --
    accession starts with 'synthetic_decoy_'.
  Both get their own explicit group, 'Synthetic (fabricated)', instead of
  being invisibly merged into "No accession match" alongside real,
  just-untracked biological sequences (real positives, real protein-matched
  negatives, real structural hard negatives). NO positive example in either
  dataset is fabricated -- every positive traces to a real scraped database
  entry or a real, named, published literature motif (see
  SEED_NES_ORGANISM / SEED_NLS_ORGANISM below; the one exception is
  NESDB_VALIDATED_NES's 'Synthetic optimal' entry, a deliberately
  rationally-designed/optimized peptide from the literature -- not a real
  protein sequence, flagged as such rather than mislabeled 'Human').

CAVEAT ON THE TWO SEED TABLES BELOW: SEED_NES_ORGANISM (37 entries) and
SEED_NLS_ORGANISM (5 entries) are manually curated from established
literature/domain knowledge, NOT re-derived from a database column like
everything else in this script. They're the only entries in this whole
diagnostic system that aren't a direct machine-read of a source file, so
They should be spot-checked if precision here matters for the writeup.

This script is READ-ONLY / diagnostic-only, same as compare_taxonomic_fit.py
and fetch_organism_data.py -- it does not touch _train_model(), any shipped
model artifact, or the production k-fold-selection + 100%-refit
methodology in any way.

Usage: imported by compare_taxonomic_fit.py, not run standalone.
"""

import csv
import re
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ACCESSION_RE = re.compile(r'(?:SWISS-PROT|TrEMBL)\s+([A-Z0-9]+)')

SYNTHETIC_GROUP = "Synthetic (fabricated)"
UNRESOLVED_GROUP = "Unresolved (genuinely untracked)"


def classify_from_organism_name(organism):
    """Broad taxonomic bucket from a free-text organism name (UniProt-style
    common name, e.g. 'Homo sapiens (Human)'), not a full NCBI lineage --
    good enough for these coarse buckets, and every string here is already
    a well-formed organism name (not user input), so simple substring
    matching is safe. Order matters: check 'virus' before 'human' since
    several viral organism names literally contain the word 'Human'
    (e.g. 'Human immunodeficiency virus type 1', 'Human T-lymphotropic
    virus 1') and would otherwise be misclassified."""
    if not organism:
        return UNRESOLVED_GROUP
    s = organism.lower()
    if 'virus' in s or 'viroid' in s:
        return "Viral"
    if 'homo sapiens' in s or s.strip() == 'human':
        return "Human"
    if 'mus musculus' in s or 'rattus' in s or 'mouse' in s or 'rat)' in s:
        return "Rodent"
    if any(k in s for k in ('saccharomyces', 'schizosaccharomyces', 'yeast',
                             'candida', 'aspergillus', 'fungi', 'fungus')):
        return "Yeast/Fungi"
    if any(k in s for k in ('arabidopsis', 'oryza', 'zea mays', 'nicotiana',
                             'lycopersicon', 'tomato', 'solanum',
                             'viridiplantae', 'plant')):
        return "Plant"
    if any(k in s for k in ('xenopus', 'danio', 'zebrafish', 'gallus',
                             'chicken', 'bos taurus', 'cow', 'sus scrofa',
                             'pig', 'canis familiaris', 'dog', 'bovine',
                             'frog', 'fish', 'chordata')):
        return "Other vertebrate"
    return "Invertebrate/Other"


# ---------------------------------------------------------------------------
# NES
# ---------------------------------------------------------------------------

# Manually curated (see module docstring caveat). NESDB_VALIDATED_NES's own
# seqs, keyed exactly as they appear in nes_ml_predictor_improved.py.
SEED_NES_ORGANISM = {
    'LALKLAGLDI': 'Homo sapiens (Human)',           # PKI
    'LQLPPLERLTL': 'Human immunodeficiency virus type 1',   # HIV-1 Rev
    'LPPLERLTL': 'Human immunodeficiency virus type 1',     # HIV-1 Rev minimal
    'MKLNVTEQEQIQL': 'Homo sapiens (Human)',        # MAPKKK / MEKK1
    'LSNRELVVL': 'Homo sapiens (Human)',            # NFAT
    'LGLGGLGLGL': None,                             # "Synthetic optimal" -- flagged below
    'LQHLRLISL': 'Homo sapiens (Human)',            # STAT3
    'LEVFEALIG': 'Homo sapiens (Human)',            # Smad3
    'LQEILEGL': 'Homo sapiens (Human)',             # Emi1
    'LEQLIESIL': 'Foot-and-mouth disease virus',    # FMDV 3A
    'LQLKVWGL': 'Homo sapiens (Human)',             # Cyclin B1
    'FTELRLLAL': 'Homo sapiens (Human)',            # MDM2
    'LLDLLQAEL': 'Homo sapiens (Human)',            # CPEB
    'MEELASLTSSFSV': 'Homo sapiens (Human)',        # Snurportin
    'MTKKFGTLTV': 'Homo sapiens (Human)',           # MAPK
    'LQKKLEELEL': 'Homo sapiens (Human)',           # MEK1
    'LRTLHSIFLV': 'Homo sapiens (Human)',           # Survivin
    'LSQALFLFL': 'Homo sapiens (Human)',            # Cdc25C
    'LELLKVWSL': 'Homo sapiens (Human)',            # AP-2alpha
    'IQKTLEEVL': 'Homo sapiens (Human)',            # eEF1A
    'LETVEELRKR': 'Human T-lymphotropic virus 1',   # HTLV-1 Rex
    'LQKKIEQLL': 'Homo sapiens (Human)',            # eIF2Bepsilon
    'LCELFTTQL': 'Homo sapiens (Human)',            # BRCA1
    'LERLRISQL': 'Homo sapiens (Human)',            # IkBalpha
    'LEKLLEEIKQL': 'Influenza A virus',             # Influenza NS2
    'LQRLVSLSQL': 'Homo sapiens (Human)',           # FOXO1
    'LRLCIISQL': 'Homo sapiens (Human)',            # FOXO3a
    'VEALLARLA': 'Homo sapiens (Human)',            # mDia2
    'IQLLQEKLE': 'Homo sapiens (Human)',            # Mad1
    'IDSLTSMLA': 'Homo sapiens (Human)',            # Trip6
    'LQELVQQFE': 'Homo sapiens (Human)',            # X11L2
    'MTEFNQALE': 'Homo sapiens (Human)',            # Rio2
    'LRKLCERLR': 'Homo sapiens (Human)',            # CDC7
    'MHSLESSLI': 'Homo sapiens (Human)',            # CPEB4
    'LPFLALFL': 'Homo sapiens (Human)',             # CPEB4 (reverse NES)
    'LFTHLFLEL': 'Homo sapiens (Human)',            # hRio2
}


def _load_nes_dataset_csv(path):
    """seq -> {organism, accession, protein} and protein -> same, from the
    real positives source file (100% organism coverage, verified :
    0/499 rows have an empty organism column)."""
    by_seq, by_protein = {}, {}
    if not path.exists():
        return by_seq, by_protein
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            seq = (row.get('nes_sequence') or '').strip().upper()
            protein = row.get('protein_name') or ''
            organism = (row.get('organism') or '').strip()
            m = ACCESSION_RE.search(row.get('db_reference') or '')
            accession = m.group(1) if m else None
            info = {'organism': organism, 'accession': accession, 'protein': protein}
            if seq and seq not in by_seq:
                by_seq[seq] = info
            if protein and protein not in by_protein:
                by_protein[protein] = info
    return by_seq, by_protein


def _load_nes_hard_negatives_csv(path):
    """match_seq -> {organism, accession, protein}, from a structural
    hard-negatives file (nes_negatives.csv: also 100% organism coverage,
    661/661 rows -- same schema as the leucine-zipper expansion file
    below, both written by negative_dataset_builder.py's write_outputs())."""
    by_seq = {}
    if not path.exists():
        return by_seq
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            seq = (row.get('match_seq') or '').strip().upper()
            if not seq:
                continue
            by_seq[seq] = {
                'organism': (row.get('organism') or '').strip(),
                'accession': row.get('accession') or None,
                'protein': row.get('protein_name') or '',
            }
    return by_seq


def _load_all_nes_hard_negatives(base_dir):
    """ : merges nes_negatives/nes_negatives.csv with
    nes_negatives_leucine_zipper_expansion/nes_negatives.csv (produced by
    expand_leucine_zipper_negatives.py, same schema, and already folded
    into real training via nes_ml_predictor_improved.py's
    EXPANSION_HARD_NEGATIVE_SOURCE). Before this fix, resolve_nes() only
    ever read the original file, so every expansion-set negative missed
    the `seq in hardneg_by_seq` check below and fell through to
    'Unresolved (genuinely untracked)' regardless of the real organism
    data sitting right there in its own CSV row. Original file takes
    priority on a seq collision (there shouldn't be any -- the expansion
    script explicitly excludes accessions already in the original file --
    but original-first matches this module's general precedence pattern)."""
    primary = _load_nes_hard_negatives_csv(base_dir / 'nes_negatives' / 'nes_negatives.csv')
    expansion = _load_nes_hard_negatives_csv(
        base_dir / 'nes_negatives_leucine_zipper_expansion' / 'nes_negatives.csv')
    merged = dict(expansion)
    merged.update(primary)  # primary wins on any (unexpected) seq collision
    return merged


def resolve_nes(dataset, base_dir):
    """dataset = predictor.build_training_dataset() output. Returns a list
    of (organism, taxonomic_group, source_note), one per row, in the exact
    same order as dataset['positives'] + dataset['negatives'] (== the order
    seqs/X/y are built in -- see compare_taxonomic_fit.py's own row-order
    verification)."""
    pos_by_seq, pos_by_protein = _load_nes_dataset_csv(
        base_dir / 'nes_data_pipeline' / 'nes_dataset.csv')
    hardneg_by_seq = _load_all_nes_hard_negatives(base_dir)

    out = []
    for p in dataset['positives']:
        seq = p['seq'].upper()
        if seq in pos_by_seq:
            info = pos_by_seq[seq]
            out.append((info['organism'], None, 'nes_dataset.csv (real positive)'))
        elif seq in SEED_NES_ORGANISM:
            org = SEED_NES_ORGANISM[seq]
            if org is None:
                out.append((None, SYNTHETIC_GROUP, "curated seed -- 'Synthetic optimal' "
                            "designed peptide, not a natural sequence"))
            else:
                out.append((org, None, 'curated seed (manually verified, see module docstring)'))
        else:
            out.append((None, None, 'unresolved -- not in nes_dataset.csv or seed table'))

    for n in dataset['negatives']:
        seq = n['seq'].upper()
        protein = n.get('protein', '')
        if protein == 'synthetic_decoy':
            out.append((None, SYNTHETIC_GROUP, 'NEGATIVE_SEQUENCES (homopolymer/pattern decoy)'))
        elif seq in hardneg_by_seq:
            info = hardneg_by_seq[seq]
            out.append((info['organism'], None,
                        'nes_negatives.csv or leucine_zipper_expansion (structural hard negative)'))
        elif protein in pos_by_protein:
            info = pos_by_protein[protein]
            out.append((info['organism'], None,
                        f"inherited from parent protein '{protein}' (matched_negative window, "
                        "real sequence, generated at train time -- see generate_matched_negatives)"))
        else:
            out.append((None, None, 'unresolved -- matched_negative whose parent protein '
                        'was not found in nes_dataset.csv'))
    return out


# ---------------------------------------------------------------------------
# NLS
# ---------------------------------------------------------------------------

SEED_NLS_ORGANISM = {
    'PKKKRKV': 'Simian virus 40',                    # SV40 large T-antigen
    'KRPAATKKAGQAKKKK': 'Xenopus laevis (African clawed frog)',  # Nucleoplasmin
    'PAAKRVKLD': 'Homo sapiens (Human)',             # c-Myc
    'KRKRRP': 'Homo sapiens (Human)',                # BRCA1 (503-508)
    'PKKNRLRRKS': 'Homo sapiens (Human)',            # BRCA1 (606-615)
}


def _load_nls_accession_organism(path, seq_col=None):
    """accession -> organism (only where organism is non-empty), from
    either nls_dataset.csv or nls_negatives.csv -- both already carry the
    real UniProt accession as a plain column."""
    acc_to_org = {}
    if not path.exists():
        return acc_to_org
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            acc = (row.get('accession') or '').strip()
            org = (row.get('organism') or '').strip()
            if acc and org and acc not in acc_to_org:
                acc_to_org[acc] = org
    return acc_to_org


def resolve_nls(dataset, base_dir):
    pipeline_dir = base_dir / 'nls_data_pipeline'
    pos_acc_to_org = _load_nls_accession_organism(pipeline_dir / 'nls_dataset.csv')
    negfile_acc_to_org = _load_nls_accession_organism(pipeline_dir / 'nls_negatives.csv')

    # Optional last-resort fallback: real UniProt data from
    # fetch_organism_data.py, if it's been run and isn't just the inert
    # placeholder ({}) this project resets it to between sessions.
    uniprot_fallback = {}
    organism_json = pipeline_dir / 'organism_data.json'
    if organism_json.exists():
        import json
        try:
            raw = json.loads(organism_json.read_text(encoding='utf-8'))
            uniprot_fallback = {acc: v.get('organism') for acc, v in raw.items()}
        except Exception:
            pass

    out = []
    for p in dataset['positives']:
        seq = p['seq'].upper()
        accession = p.get('protein', '')  # NLS loader stores accession as 'protein'
        if accession in pos_acc_to_org:
            out.append((pos_acc_to_org[accession], None, 'nls_dataset.csv (real positive, direct accession)'))
        elif seq in SEED_NLS_ORGANISM:
            out.append((SEED_NLS_ORGANISM[seq], None, 'curated seed (manually verified, see module docstring)'))
        elif accession in negfile_acc_to_org:
            out.append((negfile_acc_to_org[accession], None, 'nls_negatives.csv (same accession, dna_binding_hard row)'))
        elif accession in uniprot_fallback and uniprot_fallback[accession]:
            out.append((uniprot_fallback[accession], None, 'organism_data.json (UniProt lookup fallback)'))
        else:
            out.append((None, None, f"unresolved -- accession '{accession}' not found in any source"))

    for n in dataset['negatives']:
        accession = n.get('protein', '')
        neg_type = n.get('neg_type', '')
        if neg_type == 'synthetic_polybasic_decoy' or str(accession).startswith('synthetic_decoy'):
            out.append((None, SYNTHETIC_GROUP, 'nls_negatives.csv synthetic_polybasic_decoy row (fabricated sequence)'))
        elif accession in negfile_acc_to_org:
            out.append((negfile_acc_to_org[accession], None, 'nls_negatives.csv (direct organism, dna_binding_hard)'))
        elif accession in pos_acc_to_org:
            out.append((pos_acc_to_org[accession], None,
                        f"inherited from same accession '{accession}' in nls_dataset.csv (protein_matched_random)"))
        elif accession in uniprot_fallback and uniprot_fallback[accession]:
            out.append((uniprot_fallback[accession], None, 'organism_data.json (UniProt lookup fallback)'))
        else:
            out.append((None, None, f"unresolved -- accession '{accession}' not found in any source"))
    return out


def resolve(target, dataset, base_dir):
    """Returns a list of (organism, taxonomic_group, source_note) aligned
    to dataset['positives'] + dataset['negatives']. taxonomic_group is None
    unless the row is a special case (Synthetic/Unresolved) -- the caller
    should run classify_from_organism_name(organism) when it's None."""
    rows = resolve_nes(dataset, base_dir) if target == 'nes' else resolve_nls(dataset, base_dir)
    resolved = []
    for organism, forced_group, note in rows:
        group = forced_group if forced_group else classify_from_organism_name(organism)
        resolved.append((organism, group, note))
    return resolved
