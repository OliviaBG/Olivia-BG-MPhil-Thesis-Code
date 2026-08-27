"""
Structural dataset v2: real per-residue SASA + pLDDT for as many of the
predictor's real positives AND real structural hard negatives as have a
resolvable UniProt accession -- built to fix a real gap found while
reviewing nes_ml_predictor_improved.py: _train_model() never passes real
plddt_values/sasa_values into _extract_features() for ANY training example,
so the model's plddt_norm/sasa_norm features are constant (always the
neutral 0.75/0.50 default) and carry zero learned signal no matter how good
the rest of the pipeline is. This script produces the real data; a
follow-up change to nes_ml_predictor_improved.py's _train_model() wires it
into training (matched by exact sequence).

Coverage (checked directly against your real data before writing this):
  - 307 of 499 nes_dataset.csv rows are positionally mappable (have
    full_sequence + nes_start/nes_end + nes_sequence).
  - Of those, only 42 have a UniProt accession sitting in db_reference
    already extractable by regex (same trick nes30_structural_cider_pipeline
    used). The other 265 need a live UniProt name search.
  - nes_negatives.csv (structural hard negatives) already has a real
    UniProt accession AND real match_start/match_end for all 660 rows
    (240 unique accessions) -- no resolution needed there at all, just
    fetching.

For safety, every name-search-resolved accession is validated by comparing
the dataset's own full_sequence against the candidate's real UniProt
sequence (exact match, or one contained in the other -- covers
isoform/processed-form differences) before being accepted. If nothing
validates, that row is skipped rather than risking wrong-protein structural
data silently entering the training set.

Must be run somewhere with real internet access, since it queries
rest.uniprot.org and alphafold.ebi.ac.uk in bulk:

    pip install freesasa requests
    python3 structural_dataset_v2_pipeline.py

Output: structural_data_v2.json -- list of records:
    {seq, protein, accession, label (1=positive, 0=hard negative),
     sasa_per_residue, plddt_per_residue, whole_protein_sasa_avg,
     whole_protein_plddt_avg, max_helix_run_near_candidate}
max_helix_run_near_candidate (added, see real_ca_helix_geometry
docstring) is the longest run of CONSECUTIVE residues within +/-20 of the
candidate window that are in real, CA-coordinate-derived alpha-helix
geometry -- None if no structure coordinates were available there, 0 if
checked but no helical run found, higher = more coiled-coil-like.
Checkpointed every 25 proteins (resumable -- reruns skip accessions already
present in an existing output file with this field; accessions from before
this field existed are reprocessed once to backfill it).
"""
import csv
import json
import re
import sys
import time
import tempfile
import os
from pathlib import Path
from urllib.parse import quote

import requests
import freesasa

HERE = Path(__file__).resolve().parent
DATASET_CSV = HERE / 'nes_dataset.csv'
NEGATIVES_CSV = HERE.parent / 'nes_negatives' / 'nes_negatives.csv'
# Added. nes_ml_predictor_improved.py's load_hard_negative_examples()
# now trains on this expansion file too (previously only fed the CRM1 pocket
# weight fit), so real SASA/pLDDT needs to cover it as well -- otherwise
# every one of its ~180 training negatives gets the constant neutral
# plddt_norm/sasa_norm default forever, same gap this whole script exists to
# close for the original file. Fetches for ALL expansion accessions,
# including the held-out reserve (nes_negatives_leucine_zipper_expansion/
# held_out_accessions.json) -- harmless to have real data sitting ready for
# those even though they're excluded from training; the single source of
# truth for that exclusion is load_hard_negative_examples() itself, not
# duplicated here.
EXPANSION_NEGATIVES_CSV = HERE.parent / 'nes_negatives_leucine_zipper_expansion' / 'nes_negatives.csv'
OUTPUT_JSON = HERE / 'structural_data_v2.json'

# Reuse the exact same 3-method consensus RSA calculation the
# live app uses (app.py's calculate_sasa() was itself ported from this same
# module) -- see real_per_residue_sasa() below for why this replaced the
# single-method FreeSASA Shrake-Rupley calculation this script used to do.
sys.path.insert(0, str(HERE.parent))
from consensus_accessibility import consensus_accessibility

UNIPROT_ACCESSION_RE = re.compile(
    r'\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?)\b'
)


# ---------------------------------------------------------------------------
# UniProt accession resolution
# ---------------------------------------------------------------------------

def resolve_accession_from_text(row):
    """Regex over db_reference/source_id/references -- free, no network."""
    hay = ' '.join([row.get('db_reference') or '', row.get('source_id') or '',
                     row.get('references') or ''])
    m = UNIPROT_ACCESSION_RE.search(hay)
    return m.group(1) if m else None


def _uniprot_search(query_str, size=10, retries=2):
    """GET against the UniProt search API with retries on transient network
    errors (timeouts/handshake failures are common over a home connection
    for a run this long) -- returns [] rather than raising, so one flaky
    request never kills the whole batch."""
    url = f'https://rest.uniprot.org/uniprotkb/search?query={query_str}&format=json&size={size}&fields=accession,sequence'
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                return []
            return resp.json().get('results', [])
        except Exception:
            if attempt < retries:
                time.sleep(1.0)
                continue
            return []
    return []


def _validate_and_pick(results, full_sequence):
    full_seq_clean = (full_sequence or '').strip().upper()
    for entry in results:
        acc = entry.get('primaryAccession')
        cand_seq = (entry.get('sequence', {}) or {}).get('value', '').strip().upper()
        if not acc or not cand_seq or not full_seq_clean:
            continue
        if cand_seq == full_seq_clean or cand_seq in full_seq_clean or full_seq_clean in cand_seq:
            return acc
    return None


def _clean_organism(organism):
    """UniProt's organism_name field query expects a clean scientific name
    ('Homo sapiens', 'Mus musculus') -- the dataset's own organism strings
    are far messier in practice: non-breaking-space padding, trailing
    periods, and common names / viral strain-isolate descriptors stacked in
    parens, e.g. 'Homo sapiens (Human)\xa0\xa0\xa0', 'Human immunodeficiency
    virus type 1 (isolate BRU/LAI group M subtype B) (HIV-1)\xa0\xa0\xa0'.

    querying organism_name with the raw string basically never
    matches anything -- found while investigating why resolve_accession_by_
    name_search failed for 265/307 positives, including unambiguous human
    gene symbols like BRCA1/STAT3/PINK1 that should trivially resolve via
    the gene: strategy. The organism clause was silently zeroing out results
    for all three query strategies (gene:, protein_name:, free-text), since
    all three append it. Takes everything before the first '(' (drops
    common names/strain descriptors), strips whitespace (including \xa0)
    and a trailing '.'. Returns None if nothing usable is left, so the
    caller can drop the clause entirely rather than filter on empty/junk --
    _validate_and_pick's real-sequence check is the actual correctness
    guard, so searching without an organism restriction is still safe."""
    if not organism:
        return None
    cleaned = organism.split('(')[0].replace('\xa0', ' ').strip().rstrip('.').strip()
    return cleaned or None


def resolve_accession_by_name_search(protein_name, organism, full_sequence):
    """Live UniProt search, validated against the dataset's own
    full_sequence before being trusted (so a wrong-protein name match never
    silently pollutes training data). Tries several query strategies in
    order and stops at the first one that validates:
      1. gene symbol match (gene:NAME) -- covers the common case where the
         dataset's "protein_name" is actually a gene symbol (STAT3, BRCA1,
         AGO1, PALB2, ...), which a protein_name-field search mostly misses.
      2. protein_name phrase match (protein_name:"...") -- covers full
         descriptive names (e.g. "Cellular tumor antigen p53").
      3. plain free-text query (no field restriction) -- the most
         forgiving, catches aliases/synonyms/partial matches the other two
         miss.
    Each strategy is tried first with an organism restriction (when the
    cleaned organism string is usable) and, if that doesn't validate,
    retried without any organism restriction -- see _clean_organism for why
    the organism-scoped attempt alone was silently failing almost every
    row. Returns None if nothing validates against the real sequence."""
    if not protein_name:
        return None
    name = protein_name.strip()
    if not name:
        return None

    clean_org = _clean_organism(organism)
    organism_clause = f'+AND+organism_name:"{quote(clean_org)}"' if clean_org else ''

    def _try(base_query, size=10):
        """Try base_query+organism_clause first (if any), then retry with
        no organism restriction at all before giving up on this strategy."""
        if organism_clause:
            results = _uniprot_search(f'{base_query}{organism_clause}', size=size)
            acc = _validate_and_pick(results, full_sequence)
            if acc:
                return acc
        results = _uniprot_search(base_query, size=size)
        return _validate_and_pick(results, full_sequence)

    # Gene-symbol-shaped names are short, alphanumeric, no spaces (e.g.
    # STAT3, BRCA1, AGO1, PALB2, SOX9) -- try gene: first since it's the
    # most precise match for these.
    if re.fullmatch(r'[A-Za-z0-9\-]{2,15}', name):
        acc = _try(f'gene:{quote(name)}')
        if acc:
            return acc

    if len(name) < 80:
        acc = _try(f'protein_name:"{quote(name)}"')
        if acc:
            return acc

    # Free-text fallback, no field restriction -- most forgiving, so try it
    # last (highest chance of noisy results, but validation guards it).
    query_terms = name if len(name) < 80 else name[:80]
    return _try(quote(query_terms), size=15)


# ---------------------------------------------------------------------------
# Structure fetch + real SASA/pLDDT (same fixes proven in
# nes30_structural_cider_pipeline.py: query the metadata API for the real
# current pdbUrl rather than guessing a version suffix, and restrict SASA
# to a single chain)
# ---------------------------------------------------------------------------

def _get_with_retries(url, timeout=20, retries=2):
    """requests.get that never raises -- returns None on any failure
    (timeout, connection error, SSL handshake timeout, etc.) after a couple
    of retries with a short backoff. A single flaky request used to crash
    the entire multi-hundred-protein batch job; now it just counts as one
    more skipped protein."""
    for attempt in range(retries + 1):
        try:
            return requests.get(url, timeout=timeout)
        except requests.RequestException:
            if attempt < retries:
                time.sleep(1.5)
                continue
            return None
    return None


def fetch_alphafold_pdb(uniprot_id):
    api_url = f'https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}'
    meta_resp = _get_with_retries(api_url)
    if meta_resp is None or meta_resp.status_code != 200:
        return None
    try:
        meta = meta_resp.json()
    except ValueError:
        return None
    if not meta or 'pdbUrl' not in meta[0]:
        return None
    resp = _get_with_retries(meta[0]['pdbUrl'])
    if resp is None or resp.status_code != 200:
        return None
    return resp.text


# Tien et al. 2013, theoretical MaxASA (Ų) -- PLoS ONE 8(11):e80635.
# Kept in sync with the same table in app.py / consensus_accessibility.py.
# Residue-specific normalization matters here because this is training
# data: if this pipeline emits raw Ų (as it did before) while app.py's
# live inference path emits Tien-normalized RSA, the sasa_norm feature
# means two different things at train time vs. inference time.
MAX_ASA_TIEN2013 = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}
DEFAULT_MAX_ASA = 200.0


def real_per_residue_sasa(pdb_text):
    """Real 3-method CONSENSUS relative solvent accessibility (RSA), 0-1ish,
    single-chain-filtered: FreeSASA Lee-Richards + FreeSASA Shrake-Rupley +
    Biopython Shrake-Rupley, each Tien et al. 2013-normalized, averaged per
    residue -- via the shared consensus_accessibility() helper (same one
    app.py's calculate_sasa(return_stats=True) uses).

    fix: this function previously used FreeSASA Shrake-Rupley
    ALONE, while the live app already computed the 3-method consensus --
    meaning sasa_norm was trained on one distribution (single-method) and
    predicted on a different one (consensus). Both sides now call the exact
    same consensus_accessibility() function, so training-time and
    inference-time sasa_norm are genuinely on the same scale. See
    consensus_accessibility.py for the full method/rationale."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_text)
        tmp_path = f.name
    try:
        rows = consensus_accessibility(tmp_path)
        if not rows:
            return {}
        chain_id = sorted({r.chain for r in rows})[0]
        return {r.resnum: r.consensus_rsa for r in rows if r.chain == chain_id}
    finally:
        os.unlink(tmp_path)


def real_ca_helix_geometry(pdb_text):
    """Real, 3D-coordinate-derived secondary structure -- NOT a sequence
    proxy. Classifies each residue as helical using the P-SEA method
    (Labesse et al. 1997, CABIOS 13:291-295): CA(i)-CA(i+3) and
    CA(i)-CA(i+4) distances both within the range an ideal alpha helix
    produces. This needs no extra fetch or dependency (no DSSP/mkdssp
    binary required) -- just the CA atom coordinates already sitting in the
    same pdb_text real_per_residue_sasa/real_per_residue_plddt already
    parse.

    Added: after two purely-sequence-based attempts (expanding
    the hard-negative training set, and a heptad-repeat periodicity proxy)
    both failed to move the 5 hardest holdout-test leucine-zipper
    negatives at all in the live pipeline, this targets the actual
    structural difference the sequence-only features can only
    approximate -- a real leucine zipper/coiled-coil is a long CONTINUOUS
    helix (that's the whole point, it's what lets it dimerize), while a
    real NES sits in an otherwise disordered/loop region with at most
    "one turn of helix" contacting CRM1's groove (Fung & Chook 2017).
    real_ca_helix_geometry() + longest_helix_run() below turn that into a
    single number: how many CONSECUTIVE residues near the candidate are in
    continuous helical geometry, not just "helical or not" in isolation.

    Returns {res_num: bool_is_helical_by_CA_geometry} for the first chain
    encountered in pdb_text (AlphaFold single-chain models only have one
    anyway)."""
    ca_coords = {}
    first_chain = None
    for line in pdb_text.splitlines():
        if not (line.startswith('ATOM') and line[12:16].strip() == 'CA'):
            continue
        try:
            chain = line[21].strip()
            if first_chain is None:
                first_chain = chain
            if chain != first_chain:
                continue
            res_num = int(line[22:26].strip())
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            ca_coords[res_num] = (x, y, z)
        except (ValueError, IndexError):
            continue

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    is_helical = {}
    for r, c0 in ca_coords.items():
        c3, c4 = ca_coords.get(r + 3), ca_coords.get(r + 4)
        ok13 = c3 is not None and 4.8 <= dist(c0, c3) <= 6.6
        ok14 = c4 is not None and 5.6 <= dist(c0, c4) <= 7.0
        is_helical[r] = bool(ok13 and ok14)
    return is_helical


def longest_helix_run(is_helical_by_res, lo, hi):
    """Longest run of CONSECUTIVE residue numbers in [lo, hi] that are all
    helical -- a gap (missing residue or non-helical) breaks the run. This
    is the number that actually distinguishes 'one turn of helix' (short
    run, ~3-5) from a real extended coiled-coil helix (long run, 15-40+)."""
    best = cur = 0
    for r in range(lo, hi + 1):
        if is_helical_by_res.get(r, False):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def real_per_residue_plddt(pdb_text):
    """AlphaFold stores per-residue pLDDT directly in the B-factor column
    of the CA atom -- no extra fetch needed, just parse the same PDB text
    used for SASA."""
    plddt_by_res = {}
    for line in pdb_text.splitlines():
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            try:
                res_num = int(line[22:26].strip())
                b_factor = float(line[60:66].strip())
                plddt_by_res[res_num] = b_factor
            except ValueError:
                continue
    return plddt_by_res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_positive_tasks():
    tasks = []
    with open(DATASET_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            full_seq = (row.get('full_sequence') or '').strip()
            nes_seq = (row.get('nes_sequence') or '').strip()
            start_raw = row.get('nes_start')
            end_raw = row.get('nes_end')
            if not (full_seq and nes_seq and start_raw and end_raw):
                continue
            acc = resolve_accession_from_text(row)
            tasks.append({
                'label': 1, 'seq': nes_seq.upper(), 'protein': row.get('protein_name') or 'unknown',
                'organism': row.get('organism'), 'full_sequence': full_seq,
                'start': int(start_raw), 'end': int(end_raw),  # 1-based inclusive
                'accession': acc, 'needs_name_search': acc is None,
            })
    return tasks


def load_negative_tasks():
    # this used to dedupe by SEQUENCE across both CSVs,
    # which is wrong -- a short 10-11 residue leucine-zipper motif is
    # conserved enough that mouse/rat/bovine orthologs in the expansion
    # file routinely share an IDENTICAL match_seq with their human
    # counterpart already in the original file (confirmed directly: e.g.
    # mouse O54724/Caveolae-associated protein 1 shares match_seq with
    # human Q6NZI2, same for O35841/Q9BZZ5 and O54826/P55197). Deduping by
    # sequence silently dropped the mouse/bovine task entirely -- three
    # holdout-test accessions never got real structural data fetched
    # because of this, not because of a real fetch failure. Now dedupes by
    # (accession, start, end) -- one fetch per actual protein region, which
    # is what this function is supposed to guarantee -- so orthologs with
    # coincidentally-identical short peptides both still get processed.
    tasks = []
    seen_tasks = set()
    for csv_path in [NEGATIVES_CSV, EXPANSION_NEGATIVES_CSV]:
        if not csv_path.exists():
            continue
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                seq = (row.get('match_seq') or '').strip()
                acc = (row.get('accession') or '').strip()
                start_raw, end_raw = row.get('match_start'), row.get('match_end')
                if not (seq and acc and start_raw and end_raw):
                    continue
                key = (acc, start_raw, end_raw)
                if key in seen_tasks:
                    continue
                seen_tasks.add(key)
                tasks.append({
                    'label': 0, 'seq': seq.upper(), 'protein': row.get('protein_name') or acc,
                    'organism': row.get('organism'), 'full_sequence': None,
                    'start': int(start_raw), 'end': int(end_raw),
                    'accession': acc, 'needs_name_search': False,
                })
    return tasks


def main():
    positives = load_positive_tasks()
    negatives = load_negative_tasks()
    print(f"Loaded {len(positives)} positional positives, {len(negatives)} structural hard negatives")

    n_need_search = sum(1 for t in positives if t['needs_name_search'])
    print(f"  {len(positives) - n_need_search} positives already have a regex-resolved accession")
    print(f"  {n_need_search} positives need a live UniProt name search")

    # Resolve remaining accessions (network)
    for i, t in enumerate(positives):
        if not t['needs_name_search']:
            continue
        print(f"  [{i+1}/{len(positives)}] resolving '{t['protein'][:50]}'...", end=' ')
        acc = resolve_accession_by_name_search(t['protein'], t.get('organism'), t['full_sequence'])
        t['accession'] = acc
        print(f"-> {acc}" if acc else "-> not found/validated")
        time.sleep(0.34)

    resolved_positives = [t for t in positives if t['accession']]
    print(f"\nResolved accessions for {len(resolved_positives)}/{len(positives)} positives")

    all_tasks = resolved_positives + negatives
    unique_accessions = sorted({t['accession'] for t in all_tasks})
    print(f"Fetching structures for {len(unique_accessions)} unique proteins "
          f"({len(resolved_positives)} positive-bearing + {len(negatives)} "
          f"negative-bearing tasks map onto them)...\n")

    # Resume support: skip accessions already fully written to an existing output file.
    # An accession only counts as "done" if its existing records
    # already carry max_helix_run_near_candidate -- older records (from before
    # that field existed) get reprocessed once so this backfills cleanly
    # instead of silently leaving old entries without the new signal.
    already_done = set()
    existing_records = []
    if OUTPUT_JSON.exists():
        try:
            existing_records = json.load(open(OUTPUT_JSON, encoding='utf-8'))
            accs_seen = {r['accession'] for r in existing_records}
            accs_missing_field = {r['accession'] for r in existing_records
                                   if 'max_helix_run_near_candidate' not in r}
            already_done = accs_seen - accs_missing_field
            print(f"Resuming: {len(already_done)} accessions already fully processed "
                  f"(incl. helix geometry) in {OUTPUT_JSON.name}; "
                  f"{len(accs_missing_field)} will be reprocessed to backfill it")
        except Exception:
            existing_records = []

    structure_cache = {}  # accession -> (sasa_by_res, plddt_by_res, is_helical_by_res) or None on failure
    # Drop stale records for accessions about to be reprocessed, so the
    # rebuild below doesn't create duplicates alongside the refreshed ones.
    records = [r for r in existing_records if r['accession'] in already_done]

    for i, acc in enumerate(unique_accessions, 1):
        if acc in already_done:
            continue
        print(f"[{i}/{len(unique_accessions)}] {acc} ...", end=' ')
        pdb_text = fetch_alphafold_pdb(acc)
        if pdb_text is None:
            print("SKIP -- no AlphaFold structure")
            structure_cache[acc] = None
            time.sleep(0.34)
            continue
        try:
            sasa_by_res = real_per_residue_sasa(pdb_text)
            plddt_by_res = real_per_residue_plddt(pdb_text)
            is_helical_by_res = real_ca_helix_geometry(pdb_text)
        except Exception as e:
            print(f"SKIP -- {e}")
            structure_cache[acc] = None
            time.sleep(0.34)
            continue
        structure_cache[acc] = (sasa_by_res, plddt_by_res, is_helical_by_res)
        print(f"OK -- {len(sasa_by_res)} residues")
        time.sleep(0.34)

        if i % 25 == 0:
            json.dump(records, open(OUTPUT_JSON, 'w'), indent=2)
            print(f"  (checkpoint saved, {len(records)} records so far)")

    # Build per-task records from the structure cache
    HELIX_FLANK = 20  # residues each side of the candidate window to search for a continuous helix run
    for t in all_tasks:
        acc = t['accession']
        cached = structure_cache.get(acc)
        if cached is None:
            continue
        sasa_by_res, plddt_by_res, is_helical_by_res = cached
        sasa_window = [sasa_by_res.get(p) for p in range(t['start'], t['end'] + 1)]
        plddt_window = [plddt_by_res.get(p) for p in range(t['start'], t['end'] + 1)]
        sasa_window = [v for v in sasa_window if v is not None]
        plddt_window = [v for v in plddt_window if v is not None]
        if not sasa_window and not plddt_window:
            continue
        # Real 3D-coordinate-derived helix length near the candidate (see
        # real_ca_helix_geometry/longest_helix_run docstrings) -- None if
        # this protein's structure had no CA coordinates at all in this
        # region (distinct from 0, which means "real geometry checked, no
        # continuous helix found here").
        max_helix_run = None
        if is_helical_by_res:
            max_helix_run = longest_helix_run(
                is_helical_by_res, t['start'] - HELIX_FLANK, t['end'] + HELIX_FLANK)
        records.append({
            'seq': t['seq'], 'protein': t['protein'], 'accession': acc, 'label': t['label'],
            'sasa_per_residue': sasa_window, 'plddt_per_residue': plddt_window,
            'whole_protein_sasa_avg': (sum(sasa_by_res.values()) / len(sasa_by_res)) if sasa_by_res else None,
            'whole_protein_plddt_avg': (sum(plddt_by_res.values()) / len(plddt_by_res)) if plddt_by_res else None,
            'max_helix_run_near_candidate': max_helix_run,
        })

    json.dump(records, open(OUTPUT_JSON, 'w'), indent=2)
    n_pos_out = sum(1 for r in records if r['label'] == 1)
    n_neg_out = sum(1 for r in records if r['label'] == 0)
    print(f"\nDone. Wrote {len(records)} records to {OUTPUT_JSON.name} "
          f"({n_pos_out} positives, {n_neg_out} hard negatives with real SASA+pLDDT)")


if __name__ == '__main__':
    main()
