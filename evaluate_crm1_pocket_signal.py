#!/usr/bin/env python3
"""
evaluate_crm1_pocket_signal.py
============================================================
Empirically tests whether the two signals blended into app.py's
crm1_binding_affinity -- (1) the real fpocket-detected-cavity
compatibility score from pocket_detector.py, and (2) raw_hydrophobic_burial
(absolute A^2 of exposed anchor-residue surface) -- actually discriminate
real NES motifs from real hard negatives, using REAL AlphaFold structures
and REAL fpocket runs (not cached/reconstructed approximations).

WHY THIS EXISTS: the 70/30 blend weight between these two signals in
app.py was a guess. A follow-up test using only cached RSA data (no real
fpocket at all) found raw_hydrophobic_burial doesn't clearly help beyond
plain RSA for discriminating real NES from hard negatives -- but that test
never touched the fpocket cavity-detection term, since testing it needs
real structures and real fpocket runs. This
script does that properly: for a real sample of labeled NES positives
(nes_data_pipeline/nes_dataset.json) and hard negatives
(nes_negatives/nes_negatives.csv), it fetches each protein's real
AlphaFold structure, runs the EXACT SAME CRM1AwarePocketDetector class
app.py uses (same CRM1 reference, same shape/charge/composition/
residue-residue scoring you already fixed this project), computes both
signals for real, and reports:
  - each signal's own discriminative power (cross-validated AUC) against
    the real positive/negative label
  - a combined logistic regression fit, so the coefficients themselves
    give you a data-driven relative weight instead of a guessed split

HOW CRM1 COMPATIBILITY GETS COMPUTED HERE (so you can follow along): for
each candidate, this pulls whichever real fpocket-detected pocket(s)
overlap its residue span, and computes the exact same formula app.py uses
in unified_crm1_nes_analysis():
    fpocket_affinity = pocket['crm1_compatibility_score'] * pocket['hydrophobicity_score']
taking the max across every overlapping pocket, same as app.py's
`crm1_binding_affinity = max(crm1_binding_affinity, affinity)`.
crm1_compatibility_score itself is pocket_detector.py's real, multi-factor
score (volume + shape match to CRM1's verified 13-residue groove + charge
density + residue-composition match + residue-residue positional match) --
nothing here reimplements that logic, it's the live production code.

REQUIREMENTS (run locally; needs network access and locally installed tools):
  - real internet access to alphafold.ebi.ac.uk
  - fpocket installed on PATH (falls back to the weaker geometry-based
    detector otherwise, same as the live app would -- results will be
    noisier, but the script still runs)
  - the SAME venv as app.py (freesasa, biopython, scikit-learn, requests,
    numpy) -- see the earlier discussion this project about why mixing
    venvs here specifically can quietly change results
  - crm1_reference/CRM1_Ran_only.pdb or crm1.pdb present (whichever
    pocket_detector.py already uses)
  - nes_data_pipeline/nes_dataset.json and nes_negatives/nes_negatives.csv
    (already in this project)

USAGE:
    python3 evaluate_crm1_pocket_signal.py
    python3 evaluate_crm1_pocket_signal.py --neg-limit 200 --cache crm1_eval_results.json

This is slow (real structure downloads + real fpocket runs per protein,
maybe 10-60s each) and checkpoints to --cache after every example, so it's
safe to Ctrl-C and rerun the same command later to resume where it left
off -- already-evaluated (accession, start, end) triples are skipped. The
existing crm1_eval_results.json cache from an earlier run is still valid
and will be reused/extended, not discarded. correction: negatives used to be taken in CSV file order, which
happened to make the first 40 rows 87.5% leucine_zipper even though the
full 660-row pool is 90% coiled_coil -- an unrepresentative sample that
could by itself explain why fpocket/burial looked "backwards" (leucine
zippers are the hard-negative type most likely to present a clean,
fpocket-visible hydrophobic groove). load_negative_examples() now shuffles
with a fixed seed before sampling, --neg-limit defaults to 200 (up from
40) to actually cover both types, and the final report breaks positives
vs. leucine_zipper and positives vs. coiled_coil out separately.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / 'nes_data_pipeline'))

from structural_dataset_v2_pipeline import fetch_alphafold_pdb, MAX_ASA_TIEN2013  # noqa: E402
from pocket_detector import CRM1AwarePocketDetector  # noqa: E402
from consensus_accessibility import consensus_accessibility  # noqa: E402

PHI_ANCHORS = set('LMFIWV')
REFERENCE_A2 = 600.0

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

CRM1_REF_CANDIDATES = ['crm1_reference/CRM1_Ran_only.pdb', 'crm1.pdb']

ACCESSION_RE = re.compile(r'(?:SWISS-PROT|TrEMBL)\s+([A-Z0-9]+)')


def write_text_atomic_with_retry(path, text, retries=4, base_delay=1.5):
    """ : this script runs from /mnt/c/... under WSL (Windows
    drive mounted into Linux via the 9p protocol), which is well known to
    throw sporadic OSError('Input/output error') under sustained small-file
    write load -- exactly what this script does (a fresh PDB cache file per
    protein, plus a full results-checkpoint rewrite after every single
    example). A crash here used to kill an hours-long run outright, and a
    crash mid-write to the checkpoint file specifically could leave a
    truncated/corrupt JSON that then breaks the NEXT resume attempt too.
    Fixed two ways: (1) retry with backoff, since these errors are
    transient; (2) write to a temp file in the same directory and
    os.replace() it into place, so a failure never leaves the real target
    path partially written."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + f'.tmp{os.getpid()}')
    last_err = None
    for attempt in range(retries):
        try:
            tmp_path.write_text(text)
            os.replace(tmp_path, path)
            return
        except OSError as e:
            last_err = e
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    (transient write error on {path.name}: {e} -- retrying in {delay:.1f}s)")
                time.sleep(delay)
    raise last_err


def load_positive_examples(limit):
    """Real positive NES motifs with a resolvable accession. Two sources:
    (1) db_reference's embedded 'SWISS-PROT XXXXX'/'TrEMBL XXXXX' (NESbase
    entries only, ~42 of 307 real-coordinate entries), and (2)
    nesdb_resolved_accessions.json, produced by
    resolve_nesdb_accessions.py -- looks up the remaining ~265 NESdb
    entries (whose db_reference is just a protein-name label with no
    embedded accession at all) against real UniProt search, keeping only
    entries where the candidate's real sequence was confirmed to match
    this project's own full_sequence/nes_sequence exactly (see that
    script's docstring for why unverified name-only matches are excluded).
    If that file doesn't exist yet, this silently falls back to source (1)
    alone -- run resolve_nesdb_accessions.py first for the larger sample."""
    path = THIS_DIR / 'nes_data_pipeline' / 'nes_dataset.json'
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    resolved_path = THIS_DIR / 'nesdb_resolved_accessions.json'
    resolved_by_index = {}
    if resolved_path.exists():
        with open(resolved_path, encoding='utf-8') as f:
            for r in json.load(f):
                if r.get('accession') and 'unverified' not in (r.get('match_type') or ''):
                    resolved_by_index[r['index']] = r['accession']

    out = []
    for i, rec in enumerate(data):
        s, e = rec.get('nes_start'), rec.get('nes_end')
        if not (s and e):
            continue
        ref = rec.get('db_reference') or ''
        m = ACCESSION_RE.search(ref)
        accession = m.group(1) if m else resolved_by_index.get(i)
        if not accession:
            continue
        out.append({'label': 1, 'accession': accession, 'start': int(s), 'end': int(e)})
        if len(out) >= limit:
            break
    return out


def load_negative_examples(limit, seed=42):
    """ : nes_negatives.csv is grouped by feature_kind in file
    order -- 660 rows total, but the first 40 (what --limit 40 used to grab
    unmodified) are 35 leucine_zipper / 5 coiled_coil, while the FULL file
    is 68 leucine_zipper / 592 coiled_coil (90% coiled_coil). So the first
    evaluation run tested fpocket almost entirely against leucine-zipper
    decoys -- exactly the hard-negative type most likely to present a
    clean, fpocket-visible hydrophobic groove (regular heptad-repeat
    anchors along one helix face), and barely tested it against the
    coiled_coil majority at all. Shuffling with a fixed seed before taking
    `limit` gives a sample proportionate to the real negative pool instead
    of file order, and feature_kind is now carried through to the results
    so the two hard-negative types can be broken out separately.

    also reads nes_negatives_leucine_zipper_expansion/nes_negatives.csv
    if it exists (produced by expand_leucine_zipper_negatives.py) -- real,
    additional leucine-zipper hard negatives beyond the original 68/37-unique-
    protein pool, deliberately kept in a separate file (see that script's
    docstring) but merged in here for evaluation purposes, where oversampling
    the harder negative type is a legitimate, deliberate choice rather than
    an accidental one. Falls back to just the original file if it hasn't
    been run yet."""
    paths = [
        THIS_DIR / 'nes_negatives' / 'nes_negatives.csv',
        THIS_DIR / 'nes_negatives_leucine_zipper_expansion' / 'nes_negatives.csv',
    ]
    rows = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                acc = (row.get('accession') or '').strip()
                s, e = row.get('match_start'), row.get('match_end')
                if not (acc and s and e):
                    continue
                key = (acc, int(s), int(e))
                if key in seen:  # dedup in case a hit shows up in both files
                    continue
                seen.add(key)
                rows.append({
                    'label': 0, 'accession': acc, 'start': int(s), 'end': int(e),
                    'feature_kind': (row.get('feature_kind') or 'unknown').strip(),
                })
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def get_structure_bundle(accession, cache, pdb_cache_dir):
    """Fetch (or reuse a local cached copy of) accession's real AlphaFold
    structure, and compute real per-residue consensus RSA via the same
    consensus_accessibility() helper app.py/structural_dataset_v2_pipeline.py
    both already use. Returns None if the structure isn't available.
    Cached in-memory (per accession, since one protein can have several
    candidates) and on-disk (so a second run of this script doesn't
    re-download anything)."""
    if accession in cache:
        return cache[accession]

    cache_file = pdb_cache_dir / f'{accession}.pdb'
    if cache_file.exists():
        pdb_text = cache_file.read_text()
    else:
        print(f"    fetching {accession} from AlphaFold...")
        pdb_text = fetch_alphafold_pdb(accession)
        if pdb_text is None:
            print(f"    {accession}: no AlphaFold structure available")
            cache[accession] = None
            return None
        write_text_atomic_with_retry(cache_file, pdb_text)

    try:
        rows = consensus_accessibility(str(cache_file))
    except Exception as e:
        print(f"    {accession}: consensus_accessibility failed ({e})")
        cache[accession] = None
        return None
    if not rows:
        cache[accession] = None
        return None

    chain_id = sorted({r.chain for r in rows})[0]
    rows = sorted((r for r in rows if r.chain == chain_id), key=lambda r: r.resnum)
    residue_numbers = [r.resnum for r in rows]
    sequence = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in rows)
    rsa_by_resnum = {r.resnum: r.consensus_rsa for r in rows}
    resname_by_resnum = {r.resnum: r.resname for r in rows}

    bundle = (pdb_text, residue_numbers, sequence, rsa_by_resnum, resname_by_resnum)
    cache[accession] = bundle
    return bundle


def compute_features(detector, accession, start, end, struct_cache, pocket_cache, pdb_cache_dir):
    bundle = get_structure_bundle(accession, struct_cache, pdb_cache_dir)
    if bundle is None:
        return None
    pdb_text, residue_numbers, sequence, rsa_by_resnum, resname_by_resnum = bundle

    # Real fpocket detection, run ONCE per protein (cached) -- multiple
    # candidates on the same protein reuse the same detected pockets.
    if accession not in pocket_cache:
        try:
            pocket_cache[accession] = detector.detect_pockets(
                pdb_text, residue_numbers=residue_numbers, sequence=sequence
            )
        except Exception as e:
            print(f"    {accession}: detect_pockets failed ({e})")
            pocket_cache[accession] = []
    pockets = pocket_cache[accession]

    candidate_positions = set(range(start, end + 1))
    fpocket_affinity = 0.0
    subscores = None  # crm1_subscores of whichever pocket set fpocket_affinity
    detection_method = 'no_pockets' if not pockets else pockets[0].get('detection_method', 'unknown')
    for pocket in pockets:
        pocket_positions = set(pocket.get('residue_numbers') or [])
        if pocket_positions & candidate_positions:
            score = pocket.get('crm1_compatibility_score', 0)
            hydro = pocket.get('hydrophobicity_score', 0)
            affinity = score * hydro
            if affinity >= fpocket_affinity:
                fpocket_affinity = affinity
                subscores = pocket.get('crm1_subscores')

    # raw_hydrophobic_burial: identical formula to app.py -- real consensus
    # RSA * Tien max_ASA (reconstructs real absolute A^2), summed over Phi
    # anchor residues within the candidate span, scaled by the same 600 A^2
    # reference.
    raw_area = 0.0
    for pos in range(start, end + 1):
        if pos not in rsa_by_resnum:
            continue
        resname = resname_by_resnum.get(pos)
        aa = THREE_TO_ONE.get(resname)
        if aa in PHI_ANCHORS:
            raw_area += rsa_by_resnum[pos] * MAX_ASA_TIEN2013.get(resname, 200.0)
    burial_score = min(1.0, raw_area / REFERENCE_A2)

    return fpocket_affinity, burial_score, detection_method, subscores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=50,
                     help='How many positives to attempt (only ~42 total have a regex-resolvable '
                          'accession, so 50 just takes all of them)')
    ap.add_argument('--neg-limit', type=int, default=200,
                     help='How many negatives to attempt, RANDOMLY sampled (fixed seed) from the '
                          'full 660-row pool rather than file order -- the pool is 90% coiled_coil / '
                          '10% leucine_zipper, so a large-enough random sample is needed to actually '
                          'cover both hard-negative types. Real structure fetches + fpocket runs are '
                          'slow (10-60s each), so this will take a while; it resumes via --cache.')
    ap.add_argument('--cache', default='crm1_eval_results.json',
                     help='Where per-example results are checkpointed (resumable)')
    ap.add_argument('--pdb-cache-dir', default='crm1_eval_pdb_cache',
                     help='Directory to cache downloaded AlphaFold PDB files in')
    args = ap.parse_args()

    pdb_cache_dir = Path(args.pdb_cache_dir)
    pdb_cache_dir.mkdir(exist_ok=True)

    crm1_ref = next((p for p in CRM1_REF_CANDIDATES if (THIS_DIR / p).exists()), None)
    print(f"Using CRM1 reference: {crm1_ref or '(none found -- pocket scoring will be weaker)'}")
    # This is an offline batch script, not a live user request,
    # so it can afford to give fpocket much more time on large structures
    # than app.py's production default (60s) -- see the comment on
    # fpocket_timeout in pocket_detector.py's __init__ for why a fixed 60s
    # cap risks correlating detection_method with protein size (and
    # therefore with feature_kind, since coiled_coil-forming proteins skew
    # large) rather than with real pocket geometry.
    detector = CRM1AwarePocketDetector(
        crm1_reference_path=str(THIS_DIR / crm1_ref) if crm1_ref else None,
        fpocket_timeout=240,
    )

    results_path = Path(args.cache)
    results = []
    if results_path.exists():
        try:
            with open(results_path) as f:
                results = json.load(f)
            print(f"Resuming: {len(results)} examples already evaluated in {results_path}")
        except (json.JSONDecodeError, OSError) as e:
            backup = results_path.with_suffix(results_path.suffix + '.corrupt')
            print(f"    {results_path} is unreadable ({e}) -- saving it as {backup.name} "
                  f"and starting this cache fresh (structures/pockets already downloaded to "
                  f"{args.pdb_cache_dir}/ are untouched, so this only redoes the scoring step, "
                  f"not the slow fetches).")
            results_path.rename(backup)
            results = []
    # Drop (not just skip) any cached entry that predates
    # per-factor crm1_subscores tracking, so it gets recomputed and
    # re-appended cleanly below rather than skipped-but-incomplete or
    # duplicated. The underlying fpocket run isn't cached across script
    # invocations either (pocket_cache below is in-memory, per-run only),
    # so recomputing an old entry costs the same as a new one -- no
    # efficiency lost by dropping it here instead of trying to patch it
    # in place.
    n_before = len(results)
    results = [r for r in results if r.get('subscores') is not None]
    if n_before != len(results):
        print(f"Dropping {n_before - len(results)} cached entries from before subscore "
              f"tracking -- they'll be recomputed below.")
    done_keys = {(r['accession'], r['start'], r['end']) for r in results}

    positives = load_positive_examples(args.limit)
    negatives = load_negative_examples(args.neg_limit)
    examples = positives + negatives
    neg_kind_counts = {}
    for n in negatives:
        neg_kind_counts[n['feature_kind']] = neg_kind_counts.get(n['feature_kind'], 0) + 1
    print(f"\nAttempting {len(examples)} labeled examples "
          f"({len(positives)} positive, {len(negatives)} negative -- {neg_kind_counts})\n")

    struct_cache, pocket_cache = {}, {}
    for i, ex in enumerate(examples, 1):
        key = (ex['accession'], ex['start'], ex['end'])
        if key in done_keys:
            continue
        print(f"[{i}/{len(examples)}] {ex['accession']} {ex['start']}-{ex['end']} (label={ex['label']})")
        feats = compute_features(detector, ex['accession'], ex['start'], ex['end'],
                                  struct_cache, pocket_cache, pdb_cache_dir)
        if feats is None:
            continue
        fpocket_affinity, burial_score, detection_method, subscores = feats
        print(f"    fpocket_affinity={fpocket_affinity:.3f}  burial_score={burial_score:.3f}  "
              f"detection_method={detection_method}  subscores={subscores}")
        results.append({
            'accession': ex['accession'], 'start': ex['start'], 'end': ex['end'],
            'label': ex['label'], 'fpocket_affinity': fpocket_affinity, 'burial_score': burial_score,
            'feature_kind': ex.get('feature_kind', 'positive' if ex['label'] == 1 else 'unknown'),
            'detection_method': detection_method, 'subscores': subscores,
        })
        write_text_atomic_with_retry(results_path, json.dumps(results, indent=2))

    # ---- evaluation ----
    n_pos = sum(1 for r in results if r['label'] == 1)
    n_neg = sum(1 for r in results if r['label'] == 0)
    print(f"\n{'='*70}\nEvaluated {len(results)} real examples "
          f"({n_pos} positive, {n_neg} negative)\n{'='*70}")

    if n_pos < 5 or n_neg < 5:
        print("Not enough evaluated examples in both classes yet for a meaningful AUC -- "
              "run again (it'll resume) or raise --limit.")
        return

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    labels = np.array([r['label'] for r in results])
    fpocket = np.array([r['fpocket_affinity'] for r in results])
    burial = np.array([r['burial_score'] for r in results])
    cv = min(5, n_pos, n_neg)

    for name, sig in [('fpocket-based CRM1 pocket affinity', fpocket),
                       ('raw_hydrophobic_burial', burial)]:
        try:
            auc = cross_val_score(LogisticRegression(), sig.reshape(-1, 1), labels,
                                   cv=cv, scoring='roc_auc').mean()
        except Exception:
            auc = roc_auc_score(labels, sig)
        print(f"\n{name}:")
        print(f"  CV AUC = {auc:.3f}")
        print(f"  mean in positives = {sig[labels==1].mean():.3f}   "
              f"mean in negatives = {sig[labels==0].mean():.3f}")

    X = np.column_stack([fpocket, burial])
    try:
        auc_combined = cross_val_score(LogisticRegression(), X, labels, cv=cv, scoring='roc_auc').mean()
    except Exception:
        auc_combined = None
    print(f"\nCombined (both features): CV AUC = {auc_combined:.3f}" if auc_combined is not None else "")

    # ---- breakdown by negative feature_kind ----
    # Older cache entries (from before this field existed) fall back to
    # 'unknown'; positives are labeled 'positive' for symmetry.
    kinds = np.array([r.get('feature_kind') or ('positive' if r['label'] == 1 else 'unknown')
                       for r in results])
    print(f"\n{'-'*70}\nBreakdown: positives vs each hard-negative type separately\n"
          f"(tests whether the anti-correlation is driven by one negative type)\n{'-'*70}")
    pos_mask = labels == 1
    for kind in sorted(set(kinds[~pos_mask])):
        neg_mask = kinds == kind
        n_this_neg = neg_mask.sum()
        if n_this_neg < 5:
            print(f"\n{kind}: only {n_this_neg} examples, too few to report")
            continue
        sub_labels = np.concatenate([labels[pos_mask], labels[neg_mask]])
        print(f"\n{kind} (n={n_this_neg}) vs positives (n={pos_mask.sum()}):")
        for name, sig in [('fpocket_affinity', fpocket), ('burial_score', burial)]:
            sub_sig = np.concatenate([sig[pos_mask], sig[neg_mask]])
            try:
                auc = roc_auc_score(sub_labels, sub_sig)
            except Exception:
                auc = float('nan')
            print(f"    {name}: AUC={auc:.3f}  mean(positive)={sig[pos_mask].mean():.3f}  "
                  f"mean({kind})={sig[neg_mask].mean():.3f}")

    # ---- detection_method x feature_kind cross-tab ----
    # Checks whether real-fpocket-vs-geometry-fallback usage is confounded
    # with feature_kind/label (e.g. large coiled_coil proteins timing out
    # more often than smaller NES-containing proteins), which would muddy
    # any comparison between positives and negatives regardless of weights.
    methods = np.array([r.get('detection_method', 'unknown') for r in results])
    print(f"\n{'-'*70}\ndetection_method usage by group (checks for a size/timeout confound)\n{'-'*70}")
    for kind in sorted(set(kinds)):
        mask = kinds == kind
        n_group = mask.sum()
        if n_group == 0:
            continue
        from collections import Counter
        counts = Counter(methods[mask])
        pct = {k: f"{v}/{n_group} ({v/n_group:.0%})" for k, v in counts.items()}
        print(f"  {kind}: {pct}")

    # ---- per-factor breakdown of crm1_compatibility_score ----
    # Crm1_compatibility_score (pocket_detector.py's
    # _filter_for_crm1_compatibility) is a hand-weighted sum of 7 factors
    # (volume, hydrophobicity, shape match, druggability, charge density,
    # residue-composition match, residue-residue positional match) that
    # were never individually validated against real labels -- same
    # situation the fpocket-vs-burial blend was in before this file
    # existed. AUC is invariant to a positive linear rescaling, so testing
    # each RAW factor value (stored in pocket['crm1_subscores'] as of this
    # session) gives the same answer as testing its weighted contribution
    # would, without needing to guess at weights first. Only examples
    # where fpocket found a pocket overlapping the candidate at all have a
    # non-None subscores dict; each factor is further restricted to
    # examples where THAT factor was computable (e.g. shape_similarity
    # needs >=3 real coordinates AND a loaded CRM1 template).
    print(f"\n{'-'*70}\nPer-factor breakdown of crm1_compatibility_score\n"
          f"(which factors of the 7-part composite actually separate real NES\n"
          f" from hard negatives, tested individually against real labels)\n{'-'*70}")
    factor_names = ['volume_A3', 'hydrophobicity', 'shape_similarity', 'druggability',
                     'charge_score', 'composition_similarity', 'residue_residue_score']
    for factor in factor_names:
        vals, facs = [], []
        for r in results:
            sub = r.get('subscores')
            v = sub.get(factor) if sub else None
            if v is not None:
                vals.append(v)
                facs.append(r['label'])
        vals, facs = np.array(vals), np.array(facs)
        n_f_pos, n_f_neg = (facs == 1).sum(), (facs == 0).sum()
        if n_f_pos < 5 or n_f_neg < 5:
            print(f"  {factor}: only {n_f_pos} positive / {n_f_neg} negative examples with this "
                  f"factor computed -- too few to report")
            continue
        try:
            auc = roc_auc_score(facs, vals)
        except Exception:
            auc = float('nan')
        print(f"  {factor}: n={len(vals)} ({n_f_pos} pos / {n_f_neg} neg)  AUC={auc:.3f}  "
              f"mean(pos)={vals[facs==1].mean():.3f}  mean(neg)={vals[facs==0].mean():.3f}")

    # IMPORTANT: fit the logistic regression on STANDARDIZED features
    # (z-scored) before reading off coefficients as "relative importance".
    # Raw (unstandardized) coefficients aren't comparable across features on
    # different scales -- a feature spanning 0-1 needs a much LARGER
    # coefficient than one spanning 0-50 to have the same real effect, so
    # comparing raw magnitudes directly is invalid and was overstating
    # burial's importance in an earlier version of this analysis (before the
    # fpocket-side normalization bug in pocket_detector.py was fixed, too).
    X_scaled = StandardScaler().fit_transform(X)
    lr = LogisticRegression().fit(X_scaled, labels)
    c_fpocket, c_burial = lr.coef_[0]
    total = abs(c_fpocket) + abs(c_burial)
    print(f"\nLogistic regression coefficients on STANDARDIZED features (sign shows direction, "
          f"magnitude is now comparable across features): fpocket={c_fpocket:.3f}  burial={c_burial:.3f}")
    if total > 1e-9:
        print(f"Data-driven relative weight: fpocket {abs(c_fpocket)/total:.0%} / "
              f"raw_burial {abs(c_burial)/total:.0%}")
        print("(Compare this to app.py's current guessed 70%/30% split.)")


if __name__ == '__main__':
    main()
