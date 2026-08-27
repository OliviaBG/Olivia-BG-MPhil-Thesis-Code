#!/usr/bin/env python3
"""
resolve_nesdb_accessions.py
============================================================
Resolves real UniProt accessions for the 262 NESdb-sourced positives in
nes_data_pipeline/nes_dataset.json that evaluate_crm1_pocket_signal.py's
load_positive_examples() currently can't use.

WHY THIS EXISTS: nes_dataset.json has 307 entries with real nes_start/
nes_end, but evaluate_crm1_pocket_signal.py's ACCESSION_RE only resolves
42 of them (all 'source': 'NESbase', whose db_reference looks like
'SWISS-PROT P12345'). The other 265 are 'source': 'NESdb', whose
db_reference is just a protein-name label ('Aryl hydrocarbon receptor
\xa0\xa0\xa0\n UniProt\n \xa0\xa0\xa0') with NO embedded accession at all --
the word "UniProt" is a link-destination label from the scrape, not the ID
itself. There's no shortcut here; the accession genuinely isn't in the
local data and has to be looked up.

APPROACH: for each unresolved NESdb entry, query UniProt's REST search API
by cleaned protein_name + organism, then VERIFY every candidate by
comparing its real fetched sequence against this project's own
full_sequence field (already present in nes_dataset.json) -- so a match
here isn't a name-similarity guess, it's a confirmed identical sequence.
Only two match types are trusted enough to use for structure fetching:
  - exact_full_sequence: candidate's full UniProt sequence == our
    full_sequence exactly. Highest confidence -- same protein, same
    numbering, nes_start/nes_end from nes_dataset.json are safe to reuse
    as-is.
  - position_verified_substring: full sequences differ (isoform numbering
    etc.) but the candidate's sequence contains nes_sequence at EXACTLY
    the same start/end coordinates recorded locally. Still safe to reuse
    the coordinates.
A third type, substring_only (nes_sequence appears in the candidate's
sequence but not at the recorded position), is logged but NOT considered
resolved -- using it would risk pulling the wrong window out of the
AlphaFold structure, which is exactly the class of bug this whole project
has been hunting down all session. Better to leave it unresolved than
silently misalign a candidate.

REQUIREMENTS: real internet access to rest.uniprot.org (confirmed reachable
and returning correct results when tested manually -- e.g. "Aryl
hydrocarbon receptor" + "Homo sapiens" correctly resolves to P35869).
Run this where there is real network access.

USAGE:
    python3 resolve_nesdb_accessions.py
    python3 resolve_nesdb_accessions.py --cache nesdb_resolved_accessions.json

Checkpoints after every entry (safe to Ctrl-C / resume). Polite ~0.4s
delay between requests (UniProt is a shared public resource).
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

THIS_DIR = Path(__file__).resolve().parent
DATASET_PATH = THIS_DIR / 'nes_data_pipeline' / 'nes_dataset.json'
ALREADY_RESOLVED_RE = re.compile(r'(?:SWISS-PROT|TrEMBL)\s+([A-Z0-9]+)')

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def write_json_atomic_with_retry(path, obj, retries=4, base_delay=1.5):
    """Same pattern as evaluate_crm1_pocket_signal.py's
    write_text_atomic_with_retry -- temp file + os.replace so a failed
    write never corrupts the real cache file, with backoff for transient
    filesystem hiccups."""
    path = Path(path)
    text = json.dumps(obj, indent=2)
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
                time.sleep(base_delay * (2 ** attempt))
    raise last_err


def clean_protein_name(raw):
    """'Aryl hydrocarbon receptor \xa0\xa0\xa0\n UniProt\n \xa0\xa0\xa0'
    -> 'Aryl hydrocarbon receptor'. The scrape left the link-destination
    label ('UniProt') and non-breaking-space padding stuck to the real
    name; only the text before the first newline is the actual name."""
    if not raw:
        return ''
    first_line = raw.split('\n')[0]
    cleaned = unicodedata.normalize('NFKC', first_line).replace('\xa0', ' ')
    return ' '.join(cleaned.split()).strip()


def clean_organism(raw):
    """'Rattus norvegicus (Rat) \xa0\xa0\xa0' -> 'Rattus norvegicus' --
    UniProt's organism_name field matches scientific names best; the
    parenthetical common name and NBSP padding are scrape artifacts."""
    if not raw:
        return ''
    cleaned = unicodedata.normalize('NFKC', raw).replace('\xa0', ' ')
    cleaned = ' '.join(cleaned.split()).strip()
    paren = cleaned.find(' (')
    if paren != -1:
        cleaned = cleaned[:paren].strip()
    return cleaned


def query_uniprot(name, organism, reviewed_only, size=10, retries=2):
    query = f'protein_name:"{name}" AND organism_name:"{organism}"'
    if reviewed_only:
        query += ' AND reviewed:true'
    params = {
        'query': query,
        'fields': 'accession,sequence',
        'format': 'json',
        'size': size,
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=20)
            if resp.status_code == 200:
                return resp.json().get('results', [])
            return []
        except requests.RequestException:
            if attempt < retries:
                time.sleep(1.5)
                continue
            return []
    return []


def resolve_one(protein_name, organism, full_sequence, nes_sequence, nes_start, nes_end):
    """Returns (accession_or_None, match_type). Tries reviewed-only first
    (fewer, higher-quality candidates), falls back to all entries if that
    comes up empty (some non-model organisms only have TrEMBL records)."""
    name = clean_protein_name(protein_name)
    org = clean_organism(organism)
    if not name or not org:
        return None, 'unresolved -- missing cleaned name/organism'

    candidates = query_uniprot(name, org, reviewed_only=True)
    if not candidates:
        candidates = query_uniprot(name, org, reviewed_only=False)
    if not candidates:
        return None, 'unresolved -- no UniProt search results'

    # 0-based slice for comparing against candidate sequences (nes_start/
    # nes_end in nes_dataset.json are 1-based inclusive, same convention
    # confirmed for the rest of this project's NES pipeline).
    start0, end0 = nes_start - 1, nes_end

    substring_only_hit = None
    for c in candidates:
        acc = c.get('primaryAccession')
        seq = (c.get('sequence') or {}).get('value', '')
        if not acc or not seq:
            continue
        if full_sequence and seq == full_sequence:
            return acc, 'exact_full_sequence'
        if nes_sequence and seq[start0:end0] == nes_sequence:
            return acc, 'position_verified_substring'
        if nes_sequence and nes_sequence in seq and substring_only_hit is None:
            substring_only_hit = acc

    if substring_only_hit:
        return substring_only_hit, 'substring_only -- NOT auto-used, position unverified'
    return None, 'unresolved -- candidates found but none matched sequence'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', default='nesdb_resolved_accessions.json')
    args = ap.parse_args()

    with open(DATASET_PATH, encoding='utf-8') as f:
        data = json.load(f)

    targets = []
    for i, rec in enumerate(data):
        s, e = rec.get('nes_start'), rec.get('nes_end')
        if not (s and e):
            continue
        if ALREADY_RESOLVED_RE.search(rec.get('db_reference') or ''):
            continue  # already resolvable the old way, evaluate script has it
        targets.append({
            'index': i,
            'source': rec.get('source'),
            'source_id': rec.get('source_id'),
            'protein_name': rec.get('protein_name'),
            'organism': rec.get('organism'),
            'full_sequence': rec.get('full_sequence'),
            'nes_sequence': rec.get('nes_sequence'),
            'nes_start': int(s),
            'nes_end': int(e),
        })

    print(f"{len(targets)} unresolved positives to look up (out of {len(data)} total dataset entries)")

    results_path = Path(args.cache)
    results = []
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
            print(f"Resuming: {len(results)} already looked up")
        except (json.JSONDecodeError, OSError):
            print(f"{results_path} unreadable -- starting fresh")
    done_indices = {r['index'] for r in results}

    resolved_count = 0
    for n, t in enumerate(targets, 1):
        if t['index'] in done_indices:
            continue
        accession, match_type = resolve_one(
            t['protein_name'], t['organism'], t['full_sequence'],
            t['nes_sequence'], t['nes_start'], t['nes_end'],
        )
        status = f"-> {accession} ({match_type})" if accession else f"-> {match_type}"
        print(f"[{n}/{len(targets)}] {clean_protein_name(t['protein_name'])!r} "
              f"/ {clean_organism(t['organism'])!r} {status}")
        results.append({
            'index': t['index'], 'source_id': t['source_id'],
            'protein_name': t['protein_name'], 'organism': t['organism'],
            'nes_start': t['nes_start'], 'nes_end': t['nes_end'],
            'accession': accession, 'match_type': match_type,
        })
        if accession and 'unverified' not in match_type:
            resolved_count += 1
        write_json_atomic_with_retry(results_path, results)
        time.sleep(0.4)

    trusted = [r for r in results if r['accession'] and 'unverified' not in r['match_type']]
    print(f"\n{'='*70}\nDone. {len(trusted)}/{len(results)} confidently resolved "
          f"(exact_full_sequence or position_verified_substring).\n"
          f"Combined with the 42 already resolvable via NESbase's db_reference, "
          f"that's up to {42 + len(trusted)} usable positives for "
          f"evaluate_crm1_pocket_signal.py.\n{'='*70}")


if __name__ == '__main__':
    main()
