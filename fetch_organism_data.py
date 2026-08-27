#!/usr/bin/env python3
"""
fetch_organism_data.py

Backfills organism/taxonomic-group metadata for every unique UniProt
accession in the NES and/or NLS training datasets, producing
organism_data.json (one per pipeline) that compare_taxonomic_fit.py reads
to group out-of-fold predictions by taxonomic group.

WHY THIS IS A SEPARATE SCRIPT: same reasoning as fetch_iupred_training_data.py
-- UniProt's REST API needs real, unrestricted internet access, and this is a
one-time (or occasional, if the training sets grow) offline backfill, not
something the live training/prediction pipeline should depend on.

WHAT IT DOES, per pipeline (nes / nls):
  1. Reads the existing structural_data_v2.json / structural_data.json
     (already has 'accession' per training example).
  2. Batch-queries https://rest.uniprot.org/uniprotkb/search for
     organism_name + lineage, 90 accessions per request (well under
     UniProt's own limits -- this is NOT the proxy used during
     development, which had an unrelated ~200-character URL cap of its own).
  3. Classifies each accession into a broad taxonomic bucket from its
     lineage string (see TAXONOMIC_GROUPS below) -- broad buckets, not
     species-level, because several groups (viral, plant, invertebrate) only
     have a handful of representatives and a per-species breakdown would be
     mostly n=1 noise (confirmed empirically on a 40-accession sample before
     writing this script: 'fly' and 'rat' each appeared exactly once).
  4. Writes {pipeline}_data_pipeline/organism_data.json:
     {accession: {"organism": <full UniProt organism name>, "group": <bucket>}}

USAGE:
    pip install requests --break-system-packages   # if not already installed
    python fetch_organism_data.py                  # both pipelines
    python fetch_organism_data.py --pipeline nes    # NES only
    python fetch_organism_data.py --pipeline nls    # NLS only

AFTER RUNNING:
    Run compare_taxonomic_fit.py --target nes|nls|both to produce the
    out-of-fold per-group predictions, then generate_thesis_figures.py to
    render the figures (taxonomic-fit violin + organism-composition pie).
"""

import argparse
import json
import time
from pathlib import Path

import requests

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
BATCH_SIZE = 90  # comfortably under UniProt's own query-length limits
REQUEST_TIMEOUT = 30

PIPELINES = {
    "nes": {
        "structural_data": Path("nes_data_pipeline/structural_data_v2.json"),
        "output": Path("nes_data_pipeline/organism_data.json"),
    },
    "nls": {
        "structural_data": Path("nls_data_pipeline/structural_data.json"),
        "output": Path("nls_data_pipeline/organism_data.json"),
    },
}

# Order matters -- first match wins. Checked against each accession's
# lineage (a list of taxonomic ranks from UniProt, e.g.
# ["Eukaryota", "Metazoa",..., "Hominidae", "Homo"]) joined into one
# lowercase string for simple substring matching.
TAXONOMIC_GROUPS = [
    ("Viral", ["viruses"]),
    ("Human", ["homo"]),
    ("Rodent", ["rodentia"]),
    ("Yeast/Fungi", ["fungi"]),
    ("Plant", ["viridiplantae"]),
    # UniProt lineages don't reliably include the literal rank "Vertebrata"
    # (e.g. fish/bird/reptile lineages often go straight from Chordata to a
    # class-level rank like Actinopterygii/Aves without it) -- "chordata" is
    # the rank that's actually always present for this group.
    ("Other vertebrate", ["chordata"]),
    ("Invertebrate/Other", []),  # fallback -- matches anything (empty list = always true)
]


def classify(lineage):
    lineage_str = " ".join(lineage).lower()
    for group, needles in TAXONOMIC_GROUPS:
        if not needles or any(n in lineage_str for n in needles):
            return group
    return "Invertebrate/Other"


def fetch_batch(accessions, session):
    # A real space here, not a literal '+' -- requests.get(..., params=...)
    # URL-encodes every character in the value exactly once. A pre-built
    # '+OR+' gets that '+' encoded too (into '%2B'), so UniProt receives
    # "...%2BOR%2B..." -- a literal plus SIGN inside the query text, not an
    # encoded space -- and 400s. A plain ' OR ' lets requests do the
    # encoding correctly (space -> '+'), producing the same "...+OR+..."
    # on the wire that a manually-built URL would.
    query = " OR ".join(f"accession:{a}" for a in accessions)
    params = {"query": query, "fields": "accession,organism_name,lineage", "format": "tsv", "size": 500}
    resp = session.get(UNIPROT_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    out = {}
    for line in lines[1:]:  # skip header row
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        acc, organism, lineage_field = parts[0], parts[1], parts[2]
        lineage = [x.strip() for x in lineage_field.split(",")] if lineage_field else []
        out[acc] = {"organism": organism, "group": classify(lineage)}
    return out


def run_pipeline(name, cfg):
    print(f"\n{'=' * 70}\n{name.upper()} pipeline\n{'=' * 70}")

    if not cfg["structural_data"].exists():
        print(f"  SKIP: {cfg['structural_data']} not found -- nothing to backfill.")
        return

    records = json.loads(cfg["structural_data"].read_text(encoding="utf-8"))
    accessions = sorted({r["accession"] for r in records if r.get("accession")})
    print(f"{len(accessions)} unique UniProt accessions")

    session = requests.Session()
    result = {}
    batches = [accessions[i:i + BATCH_SIZE] for i in range(0, len(accessions), BATCH_SIZE)]
    for i, batch in enumerate(batches, 1):
        print(f"[{i}/{len(batches)}] querying {len(batch)} accessions ...", end="  ")
        try:
            found = fetch_batch(batch, session)
            result.update(found)
            missing = set(batch) - set(found)
            print(f"got {len(found)}, missing {len(missing)}"
                  + (f" ({sorted(missing)[:5]}{'...' if len(missing) > 5 else ''})" if missing else ""))
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(0.3)

    cfg["output"].write_text(json.dumps(result, indent=2))
    print(f"\nWrote {len(result)}/{len(accessions)} accessions to {cfg['output']}")

    from collections import Counter
    counts = Counter(v["group"] for v in result.values())
    print("Group breakdown (unique accessions):")
    for group, n in counts.most_common():
        print(f"  {group}: {n} ({n / len(result) * 100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["nes", "nls", "both"], default="both")
    args = ap.parse_args()

    targets = PIPELINES.keys() if args.pipeline == "both" else [args.pipeline]
    for name in targets:
        run_pipeline(name, PIPELINES[name])

    print("\nDone. Next step: python3 compare_taxonomic_fit.py --target both")


if __name__ == "__main__":
    main()
