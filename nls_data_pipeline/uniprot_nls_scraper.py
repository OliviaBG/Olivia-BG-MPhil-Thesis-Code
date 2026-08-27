"""
Production-scale UniProt NLS scraper -- run locally, with real internet
access, since it queries rest.uniprot.org in bulk.

The seed file nls_proteins_uniprot_seed.json (159 proteins, 274 NLS motif
instances) was bootstrapped from single-page-per-taxon queries. This script
re-implements the same UniProt query with proper cursor-based pagination
(the `Link` header UniProt returns for pages beyond the first ~500 results)
so that the full dataset is retrieved rather than the seed snapshot.

UniProt's REST API (https://www.uniprot.org/help/api) doesn't take an
offset parameter for paging past the first `size` results -- it returns a
`Link: <...&cursor=XXXX>; rel="next"` response header that must be
followed. That's exactly what this script does.

Usage:
    pip install requests
    python3 uniprot_nls_scraper.py --out nls_proteins_full.json
    # then re-point build_dataset.py's POS_SEED at the new file, or just
    # overwrite nls_proteins_uniprot_seed.json and rerun build_dataset.py

Also pulls the DNA-binding hard-negative set the same way
(--also-negatives, on by default).
"""
import argparse
import json
import time
from pathlib import Path

import requests

BASE = "https://rest.uniprot.org/uniprotkb/search"
HERE = Path(__file__).resolve().parent

# Same taxa used for this project's bootstrap snapshot; add more (or drop
# the taxonomy filter entirely) to widen coverage -- e.g. 7227 (fly), 7955
# (zebrafish), 6239 (C. elegans), 9031 (chicken), 8355 (X. laevis).
TAXA = {
    "human": 9606, "mouse": 10090, "yeast": 559292, "arabidopsis": 3702,
    "fly": 7227, "zebrafish": 7955, "c_elegans": 6239, "chicken": 9031,
    "xenopus": 8355, "rat": 10116,
}


def _paginate(query, fields, size=500, sleep=0.34):
    url = f"{BASE}?query={query}&format=json&size={size}&fields={fields}"
    results = []
    while url:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} -- stopping pagination for this query")
            break
        data = resp.json()
        results.extend(data.get("results", []))
        # UniProt cursor pagination via the Link header
        link = resp.headers.get("Link")
        url = None
        if link and 'rel="next"' in link:
            url = link[link.find("<") + 1: link.find(">")]
        time.sleep(sleep)
    return results


def fetch_nls_proteins(taxon_name, taxon_id):
    query = (f'ft_motif:%22nuclear%20localization%20signal%22%20AND%20'
             f'reviewed:true%20AND%20taxonomy_id:{taxon_id}')
    fields = "accession,organism_name,ft_motif,sequence"
    print(f"Fetching NLS-annotated proteins for {taxon_name} (taxid {taxon_id})...")
    entries = _paginate(query, fields)
    print(f"  {len(entries)} entries")
    out = []
    for e in entries:
        acc = e.get("primaryAccession")
        seq = (e.get("sequence") or {}).get("value")
        org = (e.get("organism") or {}).get("scientificName")
        motifs = []
        for feat in e.get("features", []):
            if feat.get("type") != "Motif":
                continue
            desc = feat.get("description") or ""
            if "nuclear localization" not in desc.lower() and "nls" not in desc.lower():
                continue
            loc = feat.get("location", {})
            start = (loc.get("start") or {}).get("value")
            end = (loc.get("end") or {}).get("value")
            evidences = feat.get("evidences") or []
            motifs.append({
                "description": desc, "start": start, "end": end,
                "bipartite": "bipartite" in desc.lower(),
                "evidence_codes": sorted({ev.get("evidenceCode") for ev in evidences if ev.get("evidenceCode")}),
                "pubmed_ids": sorted({ev.get("id") for ev in evidences if ev.get("source") == "PubMed"}),
            })
        if acc and seq and motifs:
            out.append({"accession": acc, "organism": org, "sequence": seq, "nls_motifs": motifs})
    return out


def fetch_dna_binding_proteins(taxon_name, taxon_id):
    query = f"ft_dna_bind:%2A%20AND%20reviewed:true%20AND%20taxonomy_id:{taxon_id}"
    fields = "accession,organism_name,ft_dna_bind,ft_motif,sequence"
    print(f"Fetching DNA-binding proteins (hard negatives) for {taxon_name}...")
    entries = _paginate(query, fields)
    print(f"  {len(entries)} entries")
    out = []
    for e in entries:
        acc = e.get("primaryAccession")
        seq = (e.get("sequence") or {}).get("value")
        org = (e.get("organism") or {}).get("scientificName")
        if not acc or not seq:
            continue
        dna_regions, nls_spans = [], []
        for feat in e.get("features", []):
            loc = feat.get("location", {})
            start = (loc.get("start") or {}).get("value")
            end = (loc.get("end") or {}).get("value")
            if feat.get("type") == "DNA binding":
                dna_regions.append({"start": start, "end": end, "description": feat.get("description") or ""})
            elif feat.get("type") == "Motif":
                desc = (feat.get("description") or "").lower()
                if "nuclear localization" in desc or "nls" in desc:
                    nls_spans.append((start, end))
        if dna_regions:
            out.append({"accession": acc, "organism": org, "sequence": seq,
                        "dna_bind_regions": dna_regions, "nls_spans": nls_spans})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "nls_proteins_full.json"))
    ap.add_argument("--negatives-out", default=str(HERE / "dna_bind_hard_negatives_full.json"))
    ap.add_argument("--taxa", nargs="*", default=list(TAXA.keys()),
                     help="subset of taxon names to fetch (default: all)")
    ap.add_argument("--also-negatives", action="store_true", default=True)
    args = ap.parse_args()

    all_proteins = {}
    for name in args.taxa:
        for p in fetch_nls_proteins(name, TAXA[name]):
            all_proteins[p["accession"]] = p
    json.dump(list(all_proteins.values()), open(args.out, "w"), indent=1)
    print(f"\nWrote {args.out}: {len(all_proteins)} unique proteins with real NLS annotations")

    if args.also_negatives:
        all_neg = {}
        for name in args.taxa:
            for p in fetch_dna_binding_proteins(name, TAXA[name]):
                all_neg[p["accession"]] = p
        json.dump(list(all_neg.values()), open(args.negatives_out, "w"), indent=1)
        print(f"Wrote {args.negatives_out}: {len(all_neg)} unique DNA-binding hard-negative proteins")

    print("\nNext: point build_dataset.py's POS_SEED/DNA_SEED at these files (or "
          "overwrite nls_proteins_uniprot_seed.json / dna_bind_hard_negative_seed.json), "
          "then rerun build_dataset.py and nls_ml_predictor.py train.")


if __name__ == "__main__":
    main()
