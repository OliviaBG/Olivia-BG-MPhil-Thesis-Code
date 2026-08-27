#!/usr/bin/env python3
"""
fetch_bipartite_and_nonclassical_candidates.py
============================================================
Addresses why the model misses "really easy" holdout positives even when
a real AlphaFold structure IS available. Investigation found two real,
distinct training-data gaps, not model bugs:

  1. Bipartite-type NLSs are only 16/294 (5.4%) of nls_dataset.csv's
     positive rows -- holdout recall for bipartite-classified candidates is
     ~40% (2/5) vs ~80% (12/15) for monopartite. Textbook bipartite motifs
     (TP53, RB1, Influenza NP) score well below the 0.5 survival threshold
     even with real structural data, purely from class imbalance.
  2. A handful of real, curated NLSs (HIV-2 Vpx's SYTKYRYL; hamster
     polyomavirus Large T-antigen's PKKPPPT) don't even pass scan_sequence()'s
     regex pre-filter -- their basic residues aren't in an adjacent K/R
     doublet the way MONOPARTITE_RE/detect_bipartite() require. A local
     trial confirmed this isn't a pre-filter problem: even when
     the pre-filter is loosened to admit these shapes, the trained
     classifier still scores them near zero, because it's never seen a real
     example of this shape during training.

This script re-runs the same real UniProt query uniprot_nls_scraper.py
already uses (reviewed:true Swiss-Prot, ft_motif nuclear-localization-signal
annotations, real cursor-based pagination). Every real annotated NLS motif
it finds is kept and saved: nothing is silently discarded, so this no
longer drops the ordinary-monopartite class the way earlier versions did. Each motif is bucketed by shape purely for
labeling/review, using this project's OWN regex functions (imported
directly from nls_ml_predictor.py, not reimplemented):

  - "bipartite": UniProt's own curated description contains "bipartite"
    (trusted first -- see classify_motif()'s docstring for the bugfix explaining why), OR the real annotated NLS sequence passes this
    project's detect_bipartite() -- targets gap (1) above.
  - "non_classical": the real annotated NLS sequence does NOT match
    MONOPARTITE_RE, does NOT pass detect_bipartite(), and does NOT match
    PY_NLS_RE -- i.e. a real, curated NLS this project's pre-filter would
    currently never even generate a candidate window for. Targets gap (2)
    above.
  - "monopartite": everything else -- the already-well-represented
    278/294 class. Kept for completeness/inventory even though it isn't
    the reason this script exists.

Every candidate in all three files is REAL: real UniProt accession, real
curated Motif feature, real evidence codes/PubMed IDs, real sequence
slice. Nothing here is synthesized. Accessions already present in
nls_dataset.csv are dropped (collision-checked, same as
append_viral_candidates.py) so the output is purely new material.

REQUIREMENTS: real internet access (rest.uniprot.org). Environments with
a domain-allowlisted proxy return HTTP 403 for both alphafold.ebi.ac.uk
and rest.uniprot.org -- the same limitation documented in
uniprot_nls_scraper.py's
own docstring. Run this on your own machine, same as
extract_unfiltered_nls_scores.py.

Usage (from the nls_data_pipeline directory, or pass --project-root):
    pip install requests
    python3 fetch_bipartite_and_nonclassical_candidates.py

Outputs (next to this script):
    bipartite_nls_candidates_<date>.json      -- new real bipartite candidates
    nonclassical_nls_candidates_<date>.json   -- new real non-classical candidates
    monopartite_nls_candidates_<date>.json    -- new real ordinary-monopartite candidates
    (all three: same {"accession","organism","sequence","nls_motifs":[...]}
    schema as viral_nls_candidates_2026-08-05.json)
    dna_binding_hard_candidates_<date>.json   -- new real DNA-binding-domain
    hard-negative candidates (added to fix the hard-negative
    dilution the first three buckets caused -- see
    fetch_dna_binding_proteins()'s docstring). Schema:
    {"accession","organism","sequence","dna_bind_regions":[...],"nls_spans":[...]},
    same as uniprot_nls_scraper.py's fetch_dna_binding_proteins() output.

    An append_*.py script in the same style as append_viral_candidates.py
    can merge all four into nls_dataset.csv/nls_negatives.csv once reviewed.

Nothing is merged into nls_dataset.csv/nls_negatives.csv by this script --
review the output first, same workflow as the viral candidates
(scrape -> hand review -> append script -> retrain).
"""
import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
BASE = "https://rest.uniprot.org/uniprotkb/search"
TODAY = datetime.date.today().isoformat()

# Same taxa as uniprot_nls_scraper.py, plus Viruses (10239) -- several of
# our real holdout gaps (HIV-2 Vpx, hamster polyomavirus, Influenza,
# Adenovirus, SV40) are viral, and UniProt's viral NLS annotations are
# real curated literature calls same as everything else, not lower quality.
TAXA = {
    "human": 9606, "mouse": 10090, "yeast": 559292, "arabidopsis": 3702,
    "fly": 7227, "zebrafish": 7955, "c_elegans": 6239, "chicken": 9031,
    "xenopus": 8355, "rat": 10116, "viruses": 10239,
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
                "curator_bipartite": "bipartite" in desc.lower(),
                "evidence_codes": sorted({ev.get("evidenceCode") for ev in evidences if ev.get("evidenceCode")}),
                "pubmed_ids": sorted({ev.get("id") for ev in evidences if ev.get("source") == "PubMed"}),
            })
        if acc and seq and motifs:
            out.append({"accession": acc, "organism": org, "sequence": seq, "nls_motifs": motifs})
    return out


def fetch_dna_binding_proteins(taxon_name, taxon_id):
    """Real UniProt-curated 'DNA binding' region features -- the source of
    this project's dna_binding_hard negatives (basic, K/R-rich stretches
    that are NOT nuclear import signals, the single most literature-
    documented failure mode of naive NLS predictors). Same query
    uniprot_nls_scraper.py's own fetch_dna_binding_proteins() uses.

    added to this script specifically to fix the hard-negative
    dilution this scrape's OWN earlier bipartite/non-classical/monopartite
    pull caused -- adding ~4500 new positive proteins each contributed 2
    auto-generated protein_matched_random negatives but zero new hard
    negatives, dropping dna_binding_hard from 37% to 2.4% of the negative
    training pool (see chat -- 3 of 4 post-retrain false positives were
    exactly the homeodomain/DNA-binding cases task #17's veto was built
    for). Pulling real DNA-binding proteins across the same expanded taxa
    this script already covers restores real hard-negative variety instead
    of just reweighting the existing small set.
    """
    query = f"ft_dna_bind:%2A%20AND%20reviewed:true%20AND%20taxonomy_id:{taxon_id}"
    fields = "accession,organism_name,ft_dna_bind,ft_motif,sequence"
    print(f"Fetching DNA-binding proteins (hard negatives) for {taxon_name} (taxid {taxon_id})...")
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


def classify_motif(seq, mono_re, detect_bip, py_re, curator_bipartite=False):
    """Real shape classification -- 'bipartite', 'non_classical', or
    'monopartite' (dropped by the caller, already well-represented). BUGFIX: curator_bipartite (UniProt's own curated description
    literally containing the word "bipartite") is now checked FIRST and, if
    True, always wins -- previously this function only trusted its own
    structural detect_bipartite() regex, which requires an adjacent K/R
    doublet + a 6-16aa spacer + a tight 5-residue second-cluster window.
    That's narrower than real biology: a genuine curator-confirmed
    bipartite motif that this project's own regex fails to structurally
    confirm (longer/looser spacer, non-adjacent first cluster, etc.) would
    fall through to the is_mono check -- and since a bipartite motif is by
    definition built from two basic clusters, at least one of which very
    often independently satisfies the short, loose MONOPARTITE_RE on its
    own, this silently misrouted real curator-confirmed bipartite motifs
    into the discarded "monopartite" bucket instead of keeping them.
    Caught by cross-checking the first live scrape's own output:
    574 motifs across both output files had "bipartite" in their real
    UniProt description, but only 513 landed in the bipartite bucket -- 61
    fell through to non_classical (still saved, just mislabeled), and an
    unknown further number likely fell all the way through to the
    discarded monopartite bucket (invisible -- not saved anywhere). Trusting
    the real curator annotation first fixes both leaks at once.
    """
    seq = seq.upper()
    if curator_bipartite:
        return "bipartite"
    is_bip = detect_bip(seq)[0]
    is_mono = bool(mono_re.search(seq))
    is_py = bool(py_re.search(seq))
    if is_bip:
        return "bipartite"
    if not is_mono and not is_py:
        return "non_classical"
    return "monopartite"


def load_existing_accessions(dataset_csv):
    if not dataset_csv.exists():
        return set()
    rows = list(csv.DictReader(open(dataset_csv, encoding="utf-8")))
    return {r["accession"] for r in rows}


def load_holdout_accessions(project_root):
    """ BUGFIX: the original version of this script only excluded
    accessions already present in nls_dataset.csv, never checking against
    nls_holdout_data/candidates.json -- the file that's supposed to be
    permanently held out of training for a trustworthy eval number. Result:
    the first live run of this script re-scraped 23 of the 25 holdout
    POSITIVE accessions (SV40 T-antigen, Adenovirus E1A, HIV-2 Vpx, etc.)
    straight back into nls_dataset.csv as new "real" training positives --
    caught via nls_holdout_test_results.json's
    accession_overlap_with_training_pool field after the resulting retrain
    scored a suspiciously high 92% sensitivity, and cleaned up with
    decontaminate_holdout_leakage.py. This function is the permanent fix:
    every accession in EITHER the holdout positives or holdout negatives is
    now excluded from every future scrape, the same way existing
    nls_dataset.csv accessions already were."""
    path = Path(project_root) / "nls_holdout_data" / "candidates.json"
    if not path.exists():
        print(f"  WARNING: {path} not found -- can't exclude holdout accessions from this "
              f"scrape. Pass --project-root if your AlphaFold directory isn't the parent of "
              f"this script.")
        return set()
    holdout = json.loads(path.read_text())
    return {p[1] for p in holdout.get("positives", [])} | {n[1] for n in holdout.get("negatives", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(HERE.parent),
                     help="AlphaFold project root (where nls_ml_predictor.py lives)")
    ap.add_argument("--taxa", nargs="*", default=list(TAXA.keys()))
    args = ap.parse_args()

    sys.path.insert(0, args.project_root)
    try:
        from nls_ml_predictor import MONOPARTITE_RE, detect_bipartite, PY_NLS_RE
    except ImportError as e:
        raise SystemExit(
            f"Couldn't import regex functions from nls_ml_predictor.py in "
            f"{args.project_root} ({e}). Pass --project-root pointing at your "
            f"AlphaFold directory if it's not the parent of this script."
        )

    existing_accessions = load_existing_accessions(HERE / "nls_dataset.csv")
    holdout_accessions = load_holdout_accessions(args.project_root)
    excluded_accessions = existing_accessions | holdout_accessions
    print(f"{len(existing_accessions)} accessions already in nls_dataset.csv, "
          f"{len(holdout_accessions)} accessions in the held-out eval set (nls_holdout_data/"
          f"candidates.json) -- {len(excluded_accessions)} total will be skipped so the "
          f"holdout accessions can never leak back into training (see load_holdout_accessions()'s "
          f"docstring for why this matters).\n")

    all_proteins = {}
    for name in args.taxa:
        for p in fetch_nls_proteins(name, TAXA[name]):
            all_proteins.setdefault(p["accession"], p)

    # The monopartite motifs are not silently discarded
    # either -- previously these were only counted (n_mono_dropped)
    # and thrown away. Now every real curated motif this scrape finds gets
    # saved to one of three files; nothing is discarded, the 3-way split is
    # purely for review/labeling, not for deciding what's "worth keeping".
    bipartite_out, nonclassical_out, monopartite_out = {}, {}, {}
    for acc, p in all_proteins.items():
        if acc in excluded_accessions:
            continue
        seq = p["sequence"]
        kept_motifs_bip, kept_motifs_nc, kept_motifs_mono = [], [], []
        for m in p["nls_motifs"]:
            start, end = m.get("start"), m.get("end")
            if start is None or end is None or start < 1 or end > len(seq):
                continue
            nls_seq = seq[start - 1:end]
            if not nls_seq or len(nls_seq) < 3:
                continue
            shape = classify_motif(nls_seq, MONOPARTITE_RE, detect_bipartite, PY_NLS_RE,
                                    curator_bipartite=m["curator_bipartite"])
            m_out = {
                "description": m["description"], "start": start, "end": end,
                "bipartite": int(shape == "bipartite" or m["curator_bipartite"]),
                "evidence_codes": m["evidence_codes"], "pubmed_ids": m["pubmed_ids"],
            }
            if shape == "bipartite":
                kept_motifs_bip.append(m_out)
            elif shape == "non_classical":
                kept_motifs_nc.append(m_out)
            else:
                kept_motifs_mono.append(m_out)
        if kept_motifs_bip:
            bipartite_out[acc] = {"accession": acc, "organism": p["organism"],
                                   "sequence": seq, "nls_motifs": kept_motifs_bip}
        if kept_motifs_nc:
            nonclassical_out[acc] = {"accession": acc, "organism": p["organism"],
                                      "sequence": seq, "nls_motifs": kept_motifs_nc}
        if kept_motifs_mono:
            monopartite_out[acc] = {"accession": acc, "organism": p["organism"],
                                     "sequence": seq, "nls_motifs": kept_motifs_mono}

    bip_path = HERE / f"bipartite_nls_candidates_{TODAY}.json"
    nc_path = HERE / f"nonclassical_nls_candidates_{TODAY}.json"
    mono_path = HERE / f"monopartite_nls_candidates_{TODAY}.json"
    json.dump({"candidates": list(bipartite_out.values())}, open(bip_path, "w"), indent=1)
    json.dump({"candidates": list(nonclassical_out.values())}, open(nc_path, "w"), indent=1)
    json.dump({"candidates": list(monopartite_out.values())}, open(mono_path, "w"), indent=1)

    print(f"\nWrote {bip_path.name}: {len(bipartite_out)} new proteins, "
          f"{sum(len(p['nls_motifs']) for p in bipartite_out.values())} real bipartite NLS motifs.")
    print(f"Wrote {nc_path.name}: {len(nonclassical_out)} new proteins, "
          f"{sum(len(p['nls_motifs']) for p in nonclassical_out.values())} real non-classical NLS motifs "
          f"(don't match this project's monopartite/bipartite/PY-NLS pre-filter at all).")
    print(f"Wrote {mono_path.name}: {len(monopartite_out)} new proteins, "
          f"{sum(len(p['nls_motifs']) for p in monopartite_out.values())} real ordinary monopartite motifs "
          f"(nothing dropped anymore -- kept for completeness even though this class is already "
          f"well-represented in nls_dataset.csv).")

    # Real DNA-binding-domain hard negatives, to fix the
    # dilution the bipartite/non-classical/monopartite pull above caused
    # (see fetch_dna_binding_proteins()'s docstring). Excludes accessions
    # already used as ANY negative type too (not just existing_accessions,
    # which is positives-only) so this doesn't re-fetch proteins that
    # already have a dna_binding_hard row.
    existing_negative_accessions = set()
    neg_csv = HERE / "nls_negatives.csv"
    if neg_csv.exists():
        existing_negative_accessions = {r["accession"] for r in csv.DictReader(open(neg_csv, encoding="utf-8"))}
    dna_excluded = excluded_accessions | existing_negative_accessions

    all_dna_proteins = {}
    for name in args.taxa:
        for p in fetch_dna_binding_proteins(name, TAXA[name]):
            all_dna_proteins.setdefault(p["accession"], p)

    dna_out = {acc: p for acc, p in all_dna_proteins.items() if acc not in dna_excluded}
    dna_path = HERE / f"dna_binding_hard_candidates_{TODAY}.json"
    json.dump({"candidates": list(dna_out.values())}, open(dna_path, "w"), indent=1)
    n_dna_regions = sum(len(p["dna_bind_regions"]) for p in dna_out.values())
    print(f"Wrote {dna_path.name}: {len(dna_out)} new proteins, {n_dna_regions} real curated "
          f"DNA-binding regions (candidate dna_binding_hard negatives -- final count after "
          f"excluding regions that overlap a real/candidate NLS span will be lower; that overlap "
          f"check happens at merge time, same as build_dataset.py's build_negatives() part (b)).")

    print("\nNothing merged yet -- review all four files, then run the merge step to build an "
          "append_*.py-style merge script (same pattern as append_viral_candidates.py) once "
          "you've eyeballed what came back.")


if __name__ == "__main__":
    main()
