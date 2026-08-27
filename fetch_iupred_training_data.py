#!/usr/bin/env python3
"""
fetch_iupred_training_data.py

Backfills real IUPred2A disorder + ANCHOR2 binding-region scores for the NES
and/or NLS training datasets, producing iupred_data_v2.json (NES) and/or
iupred_data.json (NLS) in the same list-of-records format load_iupred_data()
/ _load_iupred_data() in nes_ml_predictor_improved.py / nls_ml_predictor.py
already expect.

WHY THIS IS A SEPARATE SCRIPT, NOT PART OF build_training_dataset():
IUPred2A's hosted REST API (iupred2a.elte.hu) needs real, unrestricted
internet access. This is a one-time (or occasional, if the training sets
grow) offline backfill step -- run it, then retrain -- not something the
live training/prediction pipeline should depend on at request time.

WHAT IT DOES, per pipeline (nes / nls):
  1. Reads the existing structural_data_v2.json / structural_data.json
     (already has 'seq' + 'accession' + 'label' per training example --
     see structural_dataset_v2_pipeline.py / structural_dataset_pipeline.py,
     which resolved these earlier from live UniProt/AlphaFold).
  2. Fetches https://iupred2a.elte.hu/iupred2a/anchor/{accession}.json ONCE
     per unique accession (one call returns both the 'iupred2' and
     'anchor2' per-residue arrays for the full canonical UniProt sequence --
     verified against a live call during development). Results are cached
     to disk per-accession (iupred_raw_cache/<pipeline>/<accession>.json) so
     re-running this script after an interruption or a growing dataset only
     fetches what's missing.
  3. For every training record referencing that accession, finds the exact
     record['seq'] substring within the fetched full protein sequence (same
     "exact match, else substring search, else give up rather than
     misalign" logic as app.py's align_iupred_to_structure -- see that
     function's docstring for the full rationale) and computes:
       - iupred_mean / anchor2_mean: mean over the matched window
       - n_flank_iupred / n_flank_anchor2, c_flank_iupred / c_flank_anchor2:
         mean over the +/-15 residues either side (same window size as the
         existing DISORDER_PROPENSITY-based n_flank_disorder/c_flank_disorder
         features in both predictor files, for direct comparability;
         skipped -- left out of the output record, not zero-filled -- if
         fewer than MIN_FLANK_LEN residues are available, same
         under-powered-sample guard the DISORDER_PROPENSITY flank features
         already use)
  4. Writes iupred_data_v2.json / iupred_data.json.

USAGE:
    pip install requests --break-system-packages   # if not already installed
    python fetch_iupred_training_data.py                  # both pipelines
    python fetch_iupred_training_data.py --pipeline nes    # NES only
    python fetch_iupred_training_data.py --pipeline nls    # NLS only
    python fetch_iupred_training_data.py --rate-limit 1.5  # slower, more polite

AFTER RUNNING:
    Delete the existing model files so they retrain picking up the new
    features (both predictor classes auto-train on next instantiation if
    their model files are missing -- see __init__ in each):
        rm models/nes_svm_v2.pkl models/nes_scaler_v2.pkl models/nes_pssm_v2.pkl
        rm models_nls/nls_classifier.pkl models_nls/nls_scaler.pkl models_nls/nls_pssm.pkl
    Then compare models/nes_permutation_importance_v2.json and
    models_nls/nls_feature_importance.json against the backed-up versions
    (see ml_retrain_backup_*/ from before this change) to see whether
    iupred_mean/anchor2_mean etc. actually earned a place near the top, or
    whether the model just relearned what nes_disorder_mean/n_flank_disorder
    already captured.
"""

import argparse
import json
import time
from pathlib import Path

import requests

IUPRED_BASE_URL = "https://iupred2a.elte.hu/iupred2a/anchor"
FLANK_WINDOW = 15
MIN_FLANK_LEN = 9  # matches MIN_DISORDER_FLANK_LEN in both predictor files
REQUEST_TIMEOUT = 20

PIPELINES = {
    "nes": {
        "structural_data": Path("nes_data_pipeline/structural_data_v2.json"),
        "output": Path("nes_data_pipeline/iupred_data_v2.json"),
        "cache_dir": Path("iupred_raw_cache/nes"),
    },
    "nls": {
        "structural_data": Path("nls_data_pipeline/structural_data.json"),
        "output": Path("nls_data_pipeline/iupred_data.json"),
        "cache_dir": Path("iupred_raw_cache/nls"),
    },
}


def fetch_accession(accession, cache_dir, rate_limit, session):
    """Fetch (and cache) IUPred2A+ANCHOR2 for one UniProt accession.
    Returns the parsed dict ({'sequence', 'iupred2', 'anchor2'}) or None on
    failure. Resumable: if a cached copy already exists on disk, use it
    without hitting the network at all (including cached failures, marked
    with an 'error' key, so a bad accession isn't retried every run --
    delete its cache file manually if you believe it's now fixable, e.g.
    after a UniProt merge/redirect)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{accession}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if "error" in cached:
                return None
            return cached
        except Exception:
            pass  # corrupt cache entry -- refetch

    url = f"{IUPRED_BASE_URL}/{accession}.json"
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            cache_path.write_text(json.dumps({"error": f"HTTP {resp.status_code}"}))
            print(f"  Failed: {accession}: HTTP {resp.status_code}")
            return None
        data = resp.json()
        if not data.get("iupred2") or not data.get("sequence"):
            cache_path.write_text(json.dumps({"error": "missing iupred2/sequence in response"}))
            print(f"  Failed: {accession}: response missing expected fields")
            return None
        cache_path.write_text(json.dumps(data))
        print(f"  {accession}: {len(data['sequence'])} residues"
              + (" (+ANCHOR2)" if data.get("anchor2") else " (no ANCHOR2)"))
        return data
    except Exception as e:
        cache_path.write_text(json.dumps({"error": str(e)}))
        print(f"  Failed: {accession}: {e}")
        return None
    finally:
        time.sleep(rate_limit)


def compute_window_features(full_seq, iupred2, anchor2, seq):
    """Find `seq` within `full_seq` and compute window + flank means.
    Returns a dict of the 6 derived features, or None if `seq` can't be
    aligned (caller should skip the record entirely rather than write
    misaligned data -- same principle as align_iupred_to_structure in
    app.py: no data beats wrong data)."""
    offset = full_seq.find(seq)
    if offset == -1:
        return None
    end = offset + len(seq)

    window_iupred = iupred2[offset:end]
    window_anchor2 = anchor2[offset:end] if anchor2 else []

    out = {
        "iupred_mean": sum(window_iupred) / len(window_iupred) if window_iupred else None,
        "anchor2_mean": (sum(window_anchor2) / len(window_anchor2)) if window_anchor2 else None,
    }

    n_start = max(0, offset - FLANK_WINDOW)
    n_flank_iupred = iupred2[n_start:offset]
    n_flank_anchor2 = anchor2[n_start:offset] if anchor2 else []
    if len(n_flank_iupred) >= MIN_FLANK_LEN:
        out["n_flank_iupred"] = sum(n_flank_iupred) / len(n_flank_iupred)
        if n_flank_anchor2:
            out["n_flank_anchor2"] = sum(n_flank_anchor2) / len(n_flank_anchor2)

    c_end = min(len(full_seq), end + FLANK_WINDOW)
    c_flank_iupred = iupred2[end:c_end]
    c_flank_anchor2 = anchor2[end:c_end] if anchor2 else []
    if len(c_flank_iupred) >= MIN_FLANK_LEN:
        out["c_flank_iupred"] = sum(c_flank_iupred) / len(c_flank_iupred)
        if c_flank_anchor2:
            out["c_flank_anchor2"] = sum(c_flank_anchor2) / len(c_flank_anchor2)

    return out


def run_pipeline(name, cfg, rate_limit):
    print(f"\n{'=' * 70}\n{name.upper()} pipeline\n{'=' * 70}")

    if not cfg["structural_data"].exists():
        print(f"  SKIP: {cfg['structural_data']} not found -- nothing to backfill.")
        return

    records = json.loads(cfg["structural_data"].read_text(encoding="utf-8"))
    accessions = sorted({r["accession"] for r in records if r.get("accession")})
    print(f"{len(records)} training records, {len(accessions)} unique UniProt accessions")

    session = requests.Session()
    fetched = {}
    for i, acc in enumerate(accessions, 1):
        print(f"[{i}/{len(accessions)}] {acc}", end="  ")
        data = fetch_accession(acc, cfg["cache_dir"], rate_limit, session)
        if data:
            fetched[acc] = data

    print(f"\nFetched {len(fetched)}/{len(accessions)} accessions successfully.")

    output_records = []
    n_aligned, n_unaligned, n_no_fetch = 0, 0, 0
    for r in records:
        acc = r.get("accession")
        seq = r.get("seq", "").upper()
        if not acc or acc not in fetched:
            n_no_fetch += 1
            continue
        data = fetched[acc]
        feats = compute_window_features(data["sequence"], data["iupred2"], data.get("anchor2") or [], seq)
        if feats is None:
            n_unaligned += 1
            continue
        n_aligned += 1
        output_records.append({
            "seq": seq,
            "accession": acc,
            "label": r.get("label"),
            **feats,
        })

    cfg["output"].write_text(json.dumps(output_records, indent=2))
    print(f"Wrote {len(output_records)} records to {cfg['output']}")
    print(f"  aligned: {n_aligned}, unalignable: {n_unaligned}, accession fetch failed: {n_no_fetch}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["nes", "nls", "both"], default="both")
    ap.add_argument("--rate-limit", type=float, default=0.5,
                     help="Seconds to sleep between requests (default 0.5 -- be polite to iupred2a.elte.hu)")
    args = ap.parse_args()

    targets = PIPELINES.keys() if args.pipeline == "both" else [args.pipeline]
    for name in targets:
        run_pipeline(name, PIPELINES[name], args.rate_limit)

    print("\nDone. Next step: delete the existing model files so they retrain with the "
          "new features (see the module docstring's 'AFTER RUNNING' section), then "
          "compare permutation/feature importance before vs after.")


if __name__ == "__main__":
    main()
