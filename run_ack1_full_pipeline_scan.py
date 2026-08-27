#!/usr/bin/env python3
"""
run_ack1_full_pipeline_scan.py
============================================================
Whole-protein NES + NLS candidate scan through THIS project's real, live
scoring pipeline -- exactly as the running app would score it, not a
reimplementation. Same "import app.py, use Flask's test_client()" approach
as run_holdout_pipeline_test.py / run_nls_holdout_pipeline_test.py, just
aimed at every candidate window in one protein instead of a fixed
positive/negative holdout list.

Default target: human ACK1/TNK2, UniProt Q07912, 1038 aa canonical
Swiss-Prot entry (the one AlphaFold DB has a structure for -- see
prototype_cider_rsa_panels.py's docstring for why the longer TrEMBL
isoforms aren't usable here).

Pipeline exercised (real production code paths, not copies):
  NES:  GET /api/unified_crm1_nes/<model_id>?uniprot_id=...
        -> downloads the real AlphaFold structure, runs fpocket via
           CRM1AwarePocketDetector, runs the full ML NES predictor
           (nes_ml_predictor_improved.py), combines ML + pocket + sequence
           + structural (SASA/disorder/hydrophobicity/pLDDT) features into
           combined_score, filters (>0.45) and greedily removes overlaps.
  NLS:  GET /api/structure/<model_id>?uniprot_id=...  (real pLDDT + consensus
           RSA/SASA for the whole protein, same route the frontend calls on
           load)
        -> POST /api/nls_scan with that real structural data
           -> NLSPredictor.scan_sequence() (nls_ml_predictor.py) + the real
              accessibility gate, membrane-anchor veto, DNA-binding-domain
              veto, filtered (>0.5, or py_nls_shaped) and greedily
              deduplicated by score, exactly as the live app does.

This file does not import, edit, or monkeypatch anything in app.py -- it
only calls the same public routes the deployed frontend calls, via
Flask's test_client() so no separate server process is needed.

REQUIREMENTS: real internet access (AlphaFold DB + UniProt + IUPred2A) and
a working fpocket install -- i.e. run this on the pod / your own machine,
same requirement run_holdout_pipeline_test.py already documents. Won't
produce real results in an environment with no network egress.

Usage:
    python3 run_ack1_full_pipeline_scan.py
    python3 run_ack1_full_pipeline_scan.py --accession Q07912 --top-n 15
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def fetch_structural_data(client, accession):
    """Real /api/structure/<model_id> call -- same route the frontend uses
    on initial load. Returns (data_dict, status_str)."""
    model_id = f"AF-{accession}-F1"
    try:
        resp = client.get(f"/api/structure/{model_id}?uniprot_id={accession}")
    except Exception as e:
        return None, f"network_error: {e}"
    try:
        data = resp.get_json()
    except Exception as e:
        return None, f"network_error: bad JSON ({e})"
    if data is None:
        return None, "network_error: empty response"
    if "error" in data:
        return None, f"error: {data['error']}"
    if not data.get("plddt"):
        return None, "no_plddt: structure returned but plddt array empty"
    return data, "ok"


def run_nes_scan(client, accession):
    """Real /api/unified_crm1_nes/<model_id> call -- full ML + fpocket
    CRM1-pocket pipeline. Returns the already-ranked, already-overlap-
    filtered nes_motifs list straight from app.py, or [] + an error string."""
    model_id = f"AF-{accession}-F1"
    print(f"Calling /api/unified_crm1_nes/{model_id} (ML predictor + fpocket "
          f"CRM1 pocket detection -- this is the slow step, can take several "
          f"minutes for a {'~1000+' if True else ''}-residue protein)...")
    resp = client.get(f"/api/unified_crm1_nes/{model_id}?uniprot_id={accession}")
    try:
        data = resp.get_json()
    except Exception as e:
        return [], f"bad JSON: {e}"
    if data is None:
        return [], f"empty response, HTTP {resp.status_code}"
    if "error" in data:
        return [], data["error"]
    return data.get("nes_motifs", []), None


def run_nls_scan(client, accession, struct_data):
    """Real POST /api/nls_scan call using the real structural arrays fetched
    above. Returns the already-ranked, already-gated nls_binding_regions
    list straight from app.py, or [] + an error string."""
    payload = {
        "sequence": struct_data.get("sequence", ""),
        "model_id": accession,
        "uniprot_id": accession,
    }
    for key in ("plddt", "sasa", "consensus_z", "agreement_sd"):
        if struct_data.get(key):
            payload[key] = struct_data[key]

    print("Calling /api/nls_scan (ML NLS classifier + accessibility/anchor/"
          "DNA-binding gates, using real structural data)...")
    resp = client.post("/api/nls_scan", json=payload)
    try:
        data = resp.get_json()
    except Exception as e:
        return [], f"bad JSON: {e}"
    if data is None:
        return [], f"empty response, HTTP {resp.status_code}"
    if "error" in data:
        return [], data["error"]
    return data.get("nls_binding_regions", []), None


def print_nes_table(motifs, top_n):
    print(f"\nTop {min(top_n, len(motifs))} NES candidates (of {len(motifs)} surviving, "
          f"combined_score > 0.45, non-overlapping):")
    print(f"{'rank':<5}{'pos':<12}{'sequence':<20}{'score':<8}{'ml_prob':<9}"
          f"{'crm1_aff':<10}{'pocket':<8}{'hydro':<8}{'disorder':<9}{'sasa':<7}")
    for i, m in enumerate(sorted(motifs, key=lambda x: x["combined_score"], reverse=True)[:top_n], 1):
        c = m.get("components", {})
        pos = f"{m['start']}-{m['end']}"
        print(f"{i:<5}{pos:<12}{m['sequence']:<20}"
              f"{m['combined_score']:<8.3f}{c.get('ml_probability', 0):<9.3f}"
              f"{c.get('crm1_binding_affinity', 0):<10.3f}{c.get('pocket_compatibility', 0):<8.3f}"
              f"{c.get('hydrophobicity', 0):<8.3f}{c.get('disorder', 0):<9.3f}"
              f"{c.get('surface_accessibility', 0):<7.3f}")


def print_nls_table(regions, top_n):
    print(f"\nTop {min(top_n, len(regions))} NLS candidates (of {len(regions)} surviving, "
          f"nls_probability > 0.5 or py_nls_shaped):")
    print(f"{'rank':<5}{'pos':<12}{'sequence':<20}{'nls_prob':<10}{'raw_prob':<10}"
          f"{'class':<14}{'bipartite':<11}{'pssm':<8}{'rsa':<7}")
    for i, r in enumerate(sorted(regions, key=lambda x: x["nls_probability"], reverse=True)[:top_n], 1):
        pos = f"{r['start']}-{r['end']}"
        print(f"{i:<5}{pos:<12}{r['sequence']:<20}"
              f"{r['nls_probability']:<10.3f}{r['raw_nls_probability']:<10.3f}"
              f"{str(r.get('predicted_class')):<14}{str(r.get('is_bipartite')):<11}"
              f"{r.get('pssm_score', 0):<8.3f}{r.get('accessibility_rsa', 0):<7.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accession", default="Q07912",
                     help="UniProt accession (default: Q07912, human ACK1/TNK2, 1038 aa)")
    ap.add_argument("--top-n", type=int, default=10, help="how many top candidates to print/save per class")
    ap.add_argument("--fpocket-timeout", type=int, default=300,
                     help="fpocket subprocess timeout in seconds (default 300 -- ACK1 at 1038 "
                          "residues needs more than the app's default)")
    ap.add_argument("--out", default=None,
                     help="output prefix (default: '<accession>_full_pipeline_scan' in this directory)")
    args = ap.parse_args()

    print("Loading app.py (this triggers full ML/CRM1/NLS/fpocket initialization -- may take a moment)...")
    sys.path.insert(0, str(THIS_DIR))
    from app import app as flask_app, pocket_detector
    if pocket_detector is not None:
        pocket_detector.fpocket_timeout = args.fpocket_timeout
    client = flask_app.test_client()

    accession = args.accession
    out_prefix = args.out or f"{accession}_full_pipeline_scan"

    print("\n" + "=" * 100)
    print(f"NES SCAN -- {accession}")
    print("=" * 100)
    nes_motifs, nes_error = run_nes_scan(client, accession)
    if nes_error:
        print(f"  ERROR: {nes_error}")
    else:
        print_nes_table(nes_motifs, args.top_n)

    print("\n" + "=" * 100)
    print(f"NLS SCAN -- {accession}")
    print("=" * 100)
    struct_data, struct_status = fetch_structural_data(client, accession)
    if struct_status != "ok":
        print(f"  Could not fetch real structural data ({struct_status}) -- aborting NLS scan "
              f"(no sequence to scan without it).")
        nls_regions, nls_error = [], struct_status
    else:
        print(f"  Structure OK: {len(struct_data.get('sequence', ''))} residues, "
              f"mean pLDDT available, real consensus RSA available")
        nls_regions, nls_error = run_nls_scan(client, accession, struct_data)
        if nls_error:
            print(f"  ERROR: {nls_error}")
        else:
            print_nls_table(nls_regions, args.top_n)

    # ------------------------------------------------------------------
    # Write real file outputs, not just terminal text.
    # ------------------------------------------------------------------
    out_json = {
        "accession": accession,
        "nes_error": nes_error,
        "nls_error": nls_error,
        "n_nes_candidates": len(nes_motifs),
        "n_nls_candidates": len(nls_regions),
        "nes_motifs": sorted(nes_motifs, key=lambda x: x["combined_score"], reverse=True),
        "nls_binding_regions": sorted(nls_regions, key=lambda x: x["nls_probability"], reverse=True),
    }
    json_path = THIS_DIR / f"{out_prefix}.json"
    json_path.write_text(json.dumps(out_json, indent=2))

    md_lines = [
        f"# Full pipeline candidate scan -- {accession}",
        "",
        "Real production code paths (`/api/unified_crm1_nes` for NES: ML predictor + fpocket "
        "CRM1-pocket detection; `/api/structure` + `/api/nls_scan` for NLS: real structural data "
        "+ ML NLS classifier + accessibility/anchor/DNA-binding gates), run via Flask's "
        "test_client() against the imported app.py -- the exact same code path the deployed "
        "frontend hits, not a reimplementation.",
        "",
        f"NES: {len(nes_motifs)} surviving candidates" + (f" -- ERROR: {nes_error}" if nes_error else ""),
        f"NLS: {len(nls_regions)} surviving candidates" + (f" -- ERROR: {nls_error}" if nls_error else ""),
        "",
        "## Top NES candidates",
        "",
        "| Rank | Position | Sequence | combined_score | ml_probability | crm1_binding_affinity | "
        "pocket_compatibility | hydrophobicity | disorder | surface_accessibility |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, m in enumerate(sorted(nes_motifs, key=lambda x: x["combined_score"], reverse=True)[:args.top_n], 1):
        c = m.get("components", {})
        md_lines.append(
            f"| {i} | {m['start']}-{m['end']} | {m['sequence']} | {m['combined_score']:.3f} | "
            f"{c.get('ml_probability', 0):.3f} | {c.get('crm1_binding_affinity', 0):.3f} | "
            f"{c.get('pocket_compatibility', 0):.3f} | {c.get('hydrophobicity', 0):.3f} | "
            f"{c.get('disorder', 0):.3f} | {c.get('surface_accessibility', 0):.3f} |"
        )

    md_lines += [
        "",
        "## Top NLS candidates",
        "",
        "| Rank | Position | Sequence | nls_probability | raw_nls_probability | predicted_class | "
        "is_bipartite | pssm_score | accessibility_rsa |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(sorted(nls_regions, key=lambda x: x["nls_probability"], reverse=True)[:args.top_n], 1):
        md_lines.append(
            f"| {i} | {r['start']}-{r['end']} | {r['sequence']} | {r['nls_probability']:.3f} | "
            f"{r['raw_nls_probability']:.3f} | {r.get('predicted_class')} | {r.get('is_bipartite')} | "
            f"{r.get('pssm_score', 0):.3f} | {r.get('accessibility_rsa', 0)} |"
        )

    md_path = THIS_DIR / f"{out_prefix}.md"
    md_path.write_text("\n".join(md_lines))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
