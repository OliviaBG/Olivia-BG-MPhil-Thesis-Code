#!/usr/bin/env python3
"""
append_bipartite_nonclassical_monopartite_candidates.py
============================================================
merges the output of fetch_bipartite_and_nonclassical_candidates.py
into nls_dataset.csv/nls_negatives.csv. Same append-only pattern as
append_viral_candidates.py (see that script's docstring for why this project
never does a full build_dataset.py rebuild anymore -- there are hand-added
rows in both CSVs that aren't reproducible from the seed JSONs).

Nothing is dropped: this merges ALL FOUR candidate
files fetch_bipartite_and_nonclassical_candidates.py produces --
bipartite_nls_candidates_<date>.json, nonclassical_nls_candidates_<date>.json,
monopartite_nls_candidates_<date>.json (the already-well-represented
278/294 class, previously discarded, now kept on request), AND
dna_binding_hard_candidates_<date>.json ( : real DNA-binding-domain
hard negatives, added to fix the hard-negative dilution the first three
buckets caused on their own -- see fetch_dna_binding_proteins()'s
docstring in the fetch script) -- in one pass.

Every row is real: real UniProt accession, real curated Motif/DNA-binding
feature, real evidence codes/PubMed IDs, real sequence slice, taken
directly from what the fetch script pulled from rest.uniprot.org. Nothing
here is synthesized except the matched-random negative windows, which
follow the exact same rule build_dataset.py/append_viral_candidates.py
already use elsewhere in this project (protein_matched_random: a
same-length window from the same protein that doesn't overlap any real
annotated NLS span for that protein).

Usage (from the nls_data_pipeline directory):
    python3 append_bipartite_nonclassical_monopartite_candidates.py
    # or point it at specific files:
    python3 append_bipartite_nonclassical_monopartite_candidates.py \\
        --bipartite bipartite_nls_candidates_2026-08-06.json \\
        --nonclassical nonclassical_nls_candidates_2026-08-06.json \\
        --monopartite monopartite_nls_candidates_2026-08-06.json \\
        --dna-binding dna_binding_hard_candidates_2026-08-06.json
"""
import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_CSV = HERE / "nls_dataset.csv"
NEGATIVES_CSV = HERE / "nls_negatives.csv"

NEG_PER_POS = 2
RNG_SEED = 42  # same seed convention as append_viral_candidates.py / build_dataset.py,
               # own dedicated Random() instance so it can't perturb any other RNG stream


def _latest(pattern):
    matches = sorted(HERE.glob(pattern))
    return matches[-1] if matches else None


def load_candidates(path):
    if path is None or not Path(path).exists():
        return []
    return json.load(open(path, encoding="utf-8"))["candidates"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bipartite", default=None, help="defaults to the newest bipartite_nls_candidates_*.json")
    ap.add_argument("--nonclassical", default=None, help="defaults to the newest nonclassical_nls_candidates_*.json")
    ap.add_argument("--monopartite", default=None, help="defaults to the newest monopartite_nls_candidates_*.json")
    ap.add_argument("--dna-binding", default=None, help="defaults to the newest dna_binding_hard_candidates_*.json")
    args = ap.parse_args()

    bip_path = args.bipartite or _latest("bipartite_nls_candidates_*.json")
    nc_path = args.nonclassical or _latest("nonclassical_nls_candidates_*.json")
    mono_path = args.monopartite or _latest("monopartite_nls_candidates_*.json")
    dna_path = getattr(args, "dna_binding") or _latest("dna_binding_hard_candidates_*.json")

    print("Merging from:")
    print(f"  bipartite:     {bip_path}")
    print(f"  non-classical: {nc_path}")
    print(f"  monopartite:   {mono_path}")
    print(f"  dna-binding:   {dna_path}")
    if mono_path is None:
        print("\n  WARNING: no monopartite_nls_candidates_*.json found -- that file only exists "
              "if you re-ran fetch_bipartite_and_nonclassical_candidates.py AFTER the "
              "fix that stopped it from silently dropping the monopartite bucket. Re-run the fetch "
              "script first if you want monopartite rows included too.")
    if dna_path is None:
        print("\n  WARNING: no dna_binding_hard_candidates_*.json found -- that file only exists "
              "if you re-ran fetch_bipartite_and_nonclassical_candidates.py AFTER the "
              "fix that added DNA-binding hard-negative fetching. Without it, the hard-negative "
              "dilution problem from the first merge will NOT be fixed by this run.")
    print()

    candidate_proteins = {}
    for path in (bip_path, nc_path, mono_path):
        for p in load_candidates(path):
            candidate_proteins.setdefault(p["accession"], p)

    dna_proteins = {}
    for p in load_candidates(dna_path):
        dna_proteins.setdefault(p["accession"], p)

    if not candidate_proteins and not dna_proteins:
        raise SystemExit("No candidates found in any of the four files -- nothing to merge.")

    # ---- existing accessions/keys, for collision + dedup safety (same as append_viral_candidates.py) ----
    existing_pos_rows = list(csv.DictReader(open(DATASET_CSV, encoding="utf-8")))
    existing_pos_keys = {(r["accession"], r["start"], r["end"]) for r in existing_pos_rows}
    existing_accessions = {r["accession"] for r in existing_pos_rows}

    collisions = [acc for acc in candidate_proteins if acc in existing_accessions]
    if collisions:
        raise SystemExit(f"REFUSING to proceed: {len(collisions)} accession(s) (e.g. {collisions[:5]}) "
                          f"already present in nls_dataset.csv -- resolve manually first (the fetch "
                          f"script should have already excluded these; nls_dataset.csv may have "
                          f"changed since the candidates were fetched).")

    existing_negative_accessions = {r["accession"] for r in csv.DictReader(open(NEGATIVES_CSV, encoding="utf-8"))}
    dna_collisions = [acc for acc in dna_proteins if acc in existing_negative_accessions]
    if dna_collisions:
        raise SystemExit(f"REFUSING to proceed: {len(dna_collisions)} accession(s) (e.g. {dna_collisions[:5]}) "
                          f"already present in nls_negatives.csv -- resolve manually first.")

    # BUGFIX (second line of defense): the first live run of
    # fetch_bipartite_and_nonclassical_candidates.py re-scraped 23 of the 25
    # held-out eval accessions straight back into nls_dataset.csv, because
    # its own exclusion check only covered nls_dataset.csv, not the holdout
    # file -- see decontaminate_holdout_leakage.py and
    # load_holdout_accessions()'s docstring in the fetch script for the full
    # story. That fetch-side gap is now fixed, but this merge script refuses
    # independently too, in case it's ever pointed at an older/hand-edited
    # candidates file that predates that fix.
    holdout_path = HERE.parent / "nls_holdout_data" / "candidates.json"
    if holdout_path.exists():
        holdout = json.loads(holdout_path.read_text())
        holdout_accs = {p[1] for p in holdout.get("positives", [])} | {n[1] for n in holdout.get("negatives", [])}
        holdout_collisions = [acc for acc in candidate_proteins if acc in holdout_accs]
        holdout_collisions += [acc for acc in dna_proteins if acc in holdout_accs]
        if holdout_collisions:
            raise SystemExit(
                f"REFUSING to proceed: {len(holdout_collisions)} accession(s) "
                f"(e.g. {holdout_collisions[:5]}) are in nls_holdout_data/candidates.json "
                f"(the held-out eval set) -- merging these would contaminate the holdout "
                f"evaluation. Re-run fetch_bipartite_and_nonclassical_candidates.py (now fixed "
                f"to exclude these) to regenerate clean candidate files."
            )
    else:
        print(f"  WARNING: {holdout_path} not found -- can't verify these candidates don't "
              f"overlap the holdout eval set. Proceed with caution.")

    # ---- build new positive rows (same logic as build_dataset.py build_positives()) ----
    new_pos_rows = []
    seen_keys = set()
    for acc, p in candidate_proteins.items():
        seq = p["sequence"]
        for m in p["nls_motifs"]:
            start, end = m.get("start"), m.get("end")
            if start is None or end is None or start < 1 or end > len(seq):
                continue
            nls_seq = seq[start - 1:end]
            if not nls_seq or len(nls_seq) < 3:
                continue
            key = (acc, str(start), str(end))
            if key in existing_pos_keys or key in seen_keys:
                continue
            seen_keys.add(key)
            confidence = "experimental" if "ECO:0000269" in m["evidence_codes"] else "curated_rule"
            new_pos_rows.append({
                "accession": acc, "organism": p["organism"],
                "full_sequence": seq, "nls_sequence": nls_seq,
                "start": start, "end": end,
                "bipartite": int(m["bipartite"]),
                "evidence_codes": ";".join(m["evidence_codes"]),
                "pubmed_ids": ";".join(m["pubmed_ids"]),
                "confidence": confidence,
                "description": m["description"],
            })

    # ---- build matching protein_matched_random negatives, dedicated RNG (same as append_viral_candidates.py) ----
    rng = random.Random(RNG_SEED)
    by_acc = {}
    for r in new_pos_rows:
        by_acc.setdefault(r["accession"], {"full_sequence": r["full_sequence"], "spans": []})
        by_acc[r["accession"]]["spans"].append((r["start"], r["end"]))

    new_neg_rows = []
    for acc, info in by_acc.items():
        seq = info["full_sequence"]
        spans = info["spans"]
        win_len = max(4, sum(e - s + 1 for s, e in spans) // len(spans))
        made, tries = 0, 0
        while made < NEG_PER_POS and tries < NEG_PER_POS * 30 and len(seq) > win_len + 1:
            tries += 1
            i = rng.randint(0, len(seq) - win_len)
            j = i + win_len
            if any(not (j <= s - 1 or i >= e) for s, e in spans):
                continue
            new_neg_rows.append({
                "accession": acc, "organism": None, "full_sequence": seq,
                "neg_sequence": seq[i:j], "start": i + 1, "end": j,
                "neg_type": "protein_matched_random",
            })
            made += 1

    # ---- build dna_binding_hard negatives, same rule as build_dataset.py's
    # build_negatives() part (b): a real curated DNA-binding region becomes
    # a hard negative UNLESS it overlaps a real/candidate NLS span for that
    # same protein -- checked against both the candidate file's own
    # nls_spans field (UniProt Motif features on that protein) AND every
    # positive span this accession has anywhere in nls_dataset.csv
    # (existing rows plus anything just added above in this same run, in
    # case the same protein appeared in both a positive and dna-binding
    # candidate file this run). ----
    pos_spans_by_acc = {}
    for r in existing_pos_rows + new_pos_rows:
        if r.get("start") and r.get("end"):
            pos_spans_by_acc.setdefault(r["accession"], []).append((int(r["start"]), int(r["end"])))

    new_dna_neg_rows = []
    for acc, p in dna_proteins.items():
        seq = p["sequence"]
        nls_spans = [(s, e) for s, e in p.get("nls_spans", []) if s and e]
        nls_spans += pos_spans_by_acc.get(acc, [])
        for region in p["dna_bind_regions"]:
            s, e = region.get("start"), region.get("end")
            if s is None or e is None or s < 1 or e > len(seq):
                continue
            # same non-overlap test as build_dataset.py's build_negatives() part (b),
            # kept identical (not swapped for the module-level overlaps() helper
            # above) so old and new dna_binding_hard rows use one consistent
            # boundary convention rather than two subtly different ones.
            if any(not (e <= ns - 1 or s >= ne) for ns, ne in nls_spans):
                continue  # overlaps a real/candidate NLS -- skip, not a clean negative
            window = seq[s - 1:e]
            if len(window) < 3:
                continue
            new_dna_neg_rows.append({
                "accession": acc, "organism": p["organism"], "full_sequence": seq,
                "neg_sequence": window, "start": s, "end": e,
                "neg_type": "dna_binding_hard",
            })

    # ---- append (not overwrite) ----
    with open(DATASET_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(existing_pos_rows[0].keys()))
        w.writerows(new_pos_rows)

    existing_neg_rows = list(csv.DictReader(open(NEGATIVES_CSV, encoding="utf-8")))
    all_new_neg_rows = new_neg_rows + new_dna_neg_rows
    with open(NEGATIVES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(existing_neg_rows[0].keys()))
        w.writerows(all_new_neg_rows)

    n_bip = sum(1 for r in new_pos_rows if r["bipartite"] == 1)
    n_mono = len(new_pos_rows) - n_bip
    print(f"Appended {len(new_pos_rows)} positive NLS rows from {len(candidate_proteins)} "
          f"proteins to {DATASET_CSV.name} (was {len(existing_pos_rows)} rows, now "
          f"{len(existing_pos_rows) + len(new_pos_rows)})")
    print(f"  of which {n_bip} are bipartite-flagged, {n_mono} are not")
    print(f"Appended {len(all_new_neg_rows)} negatives to {NEGATIVES_CSV.name} (was "
          f"{len(existing_neg_rows)} rows, now {len(existing_neg_rows) + len(all_new_neg_rows)})")
    print(f"  of which {len(new_neg_rows)} are protein_matched_random, "
          f"{len(new_dna_neg_rows)} are dna_binding_hard (from {len(dna_proteins)} scraped proteins)")
    old_hard = sum(1 for r in existing_neg_rows if r["neg_type"] != "protein_matched_random")
    new_hard_total = old_hard + len(new_dna_neg_rows)
    new_total = len(existing_neg_rows) + len(all_new_neg_rows)
    print(f"  hard-negative share: {old_hard}/{len(existing_neg_rows)} "
          f"({100*old_hard/max(1,len(existing_neg_rows)):.1f}%) before -> "
          f"{new_hard_total}/{new_total} ({100*new_hard_total/max(1,new_total):.1f}%) after")
    print(f"\nNOTE: new class balance will need a retrain to take effect -- run "
          f"`python3 nls_ml_predictor.py train` (same as task #24 the last time this "
          f"dataset was expanded), then re-run the holdout comparison figures.")


if __name__ == "__main__":
    main()
