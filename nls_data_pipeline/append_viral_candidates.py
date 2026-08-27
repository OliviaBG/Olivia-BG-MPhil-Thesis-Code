#!/usr/bin/env python3
"""
append_viral_candidates.py
============================================================
append-only merge of the 20 curated viral NLS candidates
(viral_nls_candidates_2026-08-05.json) into nls_dataset.csv/nls_negatives.csv.

Deliberately NOT done via build_dataset.py's full rebuild: a diff against a
pre-change backup showed both nls_dataset.csv and nls_negatives.csv already
contain rows that can't be reproduced from the current seed JSON files (e.g.
4 positive rows -- P35637, P06748, P12956, P03428 -- and 20 negative rows in
neg_type categories build_negatives() doesn't even generate: linker_histone_
paralog, extreme_arg_condensin, membrane_anchor_caax, heparin_binding_
chemokine, calmodulin_pip2_effector). Those were added by hand in an earlier
session (consistent with nls_negatives_backup_before_hardneg_expansion_ .csv existing) and aren't captured anywhere build_dataset.py can
regenerate them from. Running the full rebuild would have silently dropped
all 24 of those real, already-calibrated-against rows. This script instead
appends new rows for exactly the 20 new proteins, touching nothing else --
same row schema/logic as build_dataset.py's build_positives()/build_negatives()
part (a), just scoped to the new proteins only.

Usage: python3 append_viral_candidates.py
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "viral_nls_candidates_2026-08-05.json"
DATASET_CSV = HERE / "nls_dataset.csv"
NEGATIVES_CSV = HERE / "nls_negatives.csv"

NEG_PER_POS = 2
RNG_SEED = 42  # same seed build_dataset.py uses, but its own dedicated instance


def main():
    candidates = json.load(open(CANDIDATES, encoding="utf-8"))["candidates"]

    # ---- existing accessions/keys, for collision + dedup safety ----
    existing_pos_rows = list(csv.DictReader(open(DATASET_CSV, encoding="utf-8")))
    existing_pos_keys = {(r["accession"], r["start"], r["end"]) for r in existing_pos_rows}
    existing_accessions = {r["accession"] for r in existing_pos_rows}

    collisions = [c["accession"] for c in candidates if c["accession"] in existing_accessions]
    if collisions:
        raise SystemExit(f"REFUSING to proceed: {collisions} already present in "
                          f"nls_dataset.csv -- resolve manually first.")

    # ---- build new positive rows (same logic as build_dataset.py build_positives()) ----
    new_pos_rows = []
    for p in candidates:
        seq = p["sequence"]
        for m in p["nls_motifs"]:
            start, end = m.get("start"), m.get("end")
            if start is None or end is None or start < 1 or end > len(seq):
                raise SystemExit(f"Bad motif bounds for {p['accession']}: {m}")
            nls_seq = seq[start - 1:end]
            if not nls_seq or len(nls_seq) < 3:
                raise SystemExit(f"Motif too short for {p['accession']}: {m}")
            key = (p["accession"], str(start), str(end))
            if key in existing_pos_keys:
                continue
            confidence = "experimental" if "ECO:0000269" in m["evidence_codes"] else "curated_rule"
            new_pos_rows.append({
                "accession": p["accession"], "organism": p["organism"],
                "full_sequence": seq, "nls_sequence": nls_seq,
                "start": start, "end": end,
                "bipartite": int(m["bipartite"]),
                "evidence_codes": ";".join(m["evidence_codes"]),
                "pubmed_ids": ";".join(m["pubmed_ids"]),
                "confidence": confidence,
                "description": m["description"],
            })

    # ---- build matching protein_matched_random negatives, dedicated RNG ----
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

    # ---- append (not overwrite) ----
    with open(DATASET_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(existing_pos_rows[0].keys()))
        w.writerows(new_pos_rows)

    existing_neg_rows = list(csv.DictReader(open(NEGATIVES_CSV, encoding="utf-8")))
    with open(NEGATIVES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(existing_neg_rows[0].keys()))
        w.writerows(new_neg_rows)

    print(f"Appended {len(new_pos_rows)} positive NLS windows from {len(candidates)} "
          f"proteins to {DATASET_CSV.name} (was {len(existing_pos_rows)} rows, now "
          f"{len(existing_pos_rows) + len(new_pos_rows)})")
    print(f"Appended {len(new_neg_rows)} protein_matched_random negatives to "
          f"{NEGATIVES_CSV.name} (was {len(existing_neg_rows)} rows, now "
          f"{len(existing_neg_rows) + len(new_neg_rows)})")


if __name__ == "__main__":
    main()
