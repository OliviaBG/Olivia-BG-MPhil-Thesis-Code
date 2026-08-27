"""
Merge NESbase 1.0 (nesbase_parsed.json, from nesbase_parser.py) and NESdb
(nesdb.json, from nesdb_scraper.py) into one flat, ML-ready table: one row
per experimentally-defined NES segment, with a consistent schema across both
source databases.

Usage:
    python build_dataset.py --nesbase nesbase_parsed.json --nesdb nesdb.json --out nes_dataset

Produces:
    nes_dataset.csv
    nes_dataset.json

Columns:
    source                  "NESbase" or "NESdb"
    source_id                accession / NES ID within that source
    protein_name
    organism
    full_sequence            full-length protein sequence (may be None if
                              the source page didn't expose one)
    nes_start, nes_end       1-based inclusive position of the NES within
                              full_sequence (None if unknown/only relative)
    nes_sequence              the NES peptide itself
    critical_positions       ';'-joined 1-based positions shown by
                              mutagenesis to be required for export
    critical_residues        ';'-joined wild-type residues at those positions
    mutation_codes            raw mutation notation as given in the source
                              (e.g. "L78A;L81A;L83A"), for provenance
    crm1_dependent            True / False / '' (unknown) -- CRM1/exportin-1
                              dependence (LMB-sensitivity is used as the
                              standard proxy for CRM1-dependence in both DBs)
    evidence_text              free-text evidence supporting crm1_dependent
    db_reference               UniProt/SwissProt/RefSeq accession, if given
    references                 free-text literature references
"""

import argparse
import csv
import json
import re


def load_json(path):
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  (skipping, not found: {path})")
        return []


MUT_POS_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_mutation_positions(mutation_codes):
    """'L78A' -> (78, 'L'); ignores double mutants written as 'L78D/E79L' by
    splitting on '/' first."""
    out = []
    for code in mutation_codes:
        for single in code.split("/"):
            m = MUT_POS_RE.match(single.strip())
            if m:
                wt, pos, _mut = m.groups()
                out.append((int(pos), wt))
    return out


def row_from_nesbase(rec):
    rows = []
    for seg in rec.get("nes_segments", []):
        rows.append({
            "source": "NESbase",
            "source_id": rec["accession"],
            "protein_name": rec["protein"],
            "organism": rec.get("organism"),
            "full_sequence": rec.get("full_sequence"),
            "nes_start": seg["start"],
            "nes_end": seg["end"],
            "nes_sequence": seg["sequence"],
            "critical_positions": ";".join(str(p) for p in seg["critical_positions"]),
            "critical_residues": ";".join(seg["critical_residues"]),
            "mutation_codes": "",  # NESbase doesn't give discrete mutation codes,
                                    # only the annotation-line 'M' flags already
                                    # captured above
            "crm1_dependent": rec.get("crm1_dependent"),
            "evidence_text": rec.get("crm1_evidence_text"),
            "db_reference": rec.get("db_reference"),
            "references": rec.get("references"),
        })
    if not rows:
        # keep proteins with no explicit segment (e.g. only a necessity
        # region was given, no exact start/end) as a placeholder row so the
        # necessity/sufficiency text isn't silently dropped
        rows.append({
            "source": "NESbase",
            "source_id": rec["accession"],
            "protein_name": rec["protein"],
            "organism": rec.get("organism"),
            "full_sequence": rec.get("full_sequence"),
            "nes_start": None,
            "nes_end": None,
            "nes_sequence": None,
            "critical_positions": "",
            "critical_residues": "",
            "mutation_codes": "",
            "crm1_dependent": rec.get("crm1_dependent"),
            "evidence_text": rec.get("crm1_evidence_text"),
            "db_reference": rec.get("db_reference"),
            "references": rec.get("references"),
        })
    return rows


def row_from_nesdb(rec):
    rows = []
    mut_positions = parse_mutation_positions(rec.get("mutations_affecting_export", []))

    signals = rec.get("export_signals", [])
    if not signals:
        signals = [None]

    for sig in signals:
        if sig is not None:
            start, end, seq = sig["start"], sig["end"], sig["sequence"]
            crit = [(p, wt) for p, wt in mut_positions if start <= p <= end]
        else:
            start = end = seq = None
            crit = mut_positions  # can't scope to a region, report them all

        # de-duplicate positions that show up in more than one mutation code
        # (e.g. "L78A" and "L78D/E79L" both reference position 78)
        seen_pos = set()
        dedup_crit = []
        for p, wt in crit:
            if p in seen_pos:
                continue
            seen_pos.add(p)
            dedup_crit.append((p, wt))
        crit = dedup_crit

        rows.append({
            "source": "NESdb",
            "source_id": rec["nes_id"],
            "protein_name": rec.get("name"),
            "organism": rec.get("organism"),
            "full_sequence": rec.get("full_sequence"),
            "nes_start": start,
            "nes_end": end,
            "nes_sequence": seq,
            "critical_positions": ";".join(str(p) for p, _ in crit),
            "critical_residues": ";".join(wt for _, wt in crit),
            "mutation_codes": ";".join(rec.get("mutations_affecting_export", [])),
            "crm1_dependent": rec.get("crm1_dependent"),
            "evidence_text": rec.get("crm1_evidence_text"),
            "db_reference": rec.get("full_name"),
            "references": rec.get("references"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nesbase", default="nesbase_parsed.json")
    ap.add_argument("--nesdb", default="nesdb.json")
    ap.add_argument("--out", default="nes_dataset")
    args = ap.parse_args()

    nesbase_records = load_json(args.nesbase)
    nesdb_records = load_json(args.nesdb)

    rows = []
    for rec in nesbase_records:
        rows.extend(row_from_nesbase(rec))
    for rec in nesdb_records:
        rows.extend(row_from_nesdb(rec))

    fieldnames = [
        "source", "source_id", "protein_name", "organism", "full_sequence",
        "nes_start", "nes_end", "nes_sequence", "critical_positions",
        "critical_residues", "mutation_codes", "crm1_dependent",
        "evidence_text", "db_reference", "references",
    ]

    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    n_with_seq = sum(1 for r in rows if r["nes_sequence"])
    n_crm1_true = sum(1 for r in rows if r["crm1_dependent"] is True)
    n_crm1_false = sum(1 for r in rows if r["crm1_dependent"] is False)
    n_crit = sum(1 for r in rows if r["critical_positions"])

    print(f"NESbase records: {len(nesbase_records)}, NESdb records: {len(nesdb_records)}")
    print(f"Total rows (one per NES segment): {len(rows)}")
    print(f"  rows with an explicit NES peptide sequence: {n_with_seq}")
    print(f"  rows with >=1 mutagenesis-critical residue: {n_crit}")
    print(f"  CRM1-dependent: {n_crm1_true}, CRM1-independent/other pathway: {n_crm1_false}, "
          f"unknown: {len(rows) - n_crm1_true - n_crm1_false}")
    print(f"Wrote {args.out}.csv and {args.out}.json")


if __name__ == "__main__":
    main()
