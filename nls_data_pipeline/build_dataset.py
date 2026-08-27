"""
Assembles nls_dataset.csv (positives) and nls_negatives.csv (hard negatives)
from the real UniProt data pulled into this folder:

    nls_proteins_uniprot_seed.json       -- proteins with real sequence +
        UniProt "Motif" features whose description contains "nuclear
        localization signal" (or "NLS"). Pulled live from
        rest.uniprot.org/uniprotkb/search (reviewed:true Swiss-Prot only),
        fields=accession,organism_name,ft_motif,sequence, across four taxa
        (human/mouse/yeast/Arabidopsis) plus one untargeted reviewed:true
        pull. See README.md for exact queries and how to pull far more than
        this project's snapshot.

    dna_bind_hard_negative_seed.json     -- proteins with real sequence +
        UniProt "DNA binding" region features. Used as structural/functional
        hard negatives: basic, K/R-rich stretches that are NOT nuclear
        import signals -- the single most literature-documented failure
        mode of naive NLS predictors (plain basic-patch matching gives
        ~45% accuracy per the seqNLS/NLStradamus/NucPred comparison
        studies -- see ../NLS_predictor_landscape_and_novelty.md).

    uniprot_nls_motifs_broad_563acc.json -- broader motif-only pull (563
        unique accessions, no sequence) kept for reference / provenance;
        not used directly here since it lacks sequence.

Output columns (nls_dataset.csv): accession, organism, full_sequence,
nls_sequence, start, end, bipartite, evidence_codes, pubmed_ids,
confidence (experimental | curated_rule).

Output columns (nls_negatives.csv): accession, organism, full_sequence,
neg_sequence, start, end, neg_type (dna_binding_hard | protein_matched_random
| synthetic_polybasic_decoy).
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

POS_SEED = HERE / "nls_proteins_uniprot_seed.json"
DNA_SEED = HERE / "dna_bind_hard_negative_seed.json"
DATASET_CSV = HERE / "nls_dataset.csv"
NEGATIVES_CSV = HERE / "nls_negatives.csv"


def build_positives():
    proteins = json.load(open(POS_SEED, encoding="utf-8"))
    rows = []
    seen = set()
    for p in proteins:
        seq = p["sequence"]
        for m in p["nls_motifs"]:
            start, end = m.get("start"), m.get("end")
            if start is None or end is None or start < 1 or end > len(seq):
                continue
            nls_seq = seq[start - 1:end]
            if not nls_seq or len(nls_seq) < 3:
                continue
            key = (p["accession"], start, end)
            if key in seen:
                continue
            seen.add(key)
            confidence = "experimental" if "ECO:0000269" in m["evidence_codes"] else "curated_rule"
            rows.append({
                "accession": p["accession"], "organism": p["organism"],
                "full_sequence": seq, "nls_sequence": nls_seq,
                "start": start, "end": end,
                "bipartite": int(m["bipartite"]),
                "evidence_codes": ";".join(m["evidence_codes"]),
                "pubmed_ids": ";".join(m["pubmed_ids"]),
                "confidence": confidence,
                "description": m["description"],
            })
    return rows


def build_negatives(pos_rows, neg_per_pos=2, seed=42):
    rng = random.Random(seed)
    neg_rows = []

    # ---- (a) same-protein matched random windows, outside any annotated NLS ----
    by_acc = {}
    for r in pos_rows:
        by_acc.setdefault(r["accession"], {"full_sequence": r["full_sequence"], "spans": []})
        by_acc[r["accession"]]["spans"].append((r["start"], r["end"]))

    for acc, info in by_acc.items():
        seq = info["full_sequence"]
        spans = info["spans"]
        win_len = max(4, sum(e - s + 1 for s, e in spans) // len(spans))
        made, tries = 0, 0
        while made < neg_per_pos and tries < neg_per_pos * 30 and len(seq) > win_len + 1:
            tries += 1
            i = rng.randint(0, len(seq) - win_len)
            j = i + win_len
            if any(not (j <= s - 1 or i >= e) for s, e in spans):
                continue
            neg_rows.append({
                "accession": acc, "organism": None, "full_sequence": seq,
                "neg_sequence": seq[i:j], "start": i + 1, "end": j,
                "neg_type": "protein_matched_random",
            })
            made += 1

    # ---- (b) DNA-binding hard negatives (basic patches that are NOT NLS) ----
    dna_proteins = json.load(open(DNA_SEED, encoding="utf-8"))
    pos_spans_by_acc = {}
    for r in pos_rows:
        pos_spans_by_acc.setdefault(r["accession"], []).append((r["start"], r["end"]))

    for p in dna_proteins:
        seq = p["sequence"]
        nls_spans = [(s, e) for s, e in p.get("nls_spans", []) if s and e]
        nls_spans += pos_spans_by_acc.get(p["accession"], [])
        for region in p["dna_bind_regions"]:
            s, e = region["start"], region["end"]
            if s is None or e is None or s < 1 or e > len(seq):
                continue
            if any(not (e <= ns - 1 or s >= ne) for ns, ne in nls_spans):
                continue  # overlaps a real/candidate NLS -- skip, not a clean negative
            window = seq[s - 1:e]
            if len(window) < 3:
                continue
            neg_rows.append({
                "accession": p["accession"], "organism": p["organism"], "full_sequence": seq,
                "neg_sequence": window, "start": s, "end": e,
                "neg_type": "dna_binding_hard",
            })

    # ---- (c) synthetic polybasic decoys: stress-test naive "count K/R" baselines ----
    # This is the field's most literature-documented failure mode (NucPred/
    # NLStradamus/seqNLS comparison studies report ~45% accuracy for methods
    # that lean on basic-residue density alone -- see landscape doc). These
    # decoys are K/R-rich but randomly ordered / not real recognized signals.
    # Given its own Random(seed) instance, decoupled from the rng
    # used in (a)/(b) above. Before this fix, (c) shared a single rng object
    # with (a) -- so simply appending new proteins to the seed file (adding
    # more protein-matched-random draws in (a) before reaching (c)) would
    # silently reseed/change all 80 synthetic decoys on every dataset rebuild,
    # even though nothing about the decoys themselves was meant to change.
    # A dedicated instance makes (c) reproducible independent of how many
    # real proteins are in the dataset.
    decoy_rng = random.Random(seed)
    AA_BASIC = "KR"
    AA_OTHER = "ACDEFGHILMNPQSTVWY"
    for i in range(80):
        length = decoy_rng.choice([7, 8, 9, 10, 11, 17, 18, 19, 20])
        n_basic = max(3, int(length * decoy_rng.uniform(0.4, 0.7)))
        chars = [decoy_rng.choice(AA_BASIC) for _ in range(n_basic)] + \
                [decoy_rng.choice(AA_OTHER) for _ in range(length - n_basic)]
        decoy_rng.shuffle(chars)
        neg_rows.append({
            "accession": f"synthetic_decoy_{i}", "organism": None, "full_sequence": None,
            "neg_sequence": "".join(chars), "start": None, "end": None,
            "neg_type": "synthetic_polybasic_decoy",
        })

    return neg_rows


def main():
    pos_rows = build_positives()
    neg_rows = build_negatives(pos_rows)

    with open(DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pos_rows[0].keys()))
        w.writeheader()
        w.writerows(pos_rows)

    with open(NEGATIVES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(neg_rows[0].keys()))
        w.writeheader()
        w.writerows(neg_rows)

    n_exp = sum(1 for r in pos_rows if r["confidence"] == "experimental")
    n_bip = sum(1 for r in pos_rows if r["bipartite"])
    by_type = {}
    for r in neg_rows:
        by_type[r["neg_type"]] = by_type.get(r["neg_type"], 0) + 1
    print(f"Wrote {DATASET_CSV.name}: {len(pos_rows)} positive NLS windows "
          f"({n_exp} experimental evidence, {n_bip} bipartite-labeled)")
    print(f"Wrote {NEGATIVES_CSV.name}: {len(neg_rows)} negatives {by_type}")


if __name__ == "__main__":
    main()
