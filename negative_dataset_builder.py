"""
negative_dataset_builder.py

Builds a NEGATIVE dataset for NES (Nuclear Export Signal) predictors:
hydrophobic, leucine-rich patches that match the classic NES consensus
regex but sit inside structural contexts that are NOT NESs -- coiled-coil
regions and leucine-zipper domains pulled straight from UniProt.

Rationale
---------
The general NES consensus (Kosugi et al. 2008; used by NetNES, Wregex,
LocNES, NESmapper) is a spacing pattern of 4 hydrophobic residues:

    Phi-x(2,3)-Phi-x(2,3)-Phi-x-Phi      Phi = L, I, V, F, M

This pattern is common in *any* amphipathic/hydrophobic heptad-repeat
structure, most notably coiled-coils and leucine zippers, which have
nothing to do with CRM1-mediated export. That makes them a good, cheap
source of realistic "hard negatives": real protein sequences that trip
the naive regex but are not functional NESs.

Pipeline
--------
1. Query UniProt (REST API, https://rest.uniprot.org) for reviewed
   (Swiss-Prot) entries annotated with a coiled-coil region and/or a
   leucine-zipper region.
2. Pull the annotated span + sequence for each entry.
3. Scan each annotated span (with a small flank) for regex hits against
   the NES consensus.
4. Score each hit with a lightweight, documented PSSM-style weight table
   approximating the idea behind Wregex (Fernandez-Marcos et al. 2014,
   Bioinformatics, "Prediction of nuclear export signals using weighted
   regular expressions") -- i.e. not all Phi-x-Phi-x-Phi-x-Phi matches
   are equally NES-like, so score them instead of treating every regex
   hit as equivalent.
5. Drop any hit that overlaps a UniProt-annotated "Nuclear export signal"
   motif on the same entry (removes accidental true positives).
6. Write the surviving hits out as a CSV + FASTA negative dataset.

IMPORTANT re: scoring accuracy
-------------------------------
The PSSM used here (NES_PSSM below) is a hand-built approximation based
on the published NES consensus classes (Kosugi et al. 2008) -- it is
NOT the exact proprietary matrix used by the real Wregex web tool. If
you need publication-grade scores, download the actual position weight
matrices from http://wregex.ehubio.es and drop them into `NES_PSSM`
(same shape: dict of position -> {residue: weight}). Everything else in
this pipeline (UniProt querying, region extraction, overlap filtering,
I/O) is independent of that choice.

Usage
-----
    pip install requests
    python negative_dataset_builder.py --taxon 9606 --max-entries 500 \
        --min-score 0.5 --outdir ./nes_negatives

Run with --self-test and no network access to sanity-check the regex/
scoring/filtering logic against a small built-in synthetic example.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
HYDROPHOBIC = set("LIVFM")

# General NES consensus (Kosugi et al. 2008 / Wregex / NetNES / LocNES):
#   Phi-x(2,3)-Phi-x(2,3)-Phi-x-Phi
NES_CONSENSUS_REGEX = re.compile(
    r"(?=([LIVFM].{2,3}[LIVFM].{2,3}[LIVFM].[LIVFM]))"
)

# --------------------------------------------------------------------------
# Approximate, documented PSSM (see module docstring for caveats).
# Positions are indices into the 4 "core" hydrophobic slots of a match
# (0 = Phi1, 1 = Phi2, 2 = Phi3, 3 = Phi4). Weights loosely follow the
# residue-frequency tables reported for validated NESs (Leu strongly
# favored at Phi1/Phi4, Ile/Val/Phe/Met more common at Phi2/Phi3).
# --------------------------------------------------------------------------
NES_PSSM = {
    0: {"L": 1.00, "I": 0.55, "V": 0.45, "F": 0.50, "M": 0.60},
    1: {"L": 0.70, "I": 0.75, "V": 0.80, "F": 0.55, "M": 0.65},
    2: {"L": 0.75, "I": 0.80, "V": 0.70, "F": 0.60, "M": 0.65},
    3: {"L": 1.00, "I": 0.60, "V": 0.50, "F": 0.45, "M": 0.55},
}
ACIDIC_FLANK_BONUS = 0.15  # small bonus if D/E present just upstream (common in real NESs)


@dataclass
class Region:
    kind: str          # "coiled_coil" or "leucine_zipper" or "nes"
    start: int         # 0-based, inclusive
    end: int           # 0-based, exclusive


@dataclass
class Hit:
    accession: str
    protein_name: str
    organism: str
    feature_kind: str
    feature_start: int
    feature_end: int
    match_start: int
    match_end: int
    match_seq: str
    context: str
    score: float


def score_hit(match_seq: str, upstream_flank: str) -> float:
    """Score a 4-residue-core NES-consensus hit using NES_PSSM.

    match_seq: the *four hydrophobic residues* extracted from the match,
    in order (Phi1, Phi2, Phi3, Phi4).
    upstream_flank: up to 3 residues immediately before the match, used
    for the acidic-residue bonus heuristic.
    """
    total = 0.0
    for i, aa in enumerate(match_seq):
        total += NES_PSSM.get(i, {}).get(aa, 0.0)
    max_possible = sum(max(w.values()) for w in NES_PSSM.values())
    norm = total / max_possible if max_possible else 0.0
    if any(r in "DE" for r in upstream_flank):
        norm = min(1.0, norm + ACIDIC_FLANK_BONUS)
    return round(norm, 3)


def extract_phi_positions(full_match: str) -> str:
    """Given the full regex match span (e.g. 'LDLTPLAL'), pull out just
    the 4 hydrophobic residues that anchor the consensus (first char,
    then the residue after each variable-length spacer)."""
    # The regex group captures the *entire* span from Phi1 to Phi4
    # inclusive. Re-derive spacer lengths to recover the 4 anchors.
    m = re.match(r"([LIVFM])(.{2,3})([LIVFM])(.{2,3})([LIVFM]).([LIVFM])", full_match)
    if not m:
        return ""
    return m.group(1) + m.group(3) + m.group(5) + m.group(6)


def scan_region_for_hits(
    sequence: str,
    region: Region,
    accession: str,
    protein_name: str,
    organism: str,
    flank: int = 4,
) -> list[Hit]:
    """Scan one annotated region (+ small flank) of a sequence for NES-
    consensus matches and score each one."""
    lo = max(0, region.start - flank)
    hi = min(len(sequence), region.end + flank)
    window = sequence[lo:hi]

    hits: list[Hit] = []
    for m in NES_CONSENSUS_REGEX.finditer(window):
        span = m.group(1)
        anchors = extract_phi_positions(span)
        if len(anchors) != 4:
            continue
        abs_start = lo + m.start(1)  # 0-indexed Python string offset
        abs_end = abs_start + len(span)  # 0-indexed, EXCLUSIVE (Python-slice-style)
        upstream = sequence[max(0, abs_start - 3):abs_start]
        score = score_hit(anchors, upstream)
        ctx_start = max(0, abs_start - 6)
        ctx_end = min(len(sequence), abs_end + 6)
        # match_start/match_end are written to nes_negatives.csv
        # and consumed by structural_dataset_v2_pipeline.py's
        # `range(t['start'], t['end'] + 1)` window slice, which expects
        # 1-based INCLUSIVE coordinates -- the same convention
        # load_positive_tasks() uses for nes_start/nes_end (see that
        # function's own "# 1-based inclusive" comment). abs_start/abs_end
        # above are 0-indexed with an EXCLUSIVE end, so match_start was
        # off by one (missing the 0->1-indexed +1) while match_end happened
        # to already be numerically correct (0-indexed exclusive end ==
        # 1-indexed inclusive end, by coincidence of the arithmetic).
        # Verified against Q8TDR4's real UniProt sequence: match_seq
        # "LREKLRALQL" is real residues 89-98 (1-indexed inclusive), but the
        # old code wrote match_start=88 (one too low) -- silently pulling in
        # one extra, wrong residue's SASA/pLDDT at the START of every
        # negative's structural window (~650/651 negatives affected).
        hits.append(
            Hit(
                accession=accession,
                protein_name=protein_name,
                organism=organism,
                feature_kind=region.kind,
                feature_start=region.start,
                feature_end=region.end,
                match_start=abs_start + 1,
                match_end=abs_end,
                match_seq=span,
                context=sequence[ctx_start:ctx_end],
                score=score,
            )
        )
    return hits


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def filter_out_known_nes(hits: Iterable[Hit], nes_regions: list[Region]) -> list[Hit]:
    kept = []
    for h in hits:
        if any(overlaps(h.match_start, h.match_end, n.start, n.end) for n in nes_regions):
            continue
        kept.append(h)
    return kept


# --------------------------------------------------------------------------
# UniProt querying
# --------------------------------------------------------------------------

def _require_requests():
    if requests is None:
        raise RuntimeError(
            "The 'requests' package is required. Install with: pip install requests"
        )


def build_query(taxon: str | None, want_coiled_coil: bool, want_leucine_zipper: bool) -> str:
    # NOTE: UniProt's query-field name for coiled-coil existence is
    # "ft_coiled" (NOT "ft_coiled_coil" -- that returns HTTP 400). There is
    # also no dedicated "Leucine-zipper" keyword in UniProt; zippers are
    # annotated as a Region feature with description "Leucine-zipper", so
    # we match on that field's text instead of a keyword. Verified against
    # the live UniProt REST API.
    clauses = []
    if want_coiled_coil and want_leucine_zipper:
        clauses.append('(ft_coiled:* OR ft_region:"Leucine-zipper")')
    elif want_coiled_coil:
        clauses.append("ft_coiled:*")
    elif want_leucine_zipper:
        clauses.append('ft_region:"Leucine-zipper"')
    clauses.append("reviewed:true")
    if taxon and taxon.lower() != "all":
        clauses.append(f"organism_id:{taxon}")
    return " AND ".join(clauses)


def fetch_uniprot_entries(
    taxon: str | None,
    max_entries: int,
    want_coiled_coil: bool = True,
    want_leucine_zipper: bool = True,
    page_size: int = 500,
    sleep_between_pages: float = 0.5,
) -> list[dict]:
    """Paginate through UniProt REST search results using cursor-based
    pagination (the 'Link' response header)."""
    _require_requests()
    query = build_query(taxon, want_coiled_coil, want_leucine_zipper)
    fields = "accession,protein_name,organism_name,sequence,ft_coiled,ft_region,ft_motif"
    url = UNIPROT_BASE
    params = {
        "query": query,
        "fields": fields,
        "format": "json",
        "size": str(min(page_size, 500)),
    }

    entries: list[dict] = []
    next_url = None
    while len(entries) < max_entries:
        if next_url:
            resp = requests.get(next_url, timeout=30)
        else:
            resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        entries.extend(batch)

        # was hand-parsing the Link header via link.split(","),
        # which breaks whenever the URL itself contains a literal comma --
        # which it always does here, since `fields` is a comma-separated
        # list (accession,protein_name,...). That comma got treated as a
        # link-separator, shredding the URL mid-string; part.find("<") then
        # returned -1 (not found) instead of raising, so next_url silently
        # became a garbage fragment like "ft_motif&query=...&cursor=..."
        # missing its scheme/host entirely -- requests.MissingSchema.
        # This never showed up on human-only queries because those fit in a
        # single page (no "next" link ever appeared); it only surfaces once
        # a query is large enough to paginate (e.g. --taxon all).
        # use requests' own RFC 5988-compliant Link header parser
        # instead of splitting on "," ourselves.
        next_url = resp.links.get("next", {}).get("url")
        if not next_url:
            break
        time.sleep(sleep_between_pages)

    return entries[:max_entries]


def parse_entry(entry: dict) -> tuple[str, str, str, str, list[Region], list[Region]]:
    """Returns (accession, protein_name, organism, sequence,
    structural_regions, nes_regions)."""
    accession = entry.get("primaryAccession", "?")
    protein_name = (
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "?")
    )
    organism = entry.get("organism", {}).get("scientificName", "?")
    sequence = entry.get("sequence", {}).get("value", "")

    structural_regions: list[Region] = []
    nes_regions: list[Region] = []

    for feat in entry.get("features", []):
        ftype = feat.get("type", "")
        desc = (feat.get("description") or "").lower()
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        if start is None or end is None:
            continue
        start -= 1  # UniProt is 1-based inclusive; convert to 0-based

        if ftype == "Coiled coil":
            structural_regions.append(Region("coiled_coil", start, end))
        elif ftype == "Region" and "leucine" in desc and "zipper" in desc:
            structural_regions.append(Region("leucine_zipper", start, end))
        elif ftype == "Motif" and "nuclear export" in desc:
            nes_regions.append(Region("nes", start, end))

    return accession, protein_name, organism, sequence, structural_regions, nes_regions


def build_negative_dataset(
    taxon: str | None,
    max_entries: int,
    min_score: float,
    want_coiled_coil: bool,
    want_leucine_zipper: bool,
) -> list[Hit]:
    entries = fetch_uniprot_entries(
        taxon=taxon,
        max_entries=max_entries,
        want_coiled_coil=want_coiled_coil,
        want_leucine_zipper=want_leucine_zipper,
    )

    all_hits: list[Hit] = []
    for entry in entries:
        accession, protein_name, organism, sequence, struct_regions, nes_regions = parse_entry(entry)
        if not sequence or not struct_regions:
            continue
        entry_hits: list[Hit] = []
        for region in struct_regions:
            entry_hits.extend(
                scan_region_for_hits(sequence, region, accession, protein_name, organism)
            )
        entry_hits = filter_out_known_nes(entry_hits, nes_regions)
        all_hits.extend(entry_hits)

    return [h for h in all_hits if h.score >= min_score]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_outputs(hits: list[Hit], outdir: str) -> tuple[str, str]:
    import os
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "nes_negatives.csv")
    fasta_path = os.path.join(outdir, "nes_negatives.fasta")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "accession", "protein_name", "organism", "feature_kind",
                "feature_start", "feature_end", "match_start", "match_end",
                "match_seq", "context", "score",
            ]
        )
        for h in hits:
            writer.writerow(
                [
                    h.accession, h.protein_name, h.organism, h.feature_kind,
                    h.feature_start, h.feature_end, h.match_start, h.match_end,
                    h.match_seq, h.context, h.score,
                ]
            )

    with open(fasta_path, "w") as fh:
        for h in hits:
            header = (
                f">{h.accession}|{h.feature_kind}|{h.match_start}-{h.match_end}"
                f"|score={h.score}"
            )
            fh.write(header + "\n")
            fh.write(h.context + "\n")

    return csv_path, fasta_path


# --------------------------------------------------------------------------
# Self-test (no network required)
# --------------------------------------------------------------------------

def _self_test() -> None:
    """Sanity-check regex/scoring/filtering against a synthetic entry.

    Built from named parts + len() so region boundaries are always
    correct, rather than hand-counted (easy to get wrong by eye).
    """
    filler_head = "MSTQPKLDAA"
    # Valid Phi-x(2,3)-Phi-x(2,3)-Phi-x-Phi hit embedded in a coiled-coil:
    # L-AA-L-AA-L-A-L
    coiled_coil_body = "AKLDELQ" + "LAALAALAL" + "QKLEAELDDL"
    filler_mid = "PPPPPPPPPP" + "PPPPPPP"
    # Same valid pattern, but this span will be annotated as a "known NES"
    # and must be excluded by the overlap filter even though it matches.
    fake_nes_body = "LAALAALAL"
    tail = "KKEND"

    coiled_coil_start = len(filler_head)
    coiled_coil_end = coiled_coil_start + len(coiled_coil_body)
    fake_nes_start = coiled_coil_end + len(filler_mid)
    fake_nes_end = fake_nes_start + len(fake_nes_body)

    seq = filler_head + coiled_coil_body + filler_mid + fake_nes_body + tail

    struct_regions = [
        Region("coiled_coil", coiled_coil_start, coiled_coil_end),
        Region("coiled_coil", fake_nes_start, fake_nes_end),  # deliberately also "structural"
    ]
    nes_regions = [Region("nes", fake_nes_start, fake_nes_end)]

    hits: list[Hit] = []
    for region in struct_regions:
        hits.extend(scan_region_for_hits(seq, region, "TEST1", "Test Protein", "Test organism"))

    before = len(hits)
    hits = filter_out_known_nes(hits, nes_regions)
    after = len(hits)

    assert before >= 2, f"Expected hits in both regions before filtering, got {before}"
    assert after >= 1, "Expected at least one surviving coiled-coil hit after filtering"
    assert before > after, "Expected the overlap filter to remove the fake NES-region hit"
    assert all(h.feature_kind == "coiled_coil" for h in hits)
    assert not any(overlaps(h.match_start, h.match_end, fake_nes_start, fake_nes_end) for h in hits), (
        "A hit overlapping the annotated NES region survived filtering"
    )
    assert all(0.0 <= h.score <= 1.0 for h in hits)

    print(f"[self-test] found {before} raw hits, {after} after excluding annotated NES overlap")
    for h in hits:
        print(f"  {h.accession} {h.feature_kind} {h.match_start}-{h.match_end} "
              f"seq={h.match_seq} ctx={h.context} score={h.score}")
    print("[self-test] OK")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxon", default="9606", help='NCBI taxon ID, or "all" (default: 9606 = human)')
    parser.add_argument("--max-entries", type=int, default=500, help="Max UniProt entries to pull")
    parser.add_argument("--min-score", type=float, default=0.4, help="Minimum PSSM score to keep a hit (0-1)")
    parser.add_argument("--outdir", default="./nes_negatives", help="Output directory")
    parser.add_argument("--no-coiled-coil", action="store_true", help="Skip coiled-coil regions")
    parser.add_argument("--no-leucine-zipper", action="store_true", help="Skip leucine-zipper regions")
    parser.add_argument("--self-test", action="store_true", help="Run offline sanity check and exit")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    hits = build_negative_dataset(
        taxon=args.taxon,
        max_entries=args.max_entries,
        min_score=args.min_score,
        want_coiled_coil=not args.no_coiled_coil,
        want_leucine_zipper=not args.no_leucine_zipper,
    )
    csv_path, fasta_path = write_outputs(hits, args.outdir)
    print(f"Wrote {len(hits)} negative examples to:\n  {csv_path}\n  {fasta_path}")


if __name__ == "__main__":
    main()
