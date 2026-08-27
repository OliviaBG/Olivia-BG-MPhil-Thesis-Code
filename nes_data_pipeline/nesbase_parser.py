"""
Parser for NESbase 1.0 (la Cour et al. 2003, NAR 31:393-6)
https://services.healthtech.dtu.dk/datasets/NESbase-1.0/db.html

NESbase is a small (38-entry, NES-0001..NES-0038), hand-curated flat-file
database. Each record is separated by a line containing only "//" and has
the shape:

    NES-accession:  NES-0001
    Date:           14 Aug 2002
    Protein:        cAMP-dependent Protein Kinase Inhibitor, alpha form (PKI-alpha).
    Organism:       Homo sapiens (Human) / Oryctolagus cuniculus (Rabbit).
    DB_reference:   SWISS-PROT [P04541](...), IPKA_HUMAN.
    Necessity:      Export abrogated by the following mutations: L37A/L39A/L41A, ...
    Sufficiency:    Aa 37-46 mediates export of GST.
    Pathway:        LMB-sensitive.
    Location:       -
    Regulation:     -
    Comments:       -
    References:     ...
    Sequence:       75 aa
    TDVETTYADFIASGRTGRRNAIHDILVSSASGNSNELALKLAGLDINKTEGEEDAQRSSTEQSGEAQGEAAKSES
    ....................................MaMaMaaMaM.............................
    //

The sequence block is FASTA-like but wrapped at 80 columns with a trailing
column-ruler number, followed immediately by an annotation line of the same
width:

    '.'         -> residue not part of any annotated NES
    lowercase   -> residue is part of an annotated NES region (observed: 'a')
    UPPERCASE   -> residue is additionally shown by mutagenesis to be
                   functionally critical for export (observed: 'M')

This parser is deliberately tolerant of the exact letters used (some entries
could in principle use a different lowercase/uppercase letter per NES if a
protein has more than one signal) -- it treats "any lowercase letter" as
"annotated NES" and "any uppercase letter" as "critical residue", and groups
runs of the same letter into separate NES segments.

Usage:
    python nesbase_parser.py db.html nesbase_parsed.json

`db.html` should be the saved HTML/text of
https://services.healthtech.dtu.dk/datasets/NESbase-1.0/db.html
(the page is plain preformatted text inside a couple of HTML wrapper tags,
so we strip tags rather than doing full HTML parsing).
"""

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from html import unescape
from typing import List, Optional


MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def strip_markup(text: str) -> str:
    """Strip HTML tags and collapse markdown-style links to their label."""
    text = re.sub(r"<[^>]+>", "", text)
    text = MD_LINK_RE.sub(r"\1", text)
    return unescape(text)


@dataclass
class NESSegment:
    start: int  # 1-based, inclusive, position within `full_sequence`
    end: int  # 1-based, inclusive
    sequence: str
    critical_positions: List[int] = field(default_factory=list)  # 1-based
    critical_residues: List[str] = field(default_factory=list)  # amino acids at those positions


@dataclass
class NESBaseRecord:
    accession: str
    date: Optional[str]
    protein: str
    organism: Optional[str]
    db_reference: Optional[str]
    necessity: Optional[str]
    sufficiency: Optional[str]
    pathway: Optional[str]
    location: Optional[str]
    regulation: Optional[str]
    comments: Optional[str]
    references: Optional[str]
    full_sequence: str
    nes_segments: List[NESSegment]
    crm1_dependent: Optional[bool]  # True/False/None (None = not stated)
    crm1_evidence_text: Optional[str]


FIELD_NAMES = [
    "Date", "Protein", "Organism", "DB_reference", "Necessity",
    "Sufficiency", "Pathway", "Location", "Regulation", "Comments",
    "References", "Sequence",
]

FIELD_RE = re.compile(r"^(" + "|".join(FIELD_NAMES) + r"|NES-accession):\s*(.*)$")


def classify_crm1_dependence(pathway_text: Optional[str]):
    """Heuristic classification of CRM1/exportin-1 dependence from the free-text
    'Pathway' field. Returns (bool_or_None, evidence_text)."""
    if not pathway_text or pathway_text.strip() in ("-", ""):
        return None, pathway_text
    t = pathway_text.lower()
    positive_markers = [
        "lmb-sensitiv", "lmb sensitiv", "crm-depend", "crm1-depend",
        "crm-interact", "crm1-interact", "rev-like pathway",
        "rev competition", "leptomycin",
    ]
    negative_markers = ["lmb-insensitiv", "lmb insensitiv", "crm-independ", "crm1-independ"]
    is_neg = any(m in t for m in negative_markers)
    is_pos = any(m in t for m in positive_markers)
    if is_neg:
        return False, pathway_text
    if is_pos:
        return True, pathway_text
    return None, pathway_text  # e.g. "-" or an unrecognized free-text description


def parse_sequence_block(raw_lines: List[str]):
    """`raw_lines` are the lines following 'Sequence:\t123 aa' up to (not
    including) the closing '//'. They alternate: one or more residue lines
    (each up to 80 aa, followed by whitespace and a running total), then the
    *same number* of annotation lines of identical width (dots/letters).
    Because both residue lines and annotation lines are plain text of equal
    width and appear in matching order, we detect annotation lines as those
    consisting solely of '.' and letters (no other amino-acid-only lines will
    be all-dots)."""
    seq_lines = []
    ann_lines = []
    for line in raw_lines:
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue
        # column-ruler numbers (e.g. "80") trail sequence lines; strip them.
        content = re.sub(r"\s+\d+\s*$", "", stripped)
        body = content.strip()
        if not body:
            continue
        if re.fullmatch(r"[.a-zA-Z]+", body) and ("." in body):
            ann_lines.append(body)
        else:
            seq_lines.append(body)
    full_sequence = "".join(seq_lines)
    annotation = "".join(ann_lines)
    return full_sequence, annotation


def segments_from_annotation(full_sequence: str, annotation: str) -> List[NESSegment]:
    """Walk the annotation string and group contiguous non-'.' runs into NES
    segments, recording which positions within each run are uppercase
    (critical, mutagenesis-validated residues)."""
    segments: List[NESSegment] = []
    n = min(len(full_sequence), len(annotation))
    i = 0
    while i < n:
        ch = annotation[i]
        if ch == ".":
            i += 1
            continue
        j = i
        while j < n and annotation[j] != ".":
            j += 1
        # run is [i, j)
        start, end = i + 1, j  # 1-based inclusive
        seq = full_sequence[i:j]
        crit_pos = [k + 1 for k in range(i, j) if annotation[k].isupper()]
        crit_res = [full_sequence[k] for k in range(i, j) if annotation[k].isupper()]
        segments.append(NESSegment(start=start, end=end, sequence=seq,
                                    critical_positions=crit_pos, critical_residues=crit_res))
        i = j
    return segments


def parse_nesbase(raw_text: str) -> List[NESBaseRecord]:
    text = strip_markup(raw_text)
    lines = text.splitlines()

    records: List[NESBaseRecord] = []
    cur_fields = {}
    cur_seq_lines: List[str] = []
    in_sequence = False
    accession = None

    def flush():
        nonlocal cur_fields, cur_seq_lines, in_sequence, accession
        if accession is None:
            return
        full_sequence, annotation = parse_sequence_block(cur_seq_lines)
        segments = segments_from_annotation(full_sequence, annotation)
        crm1_dep, crm1_ev = classify_crm1_dependence(cur_fields.get("Pathway"))
        rec = NESBaseRecord(
            accession=accession,
            date=cur_fields.get("Date"),
            protein=cur_fields.get("Protein", "").strip(),
            organism=cur_fields.get("Organism"),
            db_reference=cur_fields.get("DB_reference"),
            necessity=cur_fields.get("Necessity"),
            sufficiency=cur_fields.get("Sufficiency"),
            pathway=cur_fields.get("Pathway"),
            location=cur_fields.get("Location"),
            regulation=cur_fields.get("Regulation"),
            comments=cur_fields.get("Comments"),
            references=cur_fields.get("References"),
            full_sequence=full_sequence,
            nes_segments=segments,
            crm1_dependent=crm1_dep,
            crm1_evidence_text=crm1_ev,
        )
        records.append(rec)
        cur_fields = {}
        cur_seq_lines = []
        in_sequence = False
        accession = None

    current_field = None
    for line in lines:
        if line.strip() == "//":
            flush()
            current_field = None
            continue

        if in_sequence:
            # Anything until the next field keyword / "//" is sequence data.
            if FIELD_RE.match(line.strip()):
                in_sequence = False
            else:
                cur_seq_lines.append(line)
                continue

        m = FIELD_RE.match(line.strip())
        if m:
            key, value = m.group(1), m.group(2)
            if key == "NES-accession":
                if accession is not None:
                    flush()
                accession = value.strip()
                current_field = None
                continue
            current_field = key
            cur_fields[key] = value.strip()
            if key == "Sequence":
                in_sequence = True
            continue

        # continuation line of a multi-line field (e.g. wrapped Necessity text)
        if current_field and current_field in cur_fields:
            cur_fields[current_field] = (cur_fields[current_field] + " " + line.strip()).strip()

    flush()  # in case file doesn't end with a trailing //
    return records


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "nesbase_parsed.json"

    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    records = parse_nesbase(raw)
    data = [asdict(r) for r in records]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    n_segments = sum(len(r.nes_segments) for r in records)
    n_crit = sum(len(seg.critical_positions) for r in records for seg in r.nes_segments)
    print(f"Parsed {len(records)} NESbase records, {n_segments} NES segments, "
          f"{n_crit} residues flagged as mutagenesis-critical.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
