"""
Scraper for NESdb (Xu, Marchenko, Moll lab), UT Southwestern
http://prodata.swmed.edu/LRNes/

NESdb has two index pages:
  - namesGood.php  -> curated/accepted NESs (141 proteins)
  - namesDoubt.php -> NESs flagged as questionable ("in doubt", 41 proteins)

Each index page is an HTML table whose rows link to
  details.php?name=<NES_ID>
which is a per-entry summary page with a fixed sequence of section headers
(same template for every entry, only content differs):

  Summary for <NAME> (NES ID: <id>)
  Full Name
  Alternative Names
  Organism
  Experimental Evidence for CRM1-mediated Export
  Mutations That Affect Nuclear Export
  Mutations That Affect CRM1 Binding
  Functional Export Signals
  Secondary Structure of Export Signal
  Other Residues Important for Export
  Sequence  (FASTA block)
  Domain Info by CDD
  Secondary Structure by PSIPRED
  Conservation Score by AL2CO
  3D Structures in PDB
  Comments
  References

Because the exact HTML markup (tables vs. divs vs. dl/dt/dd) is an
implementation detail that could change, this scraper is deliberately
"text-first": it renders each page to plain text with BeautifulSoup's
get_text(), then slices the text between consecutive known section headers.
That makes it robust to markup changes as long as the section *labels*
stay the same, which is the safer bet for a small academic PHP site.

Requirements:
    pip install requests beautifulsoup4

Usage:
    python nesdb_scraper.py --out nesdb.json
    python nesdb_scraper.py --out nesdb.json --limit 20      # quick smoke test
    python nesdb_scraper.py --out nesdb.json --include-doubt  # also scrape namesDoubt.php

Politeness:
    - Caches every fetched page under ./nesdb_cache/ so re-runs (e.g. after a
      crash, or while you iterate on the parser) don't re-hit the server.
    - Sleeps SLEEP_SECONDS between requests.
    - Identifies itself with a descriptive User-Agent.

This will make ~180+ HTTP requests to a small academic server the first time
you run it with no --limit (141 "good" + 41 "doubt" if --include-doubt).
Please be considerate: keep SLEEP_SECONDS >= 1, and reuse the cache rather
than re-scraping from scratch.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from urllib.parse import urljoin, urlparse, parse_qs

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit(
        "This script needs `requests` and `beautifulsoup4`.\n"
        "Install with: pip install requests beautifulsoup4"
    )

BASE = "http://prodata.swmed.edu/LRNes/IndexFiles/"
INDEX_PAGES = {
    "good": "namesGood.php",
    "doubt": "namesDoubt.php",
}
CACHE_DIR = "nesdb_cache"
SLEEP_SECONDS = 1.0
HEADERS = {
    "User-Agent": "NESdb-research-scraper/1.0 (personal research use; "
                  "contact: set-your-email-here)"
}

SECTION_HEADERS = [
    "Full Name",
    "Alternative Names",
    "Organism",
    "Experimental Evidence for CRM1-mediated Export",
    "Mutations That Affect Nuclear Export",
    "Mutations That Affect CRM1 Binding",
    "Functional Export Signals",
    "Secondary Structure of Export Signal",
    "Other Residues Important for Export",
    "Sequence",
    "Domain Info by CDD",
    "Secondary Structure by PSIPRED",
    "Conservation Score by AL2CO",
    "3D Structures in PDB",
    "Comments",
    "References",
]


@dataclass
class ExportSignal:
    start: int
    end: int
    sequence: str


@dataclass
class NESdbRecord:
    nes_id: str
    name: str
    list_source: str  # "good" or "doubt"
    full_name: Optional[str] = None
    alternative_names: Optional[str] = None
    organism: Optional[str] = None
    crm1_evidence_text: Optional[str] = None
    crm1_dependent: Optional[bool] = None
    mutations_affecting_export_text: Optional[str] = None
    mutations_affecting_export: List[str] = field(default_factory=list)
    mutations_affecting_crm1_binding: Optional[str] = None
    export_signals_text: Optional[str] = None
    export_signals: List[ExportSignal] = field(default_factory=list)
    secondary_structure_of_signal: Optional[str] = None
    other_residues_important: Optional[str] = None
    full_sequence: Optional[str] = None
    comments: Optional[str] = None
    references: Optional[str] = None
    detail_url: str = ""


def _cache_path(url: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", url)[-150:]
    return os.path.join(CACHE_DIR, safe + ".html")


def fetch(url: str) -> str:
    path = _cache_path(url)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = resp.text
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    time.sleep(SLEEP_SECONDS)
    return text


def parse_index(html: str, list_source: str):
    """Return list of (name, nes_id, detail_url, list_source) from an index page."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "details.php" not in href:
            continue
        qs = parse_qs(urlparse(href).query)
        nes_id = qs.get("name", [None])[0]
        if not nes_id:
            continue
        name = a.get_text(strip=True)
        if not name:
            continue
        key = (name, nes_id)
        if key in seen:
            continue
        seen.add(key)
        detail_url = urljoin(BASE, href)
        out.append((name, nes_id, detail_url, list_source))
    return out


def _split_sections(full_text: str):
    """Slice `full_text` between consecutive SECTION_HEADERS labels.
    Returns dict {header: content_text}."""
    positions = []
    for header in SECTION_HEADERS:
        idx = full_text.find("\n" + header + "\n")
        if idx != -1:
            header_start = idx + 1  # skip the leading newline we searched for
        else:
            # header might be at the very start of a block without a
            # preceding newline (rare) -- try a looser search.
            idx = full_text.find(header)
            header_start = idx
        positions.append((idx, header_start, header))

    positions = [(i, hs, h) for i, hs, h in positions if i != -1]
    positions.sort(key=lambda p: p[0])

    sections = {}
    for k, (idx, header_start, header) in enumerate(positions):
        start = header_start + len(header)
        end = positions[k + 1][0] if k + 1 < len(positions) else len(full_text)
        content = full_text[start:end].strip(" \n\t:")
        sections[header] = content
    return sections


MUTATION_RE = re.compile(r"\b[A-Z]\d+[A-Z](?:/[A-Z]\d+[A-Z])*\b")
SIGNAL_RE = re.compile(r"(\d+)\s*([A-Z]{4,})\s*(\d+)")


def classify_crm1(evidence_text: Optional[str]):
    if not evidence_text or evidence_text.strip().lower() in ("", "unknown", "-"):
        return None
    t = evidence_text.lower()
    if "lmb resistant" in t or "lmb-insensitive" in t or "not crm1" in t or "crm1-independent" in t:
        return False
    if ("lmb sensitive" in t or "lmb-sensitive" in t or "crm1" in t
            or "leptomycin" in t or "exportin" in t):
        return True
    return None


def parse_fasta_block(seq_section_text: str) -> Optional[str]:
    lines = [l.strip() for l in seq_section_text.splitlines()]
    seq_lines = []
    in_fasta = False
    for l in lines:
        if l.startswith(">"):
            in_fasta = True
            continue
        if in_fasta:
            if not l or not re.fullmatch(r"[A-Za-z\*\-]+", l):
                if seq_lines:
                    break
                continue
            seq_lines.append(l)
    return "".join(seq_lines) if seq_lines else None


def parse_detail_page(html: str, name: str, nes_id: str, list_source: str, detail_url: str) -> NESdbRecord:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    # collapse runs of blank lines but keep single newlines (needed for header matching)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    sections = _split_sections(text)

    full_name = sections.get("Full Name")
    alt_names = sections.get("Alternative Names")
    organism = sections.get("Organism")
    crm1_evidence = sections.get("Experimental Evidence for CRM1-mediated Export")
    mut_export_text = sections.get("Mutations That Affect Nuclear Export")
    mut_crm1_text = sections.get("Mutations That Affect CRM1 Binding")
    signals_text = sections.get("Functional Export Signals")
    sec_struct = sections.get("Secondary Structure of Export Signal")
    other_res = sections.get("Other Residues Important for Export")
    seq_text = sections.get("Sequence", "")
    comments = sections.get("Comments")
    references = sections.get("References")

    mutations = MUTATION_RE.findall(mut_export_text) if mut_export_text else []

    signals = []
    if signals_text:
        for m in SIGNAL_RE.finditer(signals_text):
            start, seq, end = int(m.group(1)), m.group(2), int(m.group(3))
            signals.append(ExportSignal(start=start, end=end, sequence=seq))

    full_sequence = parse_fasta_block(seq_text)

    rec = NESdbRecord(
        nes_id=nes_id,
        name=name,
        list_source=list_source,
        full_name=full_name,
        alternative_names=alt_names,
        organism=organism,
        crm1_evidence_text=crm1_evidence,
        crm1_dependent=classify_crm1(crm1_evidence),
        mutations_affecting_export_text=mut_export_text,
        mutations_affecting_export=mutations,
        mutations_affecting_crm1_binding=mut_crm1_text,
        export_signals_text=signals_text,
        export_signals=signals,
        secondary_structure_of_signal=sec_struct,
        other_residues_important=other_res,
        full_sequence=full_sequence,
        comments=comments,
        references=references,
        detail_url=detail_url,
    )
    return rec


def scrape(include_doubt: bool, limit: Optional[int], out_path: str):
    entries = []
    good_html = fetch(urljoin(BASE, INDEX_PAGES["good"]))
    entries += parse_index(good_html, "good")
    if include_doubt:
        doubt_html = fetch(urljoin(BASE, INDEX_PAGES["doubt"]))
        entries += parse_index(doubt_html, "doubt")

    if limit:
        entries = entries[:limit]

    print(f"Found {len(entries)} entries to fetch.")
    records = []
    for i, (name, nes_id, detail_url, list_source) in enumerate(entries, 1):
        try:
            html = fetch(detail_url)
            rec = parse_detail_page(html, name, nes_id, list_source, detail_url)
            records.append(rec)
        except Exception as e:
            print(f"  [{i}/{len(entries)}] FAILED {name} ({nes_id}): {e}")
            continue
        if i % 25 == 0 or i == len(entries):
            print(f"  [{i}/{len(entries)}] fetched {name} ({nes_id})")

    data = [asdict(r) for r in records]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(records)} records to {out_path}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nesdb.json")
    ap.add_argument("--include-doubt", action="store_true",
                     help="Also scrape the 'NESs in doubt' list")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only fetch the first N entries (for a quick smoke test)")
    args = ap.parse_args()
    scrape(args.include_doubt, args.limit, args.out)


if __name__ == "__main__":
    main()
