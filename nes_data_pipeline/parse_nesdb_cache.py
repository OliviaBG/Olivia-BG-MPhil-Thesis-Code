"""
Batch-parse cached NESdb detail pages (plain text) into nesdb.json
records matching the NESdbRecord schema from nesdb_scraper.py.

Reads every nesdb_cache_text/<nes_id>.txt file, plus index_good_entries.json
(name/nes_id/detail_url) to know which id maps to which name/url, and writes
nesdb.json.

This reuses the same SECTION_HEADERS / MUTATION_RE / SIGNAL_RE / classify_crm1
/ parse_fasta_block logic as nesdb_scraper.py, adapted for cached pages
that are already clean plain text rather than raw HTML, so no
BeautifulSoup step is needed -- section headers already appear as
standalone lines in the text.
"""

import json
import os
import re
import sys

CACHE_DIR = "nesdb_cache_text"
INDEX_JSON = "index_good_entries.json"
OUT_JSON = "nesdb.json"
FAILED_LOG = "nesdb_failed.json"

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

MUTATION_RE = re.compile(r"\b[A-Z]\d+[A-Z](?:/[A-Z]\d+[A-Z])*\b")
SIGNAL_RE = re.compile(r"(\d+)\s*([A-Z]{4,})\s*(\d+)")


def _split_sections(full_text):
    positions = []
    for header in SECTION_HEADERS:
        idx = full_text.find("\n" + header + "\n")
        if idx != -1:
            header_start = idx + 1
        else:
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


def classify_crm1(evidence_text):
    if not evidence_text or evidence_text.strip().lower() in ("", "unknown", "-"):
        return None
    t = evidence_text.lower()
    if "lmb resistant" in t or "lmb-insensitive" in t or "not crm1" in t or "crm1-independent" in t:
        return False
    if ("lmb sensitive" in t or "lmb-sensitive" in t or "crm1" in t
            or "leptomycin" in t or "exportin" in t):
        return True
    return None


def parse_fasta_block(seq_section_text):
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


def parse_detail_text(text, name, nes_id, list_source, detail_url):
    # normalize: strip trailing whitespace per line, collapse blank-line runs
    # to single newlines (mirrors nesdb_scraper.py's HTML->text normalization)
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
            signals.append({"start": start, "end": end, "sequence": seq})

    full_sequence = parse_fasta_block(seq_text)

    return {
        "nes_id": nes_id,
        "name": name,
        "list_source": list_source,
        "full_name": full_name,
        "alternative_names": alt_names,
        "organism": organism,
        "crm1_evidence_text": crm1_evidence,
        "crm1_dependent": classify_crm1(crm1_evidence),
        "mutations_affecting_export_text": mut_export_text,
        "mutations_affecting_export": mutations,
        "mutations_affecting_crm1_binding": mut_crm1_text,
        "export_signals_text": signals_text,
        "export_signals": signals,
        "secondary_structure_of_signal": sec_struct,
        "other_residues_important": other_res,
        "full_sequence": full_sequence,
        "comments": comments,
        "references": references,
        "detail_url": detail_url,
    }


def main():
    with open(INDEX_JSON, encoding="utf-8") as f:
        index_entries = json.load(f)
    by_id = {e["nes_id"]: e for e in index_entries}

    records = []
    failed = []
    cached_ids = sorted(
        (fn[:-4] for fn in os.listdir(CACHE_DIR) if fn.endswith(".txt")),
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    for nes_id in cached_ids:
        entry = by_id.get(nes_id)
        name = entry["name"] if entry else nes_id
        detail_url = entry["detail_url"] if entry else f"http://prodata.swmed.edu/LRNes/IndexFiles/details.php?name={nes_id}"
        path = os.path.join(CACHE_DIR, nes_id + ".txt")
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            rec = parse_detail_text(text, name, nes_id, "good", detail_url)
            records.append(rec)
        except Exception as e:
            failed.append({"nes_id": nes_id, "name": name, "reason": str(e)})

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(FAILED_LOG, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)

    n_with_seq = sum(1 for r in records if r["full_sequence"])
    n_with_signal = sum(1 for r in records if r["export_signals"])
    n_true = sum(1 for r in records if r["crm1_dependent"] is True)
    n_false = sum(1 for r in records if r["crm1_dependent"] is False)
    n_unknown = sum(1 for r in records if r["crm1_dependent"] is None)
    print(f"Parsed {len(records)} records ({len(failed)} failed) from {len(cached_ids)} cached files.")
    print(f"  with full_sequence: {n_with_seq}")
    print(f"  with >=1 export_signal: {n_with_signal}")
    print(f"  crm1_dependent True/False/unknown: {n_true}/{n_false}/{n_unknown}")


if __name__ == "__main__":
    main()
