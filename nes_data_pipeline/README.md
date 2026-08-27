# NES data pipeline: NESbase 1.0 + NESdb predictor-ready dataset

Four scripts, run in order. All tested against real pages/records from both
sites during development (see "What was verified" below).

## 1. `nesbase_parser.py` -- NESbase 1.0 (DTU)

NESbase 1.0 (la Cour et al. 2003) is a single flat-file page with only
**38 entries** (NES-0001 ... NES-0038). Save the page and parse it:

```bash
curl -s "https://services.healthtech.dtu.dk/datasets/NESbase-1.0/db.html" -o db.html
python nesbase_parser.py db.html nesbase_parsed.json
```

Each entry has a sequence block plus an annotation line: `.` = not part of
an NES, lowercase letter = part of an annotated NES, UPPERCASE letter =
residue shown by mutagenesis to be required for export. The parser turns
this into `start`/`end`/`sequence`/`critical_positions`/`critical_residues`
per NES, and derives a `crm1_dependent` True/False/None flag from the
free-text `Pathway` field (LMB-sensitivity / CRM1-interaction = the
standard proxy for CRM1-dependence).

## 2. `nesdb_scraper.py` -- NESdb (UT Southwestern)

NESdb is much bigger (~400 proteins) and paginated: an index page
(`namesGood.php`, plus `namesDoubt.php` for lower-confidence entries) links
out to one `details.php?name=<id>` page per protein. Requires
`pip install requests beautifulsoup4`.

```bash
python nesdb_scraper.py --out nesdb.json --limit 20        # smoke test first
python nesdb_scraper.py --out nesdb.json --include-doubt   # full run
```

This makes ~400+ HTTP requests to a small academic server. It caches every
page under `nesdb_cache/` so you only ever fetch each page once, and sleeps
1s between requests -- please don't lower that. Each record captures: full
name, organism, the "Experimental Evidence for CRM1-mediated Export" text
(`crm1_dependent`), the discrete mutation codes from "Mutations That
Affect Nuclear Export" (e.g. `L78A`), the "Functional Export Signals" region
(start/end/sequence of the actual validated NES peptide), and the full
protein sequence (parsed out of the FASTA block).

Edit the `HEADERS["User-Agent"]` string in the script to include your email
before running a full scrape -- that's just good etiquette on a small lab
site, not a technical requirement.

## 3. `build_dataset.py` -- merge into one table

```bash
python build_dataset.py --nesbase nesbase_parsed.json --nesdb nesdb.json --out nes_dataset
```

Produces `nes_dataset.csv`/`.json`, one row per experimentally-defined NES
segment, same columns regardless of source: `protein_name`, `organism`,
`full_sequence`, `nes_start`, `nes_end`, `nes_sequence`,
`critical_positions`, `critical_residues`, `mutation_codes`,
`crm1_dependent`, `evidence_text`, `db_reference`, `references`.

## 4. `nes_features.py` -- features + train/test table

```bash
python nes_features.py --dataset nes_dataset.csv --out training_table.csv
```

For every row with a known NES sequence, generates the positive example
plus `--neg-per-pos` (default 3) negative windows of matching length
sampled from elsewhere in the *same* protein's sequence (excluding the real
NES region) -- negatives from the same protein are harder, and better
negatives, than random UniProt sequence. Each window gets: amino-acid
composition, hydrophobic fraction, Kyte-Doolittle hydropathy, net charge,
and -- the feature family that matters most for this problem -- the spacing
between consecutive hydrophobic (Φ = L/I/V/F/M) residues, plus a 0/1 flag
for whether the window matches the classic NES consensus
`Φ-x(2,3)-Φ-x(2,3)-Φ-x-Φ`.

`nes_features.py` is also importable (`scan_sequence`, `featurize`,
`build_training_table`) if you'd rather do your own train/test split or
plug into a different pipeline (e.g. windows across a whole proteome for
inference, not just the curated positives).

From there, `training_table.csv` drops straight into
`sklearn.linear_model.LogisticRegression` / `GradientBoostingClassifier` /
xgboost as a baseline; a CNN/transformer over raw sequence is the natural
next step once you've confirmed the classical features aren't already
saturating accuracy (they usually get you surprisingly far on this
particular problem, which is part of why the field's consensus motifs work
as well as they do).

## Notes / caveats

- **`crm1_dependent`** is inferred from free text (`Pathway` in NESbase,
  "Experimental Evidence for CRM1-mediated Export" in NESdb) using
  LMB-sensitivity / CRM1-interaction as the standard proxy. It's `None`
  where the source genuinely doesn't say -- treat `None` as "unlabeled", not
  as a third class, when training.
- The four consensus "classes" in `nes_features.py` beyond `class1_core` are
  reconstructed from memory of Kosugi et al. 2009's refined subclasses and
  are explicitly flagged as approximate in the file -- re-derive them from
  the primary paper (or NESmapper/LocNES, which reimplement Kosugi's
  classes) if you need class-level precision rather than just wider
  candidate recall.
- NESdb's `namesGood.php` vs `namesDoubt.php` split is a genuine
  confidence signal the original curators built in -- worth keeping
  `list_source` as a feature or at least excluding "doubt" entries from
  your positive set for a first pass.
- Be a polite scraper: both sites are small academic resources, not
  production APIs. The cache in `nesdb_scraper.py` means you should only
  ever need to run the full crawl once.

## What was verified during development

Both parsers were run against real fetched pages/records (not just written
blind): NESbase's PKI-alpha (NES-0001: positions 37-46, critical residues
L37/L39/L41/L44/I46) and p53 (NES-0002: 339-352, L348/L350) entries parsed
correctly; NESdb's HIV-Rev `details.php?name=2` page parsed to the correct
full name, CRM1-dependence (`LMB Sensitive` `True`), mutation codes
(`L78A, L81A, L83A, L78D/E79L`), and the exact validated signal
(`73LQLPPLERLTLD84`). `build_dataset.py` and `nes_features.py` were run
end-to-end against the real 38-entry NESbase output plus a synthetic
HIV-Rev NESdb record and produced correct merged rows and a correct
positive/negative training table (confirmed the classic consensus regex
correctly flags both known NESs).
