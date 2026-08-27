# NLS data pipeline

Companion to `../nls_ml_predictor.py`, structured the same way as
`../nes_data_pipeline/` (same conventions on purpose, for direct
comparison in the thesis) but built for nuclear *localization* signals
rather than export signals. See `../NLS_predictor_landscape_and_novelty.md`
for the literature context and why the feature set differs from the NES
model's.

## Files

- `nls_proteins_uniprot_seed.json` -- 159 real proteins (human, mouse,
  yeast, Arabidopsis + one untargeted Swiss-Prot pull), each with its full
  sequence and every UniProt `Motif` feature whose description contains
  "nuclear localization signal". Pulled live from `rest.uniprot.org`
  during development (see provenance note below). 274 total NLS motif
  instances, 33 with experimental evidence (`ECO:0000269`, PubMed-backed),
  the rest rule/sequence-analysis annotated (`ECO:0000255`).
- `dna_bind_hard_negative_seed.json` -- 93 real proteins with UniProt
  `DNA binding` region annotations, used to build hard negatives: real,
  functional, genuinely basic (K/R-rich) sequence stretches that are *not*
  nuclear import signals. This directly targets the single most literature-
  documented failure mode of NLS predictors (plain basic-residue density
  gives ~45% accuracy per the NucPred/NLStradamus/seqNLS comparison
  studies -- see the landscape doc).
- `uniprot_nls_motifs_broad_563acc.json` -- a broader motif-only pull (563
  unique accessions, positions + evidence, no sequence) kept for
  provenance/reference.
- `build_dataset.py` -- assembles the two seed files into
  `nls_dataset.csv` (positives: one row per real NLS window, with exact
  extracted sequence, organism, evidence, monopartite/bipartite flag) and
  `nls_negatives.csv` (protein-matched random windows + DNA-binding hard
  negatives + synthetic shuffled-polybasic decoys). Run it any time the
  seed JSON files change.
- `uniprot_nls_scraper.py` -- **run this locally, with real internet.**
  A domain-allowlisted network cannot reach `rest.uniprot.org` in bulk
  (the same limitation already documented in
  `../nes_data_pipeline/structural_dataset_v2_pipeline.py`). This
  session's 159/93-protein seed files came from one-off calls through a
  different, allowlisted fetch path with no real pagination support past
  ~500 results and no thousands-of-accession loop. This script does it
  properly: real `Link`-header cursor pagination, configurable taxon list
  (10 taxa wired up, easy to add more or drop the filter for everything),
  writes `nls_proteins_full.json` / `dna_bind_hard_negatives_full.json`.
  Swap those in as `build_dataset.py`'s inputs (or overwrite the seed
  files) to scale from ~250 positives to something close to NLSdb's full
  ~2,253-motif scale.
- `structural_dataset_pipeline.py` -- **also run locally.** Fetches real
  per-residue SASA (Shrake-Rupley via `freesasa`) and pLDDT (from the
  AlphaFold model B-factor column) for every real (non-synthetic) example
  in the dataset, writes `structural_data.json`. Until this is run, the
  model's `plddt_norm`/`sasa_norm` features are the constant neutral
  default (0.75/0.50) -- confirmed by this project's own permutation
  importance run, where both sat at exactly 0.0 importance.

## Provenance note on the seed files

Built from queries to `rest.uniprot.org/uniprotkb/search`
(Swiss-Prot/`reviewed:true` only), of the shape:

```
ft_motif:"nuclear localization signal" AND reviewed:true AND taxonomy_id:9606
&fields=accession,organism_name,ft_motif,sequence&size=500
```

across `taxonomy_id` 9606 (human), 10090 (mouse), 559292 (yeast), 3702
(Arabidopsis), plus one untargeted `reviewed:true` pull, and the
equivalent `ft_dna_bind:*` query for the hard-negative set. This is a
genuine (if partial -- capped by response size, not a hard result-count
limit) live pull from UniProt, not hand-typed literature recall; a small
number of textbook sequences (SV40 T-antigen, nucleoplasmin, BRCA1, c-Myc)
are kept as a hand-verified fallback in `nls_ml_predictor.py`'s
`CURATED_SEED_NLS` in case the CSVs are ever unavailable.

## Recommended next step

```
pip install requests freesasa
python3 uniprot_nls_scraper.py           # full-scale UniProt pull, ~10-30 min
python3 structural_dataset_pipeline.py   # real SASA/pLDDT, ~20-60 min
python3 build_dataset.py                 # rebuild nls_dataset.csv / nls_negatives.csv
cd .. && python3 nls_ml_predictor.py train
```
