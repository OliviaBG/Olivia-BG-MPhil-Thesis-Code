"""
Improved NES Machine Learning Predictor (v2 -- real-data edition)
Based on LocNES (Xu et al. 2015) and NESmapper (Kosugi et al. 2014) methodologies

Key improvements over v1:
1. PSSM rank-based scoring (LocNES)
2. Position-specific scoring matrices for each NES class
3. Flanking region analysis (HPR and NC from NESmapper)
4. Spacer hydrophobicity penalty (NESmapper)
5. Disorder propensity integration (LocNES)
6. Model selection (linear SVM / RBF SVM / Gradient Boosting) with a
   comprehensive feature set

What's new in v2:
- Trains on REAL, experimentally-validated NES data pulled from NESbase 1.0
  and (once you run the scraper) NESdb, in addition to the original
  hand-curated seed list -- see nes_data_pipeline/ in this same folder.
- Negative examples now include protein-matched decoys (real windows from
  the same protein, outside the annotated NES) and structural hard negatives
  (hydrophobic patches inside coiled-coil / leucine-zipper regions that match
  the NES consensus by sequence alone but sit in the wrong structural
  context -- see nes_negatives/, built by the separate
  negative_dataset_builder.py + UniProt pipeline), alongside the original
  synthetic decoys.
- IMPORTANT usage note discovered while integrating the above: the single
  most important learned feature by a wide margin is c_flank_disorder (C-
  terminal flanking disorder), because real positives and structural hard
  negatives differ enormously there (a real NES's C-terminal flank is
  disordered; a coiled-coil/zipper's is not). If you call predict(sequence)
  with ONLY the bare NES peptide and no full_sequence/nes_start, there is no
  real flanking region to measure, so this feature silently falls back to a
  neutral default (0.5) -- which this model reads as much more "negative"
  than a real NES's genuine flanking context. Sanity-checking bare motifs
  like predict("LALKLAGLDI") with no context will therefore look like a
  false regression (PKI alone scored as low as ~0.11-0.14 in bare-motif
  testing) even though the exact same sequence scores ~1.0 when given its
  real full_sequence + nes_start, exactly how app.py always calls it in
  production. Always pass full_sequence + nes_start when testing/predicting
  if you want a meaningful number; bare-peptide calls are only useful for
  the PSSM/pattern-class fields, not the final probability.
- Also fixed a real bug in _calculate_spacer_hydrophobicity found while
  investigating the above: the old "spacer region" slice
  (sequence[hydrophobic_positions[1]:hydrophobic_positions[-1]]) included
  the sequence's own interior Phi anchor residues instead of only the true
  x(2,3) linker residues between them, so it fired on ~84% of real positives
  (measured against the full 313-positive NESdb+NESbase dataset) almost as
  often as on hard negatives -- essentially no discriminative value. Now
  only counts residues strictly between consecutive Phi anchors. The
  associated hard -7.0/0.0 step-penalty feature was also found to add no
  measurable value once this was fixed (removing it changed no predictions),
  so it's now fixed at 0.0 -- the model learns the relationship from the
  continuous spacer_hydrophobicity feature directly instead.
- Actual held-out train/test split with reported accuracy/precision/recall/
  F1/ROC-AUC, plus cross-validated model comparison -- previously there was
  no validation data at all, just a fit() call.
- PSSM is now aligned on the *consensus hydrophobic register* (the last Phi
  residue of the best NES-consensus match in each sequence) instead of raw
  C-terminal string alignment. This decouples the PSSM's column count (a
  fixed window width around that anchor) from how many training sequences
  you have, and from how long any individual training sequence happens to
  be -- the old version assumed most training NESs were close to 15 aa and
  aligned by treating the literal end of the string as position 15, which
  silently misaligned the conserved hydrophobic positions for anything
  shorter/longer or with a different C-terminal tail. More training
  sequences now only make each PSSM column's amino-acid frequencies less
  noisy; they don't change the window's dimensions.

- Now trains on REAL localCIDER *linear* (sliding-window) charge/hydropathy
  profiles, not just a hand-rolled scalar NCPR average: cider_ncpr_range and
  cider_hydropathy_range (peak-to-trough spread of the real localCIDER
  profile across the candidate +/-20aa context -- catches local charge/
  hydropathy patterning that a flat average erases) and
  cider_complexity_mean (linear Wootton-Federhen sequence complexity, an
  independent CIDER descriptor -- low-complexity/repetitive stretches are a
  classic false-positive pattern). Requires the `localcider` package;
  degrades to neutral defaults (not a crash) if it isn't installed. See
  _calculate_cider_linear_features for the full rationale.

All existing consumers (app.py etc.) keep working unchanged: predict() and
predict_protein() have the same signatures and return shapes, NES_PATTERNS
class names are untouched, and details['nes_classes'] / details['pssm_score']
/ details['spacer_hydrophobicity'] are still populated the same way.
"""

import csv
import json
import random
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import joblib

# localCIDER -- used here for REAL sliding-window (linear) charge/hydropathy
# profiles, not just a single averaged/scalar value. Optional import so this
# module still works (falling back to neutral defaults) if it isn't
# installed; every caller of _calculate_cider_linear_features degrades
# gracefully rather than crashing training/prediction.
try:
    from localcider.sequenceParameters import SequenceParameters
    CIDER_AVAILABLE = True
except ImportError:
    CIDER_AVAILABLE = False

_STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# XGBoost is an optional extra (not a core sklearn dependency) -- degrade to
# leaving it out of the model comparison rather than crashing if it isn't
# installed (`pip install xgboost`), same pattern as the localCIDER import
# above.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


def _relaxed_hydrophobic_prefilter(sequence):
    """Cheap gate before predict_protein() pays for a full predict() call on
    every window. Deliberately a LOOSER version of app.py's
    validate_nes_leucine_requirement (lower hydrophobic-count/frequency
    floors, no hard "must have a leucine" rule) -- this is meant to skip
    only the windows with essentially no chance of being an NES, not to do
    real discrimination itself. The trained model (self.predict) is what
    actually decides; this just saves work. Not imported from app.py
    on purpose -- app.py imports FROM this module, so importing back
    would create a circular import.
    """
    hydrophobic = sum(sequence.count(aa) for aa in 'LIVMF')
    length = len(sequence)
    if length == 0:
        return False
    min_hydro = 2 if length <= 10 else (3 if length <= 12 else 4)
    if hydrophobic < min_hydro:
        return False
    if hydrophobic / length < 0.22:
        return False
    return True


# Hydrophobicity scales (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0
}

CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0
}

# Disorder propensity (based on literature)
DISORDER_PROPENSITY = {
    'P': 1.0, 'R': 0.9, 'E': 0.9, 'K': 0.9, 'S': 0.8,
    'Q': 0.8, 'D': 0.8, 'G': 0.7, 'A': 0.5, 'T': 0.5,
    'N': 0.5, 'H': 0.4, 'M': 0.3, 'C': 0.3, 'L': 0.2,
    'F': 0.2, 'I': 0.2, 'V': 0.2, 'W': 0.1, 'Y': 0.1,
    'X': 0.5
}

HYDROPHOBIC_SET = set('LIVFM')  # Phi positions in the classic NES consensus

# Validated NES sequences (hand-curated seed set -- kept as a supplement to
# the real scraped/parsed data loaded from nes_data_pipeline/, and as a
# fallback so this module still works even before you've run the scraper)
NESDB_VALIDATED_NES = [
    # Class 1a: Φ0-X2-Φ1-X2-Φ2-X-Φ3 (canonical PKI-like)
    {'seq': 'LALKLAGLDI', 'protein': 'PKI', 'class': '1a'},
    {'seq': 'LQLPPLERLTL', 'protein': 'HIV-1 Rev', 'class': '1a'},
    {'seq': 'LPPLERLTL', 'protein': 'HIV-1 Rev minimal', 'class': '1a'},
    {'seq': 'MKLNVTEQEQIQL', 'protein': 'MAPKKK', 'class': '1a'},
    {'seq': 'LSNRELVVL', 'protein': 'NFAT', 'class': '1a'},
    {'seq': 'LGLGGLGLGL', 'protein': 'Synthetic optimal', 'class': '1a'},
    {'seq': 'LQHLRLISL', 'protein': 'STAT3', 'class': '1a'},
    {'seq': 'LEVFEALIG', 'protein': 'Smad3', 'class': '1a'},
    {'seq': 'LQEILEGL', 'protein': 'Emi1', 'class': '1a'},
    {'seq': 'LEQLIESIL', 'protein': 'FMD virus 3A', 'class': '1a'},
    {'seq': 'LQLKVWGL', 'protein': 'Cyclin B1', 'class': '1a'},
    {'seq': 'FTELRLLAL', 'protein': 'MDM2', 'class': '1a'},
    {'seq': 'LLDLLQAEL', 'protein': 'CPEB', 'class': '1a'},
    {'seq': 'MEELASLTSSFSV', 'protein': 'Snurportin', 'class': '1a'},

    # Class 1b: Φ0-X3-Φ1-X2-Φ2-X-Φ3
    {'seq': 'MTKKFGTLTV', 'protein': 'MAPK', 'class': '1b'},
    {'seq': 'LQKKLEELEL', 'protein': 'MEK1', 'class': '1b'},
    {'seq': 'LRTLHSIFLV', 'protein': 'Survivin', 'class': '1b'},
    {'seq': 'LSQALFLFL', 'protein': 'Cdc25C', 'class': '1b'},
    {'seq': 'LELLKVWSL', 'protein': 'AP-2α', 'class': '1b'},
    {'seq': 'IQKTLEEVL', 'protein': 'eEF1A', 'class': '1b'},
    {'seq': 'LETVEELRKR', 'protein': 'HTLV-1 Rex', 'class': '1b'},

    # Class 1c: Φ0-X2-Φ1-X3-Φ2-X-Φ3
    {'seq': 'LQKKIEQLL', 'protein': 'eIF2Bε', 'class': '1c'},
    {'seq': 'LCELFTTQL', 'protein': 'BRCA1', 'class': '1c'},
    {'seq': 'LERLRISQL', 'protein': 'IκBα', 'class': '1c'},
    {'seq': 'LEKLLEEIKQL', 'protein': 'Influenza NS2', 'class': '1c'},

    # Class 1d: Φ0-X3-Φ1-X3-Φ2-X-Φ3
    {'seq': 'LQRLVSLSQL', 'protein': 'FOXO1', 'class': '1d'},
    {'seq': 'LRLCIISQL', 'protein': 'FOXO3a', 'class': '1d'},

    # Class 2: Φ0-X-Φ1-X2-Φ2-X-Φ3
    {'seq': 'LPPLERLTL', 'protein': 'HIV-1 Rev', 'class': '2'},

    # Class 3: Φ0-X2-Φ1-X3-Φ2-X2-Φ3 (from LocNES validated)
    {'seq': 'VEALLARLA', 'protein': 'mDia2', 'class': '3'},
    {'seq': 'IQLLQEKLE', 'protein': 'Mad1', 'class': '3'},
    {'seq': 'IDSLTSMLA', 'protein': 'Trip6', 'class': '3'},
    {'seq': 'LQELVQQFE', 'protein': 'X11L2', 'class': '3'},
    {'seq': 'MTEFNQALE', 'protein': 'Rio2', 'class': '3'},
    {'seq': 'LRKLCERLR', 'protein': 'CDC7', 'class': '3'},
    {'seq': 'MHSLESSLI', 'protein': 'CPEB4', 'class': '3'},

    # Reverse NES
    {'seq': 'LPFLALFL', 'protein': 'CPEB4', 'class': '1a-R'},
    {'seq': 'LFTHLFLEL', 'protein': 'hRio2', 'class': '1a-R'},
]

# Expanded negative examples (patterns that look like NES but aren't)
NEGATIVE_SEQUENCES = [
    # Homopolymers
    {'seq': 'AAAAAAAAA'}, {'seq': 'KKKKKKKKK'}, {'seq': 'DDDDDDDDDD'},
    {'seq': 'EEEEEEEEEE'}, {'seq': 'SSSSSSSSSS'}, {'seq': 'GGGGGGGGGG'},
    {'seq': 'PPPPPPPPPP'}, {'seq': 'LLLLLLLLLL'}, {'seq': 'IIIIIIIII'},
    {'seq': 'VVVVVVVVV'}, {'seq': 'MMMMMMMMMM'}, {'seq': 'FFFFFFFFF'},

    # Alternating patterns
    {'seq': 'ALALALALAL'}, {'seq': 'KVKVKVKVKV'}, {'seq': 'LELELELELE'},
    {'seq': 'LKLKLKLKLK'}, {'seq': 'LVLVLVLVLV'},

    # Random hydrophobic-rich (no proper spacing)
    {'seq': 'LLKKLLKKLL'}, {'seq': 'LLLLEEEEEE'}, {'seq': 'MMMMKKKKKK'},
    {'seq': 'LLLKKKLLL'}, {'seq': 'FFFLLLFFF'},

    # Random sequences
    {'seq': 'QHVMKTPSR'}, {'seq': 'ETFNDAKMR'}, {'seq': 'SAPLKQNHY'},
    {'seq': 'VKGDPTFNM'}, {'seq': 'LPQYRTAQL'},

    # Proline-rich (structure breakers)
    {'seq': 'LPPKPKPL'}, {'seq': 'PPPPPLPPP'}, {'seq': 'PPPPPPPPP'},

    # Charged-rich
    {'seq': 'RRRRRRRRR'}, {'seq': 'TTTTTTTTT'}, {'seq': 'NNNNNNNNNN'},
    {'seq': 'QQQQQQQQQQ'},

    # Patterns that match consensus but lack proper hydrophobic spacing
    {'seq': 'LAKAKAKAL'}, {'seq': 'LEKEKELEL'}, {'seq': 'LDKDKDLDL'},
    {'seq': 'LQKQKQLQL'}, {'seq': 'LRKRKRLRL'},

    # Signal peptide-like (very hydrophobic start)
    {'seq': 'MLLLLLLLL'}, {'seq': 'MVVVVVVVV'}, {'seq': 'MIIIIIII'},

    # Transmembrane-like (continuous hydrophobicity)
    {'seq': 'LLVVLLVVL'}, {'seq': 'IIILLLIII'}, {'seq': 'VVVIIIVVV'},
    {'seq': 'FFFVVVFFF'}, {'seq': 'LLLFFFLLLFFF'},
]

# NES consensus patterns (modified Kosugi patterns from LocNES)
NES_PATTERNS = {
    'class_1a': r'[LIVFM].{2}[LIVFM].{2}[LIVFM].[LIVFM]',
    'class_1b': r'[LIVFM].{3}[LIVFM].{2}[LIVFM].[LIVFM]',
    'class_1c': r'[LIVFM].{2}[LIVFM].{3}[LIVFM].[LIVFM]',
    'class_1d': r'[LIVFM].{3}[LIVFM].{3}[LIVFM].[LIVFM]',
    'class_2': r'[LIVFM].[LIVFM].{2}[LIVFM].[LIVFM]',
    'class_3': r'[LIVFM].{2}[LIVFM].{3}[LIVFM].{2}[LIVFM]',  # Φ-X2-Φ-X3-Φ-X2-Φ
}

# ---------------------------------------------------------------------------
# Real-data loading (nes_data_pipeline/ scripts: nesbase_parser.py,
# nesdb_scraper.py, build_dataset.py). All of this degrades gracefully to
# "found nothing" so the module still works standalone.
# ---------------------------------------------------------------------------

REAL_DATASET_CANDIDATES = ['nes_data_pipeline/nes_dataset.csv', 'nes_dataset.csv']
REAL_NESBASE_CANDIDATES = ['nes_data_pipeline/nesbase_parsed.json', 'nesbase_parsed.json']
REAL_NESDB_CANDIDATES = ['nes_data_pipeline/nesdb.json', 'nesdb.json']


def _first_existing(base_dir, candidates):
    for rel in candidates:
        p = base_dir / rel
        if p.exists():
            return p
    return None


def _load_positives_from_dataset_csv(path):
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            seq = (row.get('nes_sequence') or '').strip()
            if not seq:
                continue
            full_seq = (row.get('full_sequence') or '').strip() or None
            start_raw = row.get('nes_start')
            try:
                start0 = int(start_raw) - 1 if start_raw not in (None, '') else None
            except ValueError:
                start0 = None
            crm1_raw = (row.get('crm1_dependent') or '').strip()
            crm1 = True if crm1_raw == 'True' else (False if crm1_raw == 'False' else None)
            out.append({
                'seq': seq.upper(), 'protein': row.get('protein_name') or 'unknown',
                'full_sequence': full_seq, 'start': start0,
                'crm1_dependent': crm1, 'source': row.get('source') or 'unknown',
            })
    return out


def _load_positives_from_nesbase_json(path):
    out = []
    with open(path, encoding='utf-8') as f:
        records = json.load(f)
    for rec in records:
        full_seq = rec.get('full_sequence') or None
        for seg in rec.get('nes_segments', []):
            out.append({
                'seq': seg['sequence'].upper(), 'protein': rec.get('protein', 'unknown'),
                'full_sequence': full_seq, 'start': seg['start'] - 1,
                'crm1_dependent': rec.get('crm1_dependent'), 'source': 'NESbase',
            })
    return out


def _load_positives_from_nesdb_json(path):
    out = []
    with open(path, encoding='utf-8') as f:
        records = json.load(f)
    for rec in records:
        full_seq = rec.get('full_sequence') or None
        for sig in rec.get('export_signals', []):
            out.append({
                'seq': sig['sequence'].upper(), 'protein': rec.get('name', 'unknown'),
                'full_sequence': full_seq, 'start': sig['start'] - 1,
                'crm1_dependent': rec.get('crm1_dependent'), 'source': 'NESdb',
            })
    return out


# Minimum plausible length for a real, standalone NES peptide. Every
# canonical consensus class in NES_PATTERNS needs at least 8 residues just to
# fit its Phi-x(n)-Phi-x(n)-Phi-x-Phi spacing (class_2, the shortest, is
# [LIVFM].[LIVFM].{2}[LIVFM].[LIVFM] = 9 chars; even the loosest reasonable
# reading can't fit under 8). Found while investigating why the ML predictor
# started favoring bizarrely short, leucine-poor windows in production
# Nes_dataset.csv contains ~20 "positive" rows far below this
# floor, including 12 literal single-residue entries ('L', 'F', 'I') from
# NESbase records (MKP-7, PKC-iota, SMAD4, RFP, RPF1/Nedd4) where the source
# annotation only ever flagged an isolated mutagenesis-critical residue and
# never marked the surrounding NES region -- nesbase_parser.py's
# segments_from_annotation correctly parses that annotation convention, but
# the resulting 1-residue "segment" is a parsing artifact, not a real
# standalone NES. A few NESdb-sourced rows are similarly bogus for a
# different reason -- 'CPEB' (CPEB4) and 'LZTS' (LZTS2) are truncated gene
# symbols, not peptide sequences, suggesting a scraper field mix-up.
#
# Because the shortest real hard-structural negative is 9 residues and the
# shortest synthetic decoy is 8 (see NEGATIVE_SEQUENCES / nes_negatives.csv),
# these sub-8 "positives" had literally no negative counterexample at their
# length to balance against -- nothing in training ever taught the model
# that a short candidate CAN be negative. That's very likely why length_norm
# surfaced as a top-3 permutation-importance feature: the model could
# partly separate classes just by leaning on length itself instead of real
# NES biology, which is backwards (see nes_permutation_importance_v2.json
# from before this fix, backed up in
# models/backup_before_min_length_fix_2026-07-22/).
#
# Filtered here (at the single point all three loaders funnel through)
# rather than patching nesbase_parser.py/nesdb_scraper.py individually, so
# nes_dataset.csv/nesbase_parsed.json/nesdb.json remain faithful records of
# what each source database actually said, and any future new data source
# gets the same sanity check for free.
MIN_REAL_NES_LEN = 8


def _filter_implausibly_short(entries, source_label=""):
    kept, dropped = [], []
    for e in entries:
        if len(e['seq']) >= MIN_REAL_NES_LEN:
            kept.append(e)
        else:
            dropped.append(e)
    if dropped:
        examples = ', '.join(f"{e['seq']!r} ({e.get('protein', 'unknown')})" for e in dropped[:8])
        more = f" and {len(dropped) - 8} more" if len(dropped) > 8 else ""
        print(f"  Dropped {len(dropped)} implausibly short 'positive' entries{source_label} "
              f"(< {MIN_REAL_NES_LEN} aa -- parsing artifacts, not real NES peptides): "
              f"{examples}{more}")
    return kept


def load_real_positive_examples(base_dir):
    """Load real, experimentally-validated NES examples produced by the
    nes_data_pipeline/ scripts. Prefers the merged nes_dataset.csv; falls
    back to reading nesbase_parsed.json / nesdb.json directly; returns []
    if none of those exist yet (caller then trains on the curated seed list
    alone, same as the original module did).

    Entries shorter than MIN_REAL_NES_LEN are dropped as parsing artifacts
    (see that constant's docstring) before being returned."""
    entries = []
    ds = _first_existing(base_dir, REAL_DATASET_CANDIDATES)
    if ds is not None:
        try:
            entries = _load_positives_from_dataset_csv(ds)
        except Exception as e:
            print(f"  (could not read {ds}: {e})")

    if not entries:
        nb = _first_existing(base_dir, REAL_NESBASE_CANDIDATES)
        if nb is not None:
            try:
                entries += _load_positives_from_nesbase_json(nb)
            except Exception as e:
                print(f"  (could not read {nb}: {e})")
        ndb = _first_existing(base_dir, REAL_NESDB_CANDIDATES)
        if ndb is not None:
            try:
                entries += _load_positives_from_nesdb_json(ndb)
            except Exception as e:
                print(f"  (could not read {ndb}: {e})")

    return _filter_implausibly_short(entries, source_label=" from real-data loaders")


HARD_NEGATIVE_CANDIDATES = ['nes_negatives/nes_negatives.csv']
# Was the ONLY hard-negative source this model ever trained on.
# A separate, much larger leucine-zipper/coiled-coil pool
# (nes_negatives_leucine_zipper_expansion/nes_negatives.csv, 217 unique
# accessions) was built later but only ever got wired into the CRM1 pocket
# weight fit (crm1_eval_results.json / compute_crm1_joint_weights.py) --
# never into this model's own training data. That gap is a real, evidenced
# cause of a holdout-test failure: 5 real leucine-zipper decoys drawn from
# that expansion pool, confirmed absent from BOTH this file and the CRM1
# weight-fit pool, scored ml_probability 0.85-1.0 (mean 0.97) -- statistically
# indistinguishable from real NES positives (mean 0.996) on the same test.
# Added below as a second training source, WITH a held-out reserve
# (nes_negatives_leucine_zipper_expansion/held_out_accessions.json, ~15% of
# the pool, seed-fixed) excluded so a genuinely never-trained-on set of hard
# leucine-zipper negatives still exists for future holdout tests -- folding
# the whole expansion file in here would otherwise quietly destroy the only
# clean validation set this project has for exactly this failure mode.
EXPANSION_HARD_NEGATIVE_SOURCE = 'nes_negatives_leucine_zipper_expansion/nes_negatives.csv'
EXPANSION_HELD_OUT_ACCESSIONS_FILE = 'nes_negatives_leucine_zipper_expansion/held_out_accessions.json'
# Cap is relative to how many positives you currently have (not a fixed
# number) so it scales sensibly as nes_data_pipeline/ grows -- with only
# ~70 positives today, dumping in all ~660+ structural hits at once pushes
# the negative:positive ratio to 6:1+ and visibly drags down recall on
# real positives (see the module docstring note on this trade-off).
MAX_HARD_NEGATIVES_RATIO = 2.0
MAX_HARD_NEGATIVES_CEILING = 300  # absolute safety ceiling regardless of ratio


def _load_hard_negatives_from_csv(path, exclude_accessions=None):
    """The CSV's 'context' column is a short real-sequence window around
    match_seq (see negative_dataset_builder.py) -- a few real residues of
    flanking sequence either side, taken from the actual coiled-coil/
    leucine-zipper region in the source protein. Used here as a (narrow but
    real) full_sequence/start so _extract_features() can compute genuine
    flanking-region features for these instead of falling back to the flat
    neutral default (see generate_matched_negatives' docstring for why that
    default-vs-real asymmetry between positives and negatives is a problem).
    Falls back to seq-only (no flanking context) if 'context' doesn't
    actually contain match_seq for some row.

    exclude_accessions: optional set of accessions to skip entirely -- used
    to keep a held-out reserve out of training (see
    EXPANSION_HELD_OUT_ACCESSIONS_FILE)."""
    exclude_accessions = exclude_accessions or set()
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            accession = (row.get('accession') or '').strip()
            if accession and accession in exclude_accessions:
                continue
            seq = (row.get('match_seq') or '').strip().upper()
            if not seq:
                continue
            context = (row.get('context') or '').strip().upper()
            start_in_context = context.find(seq) if context else -1
            out.append({
                'seq': seq,
                'accession': accession or None,
                'protein': row.get('protein_name') or row.get('accession') or 'unknown',
                'feature_kind': row.get('feature_kind'),
                'score': row.get('score'),
                'full_sequence': context if start_in_context >= 0 else None,
                'start': start_in_context if start_in_context >= 0 else None,
            })
    return out


STRUCTURAL_DATA_CANDIDATES = ['nes_data_pipeline/structural_data_v2.json', 'structural_data_v2.json']


def load_structural_data(base_dir):
    """Load real per-residue SASA + pLDDT arrays (produced by
    structural_dataset_v2_pipeline.py -- real AlphaFold structures fetched
    from live UniProt/AlphaFold, real Shrake-Rupley SASA, real pLDDT from
    the PDB B-factor column), keyed by exact NES/negative-window sequence
    (uppercase). This exists to fix a real gap: _train_model() previously
    never passed real plddt_values/sasa_values into _extract_features() for
    ANY example, so plddt_norm/sasa_norm were constant (always the neutral
    0.75/0.50 default) across the whole training set and carried zero
    learned signal. Returns {} if the file hasn't been generated yet --
    callers then fall back to the same neutral defaults as before this
    existed, so nothing breaks if you haven't run the pipeline."""
    p = _first_existing(base_dir, STRUCTURAL_DATA_CANDIDATES)
    if p is None:
        return {}
    try:
        with open(p, encoding='utf-8') as f:
            records = json.load(f)
        out = {}
        for rec in records:
            seq = (rec.get('seq') or '').upper()
            if not seq:
                continue
            out[seq] = {
                'plddt': rec.get('plddt_per_residue') or [],
                'sasa': rec.get('sasa_per_residue') or [],
                # Real CA-coordinate-derived continuous helix run near this
                # candidate (see real_ca_helix_geometry in
                # structural_dataset_v2_pipeline.py) -- None if the field is
                # missing (old-schema record) or the structure had no
                # coordinates there; a real int (incl. 0) once computed.
                'max_helix_run': rec.get('max_helix_run_near_candidate'),
            }
        return out
    except Exception as e:
        print(f"  (could not read {p}: {e})")
        return {}


IUPRED_DATA_CANDIDATES = ['nes_data_pipeline/iupred_data_v2.json', 'iupred_data_v2.json']


def load_iupred_data(base_dir):
    """Load real IUPred2A disorder + ANCHOR2 binding-region scores, keyed by
    exact NES/negative-window sequence (uppercase) -- same keying convention
    as load_structural_data() above, produced by the separate
    fetch_iupred_training_data.py script (run once, offline, against
    structural_data_v2.json's accession list; NOT run automatically here
    since it needs real internet access to iupred2a.elte.hu that this
    training process itself doesn't assume).

    Each record already has iupred_mean/n_flank_iupred/c_flank_iupred/
    anchor2_mean/n_flank_anchor2/c_flank_anchor2 PRE-COMPUTED by that script
    (it has access to the full canonical UniProt sequence returned by
    IUPred2A's API and aligns this exact seq window + +/-15 residue flanks
    within it -- the same alignment approach as app.py's
    align_iupred_to_structure(), and the same 15-residue flank window
    load_real_positive_examples' DISORDER_PROPENSITY flanking features use
    above, for direct comparability).

    Returns {} if the file hasn't been generated yet -- callers fall back to
    neutral defaults, so nothing breaks if you haven't run the fetch script."""
    p = _first_existing(base_dir, IUPRED_DATA_CANDIDATES)
    if p is None:
        return {}
    try:
        with open(p, encoding='utf-8') as f:
            records = json.load(f)
        out = {}
        for rec in records:
            seq = (rec.get('seq') or '').upper()
            if not seq:
                continue
            out[seq] = {
                'iupred_mean': rec.get('iupred_mean'),
                'n_flank_iupred': rec.get('n_flank_iupred'),
                'c_flank_iupred': rec.get('c_flank_iupred'),
                'anchor2_mean': rec.get('anchor2_mean'),
                'n_flank_anchor2': rec.get('n_flank_anchor2'),
                'c_flank_anchor2': rec.get('c_flank_anchor2'),
            }
        return out
    except Exception as e:
        print(f"  (could not read {p}: {e})")
        return {}


def load_hard_negative_examples(base_dir):
    """Load structural hard negatives from BOTH nes_negatives/nes_negatives.csv
    and nes_negatives_leucine_zipper_expansion/nes_negatives.csv (built
    separately by negative_dataset_builder.py / expand_leucine_zipper_negatives.py
    against live UniProt: hydrophobic patches inside coiled-coil /
    leucine-zipper regions that match the NES consensus pattern by sequence
    alone but sit in the wrong structural context -- CRM1 needs a
    surface-exposed, largely unstructured NES, not a buried helical heptad
    repeat). These are exactly the false positives a pattern-only scanner
    would make, so they're valuable negatives distinct from both the
    synthetic decoys and the protein-matched real negatives.

    the expansion file used to be loaded ONLY for the CRM1
    pocket-weight fit, never here -- this model had literally never trained
    on any of its 217 leucine-zipper accessions, which is a well-evidenced
    cause of a holdout-test failure (see EXPANSION_HARD_NEGATIVE_SOURCE
    comment above). Now merged in, deduped by sequence, with a fixed ~15%
    accession reserve (EXPANSION_HELD_OUT_ACCESSIONS_FILE) excluded so a
    genuinely unseen validation set still exists afterward. Returns []
    (or just the original-file negatives) if the relevant file(s) haven't
    been generated yet (see negative_dataset_builder.py)."""
    out, seen_seqs = [], set()

    p = _first_existing(base_dir, HARD_NEGATIVE_CANDIDATES)
    if p is not None:
        try:
            for n in _load_hard_negatives_from_csv(p):
                if n['seq'] in seen_seqs:
                    continue
                seen_seqs.add(n['seq'])
                out.append(n)
        except Exception as e:
            print(f"  (could not read {p}: {e})")

    exp_path = base_dir / EXPANSION_HARD_NEGATIVE_SOURCE
    if exp_path.exists():
        held_out = set()
        held_out_path = base_dir / EXPANSION_HELD_OUT_ACCESSIONS_FILE
        if held_out_path.exists():
            try:
                held_out = set(json.loads(held_out_path.read_text()).get('held_out_accessions', []))
            except Exception as e:
                print(f"  (could not read {held_out_path}: {e})")
        try:
            n_before = len(out)
            for n in _load_hard_negatives_from_csv(exp_path, exclude_accessions=held_out):
                if n['seq'] in seen_seqs:
                    continue
                seen_seqs.add(n['seq'])
                out.append(n)
            print(f"  (+{len(out) - n_before} hard negatives from {EXPANSION_HARD_NEGATIVE_SOURCE}, "
                  f"{len(held_out)} accessions held out for future testing)")
        except Exception as e:
            print(f"  (could not read {exp_path}: {e})")

    return out


def generate_matched_negatives(positive_entries, neg_per_pos=2, seed=42):
    """For every positive with a known full_sequence + start position,
    sample `neg_per_pos` windows of the same length from elsewhere in the
    *same* protein (excluding the real NES region itself). These are much
    harder, more realistic negatives than pure synthetic decoys, since they
    come from real protein context.

    IMPORTANT: full_sequence + start are carried through onto each generated
    negative (not just 'seq'/'protein'). Without this, _extract_features()
    has no full_sequence to compute real flanking-region features from, so
    every one of these negatives silently fell back to a flat neutral
    default (n_flank/c_flank empty -> disorder=0.5) even though real
    flanking context was available the whole time -- positives got real
    computed flanking disorder, these negatives got a constant placeholder.
    That's not "no signal for negatives", it's "distinguishable-by-construction
    negatives" -- the model could learn to separate on "is this feature
    suspiciously exactly the neutral default" rather than real biology,
    inflating the apparent importance of flanking-region features. Fixed by
    keeping full_sequence/start so real flanking disorder gets computed for
    these the same way it does for positives."""
    rng = random.Random(seed)
    out = []
    for p in positive_entries:
        full_seq = p.get('full_sequence')
        seq = p['seq']
        start = p.get('start')
        win_len = len(seq)
        if not full_seq or len(full_seq) <= win_len + 1:
            continue
        end = (start + win_len) if start is not None else None
        made, tries = 0, 0
        while made < neg_per_pos and tries < neg_per_pos * 25:
            tries += 1
            i = rng.randint(0, len(full_seq) - win_len)
            j = i + win_len
            if start is not None and end is not None and not (j <= start or i >= end):
                continue
            out.append({'seq': full_seq[i:j].upper(), 'protein': p['protein'],
                        'full_sequence': full_seq, 'start': i})
            made += 1
    return out


# ---------------------------------------------------------------------------
# PSSM: register-anchored (not tied to raw string length / dataset size)
# ---------------------------------------------------------------------------

# Broad consensus used only to find an anchor point (the last Phi of the
# best match), not for classification -- that's what NES_PATTERNS is for.
PSSM_ANCHOR_RE = re.compile(r'[LIVFM].{1,3}[LIVFM].{1,3}[LIVFM].{1,2}[LIVFM]')
PSSM_LEFT = 12   # columns kept at/left of the anchor (inclusive of the anchor column)
PSSM_RIGHT = 3   # columns kept to the right of the anchor
PSSM_WIDTH = PSSM_LEFT + PSSM_RIGHT


class ImprovedNESPredictor:
    """
    Improved NES predictor based on LocNES and NESmapper methodologies,
    trained on real scraped/parsed NES data when available.
    """

    def __init__(self, model_dir='models', data_dir=None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Where to look for nes_data_pipeline/ output -- defaults to this
        # script's own directory so it works regardless of the caller's cwd.
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent

        self.model = None
        self.scaler = None
        self.pssm = None
        self.model_name = None
        self.metrics = None
        # Permutation importance computed on a genuine held-out split during
        # training (see _train_model) -- a materially more trustworthy
        # measure of feature importance than raw tree-impurity/coefficient
        # values for correlated features (see diagnose_feature_importance.py
        # for the full investigation of why this was added). None if it
        # hasn't been computed yet (e.g. model predates this change, or the
        # dataset was too small for a held-out split).
        self.permutation_importance_ = None

        # NOTE: filenames bumped to _v2 on purpose -- the old nes_svm_improved.pkl
        # etc. were trained on a tiny hardcoded set with a differently-aligned
        # PSSM. Loading them here would silently mix old and new assumptions,
        # so v2 always starts from a clean, freshly-trained model the first
        # time it runs in a given models/ directory.
        self.model_path = self.model_dir / 'nes_svm_v2.pkl'
        self.scaler_path = self.model_dir / 'nes_scaler_v2.pkl'
        self.pssm_path = self.model_dir / 'nes_pssm_v2.pkl'
        self.metrics_path = self.model_dir / 'nes_metrics_v2.json'
        self.meta_path = self.model_dir / 'nes_model_meta_v2.json'
        self.permutation_importance_path = self.model_dir / 'nes_permutation_importance_v2.json'

        if (self.model_path.exists() and self.scaler_path.exists()
                and self.pssm_path.exists()):
            self._load_model()
        else:
            self._train_model()

    def _load_model(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.pssm = joblib.load(self.pssm_path)
            if self.meta_path.exists():
                with open(self.meta_path) as f:
                    self.model_name = json.load(f).get('model_name')
            if self.metrics_path.exists():
                with open(self.metrics_path) as f:
                    self.metrics = json.load(f)
            if self.permutation_importance_path.exists():
                with open(self.permutation_importance_path) as f:
                    self.permutation_importance_ = json.load(f)
            print(f"Loaded improved NES model ({self.model_name or 'unknown type'})"
                  + ('' if self.permutation_importance_ else
                     ' -- no permutation importance on file yet; retrain to add it'))
        except Exception as e:
            print(f"Warning: Error loading model: {e}")
            self._train_model()

    # -- PSSM ----------------------------------------------------------

    def _find_pssm_anchor(self, seq):
        """Index (0-based) of the last hydrophobic residue of the
        rightmost consensus-pattern match in `seq`. Falls back to the last
        Phi (L/I/V/F/M) residue in the sequence if no regex match is found,
        so every sequence is alignable even when it doesn't cleanly match
        the consensus."""
        last_match = None
        for m in PSSM_ANCHOR_RE.finditer(seq):
            last_match = m
        if last_match is not None:
            return last_match.end() - 1
        phi_pos = [i for i, a in enumerate(seq) if a in HYDROPHOBIC_SET]
        return phi_pos[-1] if phi_pos else len(seq) - 1

    def _build_pssm(self, sequences):
        """
        Build a Position-Specific Scoring Matrix from validated NES
        sequences, aligned on the consensus hydrophobic register (see
        `_find_pssm_anchor`) rather than on raw string ends. Window width
        (PSSM_WIDTH) is a fixed constant independent of the number of
        training sequences or their individual lengths.
        """
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        counts = np.zeros((PSSM_WIDTH, 20))
        n_used = 0

        for entry in sequences:
            seq = entry['seq'].upper()
            if not seq:
                continue
            anchor = self._find_pssm_anchor(seq)
            for c in range(PSSM_WIDTH):
                idx = anchor - (PSSM_LEFT - 1) + c
                if 0 <= idx < len(seq) and seq[idx] in aa_to_idx:
                    counts[c, aa_to_idx[seq[idx]]] += 1
            n_used += 1

        counts += 1.0  # pseudocount
        background = 1.0 / 20
        freqs = counts / counts.sum(axis=1, keepdims=True)
        pssm = np.log2(freqs / background)

        return pssm, aa_to_idx, n_used

    def _calculate_pssm_score(self, sequence):
        """Calculate PSSM score for a sequence, anchored the same way the
        matrix itself was built."""
        if self.pssm is None:
            return 0.0

        pssm, aa_to_idx = self.pssm[0], self.pssm[1]
        seq = sequence.upper()
        if not seq:
            return 0.0

        anchor = self._find_pssm_anchor(seq)
        score = 0.0
        for c in range(PSSM_WIDTH):
            idx = anchor - (PSSM_LEFT - 1) + c
            if 0 <= idx < len(seq) and seq[idx] in aa_to_idx:
                score += pssm[c, aa_to_idx[seq[idx]]]
        return score

    def _classify_nes_pattern(self, sequence):
        """Classify which NES pattern the sequence matches"""
        matched_classes = []
        for nes_class, pattern in NES_PATTERNS.items():
            if re.search(pattern, sequence):
                matched_classes.append(nes_class)
        return matched_classes if matched_classes else ['unknown']

    # Below this many actual residues in a flank window, HPR/NC are computed
    # from too small a sample to mean anything -- this only happens when the
    # candidate sits near a protein's TRUE N- or C-terminus (the window runs
    # off the end of full_sequence). Real NESs absolutely occur at true
    # termini biologically, so a starved flank must not be treated as
    # unfavorable evidence -- that would systematically discount genuine
    # terminus-proximal NESs. 15 = 60% of the intended 25-residue window.
    MIN_FLANK_FOR_HPR_NC = 15

    def _analyze_flanking_regions(self, full_sequence, nes_start, nes_end):
        """
        Analyze flanking regions (NESmapper methodology)
        Returns HPR (hydrophobicity rate) and NC (net charge) scores

        near a true protein terminus, the N- or C-flank window is
        truncated (or absent). Previously this silently produced HPR/NC from
        whatever partial flank existed (or 0/neutral defaults when empty),
        which fed straight into hpr_ratio/nc_ratio as if it were real
        unfavorable signal. Now: if a flank has fewer than
        MIN_FLANK_FOR_HPR_NC residues, its likelihood ratio is set to neutral
        (1.0) instead of being computed -- "not enough context to judge",
        not "bad context". The other flank (if it has enough residues) still
        contributes its real ratio as before. n_flank_reliable/c_flank_reliable
        are exposed for transparency but are not fed into the ML feature
        vector (that stays hpr/nc/hpr_likelihood/nc_likelihood, unchanged in
        shape -- only their computed values differ for terminus-proximal
        candidates).
        """
        # N-terminal flanking (25 residues)
        n_flank_start = max(0, nes_start - 25)
        n_flank = full_sequence[n_flank_start:nes_start]
        n_flank_reliable = len(n_flank) >= self.MIN_FLANK_FOR_HPR_NC

        # Calculate Hydrophobicity Rate (HPR)
        hydrophobic_aas = set('AILVMFYW')
        hpr = (sum(1 for aa in n_flank if aa in hydrophobic_aas) / len(n_flank) * 100
               if n_flank else 0)

        # C-terminal flanking (25 residues)
        c_flank_end = min(len(full_sequence), nes_end + 25)
        c_flank = full_sequence[nes_end:c_flank_end]
        c_flank_reliable = len(c_flank) >= self.MIN_FLANK_FOR_HPR_NC

        # Calculate Net Charge (NC)
        positive = sum(1 for aa in c_flank if aa in 'KR')
        negative = sum(1 for aa in c_flank if aa in 'DE')
        nc = positive - negative

        # Apply likelihood ratios from NESmapper -- only when there's enough
        # flank to trust the ratio; otherwise neutral (no adjustment).
        if n_flank_reliable:
            hpr_ratio = (
                2.5 if hpr <= 30 else
                2.0 if hpr <= 40 else
                1.4 if hpr <= 50 else
                1.0 if hpr <= 60 else
                0.6 if hpr <= 80 else
                0.5
            )
        else:
            hpr_ratio = 1.0  # too close to N-terminus to judge -- neutral, not penalized

        if c_flank_reliable:
            nc_ratio = 1.8 if nc <= -4 else (0.6 if nc > 0 else 1.0)
        else:
            nc_ratio = 1.0  # too close to C-terminus to judge -- neutral, not penalized

        return {
            'hpr': hpr,
            'nc': nc,
            'hpr_likelihood': hpr_ratio,
            'nc_likelihood': nc_ratio,
            'combined_likelihood': hpr_ratio * nc_ratio,
            'n_flank_reliable': n_flank_reliable,
            'c_flank_reliable': c_flank_reliable,
        }

    def _calculate_spacer_hydrophobicity(self, sequence):
        """
        Calculate hydrophobicity rate in spacer regions
        High hydrophobicity (≥0.4) receives penalty (NESmapper)

        NOTE (v2 fix): the original implementation took
        sequence[hydrophobic_positions[1]:hydrophobic_positions[-1]] as "the
        spacer region", but that slice runs from the *2nd* Phi anchor to the
        *last* Phi anchor and so swallows every interior Phi anchor residue
        along the way -- for a compact NES like PKI (LALKLAGLDI) the "spacer"
        computed this way is "LKLAGLD", which is mostly the NES's own
        defining hydrophobic core, not the x(2,3) linker residues NESmapper
        actually means by "spacer". That made the feature fire on ~84% of
        real positives (measured against the full NESdb+NESbase dataset,
        n=313) almost as often as on structural hard negatives -- i.e. it had
        essentially no discriminative value and was actively penalizing
        genuine NESs. Fixed to only look at residues *strictly between*
        consecutive Phi anchors (excluding the anchors themselves), pooled
        across every anchor gap in the sequence.
        """
        if len(sequence) < 8:
            return 0.0

        # Find hydrophobic (Phi) anchor positions
        hydrophobic_positions = [i for i, aa in enumerate(sequence) if aa in 'LIVFM']

        if len(hydrophobic_positions) < 3:
            return 0.0

        # True spacer/linker residues: strictly between consecutive Phi
        # anchors, excluding the anchors themselves.
        gap_residues = []
        for a, b in zip(hydrophobic_positions, hydrophobic_positions[1:]):
            gap_residues.extend(sequence[a + 1:b])

        if not gap_residues:
            return 0.0

        hydrophobic_aas = set('AILVMFYW')
        hpr = sum(1 for aa in gap_residues if aa in hydrophobic_aas) / len(gap_residues)

        return hpr

    def _calculate_heptad_periodicity(self, full_sequence, nes_start, nes_end, flank=15, min_len=14):
        """Coiled-coil/leucine-zipper heptad-repeat 'peakiness': how much a
        7-residue periodic hydrophobic register (positions a+d, 3 apart,
        checked at all 7 possible phases) stands out from the other 6
        phases, over the candidate window plus its flanks. Added
        after a holdout test (real experimentally-validated viral NES motifs
        vs real hard leucine-zipper negatives, via run_holdout_pipeline_test.py)
        showed the model -- even after retraining on a much larger, better-
        covered leucine-zipper negative set -- still couldn't separate its 5
        hardest cases from real NESs. Model-free validation first
        (diagnose_heptad_periodicity.py, no ML involved): real NES flanks
        (n=307) score lower (mean 1.54) than leucine-zipper/coiled-coil
        flanks (n=1062, mean ~1.66-1.69), brute-force AUC 0.64 in the
        biologically-expected direction (real coiled-coils hold a stable
        heptad register over many residues; NES anchor spacing --
        Phi-x(2,3)-Phi-x(2,3)-Phi-x(1,3)-Phi -- has no reason to keep
        repeating on a 7-residue period outside the motif itself). NOT a
        fix for the 5 specific holdout hard cases (their peakiness sits in
        1.27-1.48, close to the positive range) -- added because it's real,
        independent signal for the GENERAL leucine-zipper class, not
        because it resolves that particular stress test.

        Returns None (caller substitutes a neutral default) if the
        available window is too short for phase averaging to mean anything
        -- same MIN-length guard philosophy as the flanking-disorder
        features above."""
        if not full_sequence:
            window = ''
        else:
            s0 = max(0, nes_start - flank)
            e0 = min(len(full_sequence), nes_end + flank)
            window = full_sequence[s0:e0]
        h = [HYDROPHOBICITY.get(aa, 0.0) for aa in window.upper() if aa in HYDROPHOBICITY]
        n = len(h)
        if n < min_len:
            return None
        phase_means = []
        for phase in range(7):
            vals = [h[i] for i in range(n) if i % 7 == phase or i % 7 == (phase + 3) % 7]
            if len(vals) < 2:
                return None
            phase_means.append(sum(vals) / len(vals))
        mu = sum(phase_means) / len(phase_means)
        sd = (sum((v - mu) ** 2 for v in phase_means) / len(phase_means)) ** 0.5
        if sd < 1e-6:
            return 0.0
        return (max(phase_means) - mu) / sd

    def _calculate_ncpr(self, seq):
        """Net charge per residue (localCIDER-style: (K+R positions) minus
        (D+E positions), divided by length). Added after analyzing 26 real
        NES-containing proteins with real AlphaFold structures + localCIDER:
        real NES regions run reliably MORE NEGATIVE than the rest of their
        own protein (paired Wilcoxon p=0.011, mean NCPR -0.09 inside the NES
        vs -0.02 across the whole protein, 20/26 proteins showing this
        direction) -- a genuine, sequence-computable signal the model didn't
        have before."""
        if not seq:
            return 0.0
        pos = sum(seq.count(aa) for aa in 'KR')
        neg = sum(seq.count(aa) for aa in 'DE')
        return (pos - neg) / len(seq)

    def _calculate_cider_linear_features(self, sequence, full_sequence=None,
                                          nes_start=0, flank=20, blob_len=5):
        """
        Real localCIDER *linear* (sliding-window, per-residue) profiles
        across the candidate NES plus its +/-flank real protein context --
        not just a single averaged charge number like ncpr_local above.

        Why this is worth adding on top of ncpr_local/ncpr_flank_contrast:
        a candidate can have a perfectly ordinary AVERAGE net charge of ~0
        while still having a sharp negative patch right where CRM1 actually
        contacts the groove and a compensating positive patch elsewhere --
        the average erases that shape entirely. These features instead
        capture the RANGE (peak-to-trough spread) of the real localCIDER
        linear NCPR and hydropathy profiles over the region, which is
        sensitive to that kind of local patterning even when the mean looks
        unremarkable. Also adds linear sequence complexity (Wootton-
        Federhen), a genuinely independent CIDER descriptor unrelated to
        charge/hydrophobicity -- low-complexity/repetitive stretches are a
        classic false-positive pattern that the amino-acid-fraction features
        elsewhere don't directly capture.

        Falls back to neutral defaults if localcider isn't installed or the
        context is too short (<5 residues) to profile meaningfully.
        """
        defaults = {'cider_ncpr_range': 0.0, 'cider_hydropathy_range': 0.0,
                    'cider_complexity_mean': 0.5}
        if not CIDER_AVAILABLE:
            return defaults

        if full_sequence and len(full_sequence) > len(sequence):
            nes_end = nes_start + len(sequence)
            ctx_start = max(0, nes_start - flank)
            ctx_end = min(len(full_sequence), nes_end + flank)
            ctx_seq = full_sequence[ctx_start:ctx_end]
        else:
            ctx_seq = sequence

        n = len(ctx_seq)
        if n < 5:
            return defaults

        try:
            clean_seq = ''.join(aa if aa in _STANDARD_AA else 'G' for aa in ctx_seq.upper())
            sp = SequenceParameters(clean_seq)
            eff_blob = min(blob_len, n if n % 2 == 1 else n - 1)
            eff_blob = max(1, eff_blob)
            _, ncpr = sp.get_linear_NCPR(blobLen=eff_blob)
            _, hydro = sp.get_linear_hydropathy(blobLen=eff_blob)
            complexity_mean = 0.5
            try:
                _, complexity = sp.get_linear_complexity(blobLen=eff_blob)
                if len(complexity):
                    complexity_mean = float(np.mean(complexity))
            except Exception:
                pass
            return {
                'cider_ncpr_range': float(max(ncpr) - min(ncpr)) if len(ncpr) else 0.0,
                'cider_hydropathy_range': float(max(hydro) - min(hydro)) if len(hydro) else 0.0,
                'cider_complexity_mean': complexity_mean,
            }
        except Exception:
            return defaults

    def _feature_names(self):
        """Names matching the exact order `_extract_features` appends in --
        used for get_feature_importance()."""
        names = ['pssm_score']
        names += [f'phi_pos_{i}_hydrophobicity' for i in range(5)]
        names += ['mean_hydro', 'var_hydro', 'max_hydro']
        names += ['nes_disorder_mean']
        names += ['n_flank_disorder', 'c_flank_disorder']
        names += ['flank_hpr_norm', 'flank_nc_norm', 'flank_hpr_likelihood', 'flank_nc_likelihood']
        names += ['spacer_hydrophobicity', 'spacer_penalty']
        names += ['frac_L', 'frac_I', 'frac_V', 'frac_phi_total']
        names += ['frac_acidic', 'frac_basic']
        names += ['ncpr_local', 'ncpr_flank_contrast']
        names += ['cider_ncpr_range', 'cider_hydropathy_range', 'cider_complexity_mean']
        names += ['plddt_norm', 'sasa_norm']
        names += ['class_1a', 'class_1b', 'class_1c', 'class_1d', 'class_2', 'class_3']
        names += ['length_norm']
        # NEW (candidate features, additive alongside nes_disorder_mean/
        # n_flank_disorder/c_flank_disorder above, NOT a replacement -- see
        # load_iupred_data() docstring). Appended at the end rather than
        # inline with the existing disorder features so this is a pure
        # addition: any code relying on the first N feature positions
        # matching the pre-IUPred layout is unaffected. Neutral defaults
        # (0.5 for the two IUPred-derived means, 0.0 for the two ANCHOR2
        # means -- see _extract_features) when iupred_data_v2.json hasn't
        # been generated yet (run fetch_iupred_training_data.py first).
        names += ['iupred_mean', 'n_flank_iupred', 'c_flank_iupred']
        names += ['anchor2_mean', 'n_flank_anchor2', 'c_flank_anchor2']
        # NEW (see _calculate_heptad_periodicity) -- appended at the end,
        # same additive-not-replacing convention as the IUPred/ANCHOR2 block
        # above.
        names += ['heptad_periodicity']
        names += ['max_helix_run_norm']
        return names

    def get_feature_importance(self, method='auto'):
        """Return {feature_name: importance} for the currently loaded model.

        method:
          'auto' (default) -- permutation importance if it was computed
            during training (self.permutation_importance_, held-out F1 drop
            when a feature's values are shuffled), falling back to impurity/
            coefficient otherwise. Permutation importance is the more
            trustworthy of the two when features are correlated -- impurity
            importance (a tree's feature_importances_) can inflate one
            feature among several correlated candidates just because it
            happened to give the best split first. See
            diagnose_feature_importance.py for the investigation that
            motivated this default (also cross-checked there against a
            model-free univariate AUC test, which agreed on this dataset).
            Falls back automatically for models loaded before this was
            added, or when the dataset was too small for a held-out split.
          'permutation' -- force permutation importance; {} if unavailable.
          'impurity' -- force the original tree-impurity/coefficient-based
            importance (works for linear SVM via coef_ and Gradient
            Boosting via feature_importances_; {} for kernels like RBF that
            expose neither).
        """
        if method not in ('auto', 'permutation', 'impurity'):
            raise ValueError(f"Unknown method {method!r}; use 'auto', 'permutation', or 'impurity'")

        if method in ('auto', 'permutation') and self.permutation_importance_:
            return dict(self.permutation_importance_)
        if method == 'permutation':
            return {}

        # impurity/coefficient fallback
        if self.model is None:
            return {}
        names = self._feature_names()
        try:
            if hasattr(self.model, 'coef_'):
                values = np.asarray(self.model.coef_).ravel()
            elif hasattr(self.model, 'feature_importances_'):
                values = np.asarray(self.model.feature_importances_).ravel()
            else:
                return {}
            return {name: float(v) for name, v in zip(names, values)}
        except Exception:
            return {}

    def _extract_features(self, sequence, full_sequence=None, nes_start=0,
                         plddt_values=None, sasa_values=None, iupred_values=None,
                         max_helix_run=None):
        """
        Extract comprehensive features based on LocNES + NESmapper

        Feature categories:
        1. PSSM-based features (LocNES)
        2. Sequence composition
        3. Hydrophobicity patterns
        4. Disorder propensity (LocNES)
        5. Flanking region analysis (NESmapper)
        6. Spacer hydrophobicity (NESmapper)
        7. Structural features
        8. NES class indicators
        """
        features = []

        # Use sequence as full_sequence if not provided
        if full_sequence is None:
            full_sequence = sequence
            nes_start = 0

        nes_end = nes_start + len(sequence)

        # 1. PSSM score
        pssm_score = self._calculate_pssm_score(sequence)
        features.append(float(pssm_score))

        # 2. Sequence composition (one-hot encoding for key positions)
        # Φ0, Φ1, Φ2, Φ3, Φ4 positions
        hydrophobic_positions = [i for i, aa in enumerate(sequence) if aa in 'LIVFM']
        if len(hydrophobic_positions) >= 4:
            for j in range(5):  # Always add exactly 5 features
                if j < len(hydrophobic_positions):
                    pos = hydrophobic_positions[j]
                    if pos < len(sequence):
                        features.append(float(HYDROPHOBICITY.get(sequence[pos], 0)))
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
        else:
            features.extend([0.0] * 5)

        # 3. Hydrophobicity features
        hydro_scores = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]
        features.append(float(np.mean(hydro_scores)) if hydro_scores else 0.0)  # mean
        features.append(float(np.var(hydro_scores)) if hydro_scores else 0.0)   # variance
        features.append(float(np.max(hydro_scores)) if hydro_scores else 0.0)   # max

        # 4. Disorder propensity (LocNES key feature)
        disorder_scores = [DISORDER_PROPENSITY.get(aa, 0.5) for aa in sequence]
        features.append(float(np.mean(disorder_scores)))  # NES region disorder

        # Flanking disorder (important from LocNES findings)
        # Same terminus-truncation issue as _analyze_flanking_regions
        # -- a candidate near a true N/C-terminus has a short-or-absent flank
        # window, and computing a "disorder mean" from 1-2 residues is noise,
        # not signal. Previously any non-empty flank (however short) was used;
        # now require at least MIN_DISORDER_FLANK_LEN residues (60% of the
        # intended 15-residue window) before trusting the computed value --
        # below that, keep the neutral 0.5 default rather than let a starved
        # sample masquerade as evidence against the candidate.
        MIN_DISORDER_FLANK_LEN = 9
        n_flank_disorder = 0.5
        c_flank_disorder = 0.5
        if full_sequence:
            n_flank_start = max(0, nes_start - 15)
            n_flank = full_sequence[n_flank_start:nes_start]
            if len(n_flank) >= MIN_DISORDER_FLANK_LEN:
                n_flank_disorder = float(np.mean([DISORDER_PROPENSITY.get(aa, 0.5) for aa in n_flank]))

            c_flank_end = min(len(full_sequence), nes_end + 15)
            c_flank = full_sequence[nes_end:c_flank_end]
            if len(c_flank) >= MIN_DISORDER_FLANK_LEN:
                c_flank_disorder = float(np.mean([DISORDER_PROPENSITY.get(aa, 0.5) for aa in c_flank]))

        features.append(float(n_flank_disorder))
        features.append(float(c_flank_disorder))

        # 5. Flanking region analysis (NESmapper)
        if full_sequence and len(full_sequence) > len(sequence):
            flanking = self._analyze_flanking_regions(full_sequence, nes_start, nes_end)
            features.append(float(flanking['hpr']) / 100.0)  # Normalize
            features.append(float(flanking['nc']) / 10.0)    # Normalize
            features.append(float(flanking['hpr_likelihood']))
            features.append(float(flanking['nc_likelihood']))
        else:
            features.extend([0.5, 0.0, 1.0, 1.0])  # Default values

        # 6. Spacer hydrophobicity (NESmapper)
        spacer_hydro = self._calculate_spacer_hydrophobicity(sequence)
        features.append(float(spacer_hydro))

        # Penalty for high spacer hydrophobicity -- was a hard -7.0/0.0 step,
        # wildly outside the scale of every other feature (which run roughly
        # -5..+5), making it act as an unlearned override rather than
        # something the model weighs itself. Tested removing it (kept as a
        # fixed 0.0 placeholder so the feature vector length / feature_names
        # order stay stable for backward compatibility): had zero measurable
        # effect on any prediction once spacer_hydrophobicity itself was
        # fixed, so it added risk without adding signal. The model now
        # learns the spacer relationship from the continuous
        # spacer_hydrophobicity feature above instead.
        spacer_penalty = 0.0
        features.append(float(spacer_penalty))

        # 7. Amino acid composition
        features.append(float(sequence.count('L')) / len(sequence))
        features.append(float(sequence.count('I')) / len(sequence))
        features.append(float(sequence.count('V')) / len(sequence))
        features.append(float(sum(sequence.count(aa) for aa in 'LIVFM')) / len(sequence))

        # Charge features
        features.append(float(sum(sequence.count(aa) for aa in 'DE')) / len(sequence))  # acidic
        features.append(float(sum(sequence.count(aa) for aa in 'KR')) / len(sequence))  # basic

        # NCPR of the candidate window itself, and its contrast against the
        # surrounding +/-20 residue protein context -- see _calculate_ncpr
        # docstring for the real-data validation behind this.
        ncpr_local = self._calculate_ncpr(sequence)
        features.append(ncpr_local)

        flank_window = 20
        if full_sequence:
            n_flank_ctx = full_sequence[max(0, nes_start - flank_window):nes_start]
            c_flank_ctx = full_sequence[nes_end:nes_end + flank_window]
            flank_ctx = n_flank_ctx + c_flank_ctx
        else:
            flank_ctx = ''
        ncpr_flank_contrast = (ncpr_local - self._calculate_ncpr(flank_ctx)) if flank_ctx else 0.0
        features.append(ncpr_flank_contrast)

        # Real localCIDER linear-profile features (range/shape, not just a
        # scalar average) -- see _calculate_cider_linear_features docstring.
        cider_feats = self._calculate_cider_linear_features(sequence, full_sequence, nes_start)
        features.append(float(cider_feats['cider_ncpr_range']))
        features.append(float(cider_feats['cider_hydropathy_range']))
        features.append(float(cider_feats['cider_complexity_mean']))

        # 8. Structural features
        features.append(float(np.mean(plddt_values)) / 100.0 if plddt_values is not None and len(plddt_values) > 0 else 0.75)
        # sasa_values is expected to already be RELATIVE solvent accessibility
        # (RSA, 0-1ish, Tien et al. 2013 residue-normalized) -- see
        # calculate_sasa() in app.py and consensus_accessibility.py. It is
        # NOT raw SASA in Ų, so no further division here. Both the live app
        # and the training data pipelines (nes_data_pipeline/*.py) must
        # produce sasa_values on this same RSA scale for this feature to be
        # meaningful; a flat /100.0 on raw Ų previously conflated residue
        # size with true burial (see PR discussion / consensus_accessibility.py).
        features.append(min(1.0, float(np.mean(sasa_values))) if sasa_values is not None and len(sasa_values) > 0 else 0.50)

        # 9. NES class indicators (one-hot)
        matched_classes = self._classify_nes_pattern(sequence)
        for nes_class in ['class_1a', 'class_1b', 'class_1c', 'class_1d', 'class_2', 'class_3']:
            features.append(1.0 if nes_class in matched_classes else 0.0)

        # 10. Sequence length
        features.append(float(len(sequence)) / 15.0)  # Normalize to typical NES length

        # 11. NEW: IUPred2A / ANCHOR2 (candidate features, additive alongside
        # the DISORDER_PROPENSITY-derived features above -- see
        # load_iupred_data() docstring for why these are kept separate
        # rather than replacing nes_disorder_mean/n_flank_disorder/
        # c_flank_disorder outright). `iupred_values` is the dict
        # build_training_dataset() looks up from load_iupred_data() (or,
        # at inference time, whatever the caller passes -- currently
        # nothing does, since wiring this into live prediction is a
        # separate step from adding it as a trainable feature). Neutral
        # defaults match the existing convention: 0.5 for the two
        # disorder-scale means (same as DISORDER_PROPENSITY's 'X' unknown
        # default just above), 0.0 for the two ANCHOR2 means (absence of
        # evidence isn't evidence of a binding region, unlike generic
        # disorder where "unknown" plausibly means "average").
        iv = iupred_values or {}
        features.append(float(iv.get('iupred_mean')) if iv.get('iupred_mean') is not None else 0.5)
        features.append(float(iv.get('n_flank_iupred')) if iv.get('n_flank_iupred') is not None else 0.5)
        features.append(float(iv.get('c_flank_iupred')) if iv.get('c_flank_iupred') is not None else 0.5)
        features.append(float(iv.get('anchor2_mean')) if iv.get('anchor2_mean') is not None else 0.0)
        features.append(float(iv.get('n_flank_anchor2')) if iv.get('n_flank_anchor2') is not None else 0.0)
        features.append(float(iv.get('c_flank_anchor2')) if iv.get('c_flank_anchor2') is not None else 0.0)

        # 12. NEW: heptad-repeat periodicity (see _calculate_heptad_periodicity
        # docstring). Neutral default 0.0 (= "no detectable periodicity
        # signal either way") when the window's too short to compute --
        # same convention as spacer_penalty's placeholder above, not a
        # judgment that short candidates are NES-like.
        heptad = self._calculate_heptad_periodicity(full_sequence, nes_start, nes_end)
        features.append(float(heptad) if heptad is not None else 0.0)

        # 13. NEW: real CA-coordinate-derived continuous helix run near the
        # candidate (see real_ca_helix_geometry/longest_helix_run in
        # structural_dataset_v2_pipeline.py). Unlike heptad_periodicity
        # above (a sequence proxy) this is measured directly from real
        # AlphaFold 3D coordinates -- how many CONSECUTIVE residues near the
        # candidate sit in genuine alpha-helix geometry. A real leucine
        # zipper/coiled-coil is a long continuous helix (that's structurally
        # what lets it dimerize); a real NES sits in an otherwise
        # disordered/loop region with at most "one turn of helix" contacting
        # CRM1's groove (Fung & Chook 2017). Normalized /30 (a long
        # coiled-coil run comfortably exceeds this; a real NES's local helix
        # turn does not), capped at 1.0. Neutral default 0.0 (no evidence
        # either way) when max_helix_run wasn't supplied (e.g. at inference
        # before this is wired into the live route, or no structure
        # coordinates were available) -- same convention as spacer_penalty/
        # heptad_periodicity above, NOT a claim the candidate is NES-like.
        features.append(min(1.0, float(max_helix_run) / 30.0) if max_helix_run is not None else 0.0)

        return np.array(features, dtype=np.float64)

    def build_training_dataset(self):
        """Assemble the exact same (X, y) training data _train_model() uses --
        same positives/negatives sources, same featurization, same PSSM --
        factored out so anything that needs to re-examine the real training
        data (e.g. a feature-importance diagnostic script) uses the identical
        pipeline instead of a separately-maintained copy that could drift out
        of sync. Also builds/overwrites self.pssm as a side effect, same as
        _train_model() did before this was factored out, since _extract_features
        depends on it.

        Returns a dict with: X, y (np.ndarray), positives, negatives (the raw
        example dicts, same order as they were featurized into X/y), and
        stats (the same counts _train_model() prints/logs).
        """
        # ---- positives: real scraped/parsed data + curated seed ----
        real_positives = load_real_positive_examples(self.data_dir)
        seed_positives = [
            {'seq': n['seq'].upper(), 'protein': n['protein'], 'full_sequence': None,
             'start': None, 'crm1_dependent': None, 'source': 'curated_seed'}
            for n in NESDB_VALIDATED_NES
        ]

        positives, seen_seqs = [], set()
        for p in real_positives + seed_positives:
            if p['seq'] in seen_seqs:
                continue
            seen_seqs.add(p['seq'])
            positives.append(p)

        n_real = sum(1 for p in positives if p['source'] != 'curated_seed')
        n_seed = len(positives) - n_real
        n_crm1_known = sum(1 for p in positives if p.get('crm1_dependent') is not None)
        print(f"Positives: {len(positives)} unique ({n_real} from real databases via "
              f"nes_data_pipeline/, {n_seed} from the curated seed list; "
              f"{n_crm1_known} have known CRM1-dependence)")
        if n_real == 0:
            print("  NOTE: no real scraped data found yet. Run nesbase_parser.py "
                  "(and, for much more data, nesdb_scraper.py) in nes_data_pipeline/ "
                  "-- see its README.md -- then delete models/*_v2.* to retrain.")

        # ---- negatives: protein-matched real negatives + synthetic decoys
        #      + structural hard negatives (coiled-coil / leucine-zipper) ----
        matched_negatives = generate_matched_negatives(positives, neg_per_pos=2)
        decoy_negatives = [{'seq': n['seq'].upper(), 'protein': 'synthetic_decoy'}
                            for n in NEGATIVE_SEQUENCES]

        pos_seqs = {p['seq'] for p in positives}
        seen_neg_seqs = {n['seq'] for n in matched_negatives} | {n['seq'] for n in decoy_negatives}
        hard_negatives_all = load_hard_negative_examples(self.data_dir)
        hard_negatives = []
        for n in hard_negatives_all:
            s = n['seq']
            if s in pos_seqs or s in seen_neg_seqs:
                continue
            seen_neg_seqs.add(s)
            hard_negatives.append(n)
        hard_neg_cap = min(MAX_HARD_NEGATIVES_CEILING,
                            int(MAX_HARD_NEGATIVES_RATIO * max(1, len(positives))))
        if len(hard_negatives) > hard_neg_cap:
            hard_negatives = random.Random(7).sample(hard_negatives, hard_neg_cap)

        negatives = matched_negatives + decoy_negatives + hard_negatives
        print(f"Negatives: {len(negatives)} ({len(matched_negatives)} protein-matched "
              f"real windows, {len(decoy_negatives)} synthetic decoys, "
              f"{len(hard_negatives)} structural hard negatives from nes_negatives/"
              f"{f' (capped from {len(hard_negatives_all)} unique hits)' if len(hard_negatives_all) > hard_neg_cap else ''})")
        if not hard_negatives_all:
            print("  NOTE: no nes_negatives/nes_negatives.csv found -- run "
                  "negative_dataset_builder.py to add coiled-coil/leucine-zipper "
                  "hard negatives, then delete models/*_v2.* to retrain.")

        # ---- PSSM (register-anchored, see module docstring) ----
        self.pssm = self._build_pssm(positives)
        print(f"PSSM: {PSSM_WIDTH}-column window anchored on the consensus "
              f"hydrophobic register, built from {int(self.pssm[2])} sequences "
              f"(window size is fixed -- more sequences only reduce noise in "
              f"each column's amino-acid frequencies)")

        # ---- real structural data (SASA + pLDDT), if generated ----
        structural_data = load_structural_data(self.data_dir)
        n_pos_structural = sum(1 for p in positives if p['seq'].upper() in structural_data)
        n_neg_structural = sum(1 for n in negatives if n['seq'].upper() in structural_data)
        print(f"Real structural data (SASA+pLDDT): {len(structural_data)} sequences loaded "
              f"({n_pos_structural}/{len(positives)} positives, "
              f"{n_neg_structural}/{len(negatives)} negatives have a match)")
        if not structural_data:
            print("  NOTE: no structural_data_v2.json found -- run "
                  "structural_dataset_v2_pipeline.py in nes_data_pipeline/ to give "
                  "plddt_norm/sasa_norm real per-example values instead of the "
                  "constant neutral default, then delete models/*_v2.* to retrain.")

        # ---- real IUPred2A/ANCHOR2 data, if generated (see load_iupred_data) ----
        iupred_data = load_iupred_data(self.data_dir)
        n_pos_iupred = sum(1 for p in positives if p['seq'].upper() in iupred_data)
        n_neg_iupred = sum(1 for n in negatives if n['seq'].upper() in iupred_data)
        print(f"Real IUPred2A/ANCHOR2 data: {len(iupred_data)} sequences loaded "
              f"({n_pos_iupred}/{len(positives)} positives, "
              f"{n_neg_iupred}/{len(negatives)} negatives have a match)")
        if not iupred_data:
            print("  NOTE: no iupred_data_v2.json found -- run "
                  "fetch_iupred_training_data.py in nes_data_pipeline/ (needs real "
                  "internet access to iupred2a.elte.hu) to give iupred_mean/"
                  "anchor2_mean etc. real per-example values instead of the "
                  "constant neutral default, then delete models/*_v2.* to retrain.")

        # ---- featurize ----
        X, y = [], []
        for p in positives:
            struct = structural_data.get(p['seq'].upper())
            plddt_vals = struct['plddt'] if struct and struct['plddt'] else None
            sasa_vals = struct['sasa'] if struct and struct['sasa'] else None
            helix_run = struct['max_helix_run'] if struct else None
            iupred_vals = iupred_data.get(p['seq'].upper())
            X.append(self._extract_features(p['seq'], p.get('full_sequence'), p.get('start') or 0,
                                              plddt_values=plddt_vals, sasa_values=sasa_vals,
                                              iupred_values=iupred_vals, max_helix_run=helix_run))
            y.append(1)
        for n in negatives:
            struct = structural_data.get(n['seq'].upper())
            plddt_vals = struct['plddt'] if struct and struct['plddt'] else None
            sasa_vals = struct['sasa'] if struct and struct['sasa'] else None
            helix_run = struct['max_helix_run'] if struct else None
            iupred_vals = iupred_data.get(n['seq'].upper())
            # full_sequence/start: real for matched_negatives (see
            # generate_matched_negatives) and, where available, hard
            # negatives (see _load_hard_negatives_from_csv's 'context'
            # column) -- None for synthetic decoys, which legitimately have
            # no real flanking biology. Previously this call omitted these
            # entirely for EVERY negative regardless of source, so even
            # negatives carrying real full_sequence/start silently fell back
            # to the flat neutral flanking default -- see
            # generate_matched_negatives' docstring for why that's a problem.
            X.append(self._extract_features(n['seq'], n.get('full_sequence'), n.get('start') or 0,
                                              plddt_values=plddt_vals, sasa_values=sasa_vals,
                                              iupred_values=iupred_vals, max_helix_run=helix_run))
            y.append(0)
        X, y = np.array(X), np.array(y)
        print(f"Feature dimension: {X.shape[1]}")

        return {
            'X': X, 'y': y,
            'positives': positives, 'negatives': negatives,
            'stats': {
                'n_positives': len(positives),
                'n_positives_real': n_real,
                'n_positives_curated_seed': n_seed,
                'n_negatives': len(negatives),
                'n_negatives_matched_real': len(matched_negatives),
                'n_negatives_synthetic': len(decoy_negatives),
                'n_negatives_hard_structural': len(hard_negatives),
                'n_negatives_hard_structural_available': len(hard_negatives_all),
                'n_positives_with_real_structural_data': n_pos_structural,
                'n_negatives_with_real_structural_data': n_neg_structural,
                'n_positives_with_real_iupred_data': n_pos_iupred,
                'n_negatives_with_real_iupred_data': n_neg_iupred,
                'feature_dim': int(X.shape[1]),
            },
        }

    def _train_model(self):
        """Train model on real (scraped/parsed) + curated-seed NES data,
        using NESTED cross-validation (see compare_split_methodology.py,
        which showed the
        single 80/20-split evaluation number was itself noisy run-to-run,
        even though classifier *selection* was already CV-based).

        Nested CV has two distinct jobs, kept deliberately separate:

        1. OUTER loop (honest performance estimate): StratifiedKFold splits
           the full dataset into outer_folds folds. Each fold's TEST
           portion is never seen by anything fit on that fold's TRAIN
           portion -- not by model fitting, not by classifier selection.
           The reported F1/AUC/etc is the mean +/- std across all outer
           folds, not one lucky-or-unlucky number from a single split.

        2. INNER loop (classifier selection, nested inside each outer
           fold's TRAIN portion): ordinary k-fold CV picks the best-F1
           classifier using only that fold's training data -- the same
           selection procedure as before, just re-run fresh inside each
           outer fold so the outer test fold can never leak into a
           selection decision (that leakage is the classic bug nested CV
           exists to prevent).

        After the outer loop's numbers are locked in, a SEPARATE final
        selection pass runs classifier-selection CV one more time on the
        FULL dataset (this can't bias the reported numbers above -- it
        doesn't feed into them) to pick which classifier type actually
        ships, then fits it on 100% of the data. The outer-fold winners
        are also tallied for a selection-stability report, mirroring
        compare_split_methodology.py's classifier_selection_counts.

        Permutation importance and the ROC/PR/confusion-matrix figures use
        the pooled out-of-fold predictions from the outer loop -- genuine
        held-out predictions for every one of the n examples, not just a
        20% slice, so they're a strict upgrade over the old approach too.
        """
        print("\n" + "=" * 70)
        print("Training Improved NES Model (v3 -- nested CV)")
        print("=" * 70)

        dataset = self.build_training_dataset()
        X, y = dataset['X'], dataset['y']
        feature_names = self._feature_names()
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        can_nest = n_pos >= 10 and n_neg >= 10

        def _build_candidates():
            # Expanded from 4 to 8 candidates for a genuinely
            # diverse comparison (margin-based, bagged trees, 3 distinct
            # boosting implementations, and one neural net) -- LSTM/
            # Transformer/LightGBM/CatBoost deliberately left out (see
            # earlier docstring history for the reasoning). Rebuilt fresh
            # (unfitted) every call so nested folds never share fitted
            # state.
            c = {
                'svm_linear': SVC(kernel='linear', C=0.01, probability=True,
                                   random_state=42, class_weight='balanced'),
                'svm_rbf': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                                random_state=42, class_weight='balanced'),
                'random_forest': RandomForestClassifier(n_estimators=300, random_state=42,
                                                          class_weight='balanced'),
                'extra_trees': ExtraTreesClassifier(n_estimators=300, random_state=42,
                                                      class_weight='balanced'),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=42),
                'mlp': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                                       random_state=42, early_stopping=True),
            }
            if XGBOOST_AVAILABLE:
                c['xgboost'] = XGBClassifier(random_state=42, eval_metric='logloss')
            return c

        def _select_best(X_s, y_sub):
            """Inner-CV classifier selection on whatever data it's given.
            Returns (best_name, best_mean_f1, per-classifier CV report)."""
            candidates = _build_candidates()
            folds = max(2, min(5, int(min(np.bincount(y_sub)))))
            report, best_name, best_score = {}, None, -1.0
            for name, mdl in candidates.items():
                try:
                    scores = cross_val_score(mdl, X_s, y_sub, cv=folds, scoring='f1')
                    report[name] = {'mean_f1': float(scores.mean()), 'std_f1': float(scores.std())}
                    if scores.mean() > best_score:
                        best_score, best_name = scores.mean(), name
                except Exception as e:
                    report[name] = {'error': str(e)}
            return best_name or 'svm_linear', best_score, report

        y_pred_for_plots, y_proba_for_plots = None, None
        nested_cv_report, cv_report = None, {}

        if can_nest:
            outer_folds = max(2, min(5, n_pos, n_neg))
            outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

            oof_proba = np.full(len(y), np.nan)
            oof_pred = np.full(len(y), np.nan)
            per_fold_rows = []
            outer_selection_counts = Counter()
            fold_perm_importances = []

            print(f"\nOuter loop: {outer_folds}-fold CV for an unbiased performance estimate "
                  f"(inner k-fold CV re-run fresh inside each outer training fold for "
                  f"classifier selection)...")
            for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                fold_scaler = StandardScaler().fit(X_tr)
                X_tr_s, X_te_s = fold_scaler.transform(X_tr), fold_scaler.transform(X_te)

                inner_name, inner_score, _ = _select_best(X_tr_s, y_tr)
                outer_selection_counts[inner_name] += 1

                fold_model = clone(_build_candidates()[inner_name]).fit(X_tr_s, y_tr)
                proba = fold_model.predict_proba(X_te_s)[:, 1]
                pred = (proba >= 0.5).astype(int)
                oof_proba[te_idx] = proba
                oof_pred[te_idx] = pred

                fold_metrics = {
                    'fold': fold_i, 'chosen_classifier': inner_name,
                    'inner_selection_f1': float(inner_score),
                    'n_test': int(len(y_te)),
                    'accuracy': float(accuracy_score(y_te, pred)),
                    'precision': float(precision_score(y_te, pred, zero_division=0)),
                    'recall': float(recall_score(y_te, pred, zero_division=0)),
                    'f1': float(f1_score(y_te, pred, zero_division=0)),
                    'roc_auc': float(roc_auc_score(y_te, proba)) if len(set(y_te)) == 2 else None,
                }
                per_fold_rows.append(fold_metrics)
                print(f"  fold {fold_i + 1}/{outer_folds}: chose {inner_name} "
                      f"(inner CV F1={inner_score:.3f}) -> outer test F1={fold_metrics['f1']:.3f}  "
                      f"ROC-AUC={fold_metrics['roc_auc']}")

                try:
                    perm_result = permutation_importance(
                        fold_model, X_te_s, y_te, n_repeats=30,
                        random_state=42, scoring='f1', n_jobs=-1)
                    fold_perm_importances.append(perm_result.importances_mean)
                except Exception as e:
                    print(f"    Warning: Could not compute permutation importance for fold {fold_i}: {e}")

            f1s = [r['f1'] for r in per_fold_rows]
            aucs = [r['roc_auc'] for r in per_fold_rows if r['roc_auc'] is not None]
            mode_clf, mode_n = outer_selection_counts.most_common(1)[0]
            nested_cv_report = {
                'outer_folds': outer_folds,
                'per_fold': per_fold_rows,
                'test_f1_mean': float(np.mean(f1s)), 'test_f1_std': float(np.std(f1s)),
                'test_roc_auc_mean': float(np.mean(aucs)) if aucs else None,
                'test_roc_auc_std': float(np.std(aucs)) if aucs else None,
                'classifier_selection_counts': dict(outer_selection_counts),
                'classifier_selection_mode': mode_clf,
                'classifier_selection_mode_frequency': float(mode_n / outer_folds),
            }
            print(f"\nNested CV estimate: F1 = {nested_cv_report['test_f1_mean']:.3f} +/- "
                  f"{nested_cv_report['test_f1_std']:.3f}   ROC-AUC = "
                  f"{nested_cv_report['test_roc_auc_mean']}   "
                  f"(outer-fold winner: {mode_clf}, chosen in {mode_n}/{outer_folds} folds)")

            y_pred_for_plots, y_proba_for_plots = oof_pred, oof_proba

            if fold_perm_importances:
                avg_perm = np.mean(fold_perm_importances, axis=0)
                self.permutation_importance_ = {
                    name: float(v) for name, v in zip(feature_names, avg_perm)
                }
                top_perm = max(self.permutation_importance_, key=self.permutation_importance_.get)
                print(f"Permutation importance (averaged across outer folds): top feature = "
                      f"{top_perm} ({self.permutation_importance_[top_perm]:+.4f} F1 drop when shuffled)")
            else:
                self.permutation_importance_ = None

            # ---- final model: separate selection pass on ALL data, then
            # fit on ALL data for shipping. This is deliberately AFTER the
            # nested CV numbers above are already locked in -- it doesn't
            # feed into any reported performance number, only into which
            # classifier type actually ships.
            print("\nFinal classifier selection (CV on 100% of the data, for shipping only)...")
            self.scaler = StandardScaler().fit(X)
            X_all_s = self.scaler.transform(X)
            self.model_name, best_score, cv_report = _select_best(X_all_s, y)
            self.model = clone(_build_candidates()[self.model_name])
            self.model.fit(X_all_s, y)
            print(f"Chosen model to ship: {self.model_name} (CV F1 on 100% data = {best_score:.3f})")

            metrics = {
                **dataset['stats'],
                'chosen_model': self.model_name,
                'cv_f1_by_model': cv_report,
                'n_train': int(len(y)),
                'training_protocol': 'nested_cv',
                'nested_cv': nested_cv_report,
                'held_out_test': None,  # no single held-out split under nested CV
            }
        else:
            print("  NOTE: dataset still small -- skipping nested CV "
                  "(would be too noisy to trust); falling back to a bare fit.")
            self.scaler = StandardScaler().fit(X)
            X_s = self.scaler.transform(X)
            self.model_name, best_score, cv_report = _select_best(X_s, y)
            self.model = clone(_build_candidates()[self.model_name])
            self.model.fit(X_s, y)
            self.permutation_importance_ = None
            metrics = {
                **dataset['stats'],
                'chosen_model': self.model_name,
                'cv_f1_by_model': cv_report,
                'n_train': int(len(y)),
                'training_protocol': 'bare_fit_small_dataset',
                'nested_cv': None,
                'held_out_test': None,
            }

        self.metrics = metrics

        # Save
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.pssm, self.pssm_path)
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        with open(self.meta_path, 'w') as f:
            json.dump({'model_name': self.model_name, 'feature_names': feature_names}, f, indent=2)
        if self.permutation_importance_ is not None:
            with open(self.permutation_importance_path, 'w') as f:
                json.dump(self.permutation_importance_, f, indent=2)

        print(f"\nModel saved to {self.model_path}")
        print(f"Metrics saved to {self.metrics_path}")
        if self.permutation_importance_ is not None:
            print(f"Permutation importance saved to {self.permutation_importance_path}")
        print("=" * 70 + "\n")

        # ---- figures for thesis/paper (feature distributions, ROC/PR,
        #      confusion matrix, CV model comparison, feature importance).
        #      y_test/y_pred/y_proba below are the pooled nested-CV
        #      out-of-fold predictions (every example, genuinely held out
        #      when predicted) rather than a single 20% split. ----
        try:
            import ml_figures
            valid = ~np.isnan(y_proba_for_plots) if y_proba_for_plots is not None else None
            ml_figures.generate_all_figures(
                model_dir=self.model_dir,
                feature_names=feature_names,
                X=X, y=y,
                cv_report=cv_report,
                importance_dict=self.get_feature_importance(),
                importance_method=('permutation' if self.permutation_importance_ else 'impurity'),
                y_test=y[valid] if valid is not None else None,
                y_pred=y_pred_for_plots[valid].astype(int) if valid is not None else None,
                y_proba=y_proba_for_plots[valid] if valid is not None else None,
            )
        except Exception as e:
            print(f"  Warning: Could not generate figures (training/model unaffected): {e}")

    def predict(self, sequence, full_sequence=None, nes_start=0,
                plddt=None, sasa=None, max_helix_run=None):
        """
        Predict NES probability with enhanced scoring

        Args:
            sequence: NES candidate sequence
            full_sequence: Full protein sequence (for flanking analysis)
            nes_start: Start position of NES in full sequence
            plddt: pLDDT scores for structure confidence
            sasa: SASA scores for accessibility
            max_helix_run: real CA-coordinate-derived continuous helix run
                (residues) near this candidate -- see
                real_ca_helix_geometry/longest_helix_run in
                structural_dataset_v2_pipeline.py. None (the default) means
                "not computed for this call", NOT "no helix" -- callers that
                have already fetched the real structure (e.g. app.py's
                unified_crm1_nes_analysis) should pass the real value;
                everything else falls back to the neutral default inside
                _extract_features.

        Returns:
            prob: Probability score (0-1)
            confidence: Confidence level
            details: Dictionary with detailed scoring
        """
        if self.model is None:
            return 0.5, 'unknown', {}

        # Extract features
        features = self._extract_features(
            sequence, full_sequence, nes_start, plddt, sasa,
            max_helix_run=max_helix_run,
        )

        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.model.predict_proba(features_scaled)[0][1]

        # Calculate detailed scores
        details = {
            'pssm_score': self._calculate_pssm_score(sequence),
            'nes_classes': self._classify_nes_pattern(sequence),
            'spacer_hydrophobicity': self._calculate_spacer_hydrophobicity(sequence),
            'ncpr_local': self._calculate_ncpr(sequence),
            'cider_linear_features': self._calculate_cider_linear_features(sequence, full_sequence, nes_start),
            'model_name': self.model_name,
        }

        # Add flanking analysis if full sequence provided
        if full_sequence and len(full_sequence) > len(sequence):
            nes_end = nes_start + len(sequence)
            flanking = self._analyze_flanking_regions(full_sequence, nes_start, nes_end)
            details['flanking_analysis'] = flanking

            # REMOVED: this used to do
            #   prob = min(1.0, max(0.0, prob * flanking['combined_likelihood']))
            # -- a post-hoc NESmapper-style likelihood-ratio multiply applied
            # directly to an already-bounded [0,1] classifier probability.
            # Two problems, found while investigating why this made scores
            # jump so aggressively: (1) hpr_likelihood/nc_likelihood (what
            # combined_likelihood is built from) are ALREADY trained-in
            # input features of this classifier (see _extract_features /
            # _feature_names, 'flank_hpr_likelihood'/'flank_nc_likelihood')
            # -- re-applying the same evidence again on top of the model's
            # own output double-counts it. (2) multiplying a probability
            # directly by a likelihood ratio isn't the mathematically
            # correct way to apply one anyway (proper Bayesian updating
            # multiplies odds, not probability), which is what made a 2-3.6x
            # factor saturate prob at the 1.0 ceiling for any candidate
            # already above ~0.3-0.4, discarding real classifier confidence
            # information. Verified via test_flanking_multiplier_ablation.py
            # against run_holdout_pipeline_test.py's held-out set before
            # removing this for real: identical confusion matrix (TP=7,
            # FP=0, FN=2, TN=8) and identical rank-based AUC (0.8889) with
            # or without this line -- the multiplier wasn't rescuing any
            # borderline call, only inflating already-correct scores.
            # flanking_analysis (hpr/nc/combined_likelihood) is still
            # computed and still attached to `details` above for display/
            # transparency -- only the post-hoc rescaling of `prob` itself
            # is gone.

        # Determine confidence
        if prob > 0.85 or prob < 0.15:
            confidence = 'very_high'
        elif prob > 0.7 or prob < 0.3:
            confidence = 'high'
        elif prob > 0.55 or prob < 0.45:
            confidence = 'medium'
        else:
            confidence = 'low'

        return prob, confidence, details

    def predict_protein(self, sequence, window_size=15, step=1,
                         plddt_values=None, sasa_values=None):
        """
        Scan protein sequence for NES candidates (LocNES approach)

        Args:
            sequence: full protein sequence to scan (single letter codes)
            window_size: sliding window length in residues (default 15)
            step: step size between window starts (default 1)
            plddt_values: optional per-residue pLDDT array, same length and
                same indexing as `sequence` (index 0 = sequence[0], etc.).
                Sliced per-window and passed into predict() so plddt_norm
                reflects this protein's real structure instead of silently
                falling back to the neutral 0.75 default. None (default)
                preserves the old no-structural-data behaviour.
            sasa_values: optional per-residue RSA (relative solvent
                accessibility, 0-1ish, Tien et al. 2013-normalized -- the
                SAME scale produced by app.py's calculate_sasa() /
                consensus_accessibility.py, NOT raw SASA in A^2) array,
                same length/indexing as `sequence`. Sliced per-window and
                passed into predict() the same way as plddt_values.

        Returns:
            List of candidate dicts, sorted by start position, each with
            'start', 'end', 'sequence', 'probability', 'confidence', 'classes'
            (and 'pssm_score' for convenience) -- overlapping windows are
            resolved by greedy selection, highest-probability candidate
            wins any contested span, same approach nls_ml_predictor.py's
            analogous scan uses.

        NOTE (reconstruction): this method's body was lost to file
        truncation (see nes_ml_predictor_improved_truncated.py for the
        broken original) and has been rewritten from: the intact predict()
        method just above, this docstring/signature (which survived), the
        return-shape contract validate_predictor.py's protein-scan test
        expects (start/end/sequence/probability/classes), a prob > 0.1
        threshold noted in an earlier code-review conversation, and the
        sibling sliding-window scan in nls_ml_predictor.py as a structural
        template. It is not recovered original code -- re-verify against
        validate_predictor.py's expected output before relying on it.

        added plddt_values/sasa_values support (previously this
        method always called predict() with no structural data at all, so
        plddt_norm/sasa_norm silently used the neutral 0.75/0.50 defaults
        for every window even when a caller had real per-residue structure
        available -- this is exactly the gap the live app.py never had,
        since app.py's own scanning loops call predict() directly with real
        local slices; this brings predict_protein() up to the same
        standard so every caller of this predictor, not just app.py,
        consistently uses the 3-method consensus RSA data end to end).
        """
        sequence = sequence.upper()
        n = len(sequence)
        candidates = []

        have_plddt = plddt_values is not None and len(plddt_values) >= n
        have_sasa = sasa_values is not None and len(sasa_values) >= n

        for i in range(0, max(0, n - window_size + 1), step):
            sub = sequence[i:i + window_size]
            if not _relaxed_hydrophobic_prefilter(sub):
                continue
            local_plddt = plddt_values[i:i + window_size] if have_plddt else None
            local_sasa = sasa_values[i:i + window_size] if have_sasa else None
            prob, confidence, details = self.predict(
                sub, full_sequence=sequence, nes_start=i,
                plddt=local_plddt, sasa=local_sasa,
            )
            if prob > 0.1:
                candidates.append({
                    'start': i,
                    'end': i + window_size - 1,
                    'sequence': sub,
                    'probability': prob,
                    'confidence': confidence,
                    'classes': details.get('nes_classes', []),
                    'pssm_score': details.get('pssm_score'),
                })

        # Greedy non-overlap selection: sort by probability descending, then
        # keep a candidate only if none of its positions were already
        # claimed by a higher-probability candidate -- same approach
        # nls_ml_predictor.py's own protein scan uses.
        candidates.sort(key=lambda c: c['probability'], reverse=True)
        selected = []
        used = set()
        for c in candidates:
            span = range(c['start'], c['end'] + 1)
            if any(p in used for p in span):
                continue
            selected.append(c)
            used.update(span)
        selected.sort(key=lambda c: c['start'])
        return selected
