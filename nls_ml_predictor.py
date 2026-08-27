"""
NLS (nuclear localization signal) predictor -- feature-engineered ML
companion to nes_ml_predictor_improved.py, built the same way (PSSM +
hand-engineered biophysical features + classical ML with a genuine
held-out split + multi-method feature importance) but adapted to NLS
biology rather than copy-pasted from the NES pipeline:

Why the feature set differs from the NES model, mechanistically:
  - NES recognition is almost entirely about a hydrophobic residue
    spacing register (Phi-x(2,3)-Phi-x(2,3)-Phi-x-Phi) docking into one
    conserved hydrophobic groove on one receptor (CRM1/XPO1). That's why
    the NES model's PSSM is anchored on hydrophobic (Phi) spacing.
  - Classical NLS recognition is the opposite chemistry: basic (K/R)
    clusters read out by the ARM-repeat grooves of importin-alpha, in
    two flavors (monopartite: a single K(K/R)X(K/R)-type cluster; and
    bipartite: two basic clusters separated by a 9-12 aa spacer, e.g.
    the nucleoplasmin-type NLS). There is no single dominant cargo
    receptor pocket to dock against the way CRM1 anchors NES validation
    (this is explicitly why this project skips fpocket/MD for NLS --
    see NLS_predictor_landscape_and_novelty.md), so this PSSM is
    anchored on the basic-cluster register instead of a hydrophobic one,
    and a bipartite spacer detector is added as its own feature family.
  - The field's best-documented failure mode is different, too: NES
    tools historically over-fire on hydrophobic-spacing pattern matches;
    NLS tools over-fire on *any* K/R-rich stretch (NucPred/NLStradamus/
    seqNLS comparison studies report ~45% accuracy for methods that lean
    on basic-residue density alone). So instead of NES's coiled-coil/
    leucine-zipper structural hard negatives, this model's hard negatives
    are real UniProt-annotated DNA-binding regions -- genuinely basic,
    genuinely functional, genuinely NOT a nuclear import signal -- a
    deliberate stress test of exactly that failure mode, plus synthetic
    shuffled-polybasic decoys.

Shared methodology with the NES pipeline (kept identical on purpose, for
direct comparability in the thesis): disorder-propensity scale (same
literature-derived residue table), real localCIDER linear (sliding-window)
charge/hydropathy/complexity features, SASA/pLDDT structural feature slots
fed by a structural_dataset_pipeline.py companion script (network-gated,
run locally, since it queries alphafold.ebi.ac.uk in bulk -- see that
script's docstring), and the same multi-method feature-importance
validation (impurity + permutation + correlation matrix).

Usage:
    python nls_ml_predictor.py train      # builds nls_data_pipeline/*.csv
                                           # datasets if missing, trains,
                                           # writes models_nls/*
    python nls_ml_predictor.py predict SEQUENCE [FULL_SEQUENCE] [START]

    from nls_ml_predictor import NLSPredictor
    p = NLSPredictor()
    p.predict("PKKKRKV")
"""
import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# XGBoost is an optional extra, same degrade-gracefully pattern as the
# localCIDER import below -- skip the candidate rather than crash if it
# isn't installed (`pip install xgboost`).
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from localcider.sequenceParameters import SequenceParameters
    CIDER_AVAILABLE = True
except ImportError:
    CIDER_AVAILABLE = False

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "nls_data_pipeline"
MODEL_DIR = HERE / "models_nls"

# ---------------------------------------------------------------------------
# Residue scales
# ---------------------------------------------------------------------------

KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2, "X": 0.0,
}

# Same disorder-propensity scale used by nes_ml_predictor_improved.py
# (literature-derived; NLS motifs, like NES motifs, must sit in surface-
# exposed/flexible regions to be read out by a transport receptor, so the
# same table applies to both problems without modification).
DISORDER_PROPENSITY = {
    'P': 1.0, 'R': 0.9, 'E': 0.9, 'K': 0.9, 'S': 0.8,
    'Q': 0.8, 'D': 0.8, 'G': 0.7, 'A': 0.5, 'T': 0.5,
    'N': 0.5, 'H': 0.4, 'M': 0.3, 'C': 0.3, 'L': 0.2,
    'F': 0.2, 'I': 0.2, 'V': 0.2, 'W': 0.1, 'Y': 0.1,
    'X': 0.5,
}

BASIC = set("KR")
ACIDIC = set("DE")
HYDROPHOBIC = set("LIVFM")

# CAAX-box (C-terminal prenylation signal) detection -- added
# after the 25+25 holdout diagnosis showed 4 of 8 remaining false positives
# (RAP1A, KRAS, HRAS, NRAS) are membrane-anchored small GTPases whose
# C-terminal polybasic patch is a CAAX-box membrane anchor, not a nuclear
# import signal: it sits immediately upstream of a real prenylated Cys
# 4 residues from the protein's true C-terminus (C-A-A-X, A = aliphatic,
# X = any -- Zhang & Casey 1996 Annu Rev Biochem consensus), and the whole
# point of that motif is lipid-anchoring the protein to a membrane, which
# is mechanistically incompatible with productive nuclear import regardless
# of how basic the preceding patch reads.
#
# NOT added as a plain ML feature: checked empirically against the full
# training pool (nls_dataset.csv + nls_negatives.csv, 768 examples) and
# ZERO of them have a CAAX-shaped C-terminal tail -- every real CAAX example
# in this project lives only in the holdout set. A feature with zero
# variance across 100% of the training data cannot be learned from (no
# split/coefficient can key off a column that's always 0), so it would be
# dead weight in the feature vector, not a usable signal. This is the same
# situation _nls_exposure_factor() (app.py) already handles the same way --
# a real physical constraint, imposed explicitly as an override rather than
# left to a learned coefficient that has no data to learn it from.
CAAX_ALIPHATIC = set("AVLIMFC")


def has_caax_box(full_sequence):
    """True if full_sequence's last 4 residues are C-A-A-X shaped (Cys, two
    aliphatic residues, then anything) -- the canonical CAAX prenylation
    motif, which by definition sits at the protein's literal C-terminus."""
    if not full_sequence or len(full_sequence) < 4:
        return False
    seq = full_sequence.upper()
    c, a1, a2 = seq[-4], seq[-3], seq[-2]
    return c == "C" and a1 in CAAX_ALIPHATIC and a2 in CAAX_ALIPHATIC


# How close a candidate window's C-terminal end must sit to the protein's
# true C-terminus for a real CAAX box to be treated as ITS membrane anchor
# (rather than an unrelated distant Cys near an unrelated basic patch in
# some other large protein). 25aa: checked against the 4 real holdout
# examples, where the polybasic decoy window ends 0-4 residues from the
# CAAX tail -- 25 is deliberately generous around that observed range
# without being so wide it starts vetoing real NLSs in shorter proteins
# that simply happen to end in a Cys-rich tail for unrelated reasons.
CAAX_PROXIMITY_WINDOW = 25

# Cap rather than hard-zero the probability when the veto fires: this is a
# strong biological prior, not logical certainty (a small fraction of
# CAAX-box proteins are reported to also undergo regulated nuclear shuttling
# before/after prenylation in some contexts), so 0.1 firmly fails the 0.5
# scan_sequence/holdout threshold without claiming absolute certainty the
# way an outright 0.0 override would.
CAAX_PROBABILITY_CAP = 0.1


def caax_membrane_anchor_veto(full_sequence, end):
    """True if this candidate window's C-terminal end is within
    CAAX_PROXIMITY_WINDOW of a real CAAX-box tail on full_sequence -- see
    has_caax_box()/module comment above for the biological rationale.
    end is 0-based, exclusive (matches scan_sequence's candidate 'end'
    convention: sequence[start:end])."""
    if not full_sequence or end is None:
        return False
    if not has_caax_box(full_sequence):
        return False
    return (len(full_sequence) - end) <= CAAX_PROXIMITY_WINDOW


# Chromatin/DNA-condensation "basic background" veto -- added
# after the 25+25 holdout diagnosis showed 4 of the remaining false
# positives (Histone H1.0, H1.2, H1.4, Sperm protamine P1) are proteins
# whose ENTIRE sequence is basic, not just the flagged window: real
# UniProt data (checked directly, not assumed) puts Histone H1.0 at 32%
# K/R across all 194 residues, H1.2 at 29% across 213, H1.4 at 30% across
# 219, and protamine P1 at 47% across all 51 residues. Compacting DNA
# *is* these proteins' whole function, so they read as basic everywhere --
# a K/R-dense stretch there isn't the kind of locally exceptional patch a
# real classical NLS represents relative to its OWN protein's typical
# composition. By contrast, this project's genuine NLS-bearing training
# examples and the other false-positive class (MARCKS, GAP-43, membrane-
# binding effectors -- see the lipidation/subcellular-location veto in
# app.py) sit in proteins with ordinary overall composition (~9-18% K/R)
# and only spike locally.
#
# The discriminator here is therefore RELATIVE, not absolute: this
# protein's own whole-sequence K/R fraction (background_frac) vs. the
# candidate window's K/R fraction (window_frac), expressed as a fold-
# enrichment ratio. Checked against the model's actual matched windows for
# all 4 chromatin false positives plus, as negative controls, the 3
# membrane-binding false positives, a real DNA-binding-but-genuinely-hard
# case (Engrailed homeodomain, a large low-background protein where the
# ratio correctly does NOT fire), and a synthetic classical-NLS positive
# control -- none of the non-chromatin cases false-trip this veto.
#
# The raw classifier-selected window is often very short (the model's own
# matched windows for MARCKS/protamine were only 5-6 residues), which
# makes a local K/R fraction computed on the bare window noisy -- a 6aa
# window can trivially read 100% just from one lucky/unlucky residue. So,
# same as the CIDER/RSA profile convention already used elsewhere in this
# project (see cider_rsa_flank in app.py's /api/nls_scan), the window is
# padded out to a minimum evaluation width using real flanking sequence
# before computing window_frac -- this is what let the veto correctly
# catch H1.4 and protamine P1's actual matched windows, which the bare,
# unpadded window did not (fold-enrichment right at/above threshold on
# the raw 5-10 residue window, clearly below it once padded to real
# neighbouring context).
#
# NOT added as a plain ML feature, same reasoning as the CAAX veto above:
# this is a real, deterministic per-protein statistic, not a pattern to
# approximate from the small number of chromatin hard-negative examples
# in the training pool.
#
# (calibration): the floor below was initially set to 0.15
# (roughly "typical protein background"), but checking it against all 254
# real training positives via build_training_dataset() -- not just the 4
# target false positives -- showed that was wrong: 46/254 (18%) of real
# NLS-bearing positives got falsely capped, because plenty of genuine
# nuclear/DNA-binding proteins (not just histones/protamines) sit in the
# 15-22% whole-protein K/R range -- basic residues are generally enriched
# in real nuclear-protein backgrounds, not just chromatin-compaction ones,
# so 0.15 wasn't actually distinguishing "chromatin protein" from "ordinary
# nuclear protein." Raised to 0.27 -- the highest whole-protein background
# among all 254 real positives is 0.261, while the 4 target chromatin false
# positives sit at 0.291-0.471 and this project's own other H1-paralog/
# protamine-paralog hard-negative training examples sit at 0.284-0.383 --
# so 0.27 sits in the real gap between "real NLS carriers, however basic"
# and "chromatin-compaction proteins," verified with actual numbers, not
# assumed. At this floor: 0/254 real positives falsely capped, 5/510 real
# hard negatives caught (all 5 are other histone-H1/protamine paralogs
# already in the training pool), plus all 4 target holdout false positives.
BASIC_BACKGROUND_FLOOR = 0.27
BASIC_BACKGROUND_FOLD_ENRICHMENT_MIN = 2.5
BASIC_BACKGROUND_MIN_WINDOW = 21
BASIC_BACKGROUND_PROBABILITY_CAP = 0.15  # kept only as the ramp's floor value, see basic_background_factor()

# (v2, continuous ramp): the original veto was a step function --
# fold_enrichment<1.8 -> hard-capped to 0.15, fold_enrichment>=1.8 -> untouched.
# That's exactly the kind of cliff the NES pipeline's coiled_coil_factor
# (app.py calculate_improved_nes_score()) deliberately avoids: a step function
# bunches every near-miss right up against the true positives with zero
# discount, while anything that trips it falls to the same floor regardless
# of how borderline it was. Replaced with a linear multiplicative ramp:
# factor=1.0 (no penalty) at/above BASIC_BACKGROUND_FOLD_ENRICHMENT_MIN,
# factor=BASIC_BACKGROUND_PROBABILITY_CAP (same floor as before) at/below
# BASIC_BACKGROUND_FOLD_ENRICHMENT_FLOOR (window no more basic than the
# protein's own background at all -- the strongest possible "this is just
# ordinary composition" reading), linear in between. Since proba is already
# in [0, 1], multiplying by a 0.15 floor factor preserves the same "never
# exceeds ~0.15" guarantee the old hard cap gave at the most severe end,
# while intermediate fold_enrichment values (e.g. 1.4-1.7, previously
# indistinguishable from 1.0) now get a proportionally smaller discount.
# The BASIC_BACKGROUND_FLOOR>proba background-frac gate is unchanged and
# stays binary -- whether the veto concept applies AT ALL to this protein
# is still a real yes/no (chromatin-like composition or not); only the
# SEVERITY of the resulting penalty is now continuous.
#
# FOLD_ENRICHMENT_MIN raised from 1.8 to 2.5 after a real
# live holdout rerun (real structural data, real Flask route) regressed
# specificity 96%->92% -- Histone H1.0 (P07305) and Histone H1.2 (P16403)
# both came back as new false positives, matched_window 168-180 and
# 204-213, with fold_enrichment 1.49 and 1.64 respectively -- squarely in
# the v2 ramp's weak middle zone (factor 0.67 and 0.83, barely enough to
# survive above 0.5 once combined with their real classifier scores).
# Confirmed via the full 254-positive check this widening costs nothing:
# EVERY real positive is already excluded by the background_frac>FLOOR
# gate before fold_enrichment is even evaluated (0/249 checkable positives
# ever reach this branch, independent of what FOLD_ENRICHMENT_MIN is set
# to), so raising this threshold can only ever strengthen the discount on
# real chromatin-type hard negatives, never touch a real positive's score.
BASIC_BACKGROUND_FOLD_ENRICHMENT_FLOOR = 1.0


def basic_background_enrichment(full_sequence, start, end, min_window=BASIC_BACKGROUND_MIN_WINDOW):
    """Returns (background_frac, window_frac, fold_enrichment) -- this
    protein's whole-sequence K/R fraction vs. the candidate window's (padded
    out to at least min_window residues using real flanking sequence, per
    the module comment above). Returns (None, None, None) if full_sequence/
    start/end aren't available."""
    if not full_sequence or start is None or end is None:
        return None, None, None
    seq = full_sequence.upper()
    if not seq:
        return None, None, None
    background_frac = sum(1 for a in seq if a in BASIC) / len(seq)

    s0, e0 = max(0, start), min(len(seq), end)
    width = e0 - s0
    if width < min_window:
        # (bugfix): a candidate near either sequence terminus has
        # less real flanking sequence available on that side than an even
        # 50/50 split wants -- the original version just silently dropped
        # the unmet half instead of pulling extra from the OTHER side,
        # producing a window shorter than min_window right at the exact
        # cases this padding was meant to stabilize (e.g. Histone H1.2's
        # real matched window, 205-213, sits 0 residues from the protein's
        # actual C-terminus at 213 -- the unpadded version only managed 15
        # of the intended 21 residues, which was enough to let fold_
        # enrichment land at 1.83, just over BASIC_BACKGROUND_FOLD_
        # ENRICHMENT_MIN=1.8 and slip through the veto on a real holdout
        # run). Redistributing the shortfall to whichever side still has
        # room fixes this without changing behaviour anywhere the protein
        # is long enough to support a full-width window on both sides.
        pad = min_window - width
        left_avail, right_avail = s0, len(seq) - e0
        left_want = pad // 2 + (pad % 2)
        right_want = pad // 2
        left_take = min(left_want, left_avail)
        right_take = min(right_want, right_avail)
        shortfall = (left_want - left_take) + (right_want - right_take)
        if shortfall > 0:
            extra_left = min(shortfall, left_avail - left_take)
            left_take += extra_left
            shortfall -= extra_left
            right_take += min(shortfall, right_avail - right_take)
        s0 -= left_take
        e0 += right_take
    window = seq[s0:e0]
    if not window:
        return background_frac, None, None
    window_frac = sum(1 for a in window if a in BASIC) / len(window)
    fold_enrichment = window_frac / max(background_frac, 0.02)
    return background_frac, window_frac, fold_enrichment


def basic_background_veto(full_sequence, start, end):
    """True if this candidate sits in a protein whose WHOLE sequence is
    already basic (background_frac > BASIC_BACKGROUND_FLOOR) AND the
    (flank-padded) candidate window isn't meaningfully more basic than
    that background (fold_enrichment < BASIC_BACKGROUND_FOLD_ENRICHMENT_MIN)
    -- i.e. the window isn't a real localized signal, it's just this
    protein's normal composition. See module comment above for the
    histone/protamine holdout evidence this threshold was checked against.
    Kept as a boolean (any penalty applied at all) for callers/reporting
    that just want a yes/no flag -- see basic_background_factor() for the
    actual continuous discount now applied to nls_probability."""
    background_frac, _, fold_enrichment = basic_background_enrichment(full_sequence, start, end)
    if background_frac is None or fold_enrichment is None:
        return False
    return background_frac > BASIC_BACKGROUND_FLOOR and fold_enrichment < BASIC_BACKGROUND_FOLD_ENRICHMENT_MIN


def basic_background_factor(full_sequence, start, end):
    """Continuous multiplicative penalty in (BASIC_BACKGROUND_PROBABILITY_CAP,
    1.0] -- see the BASIC_BACKGROUND_FOLD_ENRICHMENT_FLOOR module comment
    above for the ramp rationale. Returns 1.0 (no penalty) whenever the
    background-frac gate doesn't apply, matching basic_background_veto()'s
    gate exactly so the two stay consistent."""
    background_frac, _, fold_enrichment = basic_background_enrichment(full_sequence, start, end)
    if background_frac is None or fold_enrichment is None or background_frac <= BASIC_BACKGROUND_FLOOR:
        return 1.0
    if fold_enrichment >= BASIC_BACKGROUND_FOLD_ENRICHMENT_MIN:
        return 1.0
    if fold_enrichment <= BASIC_BACKGROUND_FOLD_ENRICHMENT_FLOOR:
        return BASIC_BACKGROUND_PROBABILITY_CAP
    span = BASIC_BACKGROUND_FOLD_ENRICHMENT_MIN - BASIC_BACKGROUND_FOLD_ENRICHMENT_FLOOR
    ramp = (fold_enrichment - BASIC_BACKGROUND_FOLD_ENRICHMENT_FLOOR) / span
    return BASIC_BACKGROUND_PROBABILITY_CAP + ramp * (1.0 - BASIC_BACKGROUND_PROBABILITY_CAP)

# Curated seed sequences -- small, hand-verified fallback/supplement kept in
# case the real scraped nls_dataset.csv is unavailable. Each entry's residue
# span and citation were cross-checked against the primary literature (see
# NLS_predictor_landscape_and_novelty.md); this list is intentionally short
# -- the real training signal comes from nls_data_pipeline/nls_dataset.csv
# (UniProt-derived, see build_dataset.py).
CURATED_SEED_NLS = [
    {"seq": "PKKKRKV", "protein": "SV40 large T-antigen", "bipartite": 0,
     "note": "Kalderon et al. 1984 -- canonical monopartite NLS"},
    {"seq": "KRPAATKKAGQAKKKK", "protein": "Nucleoplasmin (X. laevis)", "bipartite": 1,
     "note": "Robbins et al. 1991 -- prototype bipartite NLS"},
    {"seq": "PAAKRVKLD", "protein": "c-Myc", "bipartite": 0,
     "note": "Dang & Lee 1988"},
    {"seq": "KRKRRP", "protein": "BRCA1 (503-508)", "bipartite": 0,
     "note": "Chen et al. 1996, JBC -- importin-alpha binding confirmed"},
    {"seq": "PKKNRLRRKS", "protein": "BRCA1 (606-615)", "bipartite": 0,
     "note": "Chen et al. 1996, JBC -- importin-alpha binding confirmed"},
]

# Consensus patterns (Kosugi et al. 2009 JBC framework, simplified):
#   monopartite core: K(K/R)-X(0,2)-(K/R) e.g. PKKKRKV
#   bipartite: (K/R)(K/R)-X(6,16)-(K/R)x3-5 in a 5-residue window (spacer
#   widened from the literature-standard 9-12 -- see
#   detect_bipartite()'s docstring for the local trial that justified it)
MONOPARTITE_RE = re.compile(r"[KR][KR].{0,2}[KR]")

# Finer-grained classical-NLS subtyping, the NLS analog of
# NES_PATTERNS in nes_ml_predictor_improved.py. From Kosugi et al. 2009 JBC
# "Six classes of nuclear localization signals specific to different
# binding grooves of importin alpha": classes 1/2 bind the MAJOR NLS-binding
# groove of importin-alpha (monopartite), classes 3/4 bind the MINOR groove
# (still monopartite, different core chemistry). Class 5 (plant-specific)
# and class 6 (bipartite) are intentionally omitted here -- this project
# has no plant-specific training data, and bipartite is already handled by
# detect_bipartite() above, which correctly models the 2-cluster+spacer
# structure that a single short regex can't capture well.
NLS_KOSUGI_PATTERNS = {
    "class_1_monopartite": r"[KR]{4,}",        # 4+ consecutive basic residues
    "class_2_monopartite": r"K[KR].[KR]",      # K-(K/R)-X-(K/R)
    "class_3_minor_groove": r"KR.[WFY]..AF",   # K-R-X-(W/F/Y)-X-X-A-F
    "class_4_minor_groove": r"[PR]..KR[KR]",   # (P/R)-X-X-K-R-(K/R)
    # Binary presence/absence versions of class_3/4 were
    # already tried as ML features (kosugi_major_groove/kosugi_minor_groove)
    # and REMOVED -- two real nested-CV runs showed ~0/negative
    # permutation importance. Root cause: a bare yes/no flag for "does this
    # window contain a class_3-or-4-shaped substring anywhere" is almost
    # entirely redundant with is_monopartite_pattern/is_bipartite_pattern,
    # which the model already has. See class3_match_score/class4_match_score
    # below for the CONTINUOUS, non-redundant version added this project --
}

# Continuous class_3/class_4 (importin-alpha MINOR binding
# groove) match-strength scores. Motivation, from the cNLS Mapper reference
# (Kosugi et al. 2009 JBC 284:478 -- same paper NLS_KOSUGI_PATTERNS above is
# already built from): the minor groove's two documented consensuses are
# K-R-X-(W/F/Y)-X-X-A-F (class 3) and (P/R)-X-X-K-R-(K/R) (class 4), and --
# per Conti/Kuriyan 1998 (Cell 94:193), the structural paper establishing
# the two-pocket model -- classical BIPARTITE NLSs are themselves a hybrid:
# the N-terminal cluster binds the minor pocket, the C-terminal cluster
# binds the major pocket. Since this project's PSSM is a single register
# anchored on major-groove-shaped (class 1/2) examples (the large majority
# of nls_dataset.csv), it has no real signal for how MINOR-groove-shaped a
# window is -- a real class 3/4 NLS, or a bipartite window whose clusters
# don't fit the major-groove-only anchor well, currently has nothing else
# to lean on.
#
# Why continuous, not the removed binary flag: a bare presence/absence
# check is redundant with is_monopartite_pattern (any window containing a
# class_3/4 substring almost always also matches MONOPARTITE_RE somewhere).
# A continuous BEST PARTIAL MATCH score (how many of the pattern's fixed
# positions are satisfied at the best-aligned position, not just whether
# ALL of them are) carries real information a binary flag throws away --
# e.g. a window with 6/8 class_3 positions satisfied looks meaningfully
# more minor-groove-like than one with 2/8, even though neither passes the
# strict regex.
_CLASS3_TEMPLATE = [
    (0, "KR"), (2, "WFY"), (5, "A"), (6, "F"),
]  # position -> allowed-residue-set, from KR.[WFY]..AF (0-indexed within an 8-residue window)
_CLASS4_TEMPLATE = [
    (0, "PR"), (3, "KR"), (4, "KR"), (5, "KR"),
]  # from [PR]..KR[KR] (0-indexed within a 6-residue window)


def _template_match_score(seq, template, template_len):
    """Best partial-match fraction of `template` (list of (offset, allowed_chars))
    against any alignment of `seq`. Returns 0.0 if seq is shorter than the
    template."""
    seq = seq.upper()
    if len(seq) < template_len:
        return 0.0
    best = 0
    n_positions = len(template)
    for start in range(len(seq) - template_len + 1):
        hits = sum(1 for offset, allowed in template if seq[start + offset] in allowed)
        best = max(best, hits)
    return best / n_positions


def class3_match_score(seq):
    return _template_match_score(seq, _CLASS3_TEMPLATE, 8)


def class4_match_score(seq):
    return _template_match_score(seq, _CLASS4_TEMPLATE, 6)


# PY-NLS (Lee et al. 2006, Cell; "Rules for nuclear localization sequence
# recognition by karyopherin beta 2"): mechanistically DIFFERENT from the
# classical NLSs above -- recognized by Transportin-1/Karyopherin-beta2,
# NOT importin-alpha at all. Consensus: an overall basic, disordered
# region with a central hydrophobic-or-basic motif, followed by a
# C-terminal R/H/K-X(2-5)-PY motif. This project has ZERO real PY-NLS
# positives in nls_dataset.csv -- no Transportin-dependent examples were
# ever collected, since the whole feature set/PSSM/hard-negative design is
# built around importin-alpha recognition (see module docstring). So this
# regex is used ONLY to FLAG a candidate as PY-NLS-shaped for reporting --
# self.model's probability is meaningless for this class (trained
# exclusively on importin-alpha-pathway data) and must not be read as a
# confidence score when this pattern matches. Deliberately loose (no
# requirement on the upstream hydrophobic/basic motif, which is too
# degenerate to encode reliably as a short regex).
PY_NLS_RE = re.compile(r"[RHK].{2,5}PY")


def _classify_nls_pattern(seq, full_sequence=None, start=None, end=None):
    """Kosugi monopartite subclass(es) matched, PY-NLS shape (flagged
    separately -- see PY_NLS_RE docstring above for why its probability
    isn't trustworthy), and the classic bipartite signature. Returns a
    dict rather than a flat list since these are independent axes, not
    mutually exclusive single labels the way NES's classes are.

    added an optional, purely informational "potential
    tripartite" check (see detect_extra_basic_cluster docstring) -- only
    runs when a bipartite signature was found AND real full_sequence/start/
    end context is available (so it can look at genuinely NEARBY residues
    outside the candidate window itself, not just whatever happened to be
    included in this particular scan window's boundaries)."""
    seq = seq.upper()
    kosugi = [name for name, pat in NLS_KOSUGI_PATTERNS.items() if re.search(pat, seq)]
    is_bip, spacer, c1_start, c2_start = detect_bipartite(seq)
    is_py = bool(PY_NLS_RE.search(seq))

    extra_cluster = None
    if is_bip and full_sequence is not None and start is not None:
        extra_cluster = detect_extra_basic_cluster(
            full_sequence.upper(), start + c1_start, start + c2_start)

    result = {
        "kosugi_classes": kosugi if kosugi else ["none"],
        "is_bipartite": is_bip,
        "bipartite_spacer": spacer,
        "py_nls_shaped": is_py,
        "potential_tripartite": extra_cluster is not None,
    }
    if extra_cluster is not None:
        result["tripartite_extra_cluster"] = extra_cluster
        result["tripartite_note"] = (
            f"Potential tripartite NLS -- an extra basic cluster was found "
            f"{extra_cluster['side']} of this bipartite signature at residues "
            f"{extra_cluster['start'] + 1}-{extra_cluster['end'] + 1} "
            f"({extra_cluster['sequence']}). Heuristic flag only -- there is no "
            f"tripartite training data or established literature consensus behind "
            f"this the way there is for bipartite/monopartite, and nls_probability "
            f"above does NOT account for it."
        )
    return result


def detect_bipartite(seq):
    """Scan for the classic bipartite signature: a 2-residue basic cluster,
    a spacer, then a 5-residue window containing >=3 basic residues.
    Returns (found, spacer_length, cluster1_start, cluster2_start).

    spacer range widened from the literature-standard 9-12 aa
    (Kosugi et al. 2009) to 6-16 aa -- trialed locally against all 50 real
    holdout accessions (25 positives + 25 negatives, using the actual
    trained classifier, no network needed since full sequences are already
    in nls_holdout_data/candidates.json) before applying. Result: negatives
    were UNCHANGED (13 total pre-filter candidates before and after --
    zero added false-positive risk), positives gained 3 extra pre-filter
    candidates (69->72). This is a real, if modest, widening of the
    pre-filter's reach with no measured specificity cost -- NOT the same as
    the more aggressive "loosen everything" experiment that was tried
    earlier and made things worse; this only widens the SPACER a modest
    amount, it does not touch the 2-residue-cluster-adjacency requirement
    that keeps the false-positive rate low. Note this alone does not fully
    rescue every holdout miss: P03255 (Adenovirus E1A) needs a true spacer
    of 21 aa to reach its real second cluster (well outside even this
    wider range -- going that wide would flag nearly every protein), and
    even where the pre-filter now passes (e.g. P03087/SV40 VP1) the
    trained classifier's own score barely moves (0.019->0.06, still far
    below the 0.5 survival threshold) -- confirming those specific misses
    need more real bipartite training examples (only 16/294 = 5.4% of the
    training set is bipartite-labeled), not just a wider pre-filter.
    """
    n = len(seq)
    for i in range(n - 1):
        if seq[i] in BASIC and seq[i + 1] in BASIC:
            for spacer in range(6, 17):
                j = i + 2 + spacer
                if j + 5 <= n:
                    window = seq[j:j + 5]
                    if sum(1 for c in window if c in BASIC) >= 3:
                        return True, spacer, i, j
    return False, None, None, None


# How far outside a detected bipartite signature to look for a third
# qualifying basic cluster before calling it "potential tripartite" -- 15 aa
# is deliberately generous (roughly matching the wider end of the bipartite
# spacer range itself) since there's no established literature value for
# this the way there is for the 9-12aa bipartite spacer.
TRIPARTITE_SEARCH_FLANK = 15


def detect_extra_basic_cluster(full_sequence, cluster1_start_abs, cluster2_start_abs,
                                search_flank=TRIPARTITE_SEARCH_FLANK):
    """Given a bipartite hit already found in full_sequence (cluster1's basic
    pair starting at cluster1_start_abs, cluster2's 5-residue window starting
    at cluster2_start_abs -- both absolute indices into full_sequence), look
    for ANOTHER qualifying basic cluster (a K/R pair, or a 5-residue window
    with >=3 basic residues) within `search_flank` residues immediately
    before cluster1 or immediately after cluster2's window.

    This is a heuristic display-only flag, NOT a validated third NLS class --
    there's no tripartite training data anywhere in this project and no
    literature consensus on spacer/cluster geometry the way there is for
    bipartite (Kosugi 2009) or monopartite. It exists purely so a real
    nearby basic patch that a bipartite-only view would silently ignore gets
    surfaced for a human to look at, without pretending the model's score
    means anything about it.

    Returns None if nothing qualifying is found, else a dict with 'side'
    ('N-terminal' or 'C-terminal', relative to the bipartite signature),
    'start'/'end' (0-based absolute indices into full_sequence), and
    'sequence' (the extra cluster's own residues)."""
    n = len(full_sequence)

    def _scan_region(region, offset):
        # Prefer the tightest/strongest signal: a straight K/R-K/R pair first,
        # then fall back to the looser >=3-of-5 basic window (same two
        # detection shapes detect_bipartite/MONOPARTITE_RE already use
        # elsewhere in this module, just applied to the flanking region).
        for i in range(len(region) - 1):
            if region[i] in BASIC and region[i + 1] in BASIC:
                return offset + i, offset + i + 1, region[i:i + 2]
        for i in range(max(0, len(region) - 4)):
            window = region[i:i + 5]
            if len(window) == 5 and sum(1 for c in window if c in BASIC) >= 3:
                return offset + i, offset + i + 4, window
        return None

    before_start = max(0, cluster1_start_abs - search_flank)
    before_region = full_sequence[before_start:cluster1_start_abs]
    hit = _scan_region(before_region, before_start)
    if hit:
        s, e, sub = hit
        return {'side': 'N-terminal', 'start': s, 'end': e, 'sequence': sub}

    after_start = cluster2_start_abs + 5
    after_end = min(n, after_start + search_flank)
    after_region = full_sequence[after_start:after_end]
    hit = _scan_region(after_region, after_start)
    if hit:
        s, e, sub = hit
        return {'side': 'C-terminal', 'start': s, 'end': e, 'sequence': sub}

    return None


# ---------------------------------------------------------------------------
# PSSM: anchored on the basic-cluster register (analogous to how the NES
# model anchors on the hydrophobic Phi register -- see module docstring for
# why the anchor chemistry differs).
# ---------------------------------------------------------------------------

PSSM_LEFT = 8
PSSM_RIGHT = 10
PSSM_WIDTH = PSSM_LEFT + PSSM_RIGHT


class NLSPredictor:
    def __init__(self, model_dir=None, data_dir=None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.model_dir.mkdir(exist_ok=True)

        self.model = None
        self.scaler = None
        self.pssm = None
        self.model_name = None
        self.permutation_importance_ = None
        self.impurity_importance_ = None

        self.model_path = self.model_dir / "nls_classifier.pkl"
        self.scaler_path = self.model_dir / "nls_scaler.pkl"
        self.pssm_path = self.model_dir / "nls_pssm.pkl"
        self.metrics_path = self.model_dir / "nls_metrics.json"
        self.importance_path = self.model_dir / "nls_feature_importance.json"

        if self.model_path.exists() and self.scaler_path.exists() and self.pssm_path.exists():
            self._load_model()

    def _load_model(self):
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.pssm = joblib.load(self.pssm_path)
        # Metrics/importance are informational only -- a corrupt or partial
        # JSON file here must never block loading the actual model artifacts
        # above (those are what predict() needs).
        if self.metrics_path.exists():
            try:
                self.model_name = json.load(open(self.metrics_path)).get("chosen_model")
            except (json.JSONDecodeError, OSError) as e:
                print(f"  WARNING: could not read {self.metrics_path.name} ({e}); "
                      f"model still loaded, just missing the model_name label")
        if self.importance_path.exists():
            try:
                self.permutation_importance_ = json.load(open(self.importance_path))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  WARNING: could not read {self.importance_path.name} ({e}); "
                      f"model still loaded, just missing feature-importance reporting")

    # -- PSSM ---------------------------------------------------------------

    def _find_pssm_anchor(self, seq):
        """Index of the anchor column: start of the first classic
        monopartite-shaped basic cluster if present, else the start of the
        densest 4-residue basic window, else the sequence midpoint."""
        m = MONOPARTITE_RE.search(seq)
        if m:
            return m.start()
        best_i, best_score = 0, -1
        for i in range(max(1, len(seq) - 3)):
            window = seq[i:i + 4]
            score = sum(1 for c in window if c in BASIC)
            if score > best_score:
                best_score, best_i = score, i
        return best_i

    def _build_pssm(self, sequences):
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {a: i for i, a in enumerate(aa_alphabet)}
        counts = np.ones((PSSM_WIDTH, 20))  # +1 pseudocount
        n_used = 0
        for seq in sequences:
            seq = seq.upper()
            if len(seq) < 4:
                continue
            anchor = self._find_pssm_anchor(seq)
            for c in range(PSSM_WIDTH):
                idx = anchor - (PSSM_LEFT - 1) + c
                if 0 <= idx < len(seq) and seq[idx] in aa_to_idx:
                    counts[c, aa_to_idx[seq[idx]]] += 1
            n_used += 1
        freqs = counts / counts.sum(axis=1, keepdims=True)
        background = np.ones(20) / 20.0
        pssm = np.log2(freqs / background)
        return pssm, aa_to_idx, n_used

    # A best-anchor scan (try every position in the window,
    # score against the PSSM, take the max) was tried here and REVERTED.
    # Original motivation: diagnosed against the 25+25 holdout set that 7
    # real candidate windows overlapping a true NLS scored 0.01-0.29
    # instead of >0.5, and feature-attribution analysis (weighting high-
    # scorer/low-scorer differences by the shipped linear SVM's own
    # coefficients) pointed at pssm_score as the dominant driver -- the
    # hypothesis was that _find_pssm_anchor()'s single heuristic (first
    # MONOPARTITE_RE match, else the densest 4-residue basic window) was
    # locking onto the wrong reference column for non-canonically-shaped
    # NLS cores (basic doublet not at the window start, etc).
    #
    # Checked before shipping (as this file's whole history insists on):
    # computed the actual gap between the heuristic anchor's score and the
    # true best-scoring anchor for all 7 target cases. 6 of 7 had gap =
    # 0.000 -- the heuristic was ALREADY finding the best anchor. The
    # original diagnosis was wrong: these windows score low because their
    # composition genuinely doesn't match the training PSSM's mostly-
    # canonical-monopartite columns, not because of a fixable anchor-pick
    # bug. The best-anchor scan, once retrained, still moved the holdout
    # numbers (sensitivity 60%->52%, specificity 68%->76% on a full
    # pipeline run) -- but through a diffuse, unexplained mechanism: taking
    # a max over ~20+ candidate anchor positions inflates pssm_score
    # slightly almost everywhere (median gap 0.52, but nonzero for 47% of
    # the full 765-example training pool), which reshuffled the retrained
    # decision boundary in a way that happened to net positive on THIS
    # holdout without a principled reason to expect it would generalize.
    # Reverted rather than shipped on an unexplained result -- see
    # nls_pssm_anchor_and_caax_veto_2026-08-03.md for the full writeup.
    def _calculate_pssm_score(self, sequence):
        if self.pssm is None:
            return 0.0
        pssm, aa_to_idx = self.pssm[0], self.pssm[1]
        seq = sequence.upper()
        anchor = self._find_pssm_anchor(seq)
        score = 0.0
        for c in range(PSSM_WIDTH):
            idx = anchor - (PSSM_LEFT - 1) + c
            if 0 <= idx < len(seq) and seq[idx] in aa_to_idx:
                score += pssm[c, aa_to_idx[seq[idx]]]
        return score

    # -- flanking regions / disorder -----------------------------------------

    # Below this many actual residues, disorder/charge flank features are
    # computed from too small a sample to mean anything -- this only
    # happens when the candidate sits near a protein's TRUE N- or
    # C-terminus (the flank window runs off the end of full_sequence).
    # Real NLS motifs occur at true termini biologically (e.g. SV40 large
    # T-antigen's own NLS sits only a few residues from its C-terminus),
    # so a starved flank must not be treated as unfavorable evidence --
    # mirrors the matching fix in
    # nes_ml_predictor_improved.py's _analyze_flanking_regions. 6 = 60%
    # of the intended 10-residue flank_len default.
    MIN_FLANK_LEN = 6

    # Raw sequence length was being fed to the classifier
    # unbounded. Checked against the real training data: positive NLS
    # examples in nls_dataset.csv have median length 8aa (mean 10.7,
    # n=265); negative examples average 21.7aa -- almost double. Raw
    # 'length' showed real permutation importance (0.081, one of the
    # larger feature contributions), meaning the model had learned "longer
    # -> more likely negative" partly from this dataset skew (your
    # DNA-binding hard-negatives are annotated as longer regions than your
    # tightly-cropped NLS positives) rather than from real biology --
    # exactly the "penalizes longer sequences" effect flagged. Real
    # classical bipartite NLSs run up to ~20aa and PY-NLS/non-classical
    # signals can run to 30aa+, so letting raw length grow unbounded lets
    # the model keep linearly extrapolating a penalty past anything it was
    # trained on. Capping at 25 (a bit past the classical bipartite upper
    # bound) preserves the length signal within the range training data
    # actually covers, without punishing longer real signals just for
    # being longer than the shortest, most over-represented examples.
    LENGTH_CAP = 25

    def _flank_features(self, full_sequence, start, end, flank_len=10):
        """start/end are 0-based half-open into full_sequence, or None if
        unavailable (e.g. synthetic decoys). Returns disorder mean and net
        charge for the N- and C-terminal flanks. Flanks shorter than
        MIN_FLANK_LEN (candidate too close to a true terminus) fall back
        to neutral values instead of being computed from a too-small
        sample (fix, mirrors the NES side)."""
        if not full_sequence or start is None or end is None:
            return 0.5, 0.5, 0.0, 0.0
        n_flank = full_sequence[max(0, start - flank_len):start]
        c_flank = full_sequence[end:end + flank_len]
        n_disorder = (float(np.mean([DISORDER_PROPENSITY.get(a, 0.5) for a in n_flank]))
                      if len(n_flank) >= self.MIN_FLANK_LEN else 0.5)
        c_disorder = (float(np.mean([DISORDER_PROPENSITY.get(a, 0.5) for a in c_flank]))
                      if len(c_flank) >= self.MIN_FLANK_LEN else 0.5)
        n_charge = (float(sum(1 for a in n_flank if a in BASIC) - sum(1 for a in n_flank if a in ACIDIC))
                    if len(n_flank) >= self.MIN_FLANK_LEN else 0.0)
        c_charge = (float(sum(1 for a in c_flank if a in BASIC) - sum(1 for a in c_flank if a in ACIDIC))
                    if len(c_flank) >= self.MIN_FLANK_LEN else 0.0)
        return n_disorder, c_disorder, n_charge, c_charge

    # Wider than the 10aa flank above -- see
    # _wide_flank_basic_fraction docstring for why.
    WIDE_FLANK_LEN = 20

    def _wide_flank_basic_fraction(self, full_sequence, start, end, flank_len=WIDE_FLANK_LEN):
        """Fraction of basic residues (K/R) in a WIDER flank than
        _flank_features' 10aa default, added after the frac_dominant_residue
        window-local composition feature (added same session) turned out
        NOT to separate protamine/MARCKS from genuine monopartite NLSs --
        checked empirically: SV40 T-antigen's own NLS window (KKKRKV) is
        67% dominant-residue, actually HIGHER than MARCKS's 6aa window in
        one comparison, because a canonical class-1 monopartite NLS is
        *itself* a short near-homopolymeric basic burst by definition
        (Kosugi class_1: 4+ consecutive basic residues) -- window-local
        composition can't distinguish "real short NLS burst" from
        "protamine-like burst" when both ARE short homopolymeric bursts.

        What DOES separate them (checked against the same holdout
        candidates): whether the extreme basicity is LOCALIZED (real NLS:
        the 20aa flanking the core reads like a typical protein, ~0-15%
        basic) or EXTENDS well beyond the core (protamine: both 20aa
        flanks are ~40% basic, because the whole 51aa protein is basic,
        not just an NLS-shaped patch within it). Falls back to 0.0 (i.e.
        "looks non-extreme", not a penalty) when the flank is shorter than
        MIN_FLANK_LEN, same favor-the-terminus convention as
        _flank_features above."""
        if not full_sequence or start is None or end is None:
            return 0.0, 0.0
        n_flank = full_sequence[max(0, start - flank_len):start]
        c_flank = full_sequence[end:end + flank_len]
        n_frac = (float(sum(1 for a in n_flank if a in BASIC)) / len(n_flank)
                  if len(n_flank) >= self.MIN_FLANK_LEN else 0.0)
        c_frac = (float(sum(1 for a in c_flank if a in BASIC)) / len(c_flank)
                  if len(c_flank) >= self.MIN_FLANK_LEN else 0.0)
        return n_frac, c_frac

    # Longest run of consecutive K/R this window ever gets to, capped and
    # normalized -- see _feature_names() docstring for why this is worth
    # having alongside the existing binary is_monopartite_pattern flag.
    # Cap of 10: comfortably above Kosugi class 1's own "4+ consecutive"
    # threshold and above the longest real monopartite runs seen in
    # nls_dataset.csv, without letting a pathological all-basic decoy
    # (e.g. a protamine window) dominate the scale for everyone else.
    LONGEST_BASIC_RUN_CAP = 10

    def _longest_basic_run_norm(self, seq):
        best = cur = 0
        for c in seq:
            if c in BASIC:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return min(best, self.LONGEST_BASIC_RUN_CAP) / self.LONGEST_BASIC_RUN_CAP

    # (4th pass): whole-protein length as a proxy for whether
    # active NLS-mediated import is even needed -- see _feature_names()
    # docstring for the empirical justification (8/10 holdout false
    # positives are small proteins). Cap of 800: comfortably spans the
    # holdout's observed range (51-968 aa) while keeping the ~350aa
    # (~40kDa) passive-diffusion cutoff roughly mid-scale rather than at
    # an extreme, so the feature has real dynamic range on both sides of
    # that biological threshold instead of saturating most real (large)
    # NLS-bearing proteins at 1.0.
    WHOLE_PROTEIN_LENGTH_CAP = 800

    def _whole_protein_length_norm(self, full_sequence):
        if not full_sequence:
            return 0.5  # unknown -- neutral midpoint, not a penalty
        return min(len(full_sequence), self.WHOLE_PROTEIN_LENGTH_CAP) / self.WHOLE_PROTEIN_LENGTH_CAP

    # Basic-concentration ratio -- a deliberately more
    # conservative redo of the n_flank_basic_frac_wide20 feature tried and
    # REVERTED (see _feature_names() note below). That version
    # measured a FIXED 20aa flank's raw basic fraction, which raised
    # training CV F1 but dropped holdout sensitivity 56%->40% and
    # specificity 68%->64% -- it overfit to nls_dataset.csv's own
    # near-terminus composition, which isn't representative of this
    # project's viral/other holdout proteins.
    #
    # This version measures something more robust: not a raw fraction in a
    # fixed-size window (sensitive to exactly where that window's edges
    # land), but the window's own frac_basic RELATIVE TO the whole
    # protein's own baseline frac_basic. A real classical NLS is a local
    # SPIKE against an otherwise ordinary-composition protein (SV40
    # T-antigen: whole-protein ~11% basic, its NLS window ~75% basic --
    # ratio ~7x). Histones/protamine/MARCKS-family false positives are
    # basic almost everywhere (protamine: whole-protein ~65% basic), so
    # ANY window drawn from them looks basic in isolation but is only
    # mildly enriched, if at all, relative to that protein's own sky-high
    # baseline. Ratio-to-self-baseline should be far more stable across
    # different protein contexts than a fixed-flank raw fraction, since it
    # auto-normalizes against each protein's own composition instead of
    # assuming all proteins share a similar background rate.
    BASIC_CONCENTRATION_EPS = 0.05  # avoids inflating the ratio for the rare
    # near-zero-basic protein (denominator floor, not a tuned constant)
    BASIC_CONCENTRATION_CAP = 8.0   # SV40 T-antigen's own ~7x ratio (above)
    # sets the rough top of the real-NLS range seen in this project's data;
    # capped rather than left unbounded so one extreme outlier can't
    # dominate the feature's scale for everyone else -- same convention as
    # every other _norm feature in this file.

    def _basic_concentration_ratio(self, seq, full_sequence):
        if not full_sequence or len(full_sequence) == 0:
            return 0.5  # unknown whole-protein context -- neutral midpoint
        window_frac = sum(1 for c in seq if c in BASIC) / max(1, len(seq))
        protein_frac = sum(1 for c in full_sequence if c in BASIC) / len(full_sequence)
        ratio = window_frac / (protein_frac + self.BASIC_CONCENTRATION_EPS)
        return min(ratio, self.BASIC_CONCENTRATION_CAP) / self.BASIC_CONCENTRATION_CAP

    # (4th pass): count of separate K/R islands in the window --
    # see _feature_names() docstring for the empirical justification
    # (true positives average 2.4 runs, false positives 7.8). Cap of 12:
    # comfortably above protamine's 16-run outlier would exceed it, which
    # is intentional -- that pathological case should max out the scale,
    # not stretch it thin for everyone else.
    N_BASIC_RUNS_CAP = 12

    def _n_basic_runs_norm(self, seq):
        n_runs, in_run = 0, False
        for c in seq:
            if c in BASIC:
                if not in_run:
                    n_runs += 1
                    in_run = True
            else:
                in_run = False
        return min(n_runs, self.N_BASIC_RUNS_CAP) / self.N_BASIC_RUNS_CAP

    # -- CIDER ----------------------------------------------------------------

    def _cider_features(self, sequence):
        defaults = {"ncpr_range": 0.0, "hydropathy_range": 0.0, "complexity_mean": 0.5}
        if not CIDER_AVAILABLE or len(sequence) < 6:
            return defaults
        try:
            sp = SequenceParameters(sequence)
            ncpr = sp.get_linear_NCPR()[1]
            hydro = sp.get_linear_hydropathy()[1]
            complexity = sp.get_linear_complexity()[1]
            return {
                "ncpr_range": float(max(ncpr) - min(ncpr)) if len(ncpr) else 0.0,
                "hydropathy_range": float(max(hydro) - min(hydro)) if len(hydro) else 0.0,
                "complexity_mean": float(np.mean(complexity)) if len(complexity) else 0.5,
            }
        except Exception:
            return defaults

    # -- feature vector ---------------------------------------------------

    def _feature_names(self):
        return [
            "pssm_score", "length_capped25", "net_charge", "frac_basic", "frac_acidic",
            "mean_kd_hydropathy", "frac_hydrophobic", "nls_disorder_mean",
            "n_flank_disorder", "c_flank_disorder", "n_flank_net_charge",
            "c_flank_net_charge", "is_bipartite_pattern", "bipartite_spacer_norm",
            "is_monopartite_pattern", "cider_ncpr_range", "cider_hydropathy_range",
            "cider_complexity_mean", "plddt_norm", "sasa_norm",
            # NEW (candidate features, additive alongside nls_disorder_mean/
            # n_flank_disorder/c_flank_disorder above -- see
            # _load_iupred_data() docstring). Appended at the end so this is
            # a pure addition to the existing feature layout.
            "iupred_mean", "n_flank_iupred", "c_flank_iupred",
            "anchor2_mean", "n_flank_anchor2", "c_flank_anchor2",
            # Longest_basic_run_norm survived two full real
            # nested-CV runs with consistent small-but-real positive
            # permutation importance (+0.014, +0.021 -- see git-history-style
            # notes below for the rest of that day's session, since removed
            # from here). is_monopartite_pattern is a binary "4+ consecutive
            # basic residues" flag; this is the continuous version (a run of
            # 4 vs 7 scores identically under the flag, differently here).
            "longest_basic_run_norm",
            # kosugi_major_groove/kosugi_minor_groove/frac_dominant_residue/
            # plddt_std_norm/sasa_std_norm were tried and REMOVED after two
            # independent real nested-CV runs each showed them
            # at ~0 or negative permutation importance -- not hurting badly,
            # but not earning their keep either, and dead features are still
            # a real cost (noise the model has to fit around, one more thing
            # to explain in a thesis feature table). Full reasoning for each
            # is preserved in nls_ml_predictor.py's git history / this
            # session's conversation log, not repeated here.
            #
            # (4th pass): two new features, chosen from a direct
            # empirical failure-mode analysis of the 25+25 holdout set rather
            # than a priori guessing (see conversation) --
            #   whole_protein_length_norm: 8 of the 10 holdout false
            #   positives (protamine, MARCKS, RAP1A, GAP-43, the 3 histones,
            #   KRAS, LL-37) are SMALL proteins (51-332 aa), under the
            #   well-established ~40-60kDa cutoff where a protein can
            #   passively diffuse through the nuclear pore without needing
            #   receptor-mediated import at all -- i.e. several of the
            #   hardest false positives may not need an NLS to reach the
            #   nucleus in the first place, regardless of how basic their
            #   sequence looks. Imperfect (Engrailed homeodomain at 552aa is
            #   a counterexample; some small real NLS-bearing proteins exist
            #   too), but a real, cheap, biologically-grounded global-context
            #   signal, not a re-derivation of window-local composition.
            #   n_basic_runs_norm: count of separate K/R islands in the
            #   window (not just the longest one, which longest_basic_run_norm
            #   already covers). Empirically verified against this exact
            #   holdout set before adding: true positives average 2.4
            #   separate runs, false positives average 7.8 (protamine=16,
            #   the 3 histones=8-9) -- protamine is ~65% Arg overall but
            #   those residues are fragmented into 16 short islands rather
            #   than one contiguous NLS-like patch, information
            #   longest_basic_run_norm alone can't see.
            "whole_protein_length_norm", "n_basic_runs_norm",
        ]
        # Class3_match_score/class4_match_score (continuous
        # importin-alpha minor-groove match strength) and
        # _basic_concentration_ratio (window frac_basic relative to the
        # whole protein's own baseline, meant to separate a real NLS's
        # local spike from histone/protamine's uniform high baseline) were
        # BOTH tried and REJECTED before any retraining, on direct
        # pre-shipping validation against their own target cases:
        #   - basic_concentration_ratio: MARCKS (the false positive it was
        #     most meant to catch) scored 0.891 -- HIGHER than every real
        #     true positive checked (0.20-0.57). MARCKS/GAP-43's basic
        #     clusters are locally concentrated in an otherwise normal-
        #     composition protein, structurally similar to a real NLS in
        #     this respect -- unlike histones/protamine (uniformly basic
        #     throughout), which this feature WOULD separate. Composition
        #     alone can't tell a real NLS apart from MARCKS/GAP-43's
        #     effector domains; this looks like a genuine, close-to-
        #     irreducible confound for a sequence-only classifier, not a
        #     fixable feature-engineering gap.
        #   - class3/4_match_score: Nucleoplasmin (already correctly
        #     scored ~0.88-0.99) shows the same class3 score as an actual
        #     miss (SARS-CoV nucleoprotein), and SV40's class4 score
        #     exceeds every one of the 7 target misses' scores. Mostly
        #     redundant with basic-density information the model already
        #     has via pssm_score/frac_basic, not a clean new signal.
        # Both left defined (see above) but disconnected from the feature
        # vector -- documented rather than silently dropped, same
        # convention as n_flank_basic_frac_wide20 earlier in this file.
        # (second pass, same investigation): a
        # n_flank_basic_frac_wide20/c_flank_basic_frac_wide20 pair (fraction
        # of K/R in a 20aa flank, vs. the 10aa flank used elsewhere) was
        # tried and REVERTED. Rationale looked sound and cleanly separated
        # protamine (both 20aa flanks ~40% basic -- whole 51aa protein is
        # basic, not just an NLS-shaped patch) from SV40 T-antigen (flanks
        # 0%/5% basic) in isolated inspection, and it raised training CV F1
        # noticeably (0.790 -> 0.822, tighter std). But on the SAME 25+25
        # held-out set used throughout this project, it dropped sensitivity
        # 56%->40% and specificity 68%->64% -- it overfit to
        # nls_dataset.csv's own flank-context distribution (small, and
        # possibly systematically different near-terminus composition than
        # this holdout's viral/other proteins) rather than learning
        # something that generalizes. Recorded here rather than silently
        # dropped so the negative result isn't lost.

    def _extract_features(self, sequence, full_sequence=None, start=None, end=None,
                           plddt_values=None, sasa_values=None, iupred_values=None):
        seq = sequence.upper()
        n = max(1, len(seq))
        feats = []

        feats.append(float(self._calculate_pssm_score(seq)))
        feats.append(float(min(len(seq), self.LENGTH_CAP)))  # see LENGTH_CAP docstring

        n_basic = sum(1 for a in seq if a in BASIC)
        n_acidic = sum(1 for a in seq if a in ACIDIC)
        feats.append(float(n_basic - n_acidic))          # net_charge
        feats.append(n_basic / n)                          # frac_basic
        feats.append(n_acidic / n)                          # frac_acidic
        feats.append(float(np.mean([KD_SCALE.get(a, 0.0) for a in seq])))   # mean_kd_hydropathy
        feats.append(sum(1 for a in seq if a in HYDROPHOBIC) / n)           # frac_hydrophobic

        feats.append(float(np.mean([DISORDER_PROPENSITY.get(a, 0.5) for a in seq])))  # nls_disorder_mean

        n_disorder, c_disorder, n_charge, c_charge = self._flank_features(full_sequence, start, end)
        feats.append(n_disorder)
        feats.append(c_disorder)
        feats.append(n_charge)
        feats.append(c_charge)

        is_bip, spacer, _, _ = detect_bipartite(seq)
        feats.append(1.0 if is_bip else 0.0)
        # Deliberately left at /12.0 (the old max spacer) even
        # though detect_bipartite()'s own range widened to 6-16 -- this
        # trained model's coefficients were fit against spacer values that
        # never exceeded 12, so renormalizing to /16.0 here without
        # retraining would silently feed it feature values (>1.0, for the
        # new spacer=13-16 cases) from outside its learned distribution.
        # Revisit this divisor together with a retrain once real bipartite
        # training data covering the wider spacer range exists (see
        # NLS_predictor_landscape_and_novelty.md / task #50).
        feats.append((spacer / 12.0) if is_bip else 0.0)
        feats.append(1.0 if MONOPARTITE_RE.search(seq) else 0.0)

        cider = self._cider_features(seq)
        feats.append(cider["ncpr_range"])
        feats.append(cider["hydropathy_range"])
        feats.append(cider["complexity_mean"])

        # was `if plddt_values` (a truthy check), which raises
        # ValueError for a numpy array with >1 element ("truth value of an
        # array... is ambiguous") -- only ever worked before because every
        # caller happened to pass a plain list. sasa_values right below
        # already used the correct is-not-None/len() check; now both match.
        feats.append(float(np.mean(plddt_values)) / 100.0
                      if plddt_values is not None and len(plddt_values) > 0 else 0.75)
        # sasa_values is expected to already be RELATIVE solvent accessibility
        # (RSA, 0-1ish, Tien et al. 2013 residue-normalized) -- see
        # calculate_sasa() in app.py and nls_data_pipeline/structural_dataset_pipeline.py.
        # NOT raw SASA in Ų, so no further division here. NLS motifs are
        # Lys/Arg-rich, among the largest max-ASA residues, so this matters a
        # lot: a flat /100.0 on raw Ų systematically overstated their exposure.
        feats.append(min(1.0, float(np.mean(sasa_values))) if sasa_values is not None and len(sasa_values) > 0 else 0.50)

        # IUPred2A / ANCHOR2 (candidate features -- see
        # _load_iupred_data() docstring). Same neutral-default convention as
        # nes_ml_predictor_improved.py's _extract_features: 0.5 for the
        # disorder-scale means (matches nls_disorder_mean's DISORDER_PROPENSITY
        # 'unknown' default just above), 0.0 for the ANCHOR2 means (absence
        # of evidence isn't evidence of a binding region).
        iv = iupred_values or {}
        feats.append(float(iv.get('iupred_mean')) if iv.get('iupred_mean') is not None else 0.5)
        feats.append(float(iv.get('n_flank_iupred')) if iv.get('n_flank_iupred') is not None else 0.5)
        feats.append(float(iv.get('c_flank_iupred')) if iv.get('c_flank_iupred') is not None else 0.5)
        feats.append(float(iv.get('anchor2_mean')) if iv.get('anchor2_mean') is not None else 0.0)
        feats.append(float(iv.get('n_flank_anchor2')) if iv.get('n_flank_anchor2') is not None else 0.0)
        feats.append(float(iv.get('c_flank_anchor2')) if iv.get('c_flank_anchor2') is not None else 0.0)

        # kosugi_major_groove/kosugi_minor_groove/frac_dominant_residue/
        # plddt_std_norm/sasa_std_norm computation is deliberately omitted -- see
        # _feature_names() note, dropped after two real
        # nested-CV runs showed ~0/negative permutation importance).
        # n_flank_basic_frac_wide20/c_flank_basic_frac_wide20 was never wired
        # in at all (hurt held-out generalization in testing, never shipped).
        # _wide_flank_basic_fraction() is left defined above in case it's
        # worth revisiting with more training data later.

        feats.append(self._longest_basic_run_norm(seq))
        feats.append(self._whole_protein_length_norm(full_sequence))
        feats.append(self._n_basic_runs_norm(seq))

        # class3_match_score/class4_match_score/_basic_concentration_ratio
        # NOT wired in here -- see their docstrings above for why (both
        # failed pre-shipping validation against their own target cases,
        #, before any retraining was attempted).

        return feats

    # -- dataset assembly ---------------------------------------------------

    def _ensure_datasets(self):
        dataset_csv = self.data_dir / "nls_dataset.csv"
        negatives_csv = self.data_dir / "nls_negatives.csv"
        if not dataset_csv.exists() or not negatives_csv.exists():
            print("nls_dataset.csv/nls_negatives.csv not found -- running build_dataset.py ...")
            subprocess.run([sys.executable, str(self.data_dir / "build_dataset.py")], check=True)
        return dataset_csv, negatives_csv

    def _load_structural_data(self):
        path = self.data_dir / "structural_data.json"
        if not path.exists():
            return {}
        records = json.load(open(path, encoding="utf-8"))
        return {r["seq"].upper(): r for r in records}

    def _load_iupred_data(self):
        """Load real IUPred2A disorder + ANCHOR2 binding-region scores, keyed
        by exact NLS/negative-window sequence -- same pattern as
        _load_structural_data() above and nes_ml_predictor_improved.py's
        load_iupred_data(). Produced by fetch_iupred_training_data.py (run
        once, offline, against structural_data.json's accession list) -- not
        run automatically here since it needs real internet access this
        training process itself doesn't assume. Returns {} if not generated
        yet, so callers fall back to the neutral defaults in _extract_features."""
        path = self.data_dir / "iupred_data.json"
        if not path.exists():
            return {}
        try:
            records = json.load(open(path, encoding="utf-8"))
            return {r["seq"].upper(): r for r in records if r.get("seq")}
        except Exception as e:
            print(f"  (could not read {path}: {e})")
            return {}

    def build_training_dataset(self):
        import csv
        dataset_csv, negatives_csv = self._ensure_datasets()

        positives = []
        with open(dataset_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                positives.append({
                    "seq": row["nls_sequence"].upper(), "full_sequence": row.get("full_sequence") or None,
                    "start": int(row["start"]) - 1 if row.get("start") else None,
                    "end": int(row["end"]) if row.get("end") else None,
                    "protein": row["accession"], "bipartite": row.get("bipartite"),
                    "confidence": row.get("confidence"),
                })
        seen = set()
        dedup_positives = []
        for p in positives:
            if p["seq"] in seen:
                continue
            seen.add(p["seq"])
            dedup_positives.append(p)
        for s in CURATED_SEED_NLS:
            if s["seq"] not in seen:
                seen.add(s["seq"])
                dedup_positives.append({"seq": s["seq"], "full_sequence": None, "start": None,
                                         "end": None, "protein": s["protein"], "bipartite": s["bipartite"],
                                         "confidence": "curated_seed"})
        positives = dedup_positives

        negatives = []
        pos_seqs = {p["seq"] for p in positives}
        with open(negatives_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seq = row["neg_sequence"].upper()
                if seq in pos_seqs:
                    continue
                negatives.append({
                    "seq": seq, "full_sequence": row.get("full_sequence") or None,
                    "start": int(row["start"]) - 1 if row.get("start") else None,
                    "end": int(row["end"]) if row.get("end") else None,
                    "protein": row["accession"], "neg_type": row.get("neg_type"),
                })

        n_exp = sum(1 for p in positives if p.get("confidence") == "experimental")
        n_bip = sum(1 for p in positives if str(p.get("bipartite")) == "1")
        print(f"Positives: {len(positives)} unique ({n_exp} experimental evidence, {n_bip} bipartite-labeled)")
        by_type = {}
        for ng in negatives:
            by_type[ng["neg_type"]] = by_type.get(ng["neg_type"], 0) + 1
        print(f"Negatives: {len(negatives)} {by_type}")

        self.pssm = self._build_pssm([p["seq"] for p in positives])
        print(f"PSSM: {PSSM_WIDTH}-column window anchored on the basic-cluster "
              f"register, built from {int(self.pssm[2])} sequences")

        structural = self._load_structural_data()
        if not structural:
            print("  NOTE: no structural_data.json found -- run "
                  "structural_dataset_pipeline.py in nls_data_pipeline/ (locally, "
                  "with real internet) to give plddt_norm/sasa_norm real values "
                  "instead of the constant neutral default.")

        iupred = self._load_iupred_data()
        n_pos_iupred = sum(1 for p in positives if p["seq"] in iupred)
        n_neg_iupred = sum(1 for ng in negatives if ng["seq"] in iupred)
        print(f"Real IUPred2A/ANCHOR2 data: {len(iupred)} sequences loaded "
              f"({n_pos_iupred}/{len(positives)} positives, {n_neg_iupred}/{len(negatives)} negatives have a match)")
        if not iupred:
            print("  NOTE: no iupred_data.json found -- run fetch_iupred_training_data.py "
                  "in nls_data_pipeline/ (needs real internet access to iupred2a.elte.hu) "
                  "to give iupred_mean/anchor2_mean etc. real per-example values instead "
                  "of the constant neutral default.")

        X, y = [], []
        for p in positives:
            s = structural.get(p["seq"])
            plddt = s.get("plddt_per_residue") if s else None
            sasa = s.get("sasa_per_residue") if s else None
            iv = iupred.get(p["seq"])
            X.append(self._extract_features(p["seq"], p.get("full_sequence"), p.get("start"), p.get("end"),
                                              plddt_values=plddt, sasa_values=sasa, iupred_values=iv))
            y.append(1)
        for ng in negatives:
            s = structural.get(ng["seq"])
            plddt = s.get("plddt_per_residue") if s else None
            sasa = s.get("sasa_per_residue") if s else None
            iv = iupred.get(ng["seq"])
            X.append(self._extract_features(ng["seq"], ng.get("full_sequence"), ng.get("start"), ng.get("end"),
                                              plddt_values=plddt, sasa_values=sasa, iupred_values=iv))
            y.append(0)

        X, y = np.array(X), np.array(y)
        print(f"Feature dimension: {X.shape[1]}")
        return {"X": X, "y": y, "positives": positives, "negatives": negatives,
                "stats": {"n_positives": len(positives), "n_positives_experimental": n_exp,
                          "n_positives_bipartite": n_bip, "n_negatives": len(negatives),
                          "negatives_by_type": by_type, "feature_dim": int(X.shape[1])}}

    # -- training: single train/test split, k-fold CV for model selection --
    # Switched from the original 60/20/20 train/val/test design
    # (kept in nls_ml_predictor_before_kfold.py) to a single 80/20
    # train/test split with k-fold CV on the training split for model
    # selection, matching nes_ml_predictor_improved.py's _train_model()
    # exactly. Reasoning: with only ~265 positives
    # (fewer than NES's ~305), a single held-out validation slice (~20% of
    # an already-small dataset) is a noisy basis for choosing among 7-8
    # classifier candidates -- CV averages the comparison over several
    # train/validation partitions instead of trusting one. It also fixes an
    # asymmetry the old design had: the previous version's shipped model was
    # refit on train+val only, permanently withholding the test 20% from
    # training; this version refits the final shipped model on ALL data
    # (train+test) after the honest held-out evaluation is done, same as
    # NES, so no data is permanently sacrificed. Having both predictors use
    # the identical selection methodology also removes a confound when
    # comparing them side by side in the thesis.

    def _train_model(self):
        """Nested cross-validation (switch, mirrors the NES-side
        rewrite in nes_ml_predictor_improved.py -- see that file's
        _train_model docstring for the full rationale). OUTER k-fold gives
        an unbiased performance estimate (mean +/- std across outer test
        folds); INNER k-fold, re-run fresh inside each outer training fold,
        picks the classifier -- the outer test fold never leaks into a
        selection decision. A separate final selection+fit pass on 100% of
        the data (after the outer-loop numbers are locked in) decides which
        classifier type ships."""
        print("\n" + "=" * 70)
        print("Training NLS Predictor (nested CV)")
        print("=" * 70)

        dataset = self.build_training_dataset()
        X, y = dataset["X"], dataset["y"]
        feature_names = self._feature_names()
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        can_nest = n_pos >= 10 and n_neg >= 10

        def _build_candidates():
            # Expanded from 4 to 8 candidates, mirroring the NES
            # side (see nes_ml_predictor_improved.py's _train_model for the
            # full rationale on why LSTM/Transformer/LightGBM/CatBoost were
            # deliberately left out). Rebuilt fresh (unfitted) every call so
            # nested folds never share fitted state.
            c = {
                "svm_linear": SVC(kernel="linear", C=0.1, probability=True,
                                   random_state=42, class_weight="balanced"),
                "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                random_state=42, class_weight="balanced"),
                "random_forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                           class_weight="balanced"),
                "extra_trees": ExtraTreesClassifier(n_estimators=300, random_state=42,
                                                      class_weight="balanced"),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
                "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
                "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                                       random_state=42, early_stopping=True),
            }
            if XGBOOST_AVAILABLE:
                c["xgboost"] = XGBClassifier(random_state=42, eval_metric="logloss")
            return c

        # Selection was pure argmax(mean_f1), which picked
        # xgboost over svm_linear the first time xgboost was available in
        # this environment (0.792+/-0.105 vs 0.779+/-0.086) -- a margin well
        # inside one fold-to-fold standard deviation, i.e. noise, not a real
        # win. XGBoost's default hyperparameters (untuned depth/regularization,
        # no class-imbalance handling, unlike every sklearn candidate here
        # which gets class_weight="balanced") then overfit the ~750-example
        # training set hard enough that the shipped model's holdout
        # probabilities collapsed to "everything scores 0.9+" regardless of
        # true label -- confirmed against the 25+25 holdout set
        # (sensitivity jumped to 76% but specificity collapsed to 36%, with
        # both positives AND negatives clustering near 1.0 -- a miscalibration
        # signature, not genuinely better discrimination).
        #
        # select by a pessimistic lower-confidence-bound score
        # (mean_f1 - std_f1) instead of raw mean_f1, the same "1-SE rule"
        # principle glmnet uses for lambda selection -- penalize candidates
        # whose CV performance is volatile across folds, not just candidates
        # with a lower average. Recomputed against this exact run's
        # cv_f1_by_model numbers: xgboost's LCB = 0.686, svm_linear's = 0.693
        # -- svm_linear wins, as it should, without hardcoding/banning
        # xgboost outright (a future run with more data or a properly tuned
        # XGBClassifier could legitimately win under this same rule).
        SELECTION_LCB_K = 1.0

        def _select_best(X_s, y_sub):
            candidates = _build_candidates()
            folds = max(2, min(5, int(min(np.bincount(y_sub)))))
            report, best_name, best_lcb, best_mean = {}, None, -1.0, -1.0
            for name, mdl in candidates.items():
                try:
                    scores = cross_val_score(mdl, X_s, y_sub, cv=folds, scoring="f1")
                    mean_f1, std_f1 = float(scores.mean()), float(scores.std())
                    report[name] = {"mean_f1": mean_f1, "std_f1": std_f1}
                    lcb = mean_f1 - SELECTION_LCB_K * std_f1
                    if lcb > best_lcb:
                        best_lcb, best_mean, best_name = lcb, mean_f1, name
                except Exception as e:
                    report[name] = {"error": str(e)}
            # best_score returned/reported downstream is still the plain mean
            # F1 of whichever model won the LCB comparison, so
            # "Chosen model to ship: X (CV F1 on 100% data = Y)" keeps
            # reporting an intuitive, undiscounted number -- the LCB is the
            # selection mechanism, not something we want to also overload
            # the printed/logged F1 with.
            return best_name or "svm_linear", best_mean, report

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
                    "fold": fold_i, "chosen_classifier": inner_name,
                    "inner_selection_f1": float(inner_score),
                    "n_test": int(len(y_te)),
                    "accuracy": float(accuracy_score(y_te, pred)),
                    "precision": float(precision_score(y_te, pred, zero_division=0)),
                    "recall": float(recall_score(y_te, pred, zero_division=0)),
                    "f1": float(f1_score(y_te, pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_te, proba)) if len(set(y_te)) == 2 else None,
                }
                per_fold_rows.append(fold_metrics)
                print(f"  fold {fold_i + 1}/{outer_folds}: chose {inner_name} "
                      f"(inner CV F1={inner_score:.3f}) -> outer test F1={fold_metrics['f1']:.3f}  "
                      f"ROC-AUC={fold_metrics['roc_auc']}")

                try:
                    perm_result = permutation_importance(
                        fold_model, X_te_s, y_te, n_repeats=30,
                        random_state=42, scoring="f1", n_jobs=-1)
                    fold_perm_importances.append(perm_result.importances_mean)
                except Exception as e:
                    print(f"    Warning: Could not compute permutation importance for fold {fold_i}: {e}")

            f1s = [r["f1"] for r in per_fold_rows]
            aucs = [r["roc_auc"] for r in per_fold_rows if r["roc_auc"] is not None]
            mode_clf, mode_n = outer_selection_counts.most_common(1)[0]
            nested_cv_report = {
                "outer_folds": outer_folds,
                "per_fold": per_fold_rows,
                "test_f1_mean": float(np.mean(f1s)), "test_f1_std": float(np.std(f1s)),
                "test_roc_auc_mean": float(np.mean(aucs)) if aucs else None,
                "test_roc_auc_std": float(np.std(aucs)) if aucs else None,
                "classifier_selection_counts": dict(outer_selection_counts),
                "classifier_selection_mode": mode_clf,
                "classifier_selection_mode_frequency": float(mode_n / outer_folds),
            }
            print(f"\nNested CV estimate: F1 = {nested_cv_report['test_f1_mean']:.3f} +/- "
                  f"{nested_cv_report['test_f1_std']:.3f}   ROC-AUC = "
                  f"{nested_cv_report['test_roc_auc_mean']}   "
                  f"(outer-fold winner: {mode_clf}, chosen in {mode_n}/{outer_folds} folds)")

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
            # fit on ALL data for shipping (doesn't feed back into the
            # nested CV numbers above).
            print("\nFinal classifier selection (CV on 100% of the data, for shipping only)...")
            self.scaler = StandardScaler().fit(X)
            X_all_s = self.scaler.transform(X)
            self.model_name, best_score, cv_report = _select_best(X_all_s, y)
            self.model = clone(_build_candidates()[self.model_name])
            self.model.fit(X_all_s, y)
            print(f"Chosen model to ship: {self.model_name} (CV F1 on 100% data = {best_score:.3f})")

            impurity = None
            if hasattr(self.model, "feature_importances_"):
                impurity = dict(zip(feature_names, [float(v) for v in self.model.feature_importances_]))
            elif hasattr(self.model, "coef_"):
                impurity = dict(zip(feature_names, [float(v) for v in np.abs(self.model.coef_[0])]))
            self.impurity_importance_ = impurity

            metrics = {
                **dataset["stats"], "chosen_model": self.model_name,
                "cv_f1_by_model": cv_report, "n_train": int(len(y)),
                "training_protocol": "nested_cv",
                "nested_cv": nested_cv_report,
                "held_out_test": None,
                "impurity_importance": impurity,
                "permutation_importance": self.permutation_importance_,
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
            impurity = None
            if hasattr(self.model, "feature_importances_"):
                impurity = dict(zip(feature_names, [float(v) for v in self.model.feature_importances_]))
            elif hasattr(self.model, "coef_"):
                impurity = dict(zip(feature_names, [float(v) for v in np.abs(self.model.coef_[0])]))
            self.impurity_importance_ = impurity
            metrics = {
                **dataset["stats"], "chosen_model": self.model_name,
                "cv_f1_by_model": cv_report, "n_train": int(len(y)),
                "training_protocol": "bare_fit_small_dataset",
                "nested_cv": None, "held_out_test": None,
                "impurity_importance": impurity, "permutation_importance": None,
            }

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.pssm, self.pssm_path)
        if self.permutation_importance_ is not None:
            json.dump(self.permutation_importance_, open(self.importance_path, "w"), indent=2)

        json.dump(metrics, open(self.metrics_path, "w"), indent=2)
        print(f"\nSaved model/scaler/PSSM to {self.model_dir}/, metrics to {self.metrics_path.name}")
        return metrics

    # -- inference ------------------------------------------------------------

    def predict(self, sequence, full_sequence=None, start=None, end=None,
                plddt_values=None, sasa_values=None):
        if self.model is None:
            raise RuntimeError("No trained model loaded -- run NLSPredictor()._train_model() first "
                                "(or `python nls_ml_predictor.py train`).")
        feats = np.array([self._extract_features(sequence, full_sequence, start, end,
                                                   plddt_values=plddt_values, sasa_values=sasa_values)])
        feats_s = self.scaler.transform(feats)
        proba = float(self.model.predict_proba(feats_s)[0, 1])

        # CAAX-box membrane-anchor veto -- see
        # caax_membrane_anchor_veto()/has_caax_box() docstrings above for
        # why this is an explicit override rather than a learned ML feature
        # (zero variance in training data). Applied here, upstream of
        # app.py's separate structural-accessibility exposure_factor gate,
        # so this model's own nls_probability output already reflects it
        # regardless of caller (scan_sequence, single predict() calls, the
        # holdout pipeline test, etc.) -- not something a caller has to
        # remember to apply separately.
        caax_veto = caax_membrane_anchor_veto(full_sequence, end)
        raw_proba = proba
        if caax_veto:
            proba = min(proba, CAAX_PROBABILITY_CAP)

        # Chromatin/DNA-condensation "basic background" veto --
        # see basic_background_veto()/basic_background_enrichment()
        # docstrings above for the histone/protamine holdout evidence and
        # why this is a real per-protein statistic rather than a learned
        # feature. Same explicit-override placement as the CAAX veto above
        # (upstream of app.py's separate accessibility gate), so
        # nls_probability already reflects it for every caller.
        basic_bg_veto = basic_background_veto(full_sequence, start, end)
        basic_bg_factor = basic_background_factor(full_sequence, start, end)
        if basic_bg_factor < 1.0:
            proba = proba * basic_bg_factor

        is_bip, spacer, _, _ = detect_bipartite(sequence.upper())
        # full_sequence/start passed through so the (optional) tripartite
        # check can look at real nearby residues outside this candidate
        # window -- see _classify_nls_pattern/detect_extra_basic_cluster.
        classification = _classify_nls_pattern(sequence, full_sequence=full_sequence, start=start)
        result = {
            "sequence": sequence, "nls_probability": proba,
            "predicted_class": "bipartite" if is_bip else (
                "monopartite" if MONOPARTITE_RE.search(sequence.upper()) else "non-classical/uncertain"),
            "pssm_score": self._calculate_pssm_score(sequence),
            "is_bipartite": bool(is_bip),
            "kosugi_classes": classification["kosugi_classes"],
            "py_nls_shaped": classification["py_nls_shaped"],
            "potential_tripartite": classification["potential_tripartite"],
            "caax_membrane_anchor": caax_veto,
            "basic_background_veto": basic_bg_veto,
        }
        if caax_veto:
            result["raw_nls_probability_pre_caax_veto"] = raw_proba
            result["caax_caveat"] = (
                "This window sits within the C-terminal CAAX-box prenylation region "
                "(a real Cys-based membrane-anchor motif was found at the protein's true "
                "C-terminus). nls_probability has been capped -- a CAAX-anchored polybasic "
                "patch is a lipid-anchoring signal, not a nuclear import signal, regardless "
                "of how basic it reads."
            )
        if basic_bg_factor < 1.0:
            bg_frac, win_frac, fold = basic_background_enrichment(full_sequence, start, end)
            result["raw_nls_probability_pre_basic_background_veto"] = raw_proba
            result["basic_background_frac"] = round(bg_frac, 3) if bg_frac is not None else None
            result["basic_background_fold_enrichment"] = round(fold, 2) if fold is not None else None
            result["basic_background_factor"] = round(basic_bg_factor, 3)
            result["basic_background_caveat"] = (
                f"This protein's whole sequence is already {round(bg_frac * 100)}% K/R "
                f"(a chromatin-compaction-protein-like composition), and this window is only "
                f"~{round(fold, 1)}x more basic than that background -- not the kind of "
                f"locally exceptional patch a real classical NLS represents. nls_probability "
                f"has been discounted by a factor of {round(basic_bg_factor, 2)} (continuous, scaled "
                f"by how close this window's enrichment is to just being ordinary background "
                f"composition); this reads like histone/protamine-type DNA-binding composition, "
                f"not a nuclear import signal."
            )
        if classification["py_nls_shaped"]:
            result["py_nls_caveat"] = (
                "Matches the PY-NLS (Transportin-1/Karyopherin-beta2) consensus shape. "
                "nls_probability above is NOT meaningful for this class -- this model was "
                "trained exclusively on importin-alpha-pathway examples and has never seen "
                "a real PY-NLS positive."
            )
        if classification["potential_tripartite"]:
            result["tripartite_note"] = classification["tripartite_note"]
            result["tripartite_extra_cluster"] = classification["tripartite_extra_cluster"]
        return result

    # Widened from range(7,23) -- the old 22-residue ceiling made
    # it structurally impossible for scan_sequence to ever propose a PY-NLS
    # or other longer non-classical candidate (15-30aa+), independent of
    # what the classifier would say about it. 31 covers classical
    # monopartite/bipartite and the bulk of the PY-NLS range; true
    # non-classical NLSs can run past 30 still, but windows much beyond
    # this get expensive to scan across a whole protein and increasingly
    # unspecific, so 31 is a deliberate practical ceiling, not a claim that
    # nothing longer exists biologically.
    def scan_sequence(self, full_sequence, plddt_values=None, sasa_values=None,
                       window_sizes=range(6, 31), score_threshold=0.5):
        """Slide candidate windows across a whole protein sequence and return
        non-overlapping, above-threshold NLS predictions -- the structural,
        whole-protein analog of predict() (which scores one candidate
        peptide in isolation). Mirrors the NES side's unified_crm1_nes scan:
        cheap regex pre-filter first (monopartite core or bipartite
        signature), then only run the trained classifier on windows that
        pass it, then greedy non-overlap selection by score.

        plddt_values / sasa_values, if given, are the FULL per-residue
        arrays for the whole structure (same length as full_sequence,
        0-indexed) -- local slices are handed to predict() so pLDDT/SASA
        features get their real, live values here even though they're
        neutral defaults during training/single-sequence predict() calls
        (see NLS_predictor_landscape_and_novelty.md point 4).
        """
        if self.model is None:
            raise RuntimeError("No trained model loaded -- run NLSPredictor()._train_model() first "
                                "(or `python nls_ml_predictor.py train`).")
        seq = full_sequence.upper()
        n = len(seq)
        candidates = []
        for window_size in window_sizes:
            if window_size > n:
                continue
            for i in range(n - window_size + 1):
                sub = seq[i:i + window_size]
                # Added PY_NLS_RE to this pre-filter. PY-NLS
                # motifs typically have their basic residues spread singly
                # rather than in the adjacent K/R doublet both
                # MONOPARTITE_RE and detect_bipartite() require -- e.g. the
                # hnRNP A1 M9 domain matches neither, so without this it
                # would never even reach predict() regardless of window
                # size. See py_nls_shaped/PY_NLS_RE docstrings: the ML
                # score for these candidates still isn't meaningful, but
                # they should at least surface as a flagged candidate
                # rather than being silently invisible to the scan.
                if not (MONOPARTITE_RE.search(sub) or detect_bipartite(sub)[0] or PY_NLS_RE.search(sub)):
                    continue
                local_plddt = plddt_values[i:i + window_size] if plddt_values is not None and len(plddt_values) >= i + window_size else None
                local_sasa = sasa_values[i:i + window_size] if sasa_values is not None and len(sasa_values) >= i + window_size else None
                result = self.predict(sub, full_sequence=seq, start=i, end=i + window_size,
                                       plddt_values=local_plddt, sasa_values=local_sasa)
                # PY-NLS-shaped candidates are surfaced regardless of the
                # score_threshold gate on nls_probability, since that
                # probability isn't meaningful for this class (see
                # py_nls_caveat) -- gating them out here would silently
                # hide exactly the candidates this fix was meant to expose.
                if result["nls_probability"] > score_threshold or result["py_nls_shaped"]:
                    candidates.append({
                        "sequence": sub, "start": i, "end": i + window_size - 1,
                        "length": window_size,
                        "nls_probability": result["nls_probability"],
                        "predicted_class": result["predicted_class"],
                        "pssm_score": result["pssm_score"],
                        "is_bipartite": result["is_bipartite"],
                        "kosugi_classes": result["kosugi_classes"],
                        "py_nls_shaped": result["py_nls_shaped"],
                        "potential_tripartite": result["potential_tripartite"],
                        "caax_membrane_anchor": result["caax_membrane_anchor"],
                        "basic_background_veto": result["basic_background_veto"],
                        **({"py_nls_caveat": result["py_nls_caveat"]} if result["py_nls_shaped"] else {}),
                        **({"tripartite_note": result["tripartite_note"],
                            "tripartite_extra_cluster": result["tripartite_extra_cluster"]}
                           if result["potential_tripartite"] else {}),
                        **({"caax_caveat": result["caax_caveat"]} if result["caax_membrane_anchor"] else {}),
                        **({"basic_background_caveat": result["basic_background_caveat"],
                            "basic_background_frac": result["basic_background_frac"],
                            "basic_background_fold_enrichment": result["basic_background_fold_enrichment"],
                            "basic_background_factor": result["basic_background_factor"]}
                           if result["basic_background_veto"] else {}),
                    })

        # A real protein region can genuinely BE more than one
        # thing at once from a classification standpoint -- e.g. a bipartite
        # NLS's own first cluster is, in isolation, also a perfectly valid-
        # looking monopartite core. Those are two different structural
        # hypotheses about the same residues, not duplicate detections of
        # the same thing, so a single flat "highest score wins, discard
        # everything else that overlaps it" pass was wrong: it forced a
        # choice between mono/bi and always discarded whichever one didn't
        # win, even when both were real, independently-scored candidates
        # worth reporting side by side.
        #
        # classify every candidate into exactly one primary bucket
        # (bipartite / monopartite / other -- "other" covers non-classical/
        # uncertain windows, including PY-NLS-shaped ones, which are a
        # mechanistically distinct receptor pathway from classical
        # importin-alpha NLSs anyway and shouldn't be forced to compete with
        # either classical class for a span), then run non-overlap greedy
        # selection SEPARATELY within each bucket. A candidate still can't
        # duplicate/crowd out another candidate of the SAME class over the
        # same span (that would just be redundant near-identical windows),
        # but a bipartite candidate and a monopartite candidate covering
        # overlapping residues can now both surface, each with its own
        # independent score. (Tripartite has no bucket of its own here --
        # it's a heuristic annotation on a bipartite candidate, not a
        # separately trained/scored class -- see detect_extra_basic_cluster.)
        def _bucket(c):
            if c["is_bipartite"]:
                return "bipartite"
            if c["predicted_class"] == "monopartite":
                return "monopartite"
            return "other"

        buckets = {"bipartite": [], "monopartite": [], "other": []}
        for c in candidates:
            buckets[_bucket(c)].append(c)

        selected = []
        for bucket_candidates in buckets.values():
            bucket_candidates.sort(key=lambda c: c["nls_probability"], reverse=True)
            used = set()
            for c in bucket_candidates:
                span = range(c["start"], c["end"] + 1)
                if any(p in used for p in span):
                    continue
                selected.append(c)
                used.update(span)

        selected.sort(key=lambda c: c["start"])
        return selected

    def get_feature_importance(self, method="permutation"):
        if method == "permutation" and self.permutation_importance_:
            return dict(sorted(self.permutation_importance_.items(), key=lambda kv: -kv[1]))
        if method == "impurity" and self.impurity_importance_:
            return dict(sorted(self.impurity_importance_.items(), key=lambda kv: -kv[1]))
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict"])
    ap.add_argument("args", nargs="*")
    args = ap.parse_args()

    predictor = NLSPredictor()
    if args.command == "train":
        predictor._train_model()
    elif args.command == "predict":
        if not args.args:
            print("Usage: python nls_ml_predictor.py predict SEQUENCE")
            return
        result = predictor.predict(*args.args[:1])
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
