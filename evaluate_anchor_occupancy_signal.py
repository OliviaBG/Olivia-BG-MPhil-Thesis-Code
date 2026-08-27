#!/usr/bin/env python3
"""
evaluate_anchor_occupancy_signal.py
============================================================
Empirically tests whether md_refinement.py's Phi-anchor <-> CRM1 sub-pocket
occupancy metric (NESMDRefiner._run_crm1_docking's anchor_occupancy_score,
blended into binding_score via ANCHOR_OCCUPANCY_WEIGHT currently)
actually discriminates real NES motifs from real hard negatives -- using
REAL AlphaFold structures and REAL MD docking runs
(NESMDRefiner.refine_nes_candidates), not a cheap proxy for it.

WHY THIS EXISTS: ANCHOR_OCCUPANCY_WEIGHT (0.15) and the 0.5/1.5 nm
distance-to-score mapping it's built from (ANCHOR_FULL_OCCUPANCY_NM /
ANCHOR_ZERO_OCCUPANCY_NM in md_refinement.py) were both guesses -- the same
situation crm1_binding_affinity's old 70/30 fpocket/burial blend was in
before evaluate_crm1_pocket_signal.py tested it (see
CRM1_pocket_scoring_evaluation_2026-07-27.md), and the same situation
crm1_compatibility_score's 7 hand-picked factor weights were in before that
script's per-factor AUC breakdown found only 2 of the 7 actually cleared
significance. This script applies the identical methodology -- real
structures, real labels, cross-validated AUC, standardized logistic
regression for a data-driven weight -- to the newest addition instead of
just trusting it because the reasoning behind the guess sounded plausible.

THIS IS MUCH MORE EXPENSIVE than the fpocket-based eval it's modeled on:
each example needs a REAL MD docking run, not a ~10-60s fpocket call.
--duration-ns defaults far below the app's own 10 ns default specifically
to keep this feasible -- see that flag's help text for the real tradeoff
this creates (a weak result at short duration is NOT the same as a
confirmed null result). --limit/--neg-limit default much smaller than
evaluate_crm1_pocket_signal.py's for the same reason. Expect low-to-mid
single-digit MINUTES per example even at reduced duration, not seconds --
budget accordingly before raising the sample size.

REQUIREMENTS (run locally; needs network access and locally installed tools):
  - OpenMM + PDBFixer installed (conda install -c conda-forge openmm pdbfixer)
  - real internet access to alphafold.ebi.ac.uk
  - crm1_reference/CRM1_Ran_only.pdb or crm1.pdb present (same reference
    evaluate_crm1_pocket_signal.py and pocket_detector.py already use)
  - nes_data_pipeline/nes_dataset.json and nes_negatives/nes_negatives.csv
    (already in this project) -- this script reuses
    load_positive_examples()/load_negative_examples()/
    write_text_atomic_with_retry() from evaluate_crm1_pocket_signal.py
    rather than re-implementing the same sampling logic a second time

USAGE:
    python3 evaluate_anchor_occupancy_signal.py
    python3 evaluate_anchor_occupancy_signal.py --limit 5 --neg-limit 5 --duration-ns 2.0

As of this also tests, by default, whether starting each
candidate's peptide from a literature-informed idealized alpha helix
(instead of AlphaFold's isolated, unbound-state prediction) changes the
result -- see md_refinement._build_idealized_helix_pdb and
refine_nes_candidates(test_both_conformations=...) for the full rationale.
This roughly DOUBLES the MD cost per candidate (two trajectories instead
of one), which is why --limit/--neg-limit now default to 5 each (10 total)
rather than the larger samples used before this was added -- pass
--no-test-both-conformations to go back to one trajectory per candidate
and afford a larger sample instead. ADDENDUM -- specificity control (--test-specificity-control):
a real 43-example run (see anchor_occupancy_dual_hypothesis_v2.json /
session notes) found every packing-quality metric this script measures --
anchor_occupancy_score, avg_anchor_pocket_distance_nm, avg_cys528_distance_nm,
avg_groove_contacts, avg_hydrophobic_contacts -- running BACKWARDS: hard
negatives (coiled-coil/leucine-zipper fragments, which are rigid, already-
folded, generically hydrophobic helices) packed TIGHTER into CRM1's groove
than real NES motifs did, on a short (2 ns) unbiased trajectory. Testing
canonical NES-class spacing (Kosugi/LocNES classes 1a-3) and initial-vs-
final "drift" both failed to rescue this -- both still ran backwards or
were dominated by the same confound. The leading hypothesis: real NES
motifs are often intrinsically disordered pre-binding and don't have time
to settle into a tight pose on this timescale, while the rigid decoys
settle quickly into ANY plausible pocket layout regardless of whether it's
the biologically correct one -- i.e. a general stickiness/rigidity
confound, not evidence the decoys are real binders.

--test-specificity-control tests that directly: for every starting
conformation being tried, it ALSO runs a matched control where the peptide
is anchor-registered against a cyclically-WRONG sub-pocket assignment
(see NESMDRefiner._place_peptide_via_subpocket_registration's
scramble_registration docstring) instead of its own correct one, with
every other setting -- duration, forcefield, minimization/equilibration
protocol -- identical. A real NES motif is expected to show a much bigger
correct-vs-scrambled gap (it specifically fits its own registration) than
a rigid decoy (which should pack comparably well either way, since it
isn't really responding to which pocket it's aimed at). This DOUBLES MD
cost again on top of whatever --test-both-conformations already costs (4x
base cost if both are on) -- defaults to off; combine with
--no-test-both-conformations to keep this a clean 2x-cost comparison
(native vs. native-scrambled only) rather than paying for all 4 variants
at once.

Checkpoints to --cache after every example, same as
evaluate_crm1_pocket_signal.py -- safe to Ctrl-C and rerun the same command
later to resume (already-evaluated (accession, start, end) triples are
skipped).

READING THE OUTPUT: the first thing reported is the Phi-register MATCH
RATE (positives vs. negatives) -- whether _find_phi_register() even finds
a usable anchor pattern to test in the first place is itself a signal this
codebase currently computes but doesn't use anywhere (a candidate either
gets anchor-aware placement or silently falls back to the generic
approach). Only examples where a register WAS matched get an
anchor_occupancy_score at all, so watch the "n scored / n evaluated"
counts -- a low match rate on real positives would itself be a finding
worth investigating before trusting any AUC computed on the smaller,
matched-only subset.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / 'nes_data_pipeline'))

from structural_dataset_v2_pipeline import fetch_alphafold_pdb  # noqa: E402
from consensus_accessibility import consensus_accessibility  # noqa: E402
from evaluate_crm1_pocket_signal import (  # noqa: E402
    load_positive_examples, load_negative_examples,
    write_text_atomic_with_retry, THREE_TO_ONE, CRM1_REF_CANDIDATES,
)
from md_refinement import NESMDRefiner  # noqa: E402


def get_structure_bundle(accession, cache, pdb_cache_dir):
    """Real AlphaFold structure + resolved sequence + residue numbering for
    `accession`, cached in-memory and on-disk (reuses the same on-disk PDB
    cache directory evaluate_crm1_pocket_signal.py uses, so a prior run of
    that script means no re-download here). Deliberately NOT the same
    function as that script's get_structure_bundle() -- this one skips the
    RSA/fpocket-adjacent fields (rsa_by_resnum, resname_by_resnum) that
    script needs and this one doesn't, and doesn't touch its separate
    pocket_cache."""
    if accession in cache:
        return cache[accession]

    cache_file = pdb_cache_dir / f'{accession}.pdb'
    if cache_file.exists():
        pdb_text = cache_file.read_text()
    else:
        print(f"    fetching {accession} from AlphaFold...")
        pdb_text = fetch_alphafold_pdb(accession)
        if pdb_text is None:
            print(f"    {accession}: no AlphaFold structure available")
            cache[accession] = None
            return None
        write_text_atomic_with_retry(cache_file, pdb_text)

    try:
        rows = consensus_accessibility(str(cache_file))
    except Exception as e:
        print(f"    {accession}: consensus_accessibility failed ({e})")
        cache[accession] = None
        return None
    if not rows:
        cache[accession] = None
        return None

    chain_id = sorted({r.chain for r in rows})[0]
    rows = sorted((r for r in rows if r.chain == chain_id), key=lambda r: r.resnum)
    residue_numbers = [r.resnum for r in rows]
    sequence = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in rows)

    bundle = (pdb_text, residue_numbers, sequence)
    cache[accession] = bundle
    return bundle


def build_candidate(residue_numbers, sequence, start, end):
    """NESMDRefiner.refine_nes_candidates() needs a candidate dict with a
    'sequence' string matching the real residues at [start, end] (1-indexed
    inclusive, same convention nes_dataset.json/nes_negatives.csv and
    evaluate_crm1_pocket_signal.py's candidate_positions already use).
    Returns None if the structure has a gap anywhere in that exact span
    (missing density) -- same "skip, don't guess" policy the rest of this
    project's eval scripts use for incomplete structural coverage."""
    resnum_to_idx = {rn: i for i, rn in enumerate(residue_numbers)}
    idxs = [resnum_to_idx[p] for p in range(start, end + 1) if p in resnum_to_idx]
    if len(idxs) != (end - start + 1):
        return None
    cand_seq = ''.join(sequence[i] for i in idxs)
    return {
        'sequence': cand_seq,
        'start': start,
        'end': end,
        'full_sequence': sequence,
        'combined_score': 0.5,  # only used by refine_nes_candidates' own error-path fallback score
    }


def _extract_feature_fields(metrics):
    """Pulls the same set of fields out of one md_metrics dict, regardless
    of which starting conformation produced it -- shared by the
    per-conformation and best-of-both extraction below so the two can't
    silently drift apart."""
    metrics = metrics or {}
    reg = metrics.get('subpocket_registration')
    return {
        'anchor_occupancy_score': metrics.get('anchor_occupancy_score'),
        'avg_anchor_pocket_distance_nm': metrics.get('avg_anchor_pocket_distance_nm'),
        'n_anchors_matched': (reg.get('n_anchors_matched') if reg else 0) or 0,
        'matched_pockets': (reg.get('matched_pockets') if reg else []) or [],
        'anchor_fit_rmsd_nm': reg.get('anchor_fit_rmsd_nm') if reg else None,
        'raw_binding_score': metrics.get('raw_binding_score'),
        'avg_cys528_distance_nm': metrics.get('avg_cys528_distance_nm'),
        'helix_combined_score': metrics.get('helix_combined_score'),
        'avg_groove_contacts': metrics.get('avg_groove_contacts'),
        'avg_hydrophobic_contacts': metrics.get('avg_hydrophobic_contacts'),
    }


def compute_features(refiner, accession, start, end, duration_ns, struct_cache, pdb_cache_dir,
                      test_both_conformations=True, test_specificity_control=False):
    bundle = get_structure_bundle(accession, struct_cache, pdb_cache_dir)
    if bundle is None:
        return None
    pdb_text, residue_numbers, sequence = bundle

    candidate = build_candidate(residue_numbers, sequence, start, end)
    if candidate is None:
        return None

    enhanced = refiner.refine_nes_candidates(
        pdb_text, [candidate], duration_ns,
        test_both_conformations=test_both_conformations,
        test_specificity_control=test_specificity_control)
    if not enhanced:
        return None
    result = enhanced[0]

    # Best-of-both (or the only result, if test_both_conformations=False)
    # -- kept at the TOP LEVEL of the returned dict so the AUC/regression
    # analysis below doesn't need to change at all; it already looks at
    # exactly these field names.
    feats = _extract_feature_fields(result.get('md_metrics', {}))

    if test_both_conformations:
        by_conf = result.get('md_metrics_by_conformation', {}) or {}
        feats['best_starting_conformation'] = result.get('md_best_starting_conformation')
        # Added to test whether starting the peptide from a
        # literature-informed idealized alpha helix (see
        # md_refinement._build_idealized_helix_pdb), instead of AlphaFold's
        # isolated-state prediction for this stretch, changes the result --
        # motivated by candidates consistently reading as "weak binder"
        # even for real positives, and the observation that real NES
        # motifs are frequently in disordered regions where AlphaFold's
        # own isolated prediction is often not already helical, even
        # though helical engagement is the literature-documented dominant
        # (not universal) binding mode. Keeping BOTH sets of per-
        # conformation fields, not just which one "won", so this can be
        # analyzed honestly (e.g. does idealized_helix win more often for
        # real positives than for hard negatives, or does it just inflate
        # everything equally).
        for conf_label in ('native', 'idealized_helix'):
            conf_feats = _extract_feature_fields(by_conf.get(conf_label, {}))
            for key, val in conf_feats.items():
                feats[f'{key}__{conf_label}'] = val

    if test_specificity_control:
        # Pulls the SCRAMBLED-registration control run(s) out of
        # md_metrics_by_variant (see refine_nes_candidates'
        # test_specificity_control docstring) -- one per starting
        # conformation actually tested this call. Suffix convention matches
        # the native/idealized_helix one above but with "_scrambled"
        # appended, e.g. anchor_occupancy_score__native_scrambled.
        by_variant = result.get('md_metrics_by_variant', {}) or {}
        conf_labels = ('native', 'idealized_helix') if test_both_conformations else ('native',)
        for conf_label in conf_labels:
            # When test_both_conformations is False, the plain (unsuffixed)
            # top-level fields already ARE this call's only (native)
            # correct-registration result -- refine_nes_candidates always
            # picks 'native' as best_tag in that case, since it's the only
            # unscrambled variant available. Mirror it under the
            # __native suffix too so the gap computation below has a
            # uniformly-named "correct" counterpart to compare the
            # scrambled result against, regardless of which flags were set.
            if conf_label == 'native' and not test_both_conformations:
                for key, val in _extract_feature_fields(result.get('md_metrics', {})).items():
                    feats.setdefault(f'{key}__native', val)

            scrambled_feats = _extract_feature_fields(by_variant.get(f'{conf_label}_scrambled', {}))
            for key, val in scrambled_feats.items():
                feats[f'{key}__{conf_label}_scrambled'] = val

    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=5,
                     help='Positives to attempt. Kept small by default -- real MD is slow, see module '
                          'docstring, and --test-both-conformations roughly DOUBLES that cost per '
                          'candidate. Raise once a small run looks promising.')
    ap.add_argument('--neg-limit', type=int, default=5,
                     help='Negatives to attempt, randomly sampled (fixed seed, via '
                          'evaluate_crm1_pocket_signal.load_negative_examples) from the real 90%% '
                          'coiled_coil / 10%% leucine_zipper pool.')
    ap.add_argument('--duration-ns', type=float, default=2.0,
                     help='MD duration per candidate. Far shorter than the app default (10 ns) purely '
                          'to keep a first pass feasible -- NOTE this means a weak/null result here is '
                          'NOT the same as a confirmed absence of signal: the peptide may need more '
                          'time than this to settle into (or drift out of) its registered pose. Treat a '
                          'promising result at this duration as worth confirming at a longer one on a '
                          'smaller subsample before trusting it fully; treat a null result here as '
                          'inconclusive, not a green light to drop the feature.')
    ap.add_argument('--test-both-conformations', action='store_true', default=True,
                     help='Run each candidate from BOTH a native (AlphaFold isolated-state) and an '
                          'idealized-alpha-helix starting conformation (see '
                          'md_refinement._build_idealized_helix_pdb), keeping whichever scores higher '
                          'plus both raw results -- roughly doubles MD cost per candidate. Default on '
                          'since this is the whole point of this script currently; pass '
                          '--no-test-both-conformations to fall back to the original single-trajectory '
                          'behavior.')
    ap.add_argument('--no-test-both-conformations', dest='test_both_conformations', action='store_false')
    ap.add_argument('--test-specificity-control', action='store_true', default=False,
                     help='For each starting conformation tested, ALSO run a matched control where the '
                          'peptide is anchor-registered against a cyclically-WRONG sub-pocket assignment '
                          '(see NESMDRefiner._place_peptide_via_subpocket_registration scramble_registration '
                          'docstring). Tests whether candidates specifically fit their OWN correct '
                          'registration, vs. just being generically sticky/rigid helices that pack '
                          'reasonably well into any plausible pocket layout -- see the module '
                          'docstring addendum for the full motivation. DOUBLES MD cost on top of whatever '
                          '--test-both-conformations already costs. Default off; combine with '
                          '--no-test-both-conformations for a clean 2x-cost run instead of 4x.')
    ap.add_argument('--cache', default='anchor_occupancy_eval_results.json',
                     help='Where per-example results are checkpointed (resumable)')
    ap.add_argument('--pdb-cache-dir', default='crm1_eval_pdb_cache',
                     help='Reuses the same on-disk PDB cache directory as evaluate_crm1_pocket_signal.py')
    args = ap.parse_args()

    pdb_cache_dir = Path(args.pdb_cache_dir)
    pdb_cache_dir.mkdir(exist_ok=True)

    crm1_ref = next((p for p in CRM1_REF_CANDIDATES if (THIS_DIR / p).exists()), None)
    if not crm1_ref:
        print("No CRM1 reference structure found (checked: "
              f"{CRM1_REF_CANDIDATES}) -- can't run MD docking without one. Aborting.")
        return
    print(f"Using CRM1 reference: {crm1_ref}")
    print(f"MD duration per candidate: {args.duration_ns} ns (app default is 10 ns -- see --duration-ns help)")
    print(f"Testing both starting conformations (native + idealized_helix): {args.test_both_conformations} "
          f"({'~2x' if args.test_both_conformations else '1x'} MD cost per candidate)")
    print(f"Testing specificity control (correct vs. scrambled sub-pocket registration): "
          f"{args.test_specificity_control} (doubles cost again if on)")
    n_variants = (2 if args.test_both_conformations else 1) * (2 if args.test_specificity_control else 1)
    print(f"  -> {n_variants} MD trajector{'y' if n_variants == 1 else 'ies'} per candidate this run")

    refiner = NESMDRefiner(crm1_pdb_path=str(THIS_DIR / crm1_ref))

    results_path = Path(args.cache)
    results = []
    if results_path.exists():
        try:
            with open(results_path) as f:
                results = json.load(f)
            print(f"Resuming: {len(results)} examples already evaluated in {results_path}")
        except (json.JSONDecodeError, OSError) as e:
            backup = results_path.with_suffix(results_path.suffix + '.corrupt')
            print(f"    {results_path} is unreadable ({e}) -- saving it as {backup.name} and "
                  f"starting fresh (downloaded structures in {args.pdb_cache_dir}/ are untouched).")
            results_path.rename(backup)
            results = []
    done_keys = {(r['accession'], r['start'], r['end']) for r in results}

    positives = load_positive_examples(args.limit)
    negatives = load_negative_examples(args.neg_limit)
    examples = positives + negatives
    print(f"\nAttempting {len(examples)} labeled examples "
          f"({len(positives)} positive, {len(negatives)} negative)\n")

    struct_cache = {}
    for i, ex in enumerate(examples, 1):
        key = (ex['accession'], ex['start'], ex['end'])
        if key in done_keys:
            continue
        print(f"[{i}/{len(examples)}] {ex['accession']} {ex['start']}-{ex['end']} (label={ex['label']})")
        t0 = time.time()
        try:
            feats = compute_features(refiner, ex['accession'], ex['start'], ex['end'],
                                      args.duration_ns, struct_cache, pdb_cache_dir,
                                      test_both_conformations=args.test_both_conformations,
                                      test_specificity_control=args.test_specificity_control)
        except Exception as e:
            print(f"    Warning: MD run failed, skipping this example: {e}")
            feats = None
        elapsed = time.time() - t0

        if feats is None:
            print(f"    (skipped -- no structure / gap in resolved span / MD failure, {elapsed:.0f}s)")
            continue

        winner_note = (f"  winner={feats.get('best_starting_conformation')}"
                        if args.test_both_conformations else "")
        print(f"    anchor_occupancy_score={feats['anchor_occupancy_score']}  "
              f"n_anchors_matched={feats['n_anchors_matched']}  "
              f"matched_pockets={feats['matched_pockets']}{winner_note}  ({elapsed:.0f}s)")

        results.append({
            'accession': ex['accession'], 'start': ex['start'], 'end': ex['end'],
            'label': ex['label'],
            'feature_kind': ex.get('feature_kind', 'positive' if ex['label'] == 1 else 'unknown'),
            **feats,
        })
        write_text_atomic_with_retry(results_path, json.dumps(results, indent=2))

    # ---- evaluation ----
    n_pos = sum(1 for r in results if r['label'] == 1)
    n_neg = sum(1 for r in results if r['label'] == 0)
    print(f"\n{'='*70}\nEvaluated {len(results)} real examples "
          f"({n_pos} positive, {n_neg} negative)\n{'='*70}")

    if not results:
        return

    labels_all = np.array([r['label'] for r in results])
    matched = np.array([r['n_anchors_matched'] >= 3 for r in results])
    pos_match_rate = matched[labels_all == 1].mean() if n_pos else float('nan')
    neg_match_rate = matched[labels_all == 0].mean() if n_neg else float('nan')
    print(f"\nPhi-register match rate (>=3 anchors registered to sub-pockets -- see module "
          f"docstring, this is itself a signal the current code doesn't use anywhere):")
    print(f"  positives: {pos_match_rate:.0%}   negatives: {neg_match_rate:.0%}")

    if args.test_both_conformations and any('best_starting_conformation' in r for r in results):
        # Does starting from an idealized helix actually help, and does it
        # help REAL positives specifically more than it inflates hard
        # negatives too (which would mean it's just adding a generic
        # boost, not literature-informed specificity)?
        with_winner = [r for r in results if r.get('best_starting_conformation')]
        win_labels = np.array([r['label'] for r in with_winner])
        winners = np.array([r['best_starting_conformation'] for r in with_winner])
        print(f"\nStarting-conformation winner (idealized_helix vs. native, "
              f"{len(with_winner)}/{len(results)} examples with both attempted):")
        for lbl, name in ((1, 'positives'), (0, 'negatives')):
            subset = winners[win_labels == lbl]
            if len(subset) == 0:
                continue
            helix_wins = (subset == 'idealized_helix').mean()
            print(f"  {name}: idealized_helix won {helix_wins:.0%} of the time ({len(subset)} examples)")

        native_rbs = np.array([r.get('raw_binding_score__native') for r in results
                                if r.get('raw_binding_score__native') is not None])
        native_lbl = np.array([r['label'] for r in results if r.get('raw_binding_score__native') is not None])
        helix_rbs = np.array([r.get('raw_binding_score__idealized_helix') for r in results
                               if r.get('raw_binding_score__idealized_helix') is not None])
        helix_lbl = np.array([r['label'] for r in results if r.get('raw_binding_score__idealized_helix') is not None])
        if len(native_rbs) and len(helix_rbs):
            print(f"\n  mean raw_binding_score, native      : positives={native_rbs[native_lbl==1].mean() if (native_lbl==1).any() else float('nan'):.3f}  "
                  f"negatives={native_rbs[native_lbl==0].mean() if (native_lbl==0).any() else float('nan'):.3f}")
            print(f"  mean raw_binding_score, idealized_helix: positives={helix_rbs[helix_lbl==1].mean() if (helix_lbl==1).any() else float('nan'):.3f}  "
                  f"negatives={helix_rbs[helix_lbl==0].mean() if (helix_lbl==0).any() else float('nan'):.3f}")

    if args.test_specificity_control:
        # The actual specificity test -- does each candidate fit
        # its OWN correct sub-pocket registration meaningfully better than a
        # scrambled one, and does that gap (not the absolute packing score)
        # separate real NES motifs from rigid, generically-sticky decoys?
        # See the module docstring addendum for the full motivation; this is
        # a direct empirical answer, not another guess.
        conf_labels_tested = ('native', 'idealized_helix') if args.test_both_conformations else ('native',)
        print(f"\n{'='*70}\nSpecificity control: correct vs. scrambled sub-pocket registration\n{'='*70}")

        def _mw_auc(pos, neg):
            if len(pos) == 0 or len(neg) == 0:
                return float('nan')
            wins = 0.0
            for p in pos:
                for n in neg:
                    if p > n:
                        wins += 1
                    elif p == n:
                        wins += 0.5
            return wins / (len(pos) * len(neg))

        for conf_label in conf_labels_tested:
            correct_key = f'anchor_occupancy_score__{conf_label}'
            scrambled_key = f'anchor_occupancy_score__{conf_label}_scrambled'
            pairs = [(r['label'], r.get(correct_key), r.get(scrambled_key)) for r in results]
            pairs = [(lbl, c, s) for lbl, c, s in pairs if c is not None and s is not None]
            if not pairs:
                print(f"\n  {conf_label}: no examples with both correct AND scrambled "
                      f"anchor_occupancy_score computed yet")
                continue
            lbl_arr = np.array([p[0] for p in pairs])
            gap_arr = np.array([p[1] - p[2] for p in pairs])
            n_p = int((lbl_arr == 1).sum())
            n_n = int((lbl_arr == 0).sum())
            print(f"\n  {conf_label} (n={len(pairs)}, {n_p} pos / {n_n} neg):")
            print(f"    mean correct anchor_occupancy_score   : positives="
                  f"{np.array([p[1] for p in pairs])[lbl_arr==1].mean() if n_p else float('nan'):.3f}  "
                  f"negatives={np.array([p[1] for p in pairs])[lbl_arr==0].mean() if n_n else float('nan'):.3f}")
            print(f"    mean scrambled anchor_occupancy_score : positives="
                  f"{np.array([p[2] for p in pairs])[lbl_arr==1].mean() if n_p else float('nan'):.3f}  "
                  f"negatives={np.array([p[2] for p in pairs])[lbl_arr==0].mean() if n_n else float('nan'):.3f}")
            print(f"    mean specificity GAP (correct-scrambled): positives="
                  f"{gap_arr[lbl_arr==1].mean() if n_p else float('nan'):.3f}  "
                  f"negatives={gap_arr[lbl_arr==0].mean() if n_n else float('nan'):.3f}")
            if n_p and n_n:
                auc = _mw_auc(gap_arr[lbl_arr == 1], gap_arr[lbl_arr == 0])
                print(f"    specificity-gap AUC (pos gap > neg gap): {auc:.3f}  "
                      f"(0.5=no discrimination; >0.5 means real NES motifs show a BIGGER "
                      f"correct-vs-scrambled gap than decoys do, as the hypothesis predicts; "
                      f"<0.5 means the opposite)")

    scored = [r for r in results if r.get('anchor_occupancy_score') is not None]
    n_s_pos = sum(1 for r in scored if r['label'] == 1)
    n_s_neg = sum(1 for r in scored if r['label'] == 0)
    print(f"\nanchor_occupancy_score computed for {len(scored)}/{len(results)} examples "
          f"({n_s_pos} pos / {n_s_neg} neg -- the rest had no usable Phi register match)")

    if n_s_pos < 5 or n_s_neg < 5:
        print("\nNot enough SCORED examples in both classes yet for a meaningful AUC -- "
              "run again (it resumes) or raise --limit/--neg-limit. The match-rate numbers "
              "above are still real signal even without this.")
        return

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    s_labels = np.array([r['label'] for r in scored])
    occ = np.array([r['anchor_occupancy_score'] for r in scored])
    cv = min(5, n_s_pos, n_s_neg)

    try:
        auc = cross_val_score(LogisticRegression(), occ.reshape(-1, 1), s_labels,
                               cv=cv, scoring='roc_auc').mean()
    except Exception:
        auc = roc_auc_score(s_labels, occ)
    print(f"\nanchor_occupancy_score alone:")
    print(f"  CV AUC = {auc:.3f}")
    print(f"  mean(positive) = {occ[s_labels == 1].mean():.3f}   "
          f"mean(negative) = {occ[s_labels == 0].mean():.3f}")

    # Does anchor occupancy add anything NEW on top of what binding_score's
    # other, already-scored inputs (raw contacts/Cys528-distance product,
    # helix propensity) already capture -- rather than just correlating
    # with them and adding noise for free. Same "standardize before reading
    # coefficients as importance" discipline evaluate_crm1_pocket_signal.py
    # uses, for the same reason (comparing raw magnitudes across features
    # on different scales is invalid).
    raw = np.array([r.get('raw_binding_score') or 0.0 for r in scored])
    helix = np.array([r.get('helix_combined_score') or 0.0 for r in scored])
    X = np.column_stack([raw, helix, occ])
    X_scaled = StandardScaler().fit_transform(X)

    try:
        auc_combined = cross_val_score(LogisticRegression(), X_scaled, s_labels,
                                        cv=cv, scoring='roc_auc').mean()
    except Exception:
        auc_combined = None

    lr = LogisticRegression().fit(X_scaled, s_labels)
    c_raw, c_helix, c_occ = lr.coef_[0]
    total = abs(c_raw) + abs(c_helix) + abs(c_occ)

    print(f"\nCombined with raw_binding_score + helix_combined_score "
          f"(tests whether anchor occupancy adds NEW signal, not just correlated noise):")
    if auc_combined is not None:
        print(f"  CV AUC = {auc_combined:.3f}  (compare to anchor_occupancy_score alone, above)")
    print(f"  Standardized coefficients: raw_binding_score={c_raw:.3f}  "
          f"helix_combined_score={c_helix:.3f}  anchor_occupancy_score={c_occ:.3f}")
    if total > 1e-9:
        print(f"  Data-driven relative weight: raw_binding {abs(c_raw)/total:.0%}  "
              f"helix {abs(c_helix)/total:.0%}  anchor_occupancy {abs(c_occ)/total:.0%}")
        print(f"  Compare anchor_occupancy's data-driven share above to md_refinement.py's "
              f"current GUESSED ANCHOR_OCCUPANCY_WEIGHT = 0.15.")


if __name__ == '__main__':
    main()
