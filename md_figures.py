"""
Figure generation for the CRM1-docking MD refinement pipeline
(md_refinement.py / md_job_queue.py).

_run_crm1_docking() already records everything needed for standard MD
figures in candidate['md_metrics']:
  - minimization_energy_trace            (iteration, energy_kj_mol)
  - production_time_series_ps            (ps)
  - production_energy_trace_kj_mol       (kJ/mol per sample)
  - production_cys528_distance_trace_nm  (nm per sample -- NES-to-groove
                                           approach distance, this system's
                                           equivalent of a reaction coordinate)
  - production_groove_contacts_trace     (count per sample)
  - production_hydrophobic_contacts_trace(count per sample)
  - nes_peptide_rmsf                     (per-residue RMSF, nm)

This module turns that data into saved PNG figures -- never displayed
in-app, always written to disk -- so they can be dropped straight into a
thesis/paper. Call generate_job_figures() once per completed MD job (wired
into md_job_queue.py) or generate_candidate_figures() for a single
candidate. Everything degrades gracefully (prints a warning, returns None)
rather than raising, so a plotting failure never breaks a running MD job.

Figures produced per candidate (in <out_dir>/<candidate_label>/):
  01_minimization_energy.png   - energy vs. minimization iteration
  02_production_energy.png     - potential energy vs. simulation time
  03_cys528_distance.png       - NES-to-Cys528 distance vs. simulation time
  04_contacts_over_time.png    - groove + hydrophobic contact counts vs. time
  05_rmsf_per_residue.png      - per-residue RMSF bar chart

Figures produced once per job (in <out_dir>/):
  00_summary_binding_scores.png - binding score per candidate, colored by
                                   binding category (bar chart)
  00_summary_helix_vs_binding.png - helix propensity vs. binding score
                                     scatter across all candidates
"""

from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # headless -- never try to open a GUI window
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import numpy as np

_STYLE = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

CATEGORY_COLORS = {
    'strong_binder': '#2166AC',
    'moderate_binder': '#4393C3',
    'weak_binder': '#B2182B',
    'predicted_strong_binder': '#2166AC',
    'predicted_moderate_binder': '#4393C3',
    'predicted_weak_binder': '#B2182B',
    'predicted_non_binder': '#7F1D1D',
}
DEFAULT_COLOR = '#888888'


def _candidate_label(candidate):
    seq = candidate.get('sequence', 'unknown')
    start = candidate.get('start', '')
    end = candidate.get('end', '')
    return f"{seq}_{start}-{end}" if start != '' else seq


def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_minimization_energy(md_metrics, out_dir, filename='01_minimization_energy.png'):
    """Energy minimization convergence: shows the starting structure settled
    to a stable, low-energy state rather than the docking pose being
    pathologically clashed."""
    trace = md_metrics.get('minimization_energy_trace')
    if not MATPLOTLIB_AVAILABLE or not trace:
        return None
    iters = [p['iteration'] for p in trace]
    energies = [p['energy_kj_mol'] for p in trace]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.plot(iters, energies, marker='o', markersize=3, color='#2166AC', lw=1.5)
        ax.set_xlabel('Minimization iteration')
        ax.set_ylabel('Potential energy (kJ/mol)')
        ax.set_title('Energy minimization convergence')
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_production_energy(md_metrics, out_dir, filename='02_production_energy.png'):
    """Potential energy vs. simulation time during production MD -- the
    standard equilibration/stability check (should fluctuate around a
    roughly constant mean rather than trending or diverging)."""
    times = md_metrics.get('production_time_series_ps')
    energies = md_metrics.get('production_energy_trace_kj_mol')
    if not MATPLOTLIB_AVAILABLE or not times or not energies:
        return None

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(times, energies, color='#2166AC', lw=1)
        mean_e = float(np.mean(energies))
        ax.axhline(mean_e, color='#B2182B', linestyle='--', lw=1,
                   label=f'Mean = {mean_e:,.0f} kJ/mol')
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('Potential energy (kJ/mol)')
        ax.set_title('Production MD: potential energy vs. time')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_cys528_distance(md_metrics, out_dir, filename='03_cys528_distance.png'):
    """NES-centroid-to-Cys528 distance vs. time -- this system's binding
    'reaction coordinate': a peptide settling into the groove shows this
    distance decreasing then plateauing, rather than drifting or increasing."""
    times = md_metrics.get('production_time_series_ps')
    dists = md_metrics.get('production_cys528_distance_trace_nm')
    if not MATPLOTLIB_AVAILABLE or not times or not dists:
        return None
    # Distance trace can be shorter than times if Cys528 lookup failed on
    # some samples -- align defensively rather than assuming equal length.
    n = min(len(times), len(dists))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(times[:n], dists[:n], color='#2166AC', lw=1)
        avg = float(np.mean(dists[:n]))
        ax.axhline(avg, color='#B2182B', linestyle='--', lw=1, label=f'Mean = {avg:.2f} nm')
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('NES centroid - Cys528 distance (nm)')
        ax.set_title('NES approach to the Cys528 hydrophobic groove')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_contacts_over_time(md_metrics, out_dir, filename='04_contacts_over_time.png'):
    """Groove contact count and hydrophobic contact count vs. time on twin
    y-axes -- shows whether binding contacts form and are maintained
    (rather than transient) over the production trajectory."""
    times = md_metrics.get('production_time_series_ps')
    groove = md_metrics.get('production_groove_contacts_trace')
    hydro = md_metrics.get('production_hydrophobic_contacts_trace')
    if not MATPLOTLIB_AVAILABLE or not times or (not groove and not hydro):
        return None
    n = min(len(times), len(groove) if groove else len(times),
            len(hydro) if hydro else len(times))

    with plt.rc_context(_STYLE):
        fig, ax1 = plt.subplots(figsize=(5.8, 3.8))
        if groove:
            ax1.plot(times[:n], groove[:n], color='#2166AC', lw=1.2, label='Groove contacts')
        ax1.set_xlabel('Simulation time (ps)')
        ax1.set_ylabel('Groove contacts (count)', color='#2166AC')
        ax1.tick_params(axis='y', labelcolor='#2166AC')

        if hydro:
            ax2 = ax1.twinx()
            ax2.plot(times[:n], hydro[:n], color='#B2182B', lw=1.2, label='Hydrophobic contacts')
            ax2.set_ylabel('Hydrophobic contacts (count)', color='#B2182B')
            ax2.tick_params(axis='y', labelcolor='#B2182B')

        ax1.set_title('NES-CRM1 contacts vs. time')
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_rmsf_per_residue(md_metrics, out_dir, filename='05_rmsf_per_residue.png'):
    """Per-residue RMSF (root-mean-square fluctuation) of the NES peptide's
    CA atoms across the production trajectory -- low RMSF at the
    hydrophobic anchor residues alongside higher RMSF at the flexible ends
    is the classic signature of a peptide settling into a stable pose."""
    rmsf = md_metrics.get('nes_peptide_rmsf')
    if not MATPLOTLIB_AVAILABLE or not rmsf:
        return None
    labels = [r['residue'] for r in rmsf]
    values_angstrom = [r['rmsf_nm'] * 10 for r in rmsf]  # nm -> Angstrom, conventional unit

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(5, 0.4 * len(labels)), 3.8))
        ax.bar(range(len(labels)), values_angstrom, color='#2166AC')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('RMSF (Å)')
        ax.set_title('NES peptide per-residue flexibility (RMSF)')
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_rmsd(md_metrics, out_dir, filename='06_peptide_rmsd.png'):
    """Peptide backbone (CA) RMSD vs. time, relative to the first sampled
    production frame -- a converged, low, roughly flat trace indicates a
    stable bound pose; a rising or noisy trace indicates continued
    conformational drift/unfolding."""
    times = md_metrics.get('advanced_time_series_ps')
    rmsd = md_metrics.get('peptide_rmsd_trace_nm')
    if not MATPLOTLIB_AVAILABLE or not times or not rmsd:
        return None
    n = min(len(times), len(rmsd))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(times[:n], [r * 10 for r in rmsd[:n]], color='#2166AC', lw=1.3)  # nm -> Angstrom
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('CA RMSD (Å)')
        ax.set_title('NES peptide RMSD vs. time (Kabsch-aligned)')
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_radius_of_gyration(md_metrics, out_dir, filename='07_radius_of_gyration.png'):
    """Mass-weighted radius of gyration of the NES peptide vs. time --
    compactness/folding indicator, complementary to RMSD."""
    times = md_metrics.get('advanced_time_series_ps')
    rg = md_metrics.get('peptide_radius_of_gyration_trace_nm')
    if not MATPLOTLIB_AVAILABLE or not times or not rg:
        return None
    n = min(len(times), len(rg))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(times[:n], [r * 10 for r in rg[:n]], color='#2166AC', lw=1.3)  # nm -> Angstrom
        mean_rg = float(np.mean(rg[:n])) * 10
        ax.axhline(mean_rg, color='#B2182B', linestyle='--', lw=1, label=f'Mean = {mean_rg:.2f} Å')
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('Radius of gyration (Å)')
        ax.set_title('NES peptide compactness vs. time')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_hydrogen_bonds(md_metrics, out_dir, filename='08_backbone_hbonds.png'):
    """Backbone i->i+3 / i->i+4 hydrogen bond count vs. time -- a direct,
    per-frame readout of alpha-helix/3-10-helix backbone hydrogen bonding,
    companion to the DSSP-based %helix trace."""
    times = md_metrics.get('advanced_time_series_ps')
    hbonds = md_metrics.get('backbone_hbond_count_trace')
    if not MATPLOTLIB_AVAILABLE or not times or not hbonds:
        return None
    n = min(len(times), len(hbonds))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(times[:n], hbonds[:n], color='#2166AC', lw=1.2, drawstyle='steps-post')
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('Backbone H-bond count')
        ax.set_title('Helical backbone hydrogen bonds vs. time')
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_helix_fraction(md_metrics, out_dir, filename='09_dssp_helix_fraction.png'):
    """DSSP-derived fraction of NES peptide residues in a helical (H) state
    vs. time -- directly tests the pipeline's own claim that binding
    requires the NES to fold into a helix, using an actual angle-aware
    secondary-structure assignment rather than just the static sequence
    propensity score used elsewhere."""
    times = md_metrics.get('advanced_time_series_ps')
    frac = md_metrics.get('dssp_helix_fraction_trace')
    if not MATPLOTLIB_AVAILABLE or not times or not frac:
        return None
    n = min(len(times), len(frac))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(times[:n], frac[:n], color='#2166AC', lw=1.3, marker='o', markersize=3)
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('Fraction of residues helical (DSSP)')
        ax.set_ylim(0, 1.05)
        ax.set_title('Secondary structure (% helix) vs. time')
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_sasa(md_metrics, out_dir, filename='10_sasa.png'):
    """Free-peptide total SASA vs. time, plus (if available) the buried
    surface area upon binding as a reference line -- classic
    binding-interface characterization metrics."""
    times = md_metrics.get('advanced_time_series_ps')
    sasa = md_metrics.get('peptide_sasa_trace_nm2')
    if not MATPLOTLIB_AVAILABLE or not times or not sasa:
        return None
    n = min(len(times), len(sasa))
    sasa_a2 = [s * 100 for s in sasa[:n]]  # nm^2 -> Angstrom^2

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(times[:n], sasa_a2, color='#2166AC', lw=1.3, label='Free peptide SASA')
        buried = md_metrics.get('buried_sasa_nm2')
        if buried is not None:
            ax.axhline(buried * 100, color='#B2182B', linestyle='--', lw=1,
                       label=f'Buried SASA upon binding = {buried * 100:.0f} Å²\n'
                             f'(mean, representative bound-state frames)')
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('SASA (Å²)')
        ax.set_title('Peptide solvent-accessible surface area vs. time')
        ax.legend(fontsize=7)
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_mmgbsa_binding_energy(md_metrics, out_dir, filename='11_mmgbsa_binding_energy.png'):
    """System-level MM-GBSA-style binding energy (E_complex - E_peptide -
    E_CRM1, all in the same implicit-solvent forcefield) vs. time. Includes
    the implicit-solvent term (unlike the per-residue decomposition below)
    but excludes conformational entropy -- a standard single-trajectory
    MM-GBSA simplification, not a full free energy of binding."""
    times = md_metrics.get('advanced_time_series_ps')
    energy = md_metrics.get('mmgbsa_binding_energy_trace_kj_mol')
    if not MATPLOTLIB_AVAILABLE or not times or not energy:
        return None
    n = min(len(times), len(energy))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(times[:n], energy[:n], color='#2166AC', lw=1.3)
        mean_e = float(np.mean(energy[:n]))
        ax.axhline(mean_e, color='#B2182B', linestyle='--', lw=1, label=f'Mean = {mean_e:,.0f} kJ/mol')
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('MM-GBSA-style binding energy (kJ/mol)')
        ax.set_title('Binding energy vs. time (no entropy term)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_residue_interaction_energy(md_metrics, out_dir, filename='12_residue_interaction_energy.png'):
    """Per-NES-residue analytic Coulomb+LJ interaction energy with the
    CRM1 groove (averaged over representative late-trajectory frames) --
    a relative ranking of which anchor residues drive the hydrophobic-
    groove contact. Does NOT include implicit-solvent desolvation (see
    docstring in md_refinement._residue_interaction_energy)."""
    energies = md_metrics.get('residue_interaction_energy_kj_mol')
    if not MATPLOTLIB_AVAILABLE or not energies:
        return None

    items = list(energies.items())
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ['#2166AC' if v < 0 else '#B2182B' for v in values]  # negative = favorable/stabilizing

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(labels)), 4))
        ax.bar(range(len(labels)), values, color=colors)
        ax.axhline(0, color='#333333', lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Interaction energy (kJ/mol)\n(negative = favorable)')
        ax.set_title('Per-residue interaction energy with CRM1 groove\n'
                     '(vacuum Coulomb+LJ, representative frames)', fontsize=9)
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_residue_contact_map(md_metrics, out_dir, filename='13_residue_contact_map.png'):
    """Heatmap of contact FREQUENCY (fraction of sampled frames in
    contact) between each NES residue and each CRM1 groove residue --
    shows which specific anchor residues bind which specific groove
    residues, rather than just an aggregate contact count."""
    cmap_data = md_metrics.get('residue_contact_map')
    if not MATPLOTLIB_AVAILABLE or not cmap_data or not cmap_data.get('frequency'):
        return None

    freq = np.array(cmap_data['frequency'])
    nes_labels = cmap_data.get('nes_residues', [str(i) for i in range(freq.shape[0])])
    groove_labels = cmap_data.get('groove_residues', [str(i) for i in range(freq.shape[1])])
    if freq.size == 0:
        return None

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(4, 0.4 * freq.shape[1]) + 1.5, max(3, 0.35 * freq.shape[0]) + 1))
        im = ax.imshow(freq, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(groove_labels)))
        ax.set_xticklabels(groove_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(nes_labels)))
        ax.set_yticklabels(nes_labels, fontsize=8)
        ax.set_xlabel('CRM1 groove residue (index)')
        ax.set_ylabel('NES residue')
        ax.set_title('NES-CRM1 residue contact frequency')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Contact frequency')
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_free_energy_landscape(md_metrics, out_dir, filename='14_free_energy_landscape.png'):
    """
    Approximate 2D free energy landscape, F = -kT ln(P/P_max), over
    (Cys528 distance, peptide RMSD) computed from this SINGLE trajectory's
    sampling. This is a rough, qualitative landscape from one run, not a
    converged potential of mean force -- a rigorous PMF would need
    enhanced sampling (umbrella sampling / metadynamics) or many
    independent replicas. Presented here as a compact 2D summary of which
    (distance, RMSD) region the trajectory actually visited and how
    strongly it was favored, not as a quantitative free energy value.
    """
    dist_full = md_metrics.get('production_cys528_distance_trace_nm')
    rmsd = md_metrics.get('peptide_rmsd_trace_nm')
    if not MATPLOTLIB_AVAILABLE or not dist_full or not rmsd:
        return None

    # production_cys528_distance_trace_nm is sampled at full density while
    # peptide_rmsd_trace_nm is sampled at ADVANCED_ANALYSIS_STRIDE density
    # (see md_refinement.py) -- subsample the distance trace to match
    # rather than importing that constant, so this module stays decoupled
    # from md_refinement's internals.
    stride = max(1, round(len(dist_full) / len(rmsd))) if rmsd else 1
    dist = dist_full[::stride]
    n = min(len(dist), len(rmsd))
    if n < 10:
        return None  # not enough samples for a meaningful 2D histogram

    dist_a = np.array(dist[:n])
    rmsd_a = np.array(rmsd[:n]) * 10  # nm -> Angstrom

    kT = 2.494  # kJ/mol at 300 K
    bins = min(12, max(4, n // 4))
    hist, xedges, yedges = np.histogram2d(dist_a, rmsd_a, bins=bins)
    hist_max = hist.max()
    if hist_max == 0:
        return None
    with np.errstate(divide='ignore'):
        free_energy = np.where(hist > 0, -kT * np.log(hist / hist_max), np.nan)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        mesh = ax.pcolormesh(xedges, yedges, free_energy.T, cmap='viridis_r', shading='auto')
        fig.colorbar(mesh, ax=ax, label='Relative free energy (kJ/mol)')
        ax.set_xlabel('NES centroid - Cys528 distance (nm)')
        ax.set_ylabel('Peptide RMSD (Å)')
        ax.set_title('Approximate free energy landscape\n(single trajectory -- qualitative only)', fontsize=9)
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def generate_candidate_figures(candidate, out_dir):
    """Generate every single-candidate figure for one enhanced NES
    candidate dict (as returned by NESMDRefiner.refine_nes_candidates /
    _run_crm1_docking), saved into out_dir/<candidate_label>/."""
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not available -- skipping MD figure generation")
        return []

    md_metrics = candidate.get('md_metrics', {})
    if not md_metrics:
        return []

    cand_dir = _ensure_dir(Path(out_dir) / _candidate_label(candidate))
    saved = []
    try:
        saved.append(plot_minimization_energy(md_metrics, cand_dir))
        saved.append(plot_production_energy(md_metrics, cand_dir))
        saved.append(plot_cys528_distance(md_metrics, cand_dir))
        saved.append(plot_contacts_over_time(md_metrics, cand_dir))
        saved.append(plot_rmsf_per_residue(md_metrics, cand_dir))
        saved.append(plot_rmsd(md_metrics, cand_dir))
        saved.append(plot_radius_of_gyration(md_metrics, cand_dir))
        saved.append(plot_hydrogen_bonds(md_metrics, cand_dir))
        saved.append(plot_helix_fraction(md_metrics, cand_dir))
        saved.append(plot_sasa(md_metrics, cand_dir))
        saved.append(plot_mmgbsa_binding_energy(md_metrics, cand_dir))
        saved.append(plot_residue_interaction_energy(md_metrics, cand_dir))
        saved.append(plot_residue_contact_map(md_metrics, cand_dir))
        saved.append(plot_free_energy_landscape(md_metrics, cand_dir))
    except Exception as e:
        print(f"  Warning: MD figure generation failed for {_candidate_label(candidate)} "
              f"(job unaffected): {e}")
    return [p for p in saved if p is not None]


def plot_summary_binding_scores(enhanced_candidates, out_dir,
                                 filename='00_summary_binding_scores.png'):
    """Bar chart of binding score per candidate across the whole job,
    colored by binding category -- the MD-pipeline equivalent of a
    model-performance summary panel."""
    if not MATPLOTLIB_AVAILABLE or not enhanced_candidates:
        return None

    labels, scores, colors = [], [], []
    for c in enhanced_candidates:
        m = c.get('md_metrics', {})
        labels.append(_candidate_label(c))
        scores.append(m.get('binding_score', 0.0))
        colors.append(CATEGORY_COLORS.get(m.get('binding_category'), DEFAULT_COLOR))

    order = np.argsort(scores)[::-1]
    labels = [labels[i] for i in order]
    scores = [scores[i] for i in order]
    colors = [colors[i] for i in order]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), 4.2))
        ax.bar(range(len(labels)), scores, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
        ax.set_ylabel('MD binding score')
        ax.set_ylim(0, max(1.0, max(scores) * 1.1 if scores else 1.0))
        ax.set_title('Binding score by candidate (MD refinement)')

        handles = [plt.Rectangle((0, 0), 1, 1, color=col)
                   for col in dict.fromkeys(colors)]
        cat_labels = [k for k, v in CATEGORY_COLORS.items() if v in dict.fromkeys(colors)]
        seen = {}
        for c in enhanced_candidates:
            cat = c.get('md_metrics', {}).get('binding_category')
            if cat and cat not in seen:
                seen[cat] = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        if seen:
            handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in seen.values()]
            ax.legend(handles, list(seen.keys()), fontsize=7, loc='upper right')

        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def plot_summary_helix_vs_binding(enhanced_candidates, out_dir,
                                   filename='00_summary_helix_vs_binding.png'):
    """Scatter of helix-formation score vs. MD binding score across all
    candidates in a job -- visualizes the pipeline's own claim that good
    binding requires good helix formation."""
    if not MATPLOTLIB_AVAILABLE or not enhanced_candidates:
        return None

    xs, ys, colors, labels = [], [], [], []
    for c in enhanced_candidates:
        m = c.get('md_metrics', {})
        if 'helix_combined_score' not in m or 'binding_score' not in m:
            continue
        xs.append(m['helix_combined_score'])
        ys.append(m['binding_score'])
        colors.append(CATEGORY_COLORS.get(m.get('binding_category'), DEFAULT_COLOR))
        labels.append(_candidate_label(c))

    if not xs:
        return None

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(xs, ys, c=colors, s=60, edgecolor='#222222', linewidth=0.5)
        for x, y_, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y_), fontsize=6, xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel('Helix formation score')
        ax.set_ylabel('MD binding score')
        ax.set_title('Helix formation vs. binding score across candidates')
        fig.tight_layout()
        out_path = _ensure_dir(out_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


def generate_job_figures(enhanced_candidates, out_dir, per_candidate=True):
    """
    Generate all figures for a completed MD refinement job: one summary
    figure set across all candidates, plus (optionally) a per-candidate
    figure set for each individual docking trajectory.

    Wired into md_job_queue.py._run_job_thread(), called once the job's
    enhanced_candidates list is final and before/alongside saving the
    job's result JSON. Wrapped by the caller in try/except so a plotting
    failure can never fail an MD job -- the JSON results are unaffected
    either way.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not available -- no MD figures generated")
        return []
    if not enhanced_candidates:
        return []

    out_dir = _ensure_dir(out_dir)
    print(f"  Generating MD figures for {len(enhanced_candidates)} candidate(s)...")

    saved = []
    saved.append(plot_summary_binding_scores(enhanced_candidates, out_dir))
    saved.append(plot_summary_helix_vs_binding(enhanced_candidates, out_dir))

    if per_candidate:
        for candidate in enhanced_candidates:
            saved.extend(generate_candidate_figures(candidate, out_dir))

    saved = [p for p in saved if p is not None]
    print(f"  {len(saved)} MD figure(s) saved to {out_dir}")
    return saved
