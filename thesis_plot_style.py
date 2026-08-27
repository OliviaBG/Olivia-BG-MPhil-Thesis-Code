import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAM = {
    'blue': '#8EE8D8', 'light_blue': '#D1F9F1', 'warm_blue': '#00BDB6', 'dark_blue': '#133844',
    'crest': '#FD8153', 'dark_crest': '#DD3025', 'light_crest': '#FFE2C8',
    'cherry': '#CD3572', 'dark_cherry': '#911449', 'light_cherry': '#F2CAD8',
    'purple': '#A368DF', 'dark_purple': '#681FB1', 'light_purple': '#F2ECF8',
    'indigo': '#5366E0', 'dark_indigo': '#29347A', 'light_indigo': '#EBEDFB',
    'green': '#4DB78C', 'dark_green': '#13553A', 'light_green': '#DFF2EA',
    'white': '#FFFFFF',
    'slate1': '#ECEEF1', 'slate2': '#B5BDC8', 'slate3': '#546072', 'slate4': '#232830',
}

AXIS_GREY = CAM['slate3']
GRID_GREY = CAM['slate2']

# A consistent house palette is used across every
# thesis figure instead of the old rainbow CATEGORICAL (dark_blue/crest/
# purple/green/cherry/indigo) -- always Cambridge dark blue, then a light
# blue nudged darker so it's actually visible on white (the raw
# CAM['light_blue'] #D1F9F1 is nearly indistinguishable from white paper),
# then the Cambridge teal, then greys for any further categories. Two new
# keys added rather than overwriting dark_blue/light_blue so anything still
# reading those two directly (e.g. VIOLIN_PAIR below) is unaffected.
CAM['dark_blue_strong'] = '#0E2B34'   # dark_blue, nudged one step darker
CAM['light_blue_strong'] = '#4FA79A'  # light_blue, darkened to be visible (was near-white D1F9F1)

# The house order every multi-series figure (bar clusters, ROC/PR multi-
# line, grouped comparisons, taxonomic groups,...) should pull from, in
# this exact sequence: dark blue, visible light blue, Cambridge teal, then
# greys (dark-to-light so later/lower-priority series don't compete for
# attention with the first three).
CATEGORICAL = [CAM['dark_blue_strong'], CAM['light_blue_strong'], CAM['warm_blue'],
               CAM['slate3'], CAM['slate2'], CAM['slate4']]
# Kept for anything that deliberately still wants the old wider hue spread
# (e.g. a figure with >6 series where blue-monochrome would be unreadable).
CATEGORICAL_LEGACY = [CAM['dark_blue'], CAM['crest'], CAM['purple'], CAM['green'], CAM['cherry'], CAM['indigo']]
VIOLIN_PAIR = [CAM['light_blue_strong'], CAM['dark_blue_strong']]
SEQUENTIAL = [CAM['light_blue'], CAM['blue'], CAM['warm_blue'], CAM['dark_blue']]

from matplotlib.colors import LinearSegmentedColormap
CAM_DIVERGING = LinearSegmentedColormap.from_list(
    'cam_diverging', [CAM['dark_blue'], '#FFFFFF', CAM['crest']])
CAM_DIVERGING.set_bad(CAM['slate1'])

CAM_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    'cam_sequential', [CAM['white'], CAM['light_blue'], CAM['blue'], CAM['warm_blue'], CAM['dark_blue']])


def apply_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'nature'])
    except ImportError:
        plt.rcParams.update({
            'font.family': 'serif', 'font.size': 9, 'axes.linewidth': 0.8,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.major.size': 3, 'ytick.major.size': 3,
            'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'figure.dpi': 150,
        })

    plt.rcParams.update({
        'text.usetex': False,
        'text.color': AXIS_GREY,
        'axes.edgecolor': AXIS_GREY,
        'axes.labelcolor': AXIS_GREY,
        'xtick.color': AXIS_GREY,
        'ytick.color': AXIS_GREY,
        'legend.labelcolor': AXIS_GREY,
        'legend.edgecolor': GRID_GREY,
        'axes.titlecolor': AXIS_GREY,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def style_axes(ax, top_right_spines=False):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(top_right_spines)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS_GREY)
    ax.tick_params(colors=AXIS_GREY)
    ax.xaxis.label.set_color(AXIS_GREY)
    ax.yaxis.label.set_color(AXIS_GREY)
    ax.title.set_color(AXIS_GREY)


def sig_stars(p):
    if p is None:
        return 'ns'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def save_fig(fig, path, dpi=300):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close(fig)
