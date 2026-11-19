# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    Shared figure style for the Ferguson2026 analyses: matplotlib rc parameters,
    colour palette and marker order. Import and call apply_style() before
    creating figures.

"""

# =========================================================================
# Import packages
# =========================================================================

import matplotlib
import matplotlib.pyplot as plt

# =========================================================================
# Style settings
# =========================================================================

# Fonts (first available is used)
FONTS = ['Arial', 'Helvetica', 'DejaVu Sans']

# Matplotlib rc parameters
RC_PARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': FONTS,
    'font.size': 9,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.labelsize': 7,
    'axes.titleweight': 'bold',
    'axes.titlesize': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.8,
    'ytick.minor.size': 1.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'lines.linewidth': 1.6,
    'lines.markersize': 5,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'mathtext.sf': 'Arial',
    'mathtext.default': 'it',
    'savefig.dpi': 300,  # use 600 for JRSS A submission (raster minimum 350; line art 600)
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,  # embed TrueType so PDF text stays editable
}

# Colour palette for categorical or sequential colouring (used in this order)
PALETTE = ['#104485', '#f17f03', '#418c47', '#b63667', '#6b4c9a', '#2a8c8c', '#b5546b', '#7d8a2e']

# Marker shapes, shifted in step with colours (used in this order, then repeated)
MARKERS = ['o', 's', '^', 'D']

# Standard figure widths (inches)
SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.2

# =========================================================================
# Functions
# =========================================================================

def apply_style():

    """Applies the rc parameters (uses the Agg backend, as figures are only saved)."""

    matplotlib.use('Agg')
    plt.rcParams.update(RC_PARAMS)

def colour(index):

    """Colour for the series at this position (repeats after the palette ends)."""

    return PALETTE[index % len(PALETTE)]

def marker(index):

    """Marker for the series at this position (repeats after the four shapes)."""

    return MARKERS[index % len(MARKERS)]

# %% ---------- end of plot_style.py ---------- %% #
