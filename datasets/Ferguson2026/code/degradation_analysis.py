# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    Script that analyses mechanical degradation tests and generates outputs
    from the shoe tests.

"""

# =========================================================================
# Import packages
# =========================================================================

import os
import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection
import numpy as np
from scipy.signal import find_peaks

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Set paths relative to this script so it runs from any working directory
code_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(code_dir, '..', 'data')
checks_path = os.path.join(code_dir, '..', '.checks')

# Analysis settings
# -------------------------------------------------------------------------

# Minimum prominence (mm) of a displacement peak/trough for it to count as a
# cycle turning point. This is a true prominence (i.e. relative to the
# surrounding signal), so it does not depend on sampling rate or loading speed.
# Tests in this dataset cycle over ~7-12 mm, so 2 mm sits well clear of noise
# while still being well below the cycle amplitude.
min_prominence = 2.0

# Tolerance (mm) below a cycle's maximum displacement that is still treated as
# the top of the cycle. Samples within this tolerance at the top of the cycle
# (i.e. the hold) are kept in the bending phase.
hold_tolerance = 0.05

# Number of initial cycles removed to allow for seating of the shoe in the
# fixture clamps and stabilisation of the contact. Number of cycles (i.e. 20)
# was guided by Yoshida et al. (https://doi.org/10.1177/1089313X251411487)
n_seating_cycles = 20

# Number of cycles (immediately after the seating cycles) averaged to give the
# baseline value that peak force and stiffness are normalised to. Averaging
# avoids the whole curve being shifted by noise in a single cycle.
n_baseline_cycles = 10

# Displacement windows (mm) for stiffness calculation during bending
low_stiffness_start = 4.0
low_stiffness_end = 7.0
high_stiffness_start = 10.0
high_stiffness_end = 13.0

# Spacing of cycles plotted on the hysteresis loop figure
hysteresis_cycle_step = 50

# Plot settings
# -------------------------------------------------------------------------

# Set backend to use
# Agg is used as figures are only saved (not displayed), and it works on
# machines without a display
matplotlib.use('Agg')

# Set matplotlib parameters
# Fallback font avoids warnings on systems where Arial is not installed
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['legend.fontsize'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

# =========================================================================
# Define functions
# =========================================================================

# Function to read in a raw test file
# -------------------------------------------------------------------------
def read_test_file(filepath):

    """
    Reads a raw Instron test export into a dataframe with numeric Time,
    Displacement and Force columns.

    Handles both the cleaned format (header on the first line) and the raw
    export format, which can have blank lines before the header, a units row
    (e.g. "(s),(mm),(N)") after it, and quoted values. Rows that cannot be
    converted to numbers (e.g. the units row) are removed.
    """

    # Locate the header row (the first line starting with 'Time')
    header_row = None
    with open(filepath, 'r') as file:
        for line_ind, line in enumerate(file):
            if line.strip().strip('"').startswith('Time'):
                header_row = line_ind
                break
            if line_ind > 20:
                break
    if header_row is None:
        raise ValueError(f'Could not find a "Time" header row in {filepath}')

    # Read in file as dataframe
    test_data = pd.read_csv(filepath, skiprows=header_row, low_memory=False)

    # Convert all columns to numeric and drop any rows that can't be converted
    for col in ['Time', 'Displacement', 'Force']:
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    n_invalid = test_data[['Displacement', 'Force']].isna().any(axis=1).sum()
    if n_invalid > 0:
        print(f'  Removed {n_invalid} non-numeric row(s) (e.g. units row) from {os.path.basename(filepath)}')
        test_data = test_data.dropna(subset=['Displacement', 'Force'])
    test_data = test_data.reset_index(drop=True)

    return test_data

# Function to identify cycles and phases
# -------------------------------------------------------------------------
def identify_cycles(displacement):

    """
    Identifies cycles and the bending/relaxing phase within each cycle from
    the displacement trace.

    A cycle starts at a displacement trough (i.e. the start of bending) and
    runs until the next trough. Cycle 1 runs from the start of the recording
    to the first trough. Within each cycle, the bending phase (1) runs from the
    start of the cycle to the last sample at the top of the cycle (i.e. it
    includes the hold), and the relaxing phase (-1) runs from there to the end
    of the cycle.

    Returns arrays of cycle IDs and phases, along with the ID of the final
    cycle, which is incomplete as it is not closed by a trough.
    """

    # Identify troughs in displacement (start of each new cycle)
    troughs, _ = find_peaks(-displacement, prominence=min_prominence)

    # Assign cycle IDs (cycle 1 = start of data up to first trough)
    sample_inds = np.arange(len(displacement))
    cycle_ids = np.searchsorted(troughs, sample_inds, side='right') + 1

    # Identify phases within each cycle
    phases = np.full(len(displacement), -1, dtype=int)
    boundaries = np.concatenate(([0], troughs, [len(displacement)]))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        cycle_disp = displacement[start:end]
        # Bending continues until the last sample within tolerance of the top
        top_inds = np.where(cycle_disp >= cycle_disp.max() - hold_tolerance)[0]
        phases[start:start + top_inds[-1] + 1] = 1

    # Final cycle is incomplete (no closing trough)
    incomplete_cycle_id = len(troughs) + 1

    return cycle_ids, phases, incomplete_cycle_id

# Function to calculate stiffness within a displacement window
# -------------------------------------------------------------------------
def calculate_window_stiffness(displacement, force, window_start, window_end):

    """
    Calculates stiffness (N/mm) as the slope of a linear fit of force against
    displacement for all samples within the displacement window.

    Using all samples (rather than the two samples nearest the window edges)
    reduces noise and removes the error from samples not landing exactly on
    the window edges. Returns NaN if the data do not span the window.
    """

    # Check the data actually span the window
    # (allowing for up to one sample of spacing at each edge)
    in_window = (displacement >= window_start) & (displacement <= window_end)
    if in_window.sum() < 3:
        return np.nan
    step = np.median(np.abs(np.diff(displacement[in_window]))) if in_window.sum() > 1 else 0.0
    if (displacement[in_window].min() > window_start + step + 1e-9) or \
            (displacement[in_window].max() < window_end - step - 1e-9):
        return np.nan

    # Fit line and return slope
    slope = np.polyfit(displacement[in_window], force[in_window], 1)[0]

    return slope

# Function to read in and extract data from degradation test
# -------------------------------------------------------------------------
def analyse_degradation_test(filename):

    """
    Analyses a single degradation test file and saves check figures to the
    .checks folder.

    Returns:
        test_data: sample-level data (with Cycle_ID and Phase), with seating
            cycles and the final incomplete cycle removed
        peak_force: cycle-level results (peak force, stiffness and their
            normalised values)
    """

    # Read in file
    test_data = read_test_file(os.path.join(data_path, filename))

    # Get file label for naming
    filelabel = os.path.splitext(filename)[0]

    # Create folder in .checks for this sample
    check_dir = os.path.join(checks_path, filelabel)
    os.makedirs(check_dir, exist_ok=True)

    # Identify full cycles (i.e. bending and relaxation)
    # Identify bending and relaxation phases within cycle (1 = bending, -1 = relaxing)
    # Note that the bending phase includes the hold at peak displacement
    # -------------------------------------------------------------------------

    # Identify cycles and phases
    cycle_ids, phases, incomplete_cycle_id = identify_cycles(test_data['Displacement'].to_numpy())

    # Append cycle ID and phase to dataframe
    test_data['Cycle_ID'] = cycle_ids
    test_data['Phase'] = phases

    # Visual check of identification (first 10 cycles)
    # Plot data
    check_data = test_data.loc[test_data['Cycle_ID'] <= 10,]
    x = np.arange(len(check_data))
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(15,8))
    ax.plot(x, check_data['Displacement'].to_numpy(), 'k-', label='Displacement')
    ax.plot(x, (check_data['Force'] / check_data['Force'].max() * check_data['Displacement'].max()).to_numpy(),
            'r-', label='Scaled Force')
    for x_ind in np.where(np.diff(check_data['Cycle_ID'].to_numpy()) != 0)[0]:
        ax.axvline(x=x_ind, color='g', linestyle='--')
    # Collate and visualise phases
    y = np.zeros_like(x)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # reshape data to segments
    colour_map = {1: 'blue', -1: 'pink'}
    c_phases = [colour_map.get(p, 'grey') for p in check_data['Phase'].to_numpy()[:-1]]  # colours to match phases
    lc = LineCollection(segments, colors=c_phases, linewidths=4)  # create line collection
    ax.add_collection(lc)
    # Set limits and labels
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.1, check_data['Displacement'].max() +0.1)
    ax.legend(loc='lower right', framealpha=0.7, facecolor='white', edgecolor='gray')
    ax.set_title(f'{filelabel} Cycle Identification (first 10 cycles)', fontsize=12, fontweight='bold', pad=20)
    # Save to checks folder
    fig.savefig(os.path.join(check_dir,f'{filelabel}_cycle-id.png'),
                format='png',dpi=150)
    # Close figure
    plt.close('all')

    # Calculate peak force across cycles
    # -------------------------------------------------------------------------

    # Drop initial seating cycles from dataset
    # Here we remove the initial cycles to ensure that data analysis was not impacted by the initial setting of the
    # shoe in the fixture clamps and stabilisation of the contact (see n_seating_cycles above)
    # Also drop the final cycle, which is incomplete as the test ends part way through it
    test_data = test_data[(test_data['Cycle_ID'] > n_seating_cycles) &
                          (test_data['Cycle_ID'] < incomplete_cycle_id)]
    if test_data.empty:
        raise ValueError(f'{filename}: no complete cycles remain after removing {n_seating_cycles} seating cycles')

    # Calculate peak force per cycle ID
    peak_force = test_data.groupby('Cycle_ID')['Force'].max().reset_index()

    # Normalise each cycles peak force to the baseline (mean of first cycles after seating)
    baseline_peak_force = peak_force['Force'].iloc[:n_baseline_cycles].mean()
    peak_force['Normalised_Force'] = peak_force['Force'] / baseline_peak_force

    # Visualise normalised force change
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.plot(peak_force['Cycle_ID'], peak_force['Normalised_Force'], 'b-',
            label='Normalised Peak Force', lw = 1.5)
    ax.axhline(y=1.0, color='dimgrey', linestyle='--', lw=1.5)
    # Set limits and labels
    ax.set_xlim(0, test_data['Cycle_ID'].max())
    ax.set_xlabel('Cycle ID', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0.60, 1.10)
    ax.set_ylabel('Normalised Peak Force', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Normalised Peak Force Drop', fontsize=12, fontweight='bold', pad=20)
    # Save to checks folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_normalised-peak-force.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Calculate stiffness across cycles
    # This uses a similar principle to Yoshida et al. (2026), which looks at the change
    # in force between 7.0 and 7.5 mm of displacement during the bending phase
    # divided by the magnitude of displacement (i.e. 0.5 mm) to get N/mm
    # See here for Yoshida et al. paper: https://doi.org/10.1177/1089313X251411487
    # Here the displacement windows are changed to examine different regions of
    # stiffness across the displacement tested (low: 4-7 mm; high: 10-13 mm; see
    # settings above), and stiffness is taken as the slope of a linear fit to all
    # samples within each window.
    # If a test does not span a window (e.g. tests with a 10 mm peak displacement
    # never reach the high window) stiffness is returned as NaN.
    # -------------------------------------------------------------------------

    # Create lists to store calculated stiffness values
    low_stiffness = []
    high_stiffness = []

    # Loop through cycles to calculate stiffness during bending
    bending_data = test_data[test_data['Phase'] == 1]
    for cycle, cycle_data in bending_data.groupby('Cycle_ID'):
        # Get the displacement and force value arrays
        displacement = cycle_data['Displacement'].to_numpy()
        force = cycle_data['Force'].to_numpy()
        # Calculate stiffness for cycle in each window
        low_stiffness.append(calculate_window_stiffness(displacement, force,
                                                        low_stiffness_start, low_stiffness_end))
        high_stiffness.append(calculate_window_stiffness(displacement, force,
                                                         high_stiffness_start, high_stiffness_end))

    # Store stiffness values (N/mm) and relative stiffness between high and low ranges
    low_stiffness = np.asarray(low_stiffness)
    high_stiffness = np.asarray(high_stiffness)
    peak_force['Low_Stiffness'] = low_stiffness
    peak_force['High_Stiffness'] = high_stiffness
    peak_force['Relative_Stiffness'] = high_stiffness / low_stiffness

    # Warn if any stiffness windows could not be calculated
    for label, values, start, end in [('low', low_stiffness, low_stiffness_start, low_stiffness_end),
                                      ('high', high_stiffness, high_stiffness_start, high_stiffness_end)]:
        n_missing = np.isnan(values).sum()
        if n_missing > 0:
            warnings.warn(f'{filelabel}: {label} stiffness window ({start}-{end} mm) not spanned in '
                          f'{n_missing} of {len(values)} cycles; these are set to NaN')

    # Normalise each cycles stiffness to the baseline (mean of first cycles after seating)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # mean of all-NaN baseline
        low_baseline_stiffness = np.nanmean(low_stiffness[:n_baseline_cycles])
        high_baseline_stiffness = np.nanmean(high_stiffness[:n_baseline_cycles])
    peak_force['Low_Normalised_Stiffness'] = low_stiffness / low_baseline_stiffness
    peak_force['High_Normalised_Stiffness'] = high_stiffness / high_baseline_stiffness

    # Visualise normalised stiffness change
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.plot(peak_force['Cycle_ID'], peak_force['Low_Normalised_Stiffness'], 'g-',
            label=f'Normalised Stiffness (Low Range: {low_stiffness_start}-{low_stiffness_end} mm)')
    ax.plot(peak_force['Cycle_ID'], peak_force['High_Normalised_Stiffness'], 'r-',
            label=f'Normalised Stiffness (High Range: {high_stiffness_start}-{high_stiffness_end} mm)')
    ax.axhline(y=1.0, color='dimgrey', linestyle='--', lw=1.5)
    # Set limits and labels
    ax.set_xlim(0, test_data['Cycle_ID'].max())
    ax.set_xlabel('Cycle ID', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0.0, 1.20)
    ax.set_ylabel('Normalised Stiffness', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Normalised Stiffness Drop', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    # Save to checks folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_normalised-stiffness.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Create hysteresis loop across cycles
    # -------------------------------------------------------------------------

    # Visualise hysteresis loop
    # Plots the loop from every hysteresis_cycle_step cycles (plus the final complete cycle)
    # Normalises force to the baseline peak force
    # Note that the hold at peak displacement appears as a vertical line at the
    # peak displacement as force relaxes

    # Identify cycles to plot
    first_cycle = test_data['Cycle_ID'].min()
    last_cycle = test_data['Cycle_ID'].max()
    plot_cycles = list(range(first_cycle, last_cycle + 1, hysteresis_cycle_step))
    if plot_cycles[-1] != last_cycle:
        plot_cycles.append(last_cycle)

    # Create figure to plot loop on
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for cycle in plot_cycles:
        cycle_data = test_data.loc[test_data['Cycle_ID'] == cycle,]
        if cycle == first_cycle:
            # Plot first cycle on top in red
            ax.plot(cycle_data['Displacement'], cycle_data['Force'] / baseline_peak_force,
                    color='red', lw=2.0, ls='-', alpha=1.0, zorder=3
                    )
        else:
            # Plot later cycles below the first cycle, getting darker as cycles progress
            plot_alpha = 0.9 * (cycle - first_cycle) / max(last_cycle - first_cycle, 1)
            ax.plot(cycle_data['Displacement'], cycle_data['Force'] / baseline_peak_force,
                    color='blue', lw=2.0, ls='-', alpha=plot_alpha, zorder=2
                    )
    # Set limits and labels
    ax.set_xlim(-0.1, test_data['Displacement'].max()+0.1)
    ax.set_xlabel('Displacement (mm)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0.0, 1.20)
    ax.set_ylabel('Normalised Force (Baseline Peak Force)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Normalised Hysteresis Loop (every {hysteresis_cycle_step} cycles)',
                 fontsize=12, fontweight='bold', pad=20)
    # Save to checks folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_normalised-hysteresis-loop.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # TODO: quantify area of hysteresis loop...normalised version?
        # > This is the integral of the displacement phase, minus the integral of the relaxation phase
        #   Gives the area of the hysteresis loop

    return test_data, peak_force


# =========================================================================
# Run analyses
# =========================================================================

if __name__ == '__main__':

    # Run degradation analysis on each test file in the data folder
    # -------------------------------------------------------------------------

    # Get list of test files
    test_files = sorted([f for f in os.listdir(data_path) if f.endswith('_test_raw.csv')])

    # Loop through files and store results
    # Results are stored in a dictionary keyed by file label, each containing
    # the sample-level data ('data') and cycle-level results ('peak_force')
    results = {}
    for filename in test_files:
        filelabel = os.path.splitext(filename)[0]
        print(f'Analysing {filename}...')
        test_data, peak_force = analyse_degradation_test(filename)
        results[filelabel] = {'data': test_data, 'peak_force': peak_force}
        print(f'  {len(peak_force)} cycles analysed; final normalised peak force = '
              f'{peak_force["Normalised_Force"].iloc[-1]:.3f}')

    # Just some test code to produce some plots for act'ble
    # -------------------------------------------------------------------------

    # Uncomment to run (requires act_2 and bloch_3 to have been analysed above)
    # from scipy.signal import medfilt
    # act_2_data = results['act_2_test_raw']['data']
    # act_2_peak_force = results['act_2_test_raw']['peak_force']
    # bloch_3_data = results['bloch_3_test_raw']['data']
    # bloch_3_peak_force = results['bloch_3_test_raw']['peak_force']
    act_col = '#333631'
    bloch_col ='#a57463'

    # # Visualise normalised force change
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    # ax.plot(medfilt(act_2_peak_force['Normalised_Force'],kernel_size=19), ls='-', lw = 2.0, c = act_col,
    #         label="Act'Pointe")
    # ax.plot(medfilt(bloch_3_peak_force['Normalised_Force'], kernel_size=19), ls='-', lw=2.0, c=bloch_col,
    #         label='Traditional Pointe')
    # ax.axhline(y=1.0, color='red', linestyle='--', lw=1.0)
    # # Set limits and labels
    # ax.set_xlim(0, 2000)
    # ax.set_xlabel('Displacement Cycle Number', fontsize=12, fontweight='bold', labelpad=10)
    # ax.set_ylim(0.70, 1.10)
    # ax.set_ylabel('Normalised Peak Force (% Starting Value)', fontsize=12, fontweight='bold', labelpad=10)
    # plt.legend(loc='lower right')
    # # Save to checks folder
    # fig.savefig(f'act-v-trad_force-drop.png',
    #             format='png', dpi=300)
    # # Close figure
    # plt.close('all')

    # # Create figure to plot loop on
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    # cycle = 100
    # act_2_first_peak = act_2_peak_force['Force'].iloc[0]
    # bloch_3_first_peak = bloch_3_peak_force['Force'].iloc[0]
    # act_2_cycle_data = act_2_data.loc[act_2_data['Cycle_ID'] == cycle,]
    # bloch_3_cycle_data = bloch_3_data.loc[bloch_3_data['Cycle_ID'] == cycle,]
    # ax.plot(act_2_cycle_data.loc[act_2_cycle_data['Phase']==1]['Displacement'],
    #         act_2_cycle_data.loc[act_2_cycle_data['Phase']==1]['Force'] / act_2_first_peak,
    #         c=act_col, lw=2.0, ls='-', zorder=3,label="Act'Pointe"
    #         )
    # ax.plot(bloch_3_cycle_data.loc[bloch_3_cycle_data['Phase'] == 1]['Displacement'],
    #         bloch_3_cycle_data.loc[bloch_3_cycle_data['Phase'] == 1]['Force'] / bloch_3_first_peak,
    #         c=bloch_col, lw=2.0, ls='-', zorder=3, label = 'Traditional Pointe'
    #         )
    # # Set limits and labels
    # ax.set_xlim(2.0, 14)
    # ax.set_xlabel('Displacement (mm)', fontsize=12, fontweight='bold', labelpad=10)
    # ax.set_ylim(0.0, 1.00)
    # ax.set_ylabel('Normalised Force (% Peak Starting Value)', fontsize=12, fontweight='bold', labelpad=10)
    # plt.legend(loc='lower right')
    # # Save to checks folder
    # fig.savefig(f'act-v-trad_stiffness.png',
    #             format='png', dpi=300)
    # # Close figure
    # plt.close('all')

# %% ---------- end of degradation_analysis.py ---------- %% #
