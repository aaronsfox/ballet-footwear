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

import io
import os
import gzip
import json
import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection
import numpy as np
from scipy.signal import find_peaks

# Trapezoidal integration (np.trapz was renamed np.trapezoid in numpy 2.0)
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Set paths relative to this script so it runs from any working directory
code_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(code_dir, '..', 'data')
labels_path = os.path.join(code_dir, '..', 'info', 'testing_labels.csv')

# Processed results path
# Each shoe gets its own folder containing its processed data file and the check
# figures for its processing; summary figures across shoes sit in the top level
processed_path = os.path.join(code_dir, '..', 'processed')

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

# Width (degrees of bend) of the angular stiffness windows
# Low-range stiffness is taken over the first stiffness_window_width degrees after
# load onset in each cycle, and high-range stiffness over the last
# stiffness_window_width degrees before peak bend. Referencing the low window to
# load onset keeps it on the loaded part of the curve as shoes retain bend (fixed
# windows end up measuring the unloaded stretch in later cycles).
stiffness_window_width = 3.0

# Load onset detection
# Load onset is where the shoe starts carrying load during bending, taken as the
# first sample where force rises load_onset_threshold (N) above the cycle's
# minimum force before peak. A relative threshold is used as the unloaded force
# reading is not always zero (e.g. ~-2.4 N in bloch_6, ~+0.9 N in bloch-u_3).
# If onset occurs within loaded_at_start_tolerance (mm) of the start of the
# cycle, the shoe was already carrying load at the bottom of the stroke and the
# true onset is below the tested range (flagged as Loaded_At_Cycle_Start).
load_onset_threshold = 0.5
loaded_at_start_tolerance = 0.5

# Load cell zero offset (drift) correction
# The zero offset is the running median (over zero_offset_window cycles) of the
# unloaded force reading in cycles where the shoe unloads. At least
# zero_offset_min_cycles unloaded cycles are needed to estimate it.
zero_offset_window = 51
zero_offset_min_cycles = 50

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
    Calculates stiffness as the slope of a linear fit of force against
    displacement for all samples within the displacement window. Used for both
    linear stiffness (force vs. displacement; N/mm) and angular stiffness
    (moment vs. bend angle; Nm/deg).

    Using all samples (rather than the two samples nearest the window edges)
    reduces noise and removes the error from samples not landing exactly on
    the window edges. Returns NaN if the data do not span the window.
    """

    # Return NaN if no valid data (e.g. bend angle unavailable)
    if np.all(np.isnan(displacement)):
        return np.nan

    # Check the data actually span the window
    # (allowing for up to one sample of spacing at each edge)
    in_window = (displacement >= window_start) & (displacement <= window_end)
    if in_window.sum() < 3:
        return np.nan
    # Largest sample spacing is used as spacing is not constant (e.g. bend angle
    # increments shrink as angle increases)
    step = np.max(np.abs(np.diff(displacement[in_window])))
    if (displacement[in_window].min() > window_start + step + 1e-9) or \
            (displacement[in_window].max() < window_end - step - 1e-9):
        return np.nan

    # Fit line and return slope
    slope = np.polyfit(displacement[in_window], force[in_window], 1)[0]

    return slope

# Function to identify load onset within a cycle
# -------------------------------------------------------------------------
def calculate_load_onset(displacement, force):

    """
    Identifies where the shoe starts carrying load during the bending phase of
    a cycle (see load onset settings above).

    Returns the load onset position as a fractional index within the arrays
    (interpolated between the samples either side of the threshold, as samples
    are ~0.1 deg apart), the minimum (unloaded) force before peak, and whether
    the shoe was already carrying load at the start of the cycle.
    """

    # Find the minimum force before peak force as the unloaded reference
    peak_ind = np.argmax(force)
    min_ind = np.argmin(force[:peak_ind + 1])
    min_force = force[min_ind]

    # Find first sample after the minimum where force exceeds the threshold
    above = np.where(force[min_ind:peak_ind + 1] >= min_force + load_onset_threshold)[0]
    if len(above) == 0:
        return None, min_force, False
    onset_ind = min_ind + above[0]

    # Interpolate between the samples either side of the threshold crossing
    threshold = min_force + load_onset_threshold
    if onset_ind > min_ind and force[onset_ind] > force[onset_ind - 1]:
        fraction = (threshold - force[onset_ind - 1]) / (force[onset_ind] - force[onset_ind - 1])
        onset_pos = onset_ind - 1 + np.clip(fraction, 0.0, 1.0)
    else:
        onset_pos = float(onset_ind)

    # Check whether onset is at the start of the cycle (i.e. already loaded)
    onset_disp = np.interp(onset_pos, np.arange(len(displacement)), displacement)
    loaded_at_start = (onset_disp - displacement[0]) < loaded_at_start_tolerance

    return onset_pos, min_force, loaded_at_start

# Function to calculate bending angle, true force and moment
# -------------------------------------------------------------------------
def calculate_bending_geometry(test_data, lever_arm, applicator_offset):

    """
    Calculates bending angle, true force and bending moment following the
    approach of Oleson et al. (2005; Figs. 15 & 16;
    https://doi.org/10.1016/j.jbiomech.2004.08.014).

    The shoe is assumed to bend about a hinge at the clamp. The applicator acts
    vertically at a fixed horizontal distance (lever_arm; d1 in Oleson et al.)
    from the hinge. Shoes can have a starting bend, so at zero displacement the
    sole under the applicator already sits applicator_offset mm below the
    hinge. The sole angle at any point is therefore:

        angle = atan((applicator_offset + displacement) / lever_arm)

    As the sole tilts, only the component of the vertical (measured) force
    perpendicular to the sole bends it, and the contact point slides out along
    the sole so the true moment arm lengthens:

        true_force = force * cos(angle)
        true_moment_arm = lever_arm / cos(angle)    (d2 in Oleson et al.)
        moment = true_force * true_moment_arm = force * lever_arm

    Note that the cos(angle) terms cancel for the moment, so the moment does
    not depend on the applicator offset.

    Adds the following columns:
        Angle: sole angle relative to horizontal (deg)
        Bend_Angle: angle relative to the starting shape at zero displacement (deg)
        True_Force: force perpendicular to the sole (N)
        Moment: bending moment about the hinge (Nm)

    A missing applicator offset is replaced with zero (and flagged) before this
    function is called (see analyse_degradation_test).
    """

    # Calculate sole angle and angle relative to starting shape
    test_data['Angle'] = np.degrees(np.arctan((applicator_offset + test_data['Displacement']) / lever_arm))
    start_angle = np.degrees(np.arctan(applicator_offset / lever_arm))
    test_data['Bend_Angle'] = test_data['Angle'] - start_angle

    # Calculate true force and moment (lever arm converted from mm to m)
    test_data['True_Force'] = test_data['Force'] * np.cos(np.radians(test_data['Angle']))
    test_data['Moment'] = test_data['Force'] * lever_arm / 1000

    return test_data, start_angle

# Function to read in and extract data from degradation test
# -------------------------------------------------------------------------
def analyse_degradation_test(shoe_info):

    """
    Analyses a single degradation test. Saves the processed data file and check
    figures to the shoe's folder in the processed folder (processed/<id>/).

    Inputs:
        shoe_info: row from testing_labels.csv for this test, with the id
            (raw data file is <id>_test_raw.csv), and the clamp_pos_per,
            applicator_pos_per, shank_length and applicator_offset_mm set-up
            measurements. Clamp and applicator positions are percentages of
            shank length measured from the toe.

    Returns:
        test_data: sample-level data (with Cycle_ID and Phase), with seating
            cycles and the final incomplete cycle removed
        peak_force: cycle-level results (peak force, stiffness and their
            normalised values)
    """

    # Get shoe ID for naming and raw data file
    filelabel = shoe_info['id']
    filename = f'{filelabel}_test_raw.csv'

    # Read in file
    test_data = read_test_file(os.path.join(data_path, filename))

    # Create folder in processed folder for this shoe
    check_dir = os.path.join(processed_path, filelabel)
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

    # Calculate bending angle, true force and moment
    # -------------------------------------------------------------------------

    # Lever arm (mm) is the horizontal distance from the hinge (clamp) to the applicator
    lever_arm = (shoe_info['applicator_pos_per'] - shoe_info['clamp_pos_per']) / 100 * shoe_info['shank_length']

    # Applicator offset (mm)
    # If the offset has not been recorded it is assumed to be zero (i.e. no starting
    # bend) and flagged. Angles, true force and angular stiffness for these shoes will
    # change once the offset is recorded. Normalised peak true force is largely
    # unaffected, as the angle at peak is the same every cycle.
    applicator_offset = float(shoe_info['applicator_offset_mm'])
    offset_assumed = bool(np.isnan(applicator_offset))
    if offset_assumed:
        applicator_offset = 0.0
        warnings.warn(f'{filelabel}: applicator offset not recorded; assuming 0 mm (no starting bend). '
                      f'Angle-based outcomes should be re-processed once the offset is recorded')

    # Calculate geometry
    test_data, start_angle = calculate_bending_geometry(test_data, lever_arm, applicator_offset)
    print(f'  Lever arm = {lever_arm:.1f} mm; starting angle = {start_angle:.1f} deg'
          f'{" (offset assumed 0)" if offset_assumed else ""}; '
          f'bend angle range = {test_data["Bend_Angle"].min():.1f}-{test_data["Bend_Angle"].max():.1f} deg')

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
    # Save to processed folder
    fig.savefig(os.path.join(check_dir,f'{filelabel}_cycle-id.png'),
                format='png',dpi=150)
    # Close figure
    plt.close('all')

    # Drop seating and incomplete cycles
    # -------------------------------------------------------------------------

    # Drop initial seating cycles from dataset
    # Here we remove the initial cycles to ensure that data analysis was not impacted by the initial setting of the
    # shoe in the fixture clamps and stabilisation of the contact (see n_seating_cycles above)
    # Also drop the final cycle, which is incomplete as the test ends part way through it
    test_data = test_data[(test_data['Cycle_ID'] > n_seating_cycles) &
                          (test_data['Cycle_ID'] < incomplete_cycle_id)]
    if test_data.empty:
        raise ValueError(f'{filename}: no complete cycles remain after removing {n_seating_cycles} seating cycles')

    # Calculate cycle-level outcomes
    # For each cycle this calculates:
    #   Peak values: peak measured force, true force and moment
    #   Load onset: where the shoe starts carrying load during bending (see load
    #       onset settings above). As bend angle is measured from the shoe's original
    #       starting shape, increases in load onset angle indicate the shoe is not
    #       returning to its starting shape (i.e. retained bend/deformation)
    #   Angular stiffness: slope of moment against bend angle (Nm/deg; following
    #       Oleson et al., 2005) over two windows on the loading curve:
    #           Low range: the first stiffness_window_width degrees after load onset
    #           High range: the last stiffness_window_width degrees before peak bend
    #       Referencing the low window to load onset means it always sits on the
    #       loaded part of the curve, so stiffness is kept separate from the
    #       deformation captured by load onset
    #   Energy: work done by the applicator during bending (input) and returned by
    #       the shoe during relaxation (returned), from the area under the measured
    #       force-displacement curve. Energy lost is the hysteresis loop area.
    #   Hold relaxation: drop in force during the hold at peak displacement
    # -------------------------------------------------------------------------

    # Create list to store cycle outcomes
    cycle_outcomes = []

    # Loop through cycles
    for cycle, cycle_data in test_data.groupby('Cycle_ID'):

        # Get cycle arrays
        displacement = cycle_data['Displacement'].to_numpy()
        force = cycle_data['Force'].to_numpy()
        bend_angle = cycle_data['Bend_Angle'].to_numpy()
        angle = cycle_data['Angle'].to_numpy()
        moment = cycle_data['Moment'].to_numpy()
        phase = cycle_data['Phase'].to_numpy()

        # Identify bending (incl. hold) and loading (bending up to first reaching
        # peak displacement, i.e. excluding the hold) portions of the cycle
        bend_end = np.where(phase == 1)[0][-1]
        load_end = np.argmax(displacement[:bend_end + 1] >= displacement[:bend_end + 1].max() - hold_tolerance)

        # Peak values
        peak_ind = np.argmax(force)
        outcomes = {
            'Cycle_ID': cycle,
            'Force': force[peak_ind],
            'True_Force': cycle_data['True_Force'].max(),
            'Moment': moment[peak_ind],
            'Peak_Force_Angle': angle[peak_ind],
            'Peak_Bend_Angle': bend_angle[load_end],
        }

        # Load onset
        onset_pos, min_force, loaded_at_start = calculate_load_onset(displacement[:bend_end + 1],
                                                                     force[:bend_end + 1])
        sample_inds = np.arange(bend_end + 1)
        outcomes['Unloaded_Force'] = min_force
        outcomes['Loaded_At_Cycle_Start'] = loaded_at_start
        if onset_pos is None:
            outcomes['Load_Onset_Displacement'] = np.nan
            outcomes['Load_Onset_Bend_Angle'] = np.nan
        else:
            outcomes['Load_Onset_Displacement'] = np.interp(onset_pos, sample_inds, displacement[:bend_end + 1])
            outcomes['Load_Onset_Bend_Angle'] = np.interp(onset_pos, sample_inds, bend_angle[:bend_end + 1])

        # Angular stiffness windows (referenced to load onset and peak bend)
        low_start = outcomes['Load_Onset_Bend_Angle']
        high_end = outcomes['Peak_Bend_Angle']
        windows = {'Low': (low_start, low_start + stiffness_window_width),
                   'High': (high_end - stiffness_window_width, high_end)}
        for label, (start, end) in windows.items():
            outcomes[f'{label}_Window_Start'] = start
            outcomes[f'{label}_Window_End'] = end
            outcomes[f'{label}_Angular_Stiffness'] = np.nan if np.isnan(start) else \
                calculate_window_stiffness(bend_angle[:load_end + 1], moment[:load_end + 1], start, end)

        # Energy input during bending and returned during relaxation (N.mm = mJ)
        # Raw values are corrected for load cell zero offset after the loop
        outcomes['Energy_Input_Raw'] = _trapezoid(force[:bend_end + 1], displacement[:bend_end + 1])
        outcomes['Energy_Returned_Raw'] = -_trapezoid(force[bend_end:], displacement[bend_end:])
        outcomes['Displacement_Input'] = displacement[bend_end] - displacement[0]
        outcomes['Displacement_Returned'] = displacement[bend_end] - displacement[-1]

        # Force at the end of the hold (for hold relaxation)
        outcomes['Hold_End_Force'] = force[bend_end]

        # Append cycle outcomes
        cycle_outcomes.append(outcomes)

    # Convert to dataframe
    # (named peak_force for consistency with earlier versions of this script)
    peak_force = pd.DataFrame(cycle_outcomes)

    # Correct for load cell zero offset (drift)
    # When the shoe unloads at the bottom of the stroke (i.e. not loaded at cycle
    # start), the minimum force before peak is the load cell reading with no load,
    # which is not always zero and can drift over a test. The zero offset is taken
    # as a running median of these unloaded readings across cycles, interpolated
    # over cycles where the shoe stayed loaded and held constant beyond the first
    # and last unloaded cycles. Shoes that never unload (i.e. carry load at the
    # bottom of every cycle) have no unloaded reading, so no correction is possible
    # and the offset is set to zero (flagged as Zero_Offset_Estimated = False).
    # Stiffness slopes, load onset and energy lost are unaffected by a constant offset.
    # -------------------------------------------------------------------------
    unloaded = peak_force['Unloaded_Force'].where(~peak_force['Loaded_At_Cycle_Start'].astype(bool))
    if unloaded.notna().sum() >= zero_offset_min_cycles:
        smoothed = unloaded.dropna().rolling(zero_offset_window, center=True, min_periods=1).median()
        zero_offset = np.interp(peak_force['Cycle_ID'], peak_force.loc[smoothed.index, 'Cycle_ID'], smoothed)
        zero_offset_estimated = True
        print(f'  Load cell zero offset estimated from {unloaded.notna().sum()} unloaded cycles: '
              f'{zero_offset[0]:.2f} N (start) to {zero_offset[-1]:.2f} N (end)')
    else:
        zero_offset = np.zeros(len(peak_force))
        zero_offset_estimated = False
        print(f'  Shoe does not unload within the stroke; no zero offset correction applied')
    peak_force['Zero_Offset'] = zero_offset
    peak_force['Zero_Offset_Estimated'] = zero_offset_estimated

    # Corrected peak values
    # True force is corrected at the angle of peak measured force
    peak_force['Force_Corrected'] = peak_force['Force'] - peak_force['Zero_Offset']
    peak_force['True_Force_Corrected'] = peak_force['Force_Corrected'] * np.cos(np.radians(peak_force['Peak_Force_Angle']))
    peak_force['Moment_Corrected'] = peak_force['Force_Corrected'] * lever_arm / 1000

    # Drift-free force range (peak minus minimum force before peak in the cycle)
    # This is unaffected by any zero offset, so is used to check whether changes in
    # peak force in shoes that cannot be corrected are real or due to drift
    peak_force['Force_Range'] = peak_force['Force'] - peak_force['Unloaded_Force']

    # Corrected energy (mJ) and energy return
    peak_force['Energy_Input'] = peak_force['Energy_Input_Raw'] - peak_force['Zero_Offset'] * peak_force['Displacement_Input']
    peak_force['Energy_Returned'] = peak_force['Energy_Returned_Raw'] - peak_force['Zero_Offset'] * peak_force['Displacement_Returned']
    peak_force['Energy_Lost'] = peak_force['Energy_Input'] - peak_force['Energy_Returned']
    peak_force['Energy_Return_Percent'] = 100 * peak_force['Energy_Returned'] / peak_force['Energy_Input']

    # Hold relaxation (% of corrected peak force lost during the hold)
    peak_force['Hold_Relaxation_Percent'] = 100 * (peak_force['Force'] - peak_force['Hold_End_Force']) / peak_force['Force_Corrected']

    # Relative stiffness between high and low ranges
    peak_force['Relative_Angular_Stiffness'] = peak_force['High_Angular_Stiffness'] / peak_force['Low_Angular_Stiffness']

    # Warn if any stiffness windows could not be calculated
    for label in ['Low', 'High']:
        n_missing = peak_force[f'{label}_Angular_Stiffness'].isna().sum()
        if n_missing > 0:
            warnings.warn(f'{filelabel}: {label.lower()} angular stiffness could not be calculated in '
                          f'{n_missing} of {len(peak_force)} cycles; these are set to NaN')

    # Normalise to the baseline (mean of first cycles after seating)
    # The primary outcome is normalised peak true force (corrected for zero offset)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # mean of all-NaN baseline
        for var in ['Force', 'True_Force', 'Moment', 'Force_Corrected', 'True_Force_Corrected', 'Moment_Corrected',
                    'Force_Range', 'Low_Angular_Stiffness', 'High_Angular_Stiffness', 'Energy_Input',
                    'Energy_Returned', 'Energy_Lost']:
            baseline = np.nanmean(peak_force[var].iloc[:n_baseline_cycles])
            peak_force[f'Normalised_{var}'] = peak_force[var] / baseline

    # Report proportion of cycles where the shoe was already loaded at the start of the cycle
    n_loaded_at_start = int(peak_force['Loaded_At_Cycle_Start'].sum())
    if n_loaded_at_start > 0:
        print(f'  Loaded at cycle start (load onset below tested range) in {n_loaded_at_start} '
              f'of {len(peak_force)} cycles')

    # Visualise normalised peak force change (raw, corrected and drift-free)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.plot(peak_force['Cycle_ID'], peak_force['Normalised_True_Force'], color='grey', lw=1.0,
            label='Normalised Peak True Force (Raw)')
    ax.plot(peak_force['Cycle_ID'], peak_force['Normalised_True_Force_Corrected'], color='blue', lw=1.5,
            label='Normalised Peak True Force (Zero Offset Corrected)'
            if zero_offset_estimated else 'Normalised Peak True Force (No Correction Possible)')
    ax.plot(peak_force['Cycle_ID'], peak_force['Normalised_Force_Range'], color='orange', lw=1.0, ls='--',
            label='Normalised Force Range (Peak - Cycle Minimum; Drift-Free)')
    ax.axhline(y=1.0, color='dimgrey', linestyle='--', lw=1.5)
    ax.legend(loc='lower left')
    # Set limits and labels
    ax.set_xlim(0, test_data['Cycle_ID'].max())
    ax.set_xlabel('Cycle ID', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0.50, 1.20)
    ax.set_ylabel('Normalised Value', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Normalised Peak Force', fontsize=12, fontweight='bold', pad=20)
    # Save to processed folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_normalised-peak-force.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Visualise load onset change
    # Cycles where the shoe was already loaded at the start are shown in grey, as
    # the true onset is below the tested range for these
    # -------------------------------------------------------------------------
    loaded_mask = peak_force['Loaded_At_Cycle_Start'].to_numpy().astype(bool)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.scatter(peak_force.loc[~loaded_mask, 'Cycle_ID'], peak_force.loc[~loaded_mask, 'Load_Onset_Bend_Angle'],
               s=4, color='purple', label='Load Onset')
    ax.scatter(peak_force.loc[loaded_mask, 'Cycle_ID'], peak_force.loc[loaded_mask, 'Load_Onset_Bend_Angle'],
               s=4, color='grey', label='Loaded at Cycle Start (onset below tested range)')
    # Set limits and labels
    ax.set_xlim(0, test_data['Cycle_ID'].max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Cycle ID', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Load Onset Bend Angle from Starting Shape (deg)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Load Onset{" (offset assumed 0)" if offset_assumed else ""}',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='lower right', markerscale=4)
    # Save to processed folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_load-onset.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Visualise normalised stiffness change
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.plot(peak_force['Cycle_ID'], peak_force['Normalised_Low_Angular_Stiffness'], 'g-',
            label=f'Low Range (first {stiffness_window_width:g} deg after load onset)')
    ax.plot(peak_force['Cycle_ID'], peak_force['Normalised_High_Angular_Stiffness'], 'r-',
            label=f'High Range (last {stiffness_window_width:g} deg before peak bend)')
    ax.axhline(y=1.0, color='dimgrey', linestyle='--', lw=1.5)
    # Set limits and labels
    ax.set_xlim(0, test_data['Cycle_ID'].max())
    ax.set_ylim(0.0, 1.50)
    ax.set_xlabel('Cycle ID', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Normalised Angular Stiffness', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Normalised Angular Stiffness', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    # Save to processed folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_normalised-stiffness.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Visualise energy return and loss
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
    axes[0].plot(peak_force['Cycle_ID'], peak_force['Energy_Input'], color='blue', label='Energy Input (Bending)')
    axes[0].plot(peak_force['Cycle_ID'], peak_force['Energy_Returned'], color='green', label='Energy Returned (Relaxation)')
    axes[0].plot(peak_force['Cycle_ID'], peak_force['Energy_Lost'], color='red', label='Energy Lost (Hysteresis)')
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel('Energy (mJ)', fontsize=12, fontweight='bold', labelpad=10)
    axes[0].legend(loc='upper right')
    axes[1].plot(peak_force['Cycle_ID'], peak_force['Energy_Return_Percent'], color='black')
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Energy Returned (% of Input)', fontsize=12, fontweight='bold', labelpad=10)
    axes[1].set_xlim(0, test_data['Cycle_ID'].max())
    axes[1].set_xlabel('Cycle ID', fontsize=12, fontweight='bold', labelpad=10)
    axes[0].set_title(f'{filelabel} Energy Return and Loss', fontsize=12, fontweight='bold', pad=20)
    # Save to processed folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_energy.png'),
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
    baseline_peak_force = peak_force['Force'].iloc[:n_baseline_cycles].mean()
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
    # Save to processed folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_normalised-hysteresis-loop.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Visualise moment vs. bend angle loop
    # Uses the same cycles as above. The stiffness windows for the first and last
    # cycles are highlighted on the loading curve to check where they sit
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for cycle in plot_cycles:
        cycle_data = test_data.loc[test_data['Cycle_ID'] == cycle,]
        if cycle == first_cycle:
            ax.plot(cycle_data['Bend_Angle'], cycle_data['Moment'],
                    color='red', lw=2.0, ls='-', alpha=1.0, zorder=3, label=f'Cycle {cycle}')
        else:
            plot_alpha = 0.9 * (cycle - first_cycle) / max(last_cycle - first_cycle, 1)
            ax.plot(cycle_data['Bend_Angle'], cycle_data['Moment'],
                    color='blue', lw=2.0, ls='-', alpha=plot_alpha, zorder=2,
                    label=f'Cycle {cycle}' if cycle == last_cycle else None)
    for cycle in [first_cycle, last_cycle]:
        cycle_data = test_data.loc[(test_data['Cycle_ID'] == cycle) & (test_data['Phase'] == 1),]
        row = peak_force.loc[peak_force['Cycle_ID'] == cycle].iloc[0]
        for label, colour in [('Low', 'green'), ('High', 'orange')]:
            in_window = cycle_data['Bend_Angle'].between(row[f'{label}_Window_Start'], row[f'{label}_Window_End'])
            ax.plot(cycle_data.loc[in_window, 'Bend_Angle'], cycle_data.loc[in_window, 'Moment'],
                    color=colour, lw=5.0, alpha=0.6, zorder=4,
                    label=f'{label} Stiffness Window' if cycle == first_cycle else None)
    # Set limits and labels
    ax.set_xlim(0, test_data['Bend_Angle'].max() + 0.5)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel('Bend Angle from Starting Shape (deg)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Moment (Nm)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Moment vs. Bend Angle (every {hysteresis_cycle_step} cycles; '
                 f'starting angle {start_angle:.1f} deg{"; offset assumed 0" if offset_assumed else ""})',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper left')
    # Save to processed folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_moment-bend-angle.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Save processed data
    # -------------------------------------------------------------------------
    setup = {'lever_arm_mm': lever_arm,
             'applicator_offset_mm_used': applicator_offset,
             'applicator_offset_assumed': offset_assumed,
             'starting_angle_deg': start_angle,
             'zero_offset_estimated': zero_offset_estimated,
             'n_cycles_recorded': int(incomplete_cycle_id)}
    save_processed_data(os.path.join(check_dir, f'{filelabel}_processed.json.gz'),
                        shoe_info, setup, test_data, peak_force)

    return test_data, peak_force

# Functions to save and load processed data
# -------------------------------------------------------------------------

# Sample-level variables saved per cycle, with units and rounding (decimal places)
# Rounding matches the precision of the raw data (time, displacement, force) and
# keeps derived values well below measurement resolution
processed_sample_variables = {
    'Time': ('s', 4),
    'Displacement': ('mm', 4),
    'Force': ('N', 4),
    'True_Force': ('N', 4),
    'Angle': ('deg', 4),
    'Bend_Angle': ('deg', 4),
    'Moment': ('Nm', 5),
}

# Units of cycle-level results (normalised values and ratios are unitless)
processed_cycle_units = {
    'Force': 'N', 'True_Force': 'N', 'Moment': 'Nm',
    'Force_Corrected': 'N', 'True_Force_Corrected': 'N', 'Moment_Corrected': 'Nm',
    'Zero_Offset': 'N', 'Force_Range': 'N', 'Unloaded_Force': 'N', 'Hold_End_Force': 'N',
    'Peak_Force_Angle': 'deg', 'Peak_Bend_Angle': 'deg',
    'Load_Onset_Displacement': 'mm', 'Load_Onset_Bend_Angle': 'deg',
    'Low_Angular_Stiffness': 'Nm/deg', 'High_Angular_Stiffness': 'Nm/deg',
    'Low_Window_Start': 'deg', 'Low_Window_End': 'deg', 'High_Window_Start': 'deg', 'High_Window_End': 'deg',
    'Energy_Input': 'mJ', 'Energy_Returned': 'mJ', 'Energy_Lost': 'mJ',
    'Energy_Input_Raw': 'mJ', 'Energy_Returned_Raw': 'mJ',
    'Displacement_Input': 'mm', 'Displacement_Returned': 'mm',
    'Energy_Return_Percent': '%', 'Hold_Relaxation_Percent': '%',
}

def _to_json_value(value):

    """Converts numpy/pandas values to JSON-compatible values (NaN -> None)."""

    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value

def save_processed_data(filepath, shoe_info, setup, test_data, peak_force):

    """
    Saves the processed data for a shoe as a single gzip-compressed JSON file.

    The file contains:
        id, source_file
        shoe: the shoe's row from testing_labels.csv
        setup: derived set-up values (lever arm, starting angle, cycles recorded)
        settings: analysis settings used
        units: units of sample-level variables and cycle-level results
        cycles: list of analysed cycles, each with
            cycle_id
            results: cycle-level results (peak values, stiffness, load onset and
                normalised values)
            data: sample-level arrays (time, displacement, raw vertical force,
                true force, angle, bend angle, moment and phase)

    Missing values (NaN) are saved as null. Seating cycles and the final
    incomplete cycle are not included. Use load_processed_data() to read the
    file back into dataframes.
    """

    # Collate cycle-level results and sample-level data per cycle
    results_by_cycle = peak_force.set_index('Cycle_ID')
    cycles = []
    for cycle, cycle_data in test_data.groupby('Cycle_ID'):
        data = {}
        for var, (_, decimals) in processed_sample_variables.items():
            values = cycle_data[var].round(decimals).to_numpy()
            data[var] = [None if np.isnan(v) else float(v) for v in values] \
                if np.isnan(values).any() else values.tolist()
        data['Phase'] = cycle_data['Phase'].astype(int).tolist()
        cycles.append({
            'cycle_id': int(cycle),
            'results': {key: _to_json_value(val) for key, val in results_by_cycle.loc[cycle].items()},
            'data': data,
        })

    # Collate file contents
    processed = {
        'id': shoe_info['id'],
        'source_file': f'{shoe_info["id"]}_test_raw.csv',
        'shoe': {key: _to_json_value(val) for key, val in shoe_info.items()},
        'setup': {key: _to_json_value(val) for key, val in setup.items()},
        'settings': {
            'min_prominence_mm': min_prominence,
            'hold_tolerance_mm': hold_tolerance,
            'n_seating_cycles': n_seating_cycles,
            'n_baseline_cycles': n_baseline_cycles,
            'stiffness_window_width_deg': stiffness_window_width,
            'stiffness_windows': 'low = first window width after load onset; high = last window width before peak bend',
            'load_onset_threshold_N': load_onset_threshold,
            'loaded_at_start_tolerance_mm': loaded_at_start_tolerance,
            'zero_offset_window_cycles': zero_offset_window,
            'zero_offset_min_cycles': zero_offset_min_cycles,
        },
        'units': {
            'data': {var: units for var, (units, _) in processed_sample_variables.items()} |
                    {'Phase': '1 = bending (incl. hold), -1 = relaxing'},
            'results': processed_cycle_units,
        },
        'cycles': cycles,
    }

    # Save as compressed JSON, overwriting any existing file
    # The gzip timestamp is fixed (mtime=0) and no run timestamp is stored, so
    # re-running with unchanged results produces an identical file (i.e. git
    # only sees a change when the processed results actually change)
    with open(filepath, 'wb') as raw_file:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw_file, compresslevel=6, mtime=0) as gz_file:
            with io.TextIOWrapper(gz_file, encoding='utf-8') as file:
                json.dump(processed, file, separators=(',', ':'))

def load_processed_data(shoe_id):

    """
    Loads a shoe's processed data file (see save_processed_data()).

    Returns:
        info: dictionary of file metadata (id, shoe, setup, settings, units)
        sample_data: dataframe of sample-level data with Cycle_ID
        cycle_results: dataframe of cycle-level results with Cycle_ID
    """

    # Read file
    filepath = os.path.join(processed_path, shoe_id, f'{shoe_id}_processed.json.gz')
    with gzip.open(filepath, 'rt', encoding='utf-8') as file:
        processed = json.load(file)

    # Build sample-level and cycle-level dataframes
    sample_data = pd.concat([pd.DataFrame(cycle['data']).assign(Cycle_ID=cycle['cycle_id'])
                             for cycle in processed['cycles']], ignore_index=True)
    cycle_results = pd.DataFrame([{'Cycle_ID': cycle['cycle_id']} | cycle['results']
                                  for cycle in processed['cycles']])
    info = {key: val for key, val in processed.items() if key != 'cycles'}

    return info, sample_data, cycle_results


# =========================================================================
# Run analyses
# =========================================================================

if __name__ == '__main__':

    # Run degradation analysis on each test file in the data folder
    # -------------------------------------------------------------------------

    # Read in testing labels (set-up measurements and inclusion)
    # File is cp1252 encoded (e.g. the apostrophe in Act'Pointe)
    testing_labels = pd.read_csv(labels_path, encoding='cp1252')

    # Get tests marked for inclusion
    included_tests = testing_labels.loc[testing_labels['include'] == 'Y']
    excluded_ids = testing_labels.loc[testing_labels['include'] != 'Y', 'id'].tolist()
    print(f'Analysing {len(included_tests)} included tests (excluded: {", ".join(excluded_ids)})')

    # Loop through included tests and store results
    # Results are stored in a dictionary keyed by shoe ID, each containing
    # the sample-level data ('data') and cycle-level results ('peak_force')
    # Processed data and check figures for each shoe are saved to processed/<id>/
    results = {}
    for _, shoe_info in included_tests.iterrows():
        filename = f'{shoe_info["id"]}_test_raw.csv'
        if not os.path.exists(os.path.join(data_path, filename)):
            warnings.warn(f'{filename} not found in data folder; skipping')
            continue
        print(f'Analysing {filename}...')
        test_data, peak_force = analyse_degradation_test(shoe_info)
        results[shoe_info['id']] = {'data': test_data, 'peak_force': peak_force}
        print(f"  {len(peak_force)} cycles analysed; final normalised peak true force (corrected) = "
              f"{peak_force['Normalised_True_Force_Corrected'].iloc[-1]:.3f}")

    # Summary check of moment vs. bend angle across shoes
    # -------------------------------------------------------------------------

    # Plots the bending phase of the first (solid) and last (dashed) analysed cycle
    # for each shoe with a recorded applicator offset, with the bend angle stiffness
    # windows shaded. Used to check the windows are covered by all shoes and sit
    # on usable parts of the curves.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    colours = plt.get_cmap('tab20')
    for ind, (filelabel, result) in enumerate(results.items()):
        test_data = result['data']
        if test_data['Bend_Angle'].isna().all():
            continue
        colour = colours(ind % 20)
        bending_data = test_data.loc[test_data['Phase'] == 1]
        for cycle, line_style in zip([bending_data['Cycle_ID'].min(), bending_data['Cycle_ID'].max()], ['-', '--']):
            cycle_data = bending_data.loc[bending_data['Cycle_ID'] == cycle]
            ax.plot(cycle_data['Bend_Angle'], cycle_data['Moment'], color=colour, ls=line_style, lw=1.5,
                    label=filelabel.replace('_test_raw', '') if line_style == '-' else None)
    # Set limits and labels
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Bend Angle from Starting Shape (deg)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Moment (Nm)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Moment vs. Bend Angle During Bending (solid = first analysed cycle; dashed = last cycle)',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper left', ncol=2)
    # Save to processed folder
    fig.savefig(os.path.join(processed_path, 'summary_moment-bend-angle.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Summary check of load onset across shoes
    # -------------------------------------------------------------------------

    # Plots load onset bend angle across cycles for each shoe with a recorded
    # applicator offset. Cycles where the shoe was already loaded at the start of
    # the cycle are not shown, as the true onset is below the tested range.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for ind, (filelabel, result) in enumerate(results.items()):
        peak_force = result['peak_force']
        if peak_force['Load_Onset_Bend_Angle'].isna().all():
            continue
        onset = peak_force['Load_Onset_Bend_Angle'].where(~peak_force['Loaded_At_Cycle_Start'])
        pct_loaded = 100 * peak_force['Loaded_At_Cycle_Start'].mean()
        label = filelabel.replace('_test_raw', '')
        if pct_loaded > 0:
            label += f' ({pct_loaded:.0f}% cycles loaded at start)'
        ax.plot(peak_force['Cycle_ID'], onset, color=colours(ind % 20), lw=1.5, label=label)
    # Set limits and labels
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Cycle ID', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Load Onset Bend Angle from Starting Shape (deg)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Load Onset Across Cycles', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper left', ncol=2, fontsize=8)
    # Save to processed folder
    fig.savefig(os.path.join(processed_path, 'summary_load-onset.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Print summary of load onset across shoes
    print('\nLoad onset summary (bend angle from starting shape; first/last analysed cycle):')
    for filelabel, result in results.items():
        peak_force = result['peak_force']
        first, last = peak_force.iloc[0], peak_force.iloc[-1]
        fmt = lambda row: ('loaded at start' if row['Loaded_At_Cycle_Start']
                           else f'{row["Load_Onset_Bend_Angle"]:.1f} deg / {row["Load_Onset_Displacement"]:.1f} mm')
        print(f'  {filelabel.replace("_test_raw", ""):10s} first: {fmt(first):24s} last: {fmt(last):24s} '
              f'unloaded force first/last: {first["Unloaded_Force"]:.2f}/{last["Unloaded_Force"]:.2f} N')

    # Just some test code to produce some plots for act'ble
    # -------------------------------------------------------------------------

    # Uncomment to run (requires act_2 and bloch_3 to have been analysed above)
    # from scipy.signal import medfilt
    # act_2_data = results['act_2']['data']
    # act_2_peak_force = results['act_2']['peak_force']
    # bloch_3_data = results['bloch_3']['data']
    # bloch_3_peak_force = results['bloch_3']['peak_force']
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
