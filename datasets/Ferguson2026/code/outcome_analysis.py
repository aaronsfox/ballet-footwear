# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    Script that extracts outcome measures from the processed degradation tests
    (see degradation_analysis.py) and generates the figures and tables for each
    aim of the study:

        Aim 1: comparison of shoe models (normalised profiles)
        Aim 2: stronger shank vs. standard Heritage
        Aim 3: where a used shoe sits against the new profile of the same model

    Outputs are saved to the results folder next to the data folder.

"""

# =========================================================================
# Import packages
# =========================================================================

import os
import gzip
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import processing script for paths, settings and plot settings
import degradation_analysis as da

# =========================================================================
# Set-up
# =========================================================================

# Paths
# -------------------------------------------------------------------------
results_path = os.path.join(da.code_dir, '..', 'results')

# Groups for each aim
# -------------------------------------------------------------------------
aim1_groups = {
    'Bloch Heritage': ['bloch_2', 'bloch_3', 'bloch_4'],
    'Bloch European': ['bloch_5', 'bloch_6'],
    'Gaynor Minden': ['gm_1', 'gm_2'],
    "Act'Pointe": ['act_2', 'act_3', 'act_4'],
}
aim2_groups = {
    'Bloch Heritage': ['bloch_2', 'bloch_3', 'bloch_4'],
    'Bloch Heritage (Strong)': ['bloch-s_1', 'bloch-s_2'],
}
aim3_new = ['bloch_5', 'bloch_6']
aim3_used = 'bloch-u_1'

# Group colours (fixed order; validated for colour vision deficiency separation)
group_colours = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']

# Outcome extraction settings
# -------------------------------------------------------------------------

# Running median window (cycles) used to smooth each shoe's curve before
# averaging across shoes (~1% of the test). Cycle-to-cycle variation is small
# (~0.1% for force, 0.1-0.4% for stiffness), so this is for clean curves and
# robust extraction rather than heavy smoothing
smoothing_window = 25

# Common horizon: all tests reach this cycle, so group curves and comparisons stop here
last_common_cycle = 2499

# Checkpoint bins (cycles) where values are extracted as the median of the bin
checkpoints = {
    'Start': (21, 70),
    '500': (476, 525),
    '1000': (976, 1025),
    '1500': (1476, 1525),
    '2000': (1976, 2025),
    '2500': (2450, 2499),
}

# Outcome variables
# (column in processed results, label, units, whether normalised to baseline)
outcomes = {
    'peak_true_force': ('Normalised_True_Force_Corrected', 'Normalised Peak True Force', '', True),
    'force_range': ('Normalised_Force_Range', 'Normalised Force Range (Drift-Free)', '', True),
    'low_stiffness': ('Normalised_Low_Angular_Stiffness', 'Normalised Low-Range Stiffness', '', True),
    'high_stiffness': ('Normalised_High_Angular_Stiffness', 'Normalised High-Range Stiffness', '', True),
    'deformation': ('Load_Onset_Bend_Angle', 'Retained Bend (Load Onset Angle)', 'deg', False),
    'energy_return': ('Energy_Return_Percent', 'Energy Returned', '%', False),
    'energy_lost': ('Normalised_Energy_Lost', 'Normalised Energy Lost (Hysteresis)', '', True),
    'hold_relaxation': ('Hold_Relaxation_Percent', 'Force Relaxation During Hold', '%', False),
}

# Absolute measures for Aims 2 and 3
# (column, label, units)
absolute_measures = {
    'moment': ('Moment_Corrected', 'Peak Moment', 'Nm'),
    'low_stiffness_abs': ('Low_Angular_Stiffness', 'Low-Range Stiffness', 'Nm/deg'),
    'high_stiffness_abs': ('High_Angular_Stiffness', 'High-Range Stiffness', 'Nm/deg'),
    'energy_input': ('Energy_Input', 'Energy Input', 'mJ'),
    'energy_return_abs': ('Energy_Return_Percent', 'Energy Returned', '%'),
    'hold_relaxation_abs': ('Hold_Relaxation_Percent', 'Force Relaxation During Hold', '%'),
    'onset_displacement': ('Load_Onset_Displacement', 'Load Onset Displacement', 'mm'),
}

# Plot style tweaks (on top of degradation_analysis settings)
plt.rcParams.update({'font.weight': 'normal', 'axes.labelweight': 'normal',
                     'axes.titleweight': 'normal', 'axes.titlesize': 12})
grid_colour = '#e6e6e2'

# =========================================================================
# Define functions
# =========================================================================

# Function to read cycle-level results (and selected cycles of sample data)
# -------------------------------------------------------------------------
def read_processed(shoe_id, sample_cycles=()):

    """
    Reads a shoe's processed file, returning the file info, cycle-level results
    (as numeric dataframe) and sample-level data for the requested cycles only
    (dictionary of dataframes keyed by cycle ID). Reading only selected cycles
    of sample data keeps memory use low.
    """

    filepath = os.path.join(da.processed_path, shoe_id, f'{shoe_id}_processed.json.gz')
    with gzip.open(filepath, 'rt', encoding='utf-8') as file:
        processed = json.load(file)
    results = pd.DataFrame([{'Cycle_ID': cycle['cycle_id']} | cycle['results'] for cycle in processed['cycles']])
    for col in results.columns:
        if col != 'Loaded_At_Cycle_Start' and col != 'Zero_Offset_Estimated':
            results[col] = pd.to_numeric(results[col], errors='coerce')
    results['Loaded_At_Cycle_Start'] = results['Loaded_At_Cycle_Start'].astype(bool)
    samples = {cycle['cycle_id']: pd.DataFrame(cycle['data']).apply(pd.to_numeric, errors='coerce')
               for cycle in processed['cycles'] if cycle['cycle_id'] in sample_cycles}
    info = {key: val for key, val in processed.items() if key != 'cycles'}
    return info, results, samples

# Function to get an outcome series for a shoe
# -------------------------------------------------------------------------
def get_series(results, column, mask_loaded_at_start=False, last_cycle=None):

    """
    Returns an outcome as a series indexed by cycle ID. For load onset, cycles
    where the shoe was loaded at the start of the cycle are masked (NaN), as
    the true onset is below the tested range.
    """

    series = results.set_index('Cycle_ID')[column].astype(float)
    if mask_loaded_at_start:
        series = series.where(~results.set_index('Cycle_ID')['Loaded_At_Cycle_Start'])
    if last_cycle is not None:
        series = series.loc[:last_cycle]
    return series

# Function to smooth a series with a centred running median
# -------------------------------------------------------------------------
def smooth(series):

    """Centred running median over smoothing_window cycles."""

    return series.rolling(smoothing_window, center=True, min_periods=smoothing_window // 2 + 1).median()

# Function to extract checkpoint values
# -------------------------------------------------------------------------
def extract_checkpoints(series):

    """Median of the series within each checkpoint bin."""

    return {label: series.loc[start:end].median() for label, (start, end) in checkpoints.items()}

# Function to plot group curves
# -------------------------------------------------------------------------
def plot_group_curves(ax, groups, curves, ylabel, reference=None):

    """
    Plots each shoe's smoothed curve (thin) and the group mean (thick) for each
    group, with a legend naming the groups and their shoes.
    """

    for (group, shoes), colour in zip(groups.items(), group_colours):
        group_curves = pd.concat({shoe: curves[shoe] for shoe in shoes if shoe in curves}, axis=1)
        if group_curves.dropna(how='all').empty:
            continue
        for shoe in group_curves.columns:
            ax.plot(group_curves.index, group_curves[shoe], color=colour, lw=0.8, alpha=0.45)
        # Group mean only where every shoe in the group has a value (avoids jumps
        # as shoes enter/leave, e.g. when load onset becomes measurable)
        ax.plot(group_curves.index, group_curves.mean(axis=1, skipna=False), color=colour, lw=2.5,
                label=f'{group} (n={group_curves.shape[1]})')
    if reference is not None:
        ax.axhline(reference, color='dimgrey', ls='--', lw=1.0)
    ax.set_xlim(0, last_common_cycle + 1)
    ax.set_xlabel('Cycle')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color=grid_colour, lw=0.8)
    ax.set_axisbelow(True)

# Decay models for Aim 3
# -------------------------------------------------------------------------
def log_model(n, a, b):
    return a + b * np.log(n)

def power_model(n, a, b):
    return a * np.power(n, b)

def exp_plateau_model(n, c, y0, tau):
    return c + (y0 - c) * np.exp(-(n - checkpoints['Start'][0]) / tau)

# Equivalent cycles beyond this are reported as not reached
max_equivalent_cycles = 1e7

decay_models = {
    'Logarithmic': (log_model, lambda y, n: (np.mean(y), 0.0)),
    'Power': (power_model, lambda y, n: (y[0], 0.0)),
    'Exponential plateau': (exp_plateau_model, lambda y, n: (y[-1], y[0], 500.0)),
}

def solve_model(model_name, params, target):

    """
    Solves a fitted decay model for the cycle at which it reaches the target
    value. Returns np.inf if the model never reaches the target within
    max_equivalent_cycles (e.g. target is beyond the plateau of the exponential
    model, or the model trends away from the target so the solution lies before
    the first analysed cycle).
    """

    cycle = _solve_model(model_name, params, target)
    if not np.isfinite(cycle) or cycle < checkpoints['Start'][0] or cycle > max_equivalent_cycles:
        return np.inf
    return cycle

def _solve_model(model_name, params, target):

    if model_name == 'Logarithmic':
        a, b = params
        return np.exp((target - a) / b) if b != 0 else np.inf
    if model_name == 'Power':
        a, b = params
        if b == 0 or target / a <= 0:
            return np.inf
        return (target / a) ** (1 / b)
    if model_name == 'Exponential plateau':
        c, y0, tau = params
        ratio = (target - c) / (y0 - c)
        if ratio <= 0 or ratio > 1 or tau <= 0:
            return np.inf
        return checkpoints['Start'][0] - tau * np.log(ratio)

def fit_decay(cycles, values):

    """
    Fits each decay model to a shoe's cycle-level values. Returns a dictionary of
    model name -> (parameters, R squared, AIC), skipping models that fail to fit.
    """

    fits = {}
    valid = ~np.isnan(values)
    n, y = cycles[valid].astype(float), values[valid]
    for name, (model, initial) in decay_models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                params, _ = curve_fit(model, n, y, p0=initial(y, n), maxfev=20000)
            residuals = y - model(n, *params)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - ss_res / np.sum((y - y.mean()) ** 2)
            aic = len(y) * np.log(ss_res / len(y)) + 2 * len(params)
            fits[name] = (params, r_squared, aic)
        except (RuntimeError, ValueError):
            continue
    return fits

# =========================================================================
# Run analyses
# =========================================================================

if __name__ == '__main__':

    # Read in processed results
    # -------------------------------------------------------------------------

    # Get all shoes used in the aims
    all_shoes = sorted(set(sum(aim1_groups.values(), []) + sum(aim2_groups.values(), []) + aim3_new + [aim3_used]))

    # Read cycle-level results (and first/last common cycle samples for hysteresis loops)
    info, results, samples = {}, {}, {}
    for shoe in all_shoes:
        print(f'Reading {shoe}...')
        info[shoe], results[shoe], samples[shoe] = read_processed(
            shoe, sample_cycles=(checkpoints['Start'][0], last_common_cycle))

    # Create results folders
    for folder in ['aim1', 'aim2', 'aim3', 'checks']:
        os.makedirs(os.path.join(results_path, folder), exist_ok=True)

    # Extract smoothed curves and checkpoint values for each outcome
    # -------------------------------------------------------------------------
    curves = {outcome: {} for outcome in outcomes}
    checkpoint_rows = []
    for shoe in all_shoes:
        for outcome, (column, label, units, normalised) in outcomes.items():
            series = get_series(results[shoe], column, mask_loaded_at_start=(outcome == 'deformation'),
                                last_cycle=last_common_cycle)
            curves[outcome][shoe] = smooth(series)
            values = extract_checkpoints(series)
            # Proportion of cycles where load onset was below the tested range
            if outcome == 'deformation':
                values['Pct_Below_Detection'] = 100 * results[shoe].loc[
                    results[shoe]['Cycle_ID'] <= last_common_cycle, 'Loaded_At_Cycle_Start'].mean()
            checkpoint_rows.append({'shoe': shoe, 'outcome': outcome, 'label': label, 'units': units} | values)
        # Absolute values at checkpoints (for context and Aims 2 and 3)
        for measure, (column, label, units) in absolute_measures.items():
            series = get_series(results[shoe], column, mask_loaded_at_start=(measure == 'onset_displacement'),
                                last_cycle=last_common_cycle)
            checkpoint_rows.append({'shoe': shoe, 'outcome': measure, 'label': label, 'units': units}
                                   | extract_checkpoints(series))

    # Save per-shoe checkpoint table
    shoe_checkpoints = pd.DataFrame(checkpoint_rows)
    shoe_checkpoints.to_csv(os.path.join(results_path, 'shoe_checkpoints.csv'), index=False, float_format='%.4f')

    # Function to summarise checkpoints by group (mean, min, max)
    def group_checkpoint_table(groups):
        rows = []
        for group, shoes in groups.items():
            group_data = shoe_checkpoints.loc[shoe_checkpoints['shoe'].isin(shoes)]
            for outcome, outcome_data in group_data.groupby('outcome', sort=False):
                row = {'group': group, 'n': len(shoes), 'outcome': outcome,
                       'label': outcome_data['label'].iloc[0], 'units': outcome_data['units'].iloc[0]}
                for checkpoint in checkpoints:
                    values = outcome_data[checkpoint]
                    row[f'{checkpoint}_mean'] = values.mean()
                    row[f'{checkpoint}_min'] = values.min()
                    row[f'{checkpoint}_max'] = values.max()
                rows.append(row)
        return pd.DataFrame(rows)

    # Aim 1: comparison of shoe models
    # -------------------------------------------------------------------------
    print('Aim 1...')
    aim1_dir = os.path.join(results_path, 'aim1')
    group_checkpoint_table(aim1_groups).to_csv(os.path.join(aim1_dir, 'aim1_group_checkpoints.csv'),
                                               index=False, float_format='%.4f')

    # Primary outcome: normalised peak true force
    fig, ax = plt.subplots(figsize=(12, 6.5))
    plot_group_curves(ax, aim1_groups, curves['peak_true_force'], 'Normalised peak true force', reference=1.0)
    ax.set_ylim(0.6, 1.15)
    ax.legend(loc='lower left', frameon=False)
    ax.set_title('Peak true force relative to cycles 21-30 (group mean thick, individual shoes thin; '
                 f'{smoothing_window}-cycle running median)', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(aim1_dir, 'aim1_normalised-peak-true-force.png'), format='png', dpi=200)
    plt.close('all')

    # Stiffness: low and high range
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, outcome, title in zip(axes, ['low_stiffness', 'high_stiffness'],
                                  [f'Low range: first {da.stiffness_window_width:g}° after load onset',
                                   f'High range: last {da.stiffness_window_width:g}° before peak bend']):
        plot_group_curves(ax, aim1_groups, curves[outcome], 'Normalised angular stiffness', reference=1.0)
        ax.set_title(title, loc='left')
    axes[0].set_ylim(0.6, 1.9)
    axes[0].legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(aim1_dir, 'aim1_normalised-stiffness.png'), format='png', dpi=200)
    plt.close('all')

    # Deformation: retained bend (load onset angle)
    # Shoes that carry load at the bottom of every cycle have no measurable onset;
    # their detection limit (bend angle at the bottom of the stroke) is shown dotted
    fig, ax = plt.subplots(figsize=(12, 6.5))
    plot_group_curves(ax, aim1_groups, curves['deformation'],
                      'Load onset bend angle from original starting shape (deg)')
    for (group, shoes), colour in zip(aim1_groups.items(), group_colours):
        if all(results[shoe]['Loaded_At_Cycle_Start'].all() for shoe in shoes):
            limit = np.mean([results[shoe]['Load_Onset_Bend_Angle'].median() for shoe in shoes])
            ax.axhline(limit, color=colour, ls=':', lw=2.0,
                       label=f'{group}: below detection limit (< {limit:.1f}°) in all cycles')
    ax.set_ylim(0, 9)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title('Retained bend: angle at which the shoe starts carrying load in each cycle', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(aim1_dir, 'aim1_deformation.png'), format='png', dpi=200)
    plt.close('all')

    # Energy: returned (% of input) and normalised energy lost
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    plot_group_curves(axes[0], aim1_groups, curves['energy_return'], 'Energy returned (% of energy input)')
    axes[0].set_ylim(30, 75)
    axes[0].set_title('Energy returned during relaxation', loc='left')
    plot_group_curves(axes[1], aim1_groups, curves['energy_lost'], 'Normalised energy lost (hysteresis area)',
                      reference=1.0)
    axes[1].set_ylim(0.5, 1.3)
    axes[1].set_title('Energy lost per cycle relative to cycles 21-30', loc='left')
    axes[0].legend(loc='lower left', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(aim1_dir, 'aim1_energy.png'), format='png', dpi=200)
    plt.close('all')

    # Hysteresis loops: moment vs. bend angle at first and last common cycle for
    # each shoe, with stiffness windows highlighted and energy return annotated
    for groups, aim, filename in [(aim1_groups, 'aim1', 'aim1_hysteresis-loops.png'),
                                  (aim2_groups, 'aim2', 'aim2_hysteresis-loops.png')]:
        n_cols = max(len(shoes) for shoes in groups.values())
        fig, axes = plt.subplots(len(groups), n_cols, figsize=(5 * n_cols, 4 * len(groups)), squeeze=False)
        for row, ((group, shoes), colour) in enumerate(zip(groups.items(), group_colours)):
            for col in range(n_cols):
                ax = axes[row, col]
                if col >= len(shoes):
                    ax.axis('off')
                    continue
                shoe = shoes[col]
                cycle_results = results[shoe].set_index('Cycle_ID')
                for cycle, line_style, alpha in [(checkpoints['Start'][0], '-', 1.0), (last_common_cycle, '--', 0.9)]:
                    data = samples[shoe][cycle].copy()
                    row_results = cycle_results.loc[cycle]
                    # Correct moment for load cell zero offset
                    data['Moment'] = data['Moment'] - row_results['Zero_Offset'] * info[shoe]['setup']['lever_arm_mm'] / 1000
                    ax.plot(data['Bend_Angle'], data['Moment'], color=colour, ls=line_style, lw=1.5, alpha=alpha,
                            label=f'Cycle {cycle}: {row_results["Energy_Return_Percent"]:.0f}% returned')
                    bending = data.loc[data['Phase'] == 1]
                    for window, window_colour in [('Low', '#555555'), ('High', '#111111')]:
                        in_window = bending['Bend_Angle'].between(row_results[f'{window}_Window_Start'],
                                                                  row_results[f'{window}_Window_End'])
                        ax.plot(bending.loc[in_window, 'Bend_Angle'], bending.loc[in_window, 'Moment'],
                                color=window_colour, lw=4, alpha=0.35, solid_capstyle='round')
                ax.set_title(f'{shoe} ({group})', loc='left', fontsize=10)
                ax.set_xlabel('Bend angle (deg)')
                ax.set_ylabel('Moment (Nm)')
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
                ax.grid(axis='y', color=grid_colour, lw=0.8)
                ax.legend(loc='upper left', frameon=False, fontsize=8)
        fig.suptitle('Moment vs. bend angle loops, corrected for load cell zero offset (solid = cycle 21, '
                     'dashed = cycle 2499; grey bands = low/high stiffness windows)', x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(results_path, aim, filename), format='png', dpi=150)
        plt.close('all')

    # Aim 2: stronger shank vs. standard Heritage
    # -------------------------------------------------------------------------
    print('Aim 2...')
    aim2_dir = os.path.join(results_path, 'aim2')
    group_checkpoint_table(aim2_groups).to_csv(os.path.join(aim2_dir, 'aim2_group_checkpoints.csv'),
                                               index=False, float_format='%.4f')

    # Normalised outcomes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, outcome, ylim in zip(axes.flat, ['peak_true_force', 'low_stiffness', 'high_stiffness', 'energy_return'],
                                 [(0.6, 1.15), (0.6, 1.5), (0.6, 1.5), (30, 75)]):
        column, label, units, normalised = outcomes[outcome]
        plot_group_curves(ax, aim2_groups, curves[outcome], label + (f' ({units})' if units else ''),
                          reference=1.0 if normalised else None)
        ax.set_ylim(*ylim)
    axes[0, 0].legend(loc='lower left', frameon=False)
    fig.suptitle('Strong vs. standard Heritage shank (strong shank applicator offset assumed 0 mm; '
                 'normalised force unaffected)', x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(aim2_dir, 'aim2_normalised-outcomes.png'), format='png', dpi=200)
    plt.close('all')

    # Absolute outcomes (same model, size and shank length, so directly comparable)
    # Load onset is shown as displacement (mm), which does not depend on the offset
    absolute_curves = {}
    for measure, (column, label, units) in absolute_measures.items():
        absolute_curves[measure] = {shoe: smooth(get_series(results[shoe], column,
                                                            mask_loaded_at_start=(measure == 'onset_displacement'),
                                                            last_cycle=last_common_cycle))
                                    for shoe in all_shoes}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, measure in zip(axes.flat, ['moment', 'low_stiffness_abs', 'high_stiffness_abs', 'onset_displacement']):
        column, label, units = absolute_measures[measure]
        plot_group_curves(ax, aim2_groups, absolute_curves[measure], f'{label} ({units})')
        ax.set_ylim(bottom=0)
    axes[1, 1].set_ylim(0, 9)
    axes[0, 0].legend(loc='lower left', frameon=False)
    fig.suptitle('Strong vs. standard Heritage shank: absolute values (angle-based stiffness for strong shank '
                 'uses assumed 0 mm offset)', x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(aim2_dir, 'aim2_absolute-outcomes.png'), format='png', dpi=200)
    plt.close('all')

    # Aim 3: used vs. new European
    # -------------------------------------------------------------------------
    print('Aim 3...')
    aim3_dir = os.path.join(results_path, 'aim3')

    # Measures used to match the used shoe against the new shoes' profiles
    # (raw values; the used shoe's early cycles represent its state on arrival)
    aim3_measures = ['moment', 'low_stiffness_abs', 'high_stiffness_abs', 'energy_input',
                     'energy_return_abs', 'hold_relaxation_abs']

    # Fit decay models to each new shoe and solve for the equivalent cycle
    equivalence_rows = []
    fits_all = {}
    for measure in aim3_measures:
        column, label, units = absolute_measures[measure]
        used_series = get_series(results[aim3_used], column)
        used_value = used_series.loc[checkpoints['Start'][0]:checkpoints['Start'][1]].median()
        for shoe in aim3_new:
            series = get_series(results[shoe], column).dropna()
            fits = fit_decay(series.index.to_numpy(), series.to_numpy())
            fits_all[(measure, shoe)] = fits
            # Observed crossing (first cycle where the smoothed curve reaches the used value)
            smoothed = smooth(series).dropna()
            start_value = smoothed.iloc[0]
            crossed = smoothed[(smoothed - used_value) * np.sign(start_value - used_value) <= 0]
            observed = int(crossed.index[0]) if len(crossed) else None
            row = {'measure': measure, 'label': label, 'units': units, 'new_shoe': shoe,
                   'used_start_value': used_value, 'new_start_value': start_value,
                   'new_end_value': smoothed.iloc[-1], 'new_last_cycle': int(smoothed.index[-1]),
                   'observed_crossing_cycle': observed}
            for name, (params, r_squared, aic) in fits.items():
                row[f'{name}_R2'] = r_squared
                row[f'{name}_AIC'] = aic
                row[f'{name}_equivalent_cycle'] = solve_model(name, params, used_value)
            # Summary using the best fitting model (lowest AIC)
            if fits:
                best = min(fits, key=lambda name: fits[name][2])
                cycle = row[f'{best}_equivalent_cycle']
                row['best_model'] = best
                row['best_model_R2'] = fits[best][1]
                row['summary'] = (f'~{cycle:,.0f} cycles ({best}, R2 = {fits[best][1]:.2f})' if np.isfinite(cycle)
                                  else f'not reached within {max_equivalent_cycles:,.0f} cycles ({best}, '
                                       f'R2 = {fits[best][1]:.2f})')
            equivalence_rows.append(row)
    equivalence = pd.DataFrame(equivalence_rows)
    equivalence.to_csv(os.path.join(aim3_dir, 'aim3_equivalent_cycles.csv'), index=False, float_format='%.4f')

    # Visualise decay fits and the used shoe's starting value (log cycle axis to show extrapolation)
    extrapolate_to = 1e6
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    model_styles = {'Logarithmic': '--', 'Power': '-.', 'Exponential plateau': ':'}
    for ax, measure in zip(axes.flat, aim3_measures):
        column, label, units = absolute_measures[measure]
        used_value = equivalence.loc[equivalence['measure'] == measure, 'used_start_value'].iloc[0]
        for shoe, colour in zip(aim3_new, group_colours[:2]):
            series = get_series(results[shoe], column).dropna()
            ax.plot(series.index, smooth(series), color=colour, lw=2, label=f'{shoe} (new)')
            n_ext = np.logspace(np.log10(checkpoints['Start'][0]), np.log10(extrapolate_to), 300)
            for name, (params, r_squared, aic) in fits_all[(measure, shoe)].items():
                model = decay_models[name][0]
                ax.plot(n_ext, model(n_ext, *params), color=colour, lw=1, ls=model_styles[name], alpha=0.8,
                        label=f'{name} fit' if shoe == aim3_new[0] else None)
        used_series = smooth(get_series(results[aim3_used], column).dropna())
        ax.plot(used_series.index, used_series, color='#555555', lw=1.5, label=f'{aim3_used} (used) own test')
        ax.axhline(used_value, color='black', lw=1.2, label=f'{aim3_used} start value')
        ax.axvspan(checkpoints['Start'][0], max(results[s]['Cycle_ID'].max() for s in aim3_new),
                   color='#d9d9d4', alpha=0.25, lw=0)
        ax.set_xscale('log')
        ax.set_xlim(checkpoints['Start'][0], extrapolate_to)
        ax.set_xlabel('Cycle (log scale; shaded = tested range)')
        ax.set_ylabel(f'{label} ({units})')
        ax.grid(axis='y', color=grid_colour, lw=0.8)
    axes[0, 0].legend(loc='lower left', frameon=False, fontsize=8)
    fig.suptitle('Used vs. new Bloch European: decay fits to new shoes and the used shoe\'s starting value',
                 x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(aim3_dir, 'aim3_decay-fits.png'), format='png', dpi=150)
    plt.close('all')

    # Starting shape vs. deformation
    # The resting sole angle is the starting angle (from the applicator offset) plus
    # the load onset bend angle (retained bend) in each cycle. For the new shoes this
    # shows how the test moves the shoe's resting shape; the used shoe's starting
    # angle shows where wear has left it
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for shoe, colour in zip(aim3_new + [aim3_used], group_colours[:2] + ['#555555']):
        start_angle = info[shoe]['setup']['starting_angle_deg']
        onset = smooth(get_series(results[shoe], 'Load_Onset_Bend_Angle', mask_loaded_at_start=True))
        ax.plot(onset.index, start_angle + onset, color=colour, lw=2,
                label=f'{shoe} resting angle during test (starting angle {start_angle:.1f}°)')
        ax.axhline(start_angle, color=colour, ls=':', lw=1.5)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 25)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Sole angle at load onset (deg from horizontal)')
    ax.grid(axis='y', color=grid_colour, lw=0.8)
    ax.legend(loc='center right', frameon=False)
    ax.set_title('Resting sole angle: starting angle (dotted) plus retained bend during testing (solid)', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(aim3_dir, 'aim3_starting-angle-vs-deformation.png'), format='png', dpi=200)
    plt.close('all')

    # Normalised profiles (rate of degradation) for used vs. new
    aim3_groups = {'Bloch European (new)': aim3_new, 'Bloch European (used)': [aim3_used]}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, outcome, ylim in zip(axes, ['peak_true_force', 'low_stiffness', 'high_stiffness'],
                                 [(0.6, 1.15), (0.6, 1.5), (0.6, 1.5)]):
        plot_group_curves(ax, aim3_groups, curves[outcome], outcomes[outcome][1], reference=1.0)
        ax.set_ylim(*ylim)
    axes[0].legend(loc='lower left', frameon=False)
    fig.suptitle('Used vs. new Bloch European: normalised profiles', x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(aim3_dir, 'aim3_normalised-profiles.png'), format='png', dpi=200)
    plt.close('all')
    group_checkpoint_table(aim3_groups).to_csv(os.path.join(aim3_dir, 'aim3_group_checkpoints.csv'),
                                               index=False, float_format='%.4f')

    # Checks: load cell zero offset and drift-free force range
    # -------------------------------------------------------------------------
    print('Checks...')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for shoe in all_shoes:
        offset = get_series(results[shoe], 'Zero_Offset')
        if results[shoe]['Zero_Offset_Estimated'].iloc[0] in (True, 'True', 1):
            axes[0].plot(offset.index, offset, lw=1.2, label=shoe)
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('Estimated load cell zero offset (N)')
    axes[0].set_title('Zero offset from unloaded readings (shoes that unload only)', loc='left')
    axes[0].legend(loc='upper left', frameon=False, fontsize=8, ncol=2)
    axes[0].grid(axis='y', color=grid_colour, lw=0.8)
    for (group, shoes), colour in zip(aim1_groups.items(), group_colours):
        for shoe in shoes:
            axes[1].plot(curves['peak_true_force'][shoe].index, curves['peak_true_force'][shoe], color=colour, lw=1.5,
                         label=f'{group}' if shoe == shoes[0] else None)
            axes[1].plot(curves['force_range'][shoe].index, curves['force_range'][shoe], color=colour, lw=1.0,
                         ls='--')
    axes[1].axhline(1.0, color='dimgrey', ls='--', lw=1.0)
    axes[1].set_xlim(0, last_common_cycle + 1)
    axes[1].set_xlabel('Cycle')
    axes[1].set_ylabel('Normalised value')
    axes[1].set_title('Peak true force (solid) vs. drift-free force range (dashed)', loc='left')
    axes[1].legend(loc='lower left', frameon=False)
    axes[1].grid(axis='y', color=grid_colour, lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_path, 'checks', 'drift-check.png'), format='png', dpi=150)
    plt.close('all')

    print('Done.')

# %% ---------- end of outcome_analysis.py ---------- %% #
