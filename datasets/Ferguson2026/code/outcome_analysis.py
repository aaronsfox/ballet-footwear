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

# Import processing script for paths and settings, and shared figure style
import degradation_analysis as da
import plot_style as ps

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

# Group colours and markers come from the shared style (plot_style.PALETTE and
# plot_style.MARKERS), assigned in group order

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

# Apply shared figure style (overrides the processing script's check-figure settings)
ps.apply_style()

# Spacing (cycles) of markers on group mean curves; markers are staggered between
# groups so they do not overlap
marker_spacing = 250

# Figure height (inches) for a single row of panels
row_height = 2.6

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

# Function to save figures
# -------------------------------------------------------------------------
def save_figure(fig, filepath_no_ext):

    """Saves a figure as PNG and PDF (editable text) and closes it."""

    fig.savefig(f'{filepath_no_ext}.png', format='png')
    # No creation date in the PDF, so re-running unchanged figures gives identical files
    fig.savefig(f'{filepath_no_ext}.pdf', format='pdf', metadata={'CreationDate': None})
    plt.close(fig)

# Function to plot a curve with sparse markers
# -------------------------------------------------------------------------
def plot_marked(ax, x, y, index, label=None, **kwargs):

    """
    Plots a line in the palette colour for this index, with the matching marker
    shape every marker_spacing points (staggered by index so markers from
    different series do not overlap).
    """

    offset = (index * marker_spacing // 4) % marker_spacing
    ax.plot(x, y, color=ps.colour(index), marker=ps.marker(index), markersize=3.5,
            markevery=(offset, marker_spacing), label=label, **kwargs)

# Function to plot group curves
# -------------------------------------------------------------------------
def plot_group_curves(ax, groups, curves, ylabel, reference=None, markers=True, scale=1.0):

    """
    Plots each shoe's smoothed curve (thin) and the group mean (thick, with
    markers unless markers=False) for each group, labelled with the group name
    and number of shoes. Values are multiplied by scale (e.g. 100 for %).
    """

    for index, (group, shoes) in enumerate(groups.items()):
        group_curves = pd.concat({shoe: curves[shoe] for shoe in shoes if shoe in curves}, axis=1) * scale
        if group_curves.dropna(how='all').empty:
            continue
        for shoe in group_curves.columns:
            ax.plot(group_curves.index, group_curves[shoe], color=ps.colour(index), lw=0.6, alpha=0.4)
        # Group mean only where every shoe in the group has a value (avoids jumps
        # as shoes enter/leave, e.g. when load onset becomes measurable)
        group_mean = group_curves.mean(axis=1, skipna=False)
        label = f'{group} (n={group_curves.shape[1]})'
        if markers:
            plot_marked(ax, group_curves.index, group_mean, index, label=label)
        else:
            ax.plot(group_curves.index, group_mean, color=ps.colour(index), label=label)
    if reference is not None:
        ax.axhline(reference, color='dimgrey', ls='--', lw=0.8)
    ax.set_xlim(0, last_common_cycle + 1)
    ax.set_xlabel('Cycle')
    ax.set_ylabel(ylabel)

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

    # Primary outcome: normalised peak true force (% of cycles 21-30)
    # No markers, legend or title (groups and details are given in the caption)
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN, row_height + 0.3))
    plot_group_curves(ax, aim1_groups, curves['peak_true_force'], 'Normalised Peak Force (%)',
                      reference=100, markers=False, scale=100)
    ax.set_ylim(60, 115)
    save_figure(fig, os.path.join(aim1_dir, 'aim1_normalised-peak-true-force'))

    # Key metrics at checkpoints: mean +/- SD across shoes in each model
    # Each shoe's value is the median of the 50-cycle checkpoint bin (see
    # checkpoints); points are the model mean and error bars the sample SD across
    # shoes (n = 2-3 per model, so SD is indicative only). Normalised metrics are
    # expressed as % of cycles 21-30; energy return is % of energy input
    point_metrics = {
        'peak_true_force': ('Normalised Peak Force (%)', 100),
        'low_stiffness': ('Normalised Low-Range Stiffness (%)', 100),
        'high_stiffness': ('Normalised High-Range Stiffness (%)', 100),
        'energy_return': ('Energy Returned (% of Input)', 1),
    }
    checkpoint_cycles = {label: (start + end) / 2 for label, (start, end) in checkpoints.items()}
    point_rows = []
    for metric, (ylabel, scale) in point_metrics.items():
        for group, shoes in aim1_groups.items():
            values = shoe_checkpoints.loc[(shoe_checkpoints['outcome'] == metric) &
                                          shoe_checkpoints['shoe'].isin(shoes)].set_index('shoe')
            for checkpoint, (start, end) in checkpoints.items():
                shoe_values = values.loc[shoes, checkpoint] * scale
                point_rows.append({
                    'metric': metric, 'label': ylabel, 'model': group, 'checkpoint': checkpoint,
                    'cycle_bin': f'{start}-{end}', 'cycle_mid': checkpoint_cycles[checkpoint],
                    'n': int(shoe_values.notna().sum()), 'mean': shoe_values.mean(), 'sd': shoe_values.std(ddof=1),
                    'mean_sd': f'{shoe_values.mean():.1f} ± {shoe_values.std(ddof=1):.1f}',
                    'shoe_values': '; '.join(f'{shoe}: {value:.2f}' for shoe, value in shoe_values.items()),
                })
    point_summary = pd.DataFrame(point_rows)
    point_summary.to_csv(os.path.join(aim1_dir, 'aim1_checkpoint-mean-sd.csv'), index=False, float_format='%.3f')

    # Point plot (models dodged horizontally so error bars do not overlap)
    dodge = 45
    fig, axes = plt.subplots(2, 2, figsize=(ps.DOUBLE_COLUMN, 2 * row_height))
    for ax, (metric, (ylabel, scale)) in zip(axes.flat, point_metrics.items()):
        metric_data = point_summary.loc[point_summary['metric'] == metric]
        for index, group in enumerate(aim1_groups):
            group_data = metric_data.loc[metric_data['model'] == group]
            x = group_data['cycle_mid'] + (index - (len(aim1_groups) - 1) / 2) * dodge
            ax.errorbar(x, group_data['mean'], yerr=group_data['sd'], color=ps.colour(index),
                        marker=ps.marker(index), markersize=4, lw=1.0, capsize=2, capthick=0.8, elinewidth=0.8,
                        label=f'{group} (n={len(aim1_groups[group])})')
        if scale == 100:
            ax.axhline(100, color='dimgrey', ls='--', lw=0.8, zorder=0)
        ax.set_xticks(list(range(0, last_common_cycle + 2, 500)))
        ax.set_xlim(-150, last_common_cycle + 150)
        ax.set_xlabel('Cycle')
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    save_figure(fig, os.path.join(aim1_dir, 'aim1_checkpoint-mean-sd'))

    # Stiffness: low and high range
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN, row_height), sharey=True)
    for ax, outcome, title in zip(axes, ['low_stiffness', 'high_stiffness'],
                                  [f'Low range (first {da.stiffness_window_width:g}° after load onset)',
                                   f'High range (last {da.stiffness_window_width:g}° before peak bend)']):
        plot_group_curves(ax, aim1_groups, curves[outcome], 'Normalised angular stiffness', reference=1.0)
        ax.set_title(title)
    axes[1].set_ylabel('')
    axes[0].set_ylim(0.6, 1.9)
    axes[0].legend(loc='upper left')
    save_figure(fig, os.path.join(aim1_dir, 'aim1_normalised-stiffness'))

    # Deformation: retained bend (load onset angle)
    # Shoes that carry load at the bottom of every cycle have no measurable onset;
    # their detection limit (bend angle at the bottom of the stroke) is shown dotted
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN, row_height + 0.3))
    plot_group_curves(ax, aim1_groups, curves['deformation'], 'Load onset angle (deg)')
    cycles = np.arange(checkpoints['Start'][0], last_common_cycle + 1)
    for index, (group, shoes) in enumerate(aim1_groups.items()):
        if all(results[shoe]['Loaded_At_Cycle_Start'].all() for shoe in shoes):
            limit = np.mean([results[shoe]['Load_Onset_Bend_Angle'].median() for shoe in shoes])
            plot_marked(ax, cycles, np.full(len(cycles), limit), index, ls=':', lw=1.2,
                        label=f'{group}: < {limit:.1f}° (below limit)')
    ax.set_ylim(0, 9)
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title('Retained bend')
    save_figure(fig, os.path.join(aim1_dir, 'aim1_deformation'))

    # Energy: returned (% of input) and normalised energy lost
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN, row_height))
    plot_group_curves(axes[0], aim1_groups, curves['energy_return'], 'Energy returned (% of input)')
    axes[0].set_ylim(30, 75)
    axes[0].set_title('Energy returned')
    plot_group_curves(axes[1], aim1_groups, curves['energy_lost'], 'Normalised energy lost', reference=1.0)
    axes[1].set_ylim(0.5, 1.3)
    axes[1].set_title('Energy lost (hysteresis)')
    axes[0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.22), ncol=4)
    save_figure(fig, os.path.join(aim1_dir, 'aim1_energy'))

    # Hysteresis loops: moment vs. bend angle at first and last common cycle for
    # each shoe, with stiffness windows highlighted (solid = cycle 21; dashed =
    # cycle 2499; grey bands = stiffness windows). No legends or markers; curves are
    # described in the caption. Shoes are titled by model and number, and the
    # y-axis is shared within each row (model)
    for groups, aim, filename in [(aim1_groups, 'aim1', 'aim1_hysteresis-loops'),
                                  (aim2_groups, 'aim2', 'aim2_hysteresis-loops')]:
        n_cols = max(len(shoes) for shoes in groups.values())
        fig, axes = plt.subplots(len(groups), n_cols, figsize=(ps.DOUBLE_COLUMN, 2.1 * len(groups)),
                                 squeeze=False, sharey='row')
        for row, (group, shoes) in enumerate(groups.items()):
            for col in range(n_cols):
                ax = axes[row, col]
                if col >= len(shoes):
                    ax.axis('off')
                    continue
                shoe = shoes[col]
                cycle_results = results[shoe].set_index('Cycle_ID')
                for cycle, line_style in [(checkpoints['Start'][0], '-'), (last_common_cycle, '--')]:
                    data = samples[shoe][cycle].copy()
                    row_results = cycle_results.loc[cycle]
                    # Correct moment for load cell zero offset
                    data['Moment'] = data['Moment'] - row_results['Zero_Offset'] * info[shoe]['setup']['lever_arm_mm'] / 1000
                    bending = data.loc[data['Phase'] == 1]
                    for window in ['Low', 'High']:
                        in_window = bending['Bend_Angle'].between(row_results[f'{window}_Window_Start'],
                                                                  row_results[f'{window}_Window_End'])
                        ax.plot(bending.loc[in_window, 'Bend_Angle'], bending.loc[in_window, 'Moment'],
                                color='#9a9a9a', lw=4, alpha=0.45, solid_capstyle='round', zorder=1)
                    ax.plot(data['Bend_Angle'], data['Moment'], color=ps.colour(row), ls=line_style, lw=1.0, zorder=2)
                ax.set_title(f'{group} {col + 1}', fontsize=8)
                ax.set_xlabel('Bend Angle (deg)')
                if col == 0:
                    ax.set_ylabel('Moment (Nm)')
                ax.set_xlim(left=0)
            axes[row, 0].set_ylim(bottom=min(0, axes[row, 0].get_ylim()[0]))
        fig.tight_layout()
        save_figure(fig, os.path.join(results_path, aim, filename))

    # Aim 2: stronger shank vs. standard Heritage
    # -------------------------------------------------------------------------
    print('Aim 2...')
    aim2_dir = os.path.join(results_path, 'aim2')
    group_checkpoint_table(aim2_groups).to_csv(os.path.join(aim2_dir, 'aim2_group_checkpoints.csv'),
                                               index=False, float_format='%.4f')

    # Normalised outcomes
    fig, axes = plt.subplots(2, 2, figsize=(ps.DOUBLE_COLUMN, 2 * row_height))
    for ax, outcome, ylim, title in zip(axes.flat,
                                        ['peak_true_force', 'low_stiffness', 'high_stiffness', 'energy_return'],
                                        [(0.6, 1.15), (0.6, 1.5), (0.6, 1.5), (30, 75)],
                                        ['Peak true force', 'Low-range stiffness', 'High-range stiffness',
                                         'Energy returned']):
        column, label, units, normalised = outcomes[outcome]
        plot_group_curves(ax, aim2_groups, curves[outcome],
                          'Energy returned (% of input)' if outcome == 'energy_return' else 'Normalised value',
                          reference=1.0 if normalised else None)
        ax.set_ylim(*ylim)
        ax.set_title(title)
    axes[0, 0].legend(loc='lower left')
    fig.tight_layout()
    save_figure(fig, os.path.join(aim2_dir, 'aim2_normalised-outcomes'))

    # Absolute outcomes (same model, size and shank length, so directly comparable)
    # Load onset is shown as displacement (mm), which does not depend on the offset
    absolute_curves = {}
    for measure, (column, label, units) in absolute_measures.items():
        absolute_curves[measure] = {shoe: smooth(get_series(results[shoe], column,
                                                            mask_loaded_at_start=(measure == 'onset_displacement'),
                                                            last_cycle=last_common_cycle))
                                    for shoe in all_shoes}
    fig, axes = plt.subplots(2, 2, figsize=(ps.DOUBLE_COLUMN, 2 * row_height))
    for ax, measure in zip(axes.flat, ['moment', 'low_stiffness_abs', 'high_stiffness_abs', 'onset_displacement']):
        column, label, units = absolute_measures[measure]
        plot_group_curves(ax, aim2_groups, absolute_curves[measure], f'{label} ({units})')
        ax.set_ylim(bottom=0)
        ax.set_title(label)
    axes[1, 1].set_ylim(0, 9)
    axes[0, 0].legend(loc='lower left')
    fig.tight_layout()
    save_figure(fig, os.path.join(aim2_dir, 'aim2_absolute-outcomes'))

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

    # Shoe order for colours/markers in Aim 3 figures (new shoes, then used shoe)
    aim3_shoes = aim3_new + [aim3_used]

    # Visualise decay fits and the used shoe's starting value (log cycle axis to show extrapolation)
    extrapolate_to = 1e6
    model_styles = {'Logarithmic': '--', 'Power': '-.', 'Exponential plateau': ':'}
    fig, axes = plt.subplots(2, 3, figsize=(ps.DOUBLE_COLUMN, 2 * row_height))
    for ax, measure in zip(axes.flat, aim3_measures):
        column, label, units = absolute_measures[measure]
        used_value = equivalence.loc[equivalence['measure'] == measure, 'used_start_value'].iloc[0]
        ax.axvspan(checkpoints['Start'][0], max(results[shoe]['Cycle_ID'].max() for shoe in aim3_new),
                   color='#e8e8e4', lw=0, zorder=0)
        n_ext = np.logspace(np.log10(checkpoints['Start'][0]), np.log10(extrapolate_to), 300)
        for index, shoe in enumerate(aim3_shoes):
            series = smooth(get_series(results[shoe], column).dropna()).dropna()
            ax.plot(series.index, series, color=ps.colour(index), lw=1.4, marker=ps.marker(index), markersize=3,
                    markevery=0.1, label=f'{shoe} ({"used" if shoe == aim3_used else "new"})')
            if shoe in aim3_new:
                for name, (params, r_squared, aic) in fits_all[(measure, shoe)].items():
                    ax.plot(n_ext, decay_models[name][0](n_ext, *params), color=ps.colour(index), lw=0.7,
                            ls=model_styles[name])
        ax.axhline(used_value, color='black', lw=0.8, label=f'{aim3_used} start value')
        ax.set_xscale('log')
        ax.set_xlim(checkpoints['Start'][0], extrapolate_to)
        ax.set_xlabel('Cycle (log scale)')
        ax.set_ylabel(f'{label} ({units})')
        ax.set_title(label, fontsize=8)
    # Legend: shoes (colour and marker), fit models (line style, shown in grey as each
    # shoe's fits are drawn in its own colour) and the used shoe's start value
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color='grey', lw=0.7, ls=style) for style in model_styles.values()]
    labels += [f'{name} fit' for name in model_styles]
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save_figure(fig, os.path.join(aim3_dir, 'aim3_decay-fits'))

    # Starting shape vs. deformation
    # The resting sole angle is the starting angle (from the applicator offset) plus
    # the load onset bend angle (retained bend) in each cycle. For the new shoes this
    # shows how the test moves the shoe's resting shape; the used shoe's starting
    # angle shows where wear has left it
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN, row_height + 0.4))
    for index, shoe in enumerate(aim3_shoes):
        start_angle = info[shoe]['setup']['starting_angle_deg']
        onset = smooth(get_series(results[shoe], 'Load_Onset_Bend_Angle', mask_loaded_at_start=True))
        plot_marked(ax, onset.index, start_angle + onset, index,
                    label=f'{shoe} ({"used" if shoe == aim3_used else "new"}; start {start_angle:.1f}°)')
        ax.axhline(start_angle, color=ps.colour(index), ls=':', lw=1.0)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 25)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Sole angle at load onset (deg)')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title('Resting angle vs. starting angle (dotted)')
    save_figure(fig, os.path.join(aim3_dir, 'aim3_starting-angle-vs-deformation'))

    # Normalised profiles (rate of degradation) for used vs. new
    aim3_groups = {'Bloch European (new)': aim3_new, 'Bloch European (used)': [aim3_used]}
    fig, axes = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN, row_height))
    for ax, outcome, ylim, title in zip(axes, ['peak_true_force', 'low_stiffness', 'high_stiffness'],
                                        [(0.6, 1.15), (0.6, 1.5), (0.6, 1.5)],
                                        ['Peak true force', 'Low-range stiffness', 'High-range stiffness']):
        plot_group_curves(ax, aim3_groups, curves[outcome], 'Normalised value', reference=1.0)
        ax.set_ylim(*ylim)
        ax.set_title(title)
    axes[0].legend(loc='lower left', fontsize=7)
    fig.tight_layout()
    save_figure(fig, os.path.join(aim3_dir, 'aim3_normalised-profiles'))
    group_checkpoint_table(aim3_groups).to_csv(os.path.join(aim3_dir, 'aim3_group_checkpoints.csv'),
                                               index=False, float_format='%.4f')

    # Checks: load cell zero offset and drift-free force range
    # -------------------------------------------------------------------------
    print('Checks...')
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN, row_height + 0.4))
    offset_shoes = [shoe for shoe in all_shoes
                    if results[shoe]['Zero_Offset_Estimated'].iloc[0] in (True, 'True', 1)]
    for index, shoe in enumerate(offset_shoes):
        offset = get_series(results[shoe], 'Zero_Offset')
        plot_marked(axes[0], offset.index, offset, index, lw=1.0, label=shoe)
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('Estimated zero offset (N)')
    axes[0].set_title('Load cell zero offset')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=7, ncol=4)
    for index, (group, shoes) in enumerate(aim1_groups.items()):
        for shoe in shoes:
            plot_marked(axes[1], curves['peak_true_force'][shoe].index, curves['peak_true_force'][shoe], index,
                        lw=1.0, label=group if shoe == shoes[0] else None)
            axes[1].plot(curves['force_range'][shoe].index, curves['force_range'][shoe], color=ps.colour(index),
                         lw=0.7, ls='--')
    axes[1].axhline(1.0, color='dimgrey', ls='--', lw=0.8)
    axes[1].set_xlim(0, last_common_cycle + 1)
    axes[1].set_xlabel('Cycle')
    axes[1].set_ylabel('Normalised value')
    axes[1].set_title('Peak true force (solid) vs. force range (dashed)')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, os.path.join(results_path, 'checks', 'drift-check'))

    print('Done.')

# %% ---------- end of outcome_analysis.py ---------- %% #
