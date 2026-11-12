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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.signal import medfilt

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Set test data path
data_path = os.path.join('..','data')

# Plot settings
# -------------------------------------------------------------------------

# Set matplotlib parameters
from matplotlib import rcParams
import matplotlib

# Set backend to use
matplotlib.use('TkAgg')
plt.ion()

# rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
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

# Function to read in and extract data from degradation test
# -------------------------------------------------------------------------
def analyse_degradation_test(filename):

    # Read in file as dataframe
    test_data = pd.read_csv(os.path.join(data_path,filename))

    # Get file label for naming
    filelabel = os.path.splitext(filename)[0]

    # Create folder in .checks for this sample
    os.makedirs(os.path.join('..','.checks',filelabel), exist_ok=True)
    check_dir = os.path.join('..','.checks',filelabel)

    # Extract displacement values for cycle identification
    values = pd.to_numeric(test_data['Displacement'], errors='coerce').values

    # Identify full cycles (i.e. bending and relaxation)
    # Identify bending and relaxation phases within cycle (1 = bending, -1 = relaxing)
    # -------------------------------------------------------------------------

    # Set prominence for detecting displacement signifying new cycle
    min_prominence = 0.05

    # Set cycle identification variables
    cycle_ids = []
    phases = []
    current_id = 1
    is_increasing = True
    prev_disp = values[0] if len(values) > 0 else 0

    # Loop through values for cycle identification
    for val in values:
        if np.isnan(val):
            cycle_ids.append("Cycle_ID_Error")
            continue

        # Logic for cycle identification
        if is_increasing and (val < (prev_disp - min_prominence)):
            is_increasing = False
        elif not is_increasing and (val > (prev_disp + min_prominence)):
            is_increasing = True
            current_id += 1

        # Append cycle ID and phase
        cycle_ids.append(current_id)
        phases.append(1 if is_increasing else -1)
        prev_disp = val

    # Append cycle ID and phase to dataframe
    test_data['Cycle_ID'] = cycle_ids
    test_data['Phase'] = phases

    # Visual check of identification (first XX cycles)
    # Plot data
    check_data = test_data.loc[test_data['Cycle_ID'] <= 10,]
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(15,8))
    ax.plot(check_data['Displacement'], 'k-', label='Displacement')
    ax.plot(check_data['Force'] / check_data['Force'].max() * check_data['Displacement'].max(),
            'r-', label='Scaled Force')
    for x_ind in np.where(np.diff(check_data['Cycle_ID'].to_numpy()) != 0)[0]:
        ax.axvline(x=x_ind, color='g', linestyle='--')
    # Collate and visualise phases
    x = np.arange(len(check_data['Phase']))
    y = np.zeros_like(x)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # reshape data to segments
    colour_map = {1: 'blue', -1: 'pink'}
    c_phases = [colour_map.get(p, 'grey') for p in phases[:-1]]  # create colour map to match with phases
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

    # Drop initial cycles from dataset
    # Here we remove the initial 20 cycles to ensure that data analysis was not impacted by the initial setting of the
    # shoe in the fixture clamps and stabilisation of the contact. Number of cycles (i.e. 20) was guided by
    # Yoshida et al. (https://doi.org/10.1177/1089313X251411487)
    test_data = test_data[test_data['Cycle_ID'] > 20]

    # Calculate peak force per cycle ID
    peak_force = test_data.groupby('Cycle_ID')['Force'].max().reset_index()

    # Normalise each cycles peak force to the first cycle
    # Note that this is now actually a later cycle given the removal of the early seating cycles
    first_cycle_peak_force = peak_force['Force'].iloc[0]
    peak_force['Normalised_Force'] = peak_force['Force'] / first_cycle_peak_force

    # Visualise normalised force change
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.plot(peak_force['Normalised_Force'], 'b-',
            label='Normalised Peak Force', lw = 1.5)
    # ax.plot(medfilt(peak_force['Normalised_Force'], kernel_size=19), 'r-',
    #         label='Median Filtered (n=5) Normalised Peak Force', lw = 1.5)
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
    # This uses a similar method as Yoshida et al. (2026) which looks at the change
    # in force between 7.5 and 7.0 mm of displacement during the bending phase
    # divided by the magnitude of displacement (i.e. 0.5 mm) to get N/mm
    # See here for Yoshida et al. paper: https://doi.org/10.1177/1089313X251411487
    # The principle here is the same, but the displacement window is changed. In
    # particular we look at some different regions of stiffness across the displacement
    # tested.
    # -------------------------------------------------------------------------

    # Set parameters for where to examine stiffness
    low_stiffness_start = 4.0
    low_stiffness_end = 7.0
    high_stiffness_start = 10.0
    high_stiffness_end = 13.0

    # Create list to store calculated stiffness values
    low_stiffness = []
    high_stiffness = []
    relative_stiffness = []

    # Loop through cycles to calculate stiffness during bending
    for cycle in peak_force['Cycle_ID']:
        # Extract cycle data during bending phase
        cycle_data = test_data[(test_data['Cycle_ID']==cycle) & (test_data['Phase']==1)]
        # Get the displacement and force value arrays
        displacement = cycle_data['Displacement'].to_numpy()
        force = cycle_data['Force'].to_numpy()
        # Identify where displacement reaches the stiffness start and end parameters
        low_start_ind = np.argmax(displacement >= low_stiffness_start)
        low_end_ind = np.argmax(displacement >= low_stiffness_end)
        high_start_ind = np.argmax(displacement >= high_stiffness_start)
        high_end_ind = np.argmax(displacement >= high_stiffness_end)
        # Calculate stiffness for cycle
        low_val = (force[low_end_ind] - force[low_start_ind]) / (low_stiffness_end - low_stiffness_start)
        high_val = (force[high_end_ind] - force[high_start_ind]) / (high_stiffness_end - high_stiffness_start)
        # Calculate relative stiffness between low and high
        rel_val = high_val / low_val
        # Append to lists
        low_stiffness.append(low_val)
        high_stiffness.append(high_val)
        relative_stiffness.append(rel_val)

    # Normalise each cycles stiffness to the first cycle
    # Note that this is now actually the second cycle given the first seating cycle is removed
    low_first_cycle_stiffness = low_stiffness[0]
    high_first_cycle_stiffness = high_stiffness[0]
    peak_force['Low_Normalised_Stiffness'] = low_stiffness / low_first_cycle_stiffness
    peak_force['High_Normalised_Stiffness'] = high_stiffness / high_first_cycle_stiffness

    # Visualise normalised force change
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.plot(peak_force['Low_Normalised_Stiffness'], 'g-', label='Normalised Peak Stiffness (Low Range)')
    ax.plot(peak_force['High_Normalised_Stiffness'], 'r-', label='Normalised Peak Stiffness (High Range)')
    ax.axhline(y=1.0, color='dimgrey', linestyle='--', lw=1.5)
    # Set limits and labels
    ax.set_xlim(-5, test_data['Cycle_ID'].max())
    ax.set_xlabel('Cycle ID', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0.0, 1.20)
    ax.set_ylabel('Normalised Stiffness', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Normalised Stiffness Drop', fontsize=12, fontweight='bold', pad=20)
    plt.legend(loc='lower right')
    # Save to checks folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_normalised-stiffness.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # Create hysteresis loop across cycles
    # -------------------------------------------------------------------------

    # Visualise hysteresis loop
    # Currently this plots the loop from every 250 trials
    # Normalises data to the peak force of the first cycle

    # Create figure to plot loop on
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    # Loop through every 50 cycles starting at first in the dataset
    cycle = test_data['Cycle_ID'].min()
    while cycle < test_data['Cycle_ID'].max():
        # Extract cycle and identify phases
        # First cycle index is incorrect as it starts at 2 (not zero) and hence this needs to be checked
        if cycle == test_data['Cycle_ID'].min():
            cycle_data = test_data.loc[test_data['Cycle_ID'] == cycle,]
            plot_alpha = 1.0  # to demonstrate first cycle
            plot_col = 'red'  # to demonstrate first cycle
            # Plot bending and relaxing data cycle on top
            ax.plot(cycle_data['Displacement'], cycle_data['Force'] / first_cycle_peak_force,
                    color=plot_col, lw=2.0, ls='-', alpha=plot_alpha, zorder=3
                    )
        else:
            cycle_data = test_data.loc[test_data['Cycle_ID'] == cycle,]
            plot_alpha = (0.9 * (cycle - test_data['Cycle_ID'].min()) / (test_data['Cycle_ID'].max() - test_data['Cycle_ID'].min()))
            plot_col = 'blue'
            # Plot bending and relaxing data cycle below initial cycle
            ax.plot(cycle_data['Displacement'], cycle_data['Force'] / first_cycle_peak_force,
                    color=plot_col, lw=2.0, ls='-', alpha=plot_alpha, zorder=2
                    )
        # Add to cycle index
        cycle += 50
    # Set limits and labels
    ax.set_xlim(-0.1, test_data['Displacement'].max()+0.1)
    ax.set_xlabel('Displacement (mm)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0.0, 1.20)
    ax.set_ylabel('Force (N)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{filelabel} Normalised Peak Force Hysteresis Loop (every 50 cycles)', fontsize=12, fontweight='bold', pad=20)
    # Save to checks folder
    fig.savefig(os.path.join(check_dir, f'{filelabel}_normalised-hysteresis-loop.png'),
                format='png', dpi=150)
    # Close figure
    plt.close('all')

    # TODO: quantify area of hysteresis loop...normalised version?
        # > This is the integral of the displacement phase, minus the integral of the relaxation phase
        #   Gives the area of the hysteresis loop




# =========================================================================
# TODO: write appropriate header for here
# =========================================================================

if __name__ == '__main__':

    # TODO: run any appropriate analyses here
    # -------------------------------------------------------------------------

    # TODO: add functions to run

    # Just some test code to produce some plots for act'ble
    # -------------------------------------------------------------------------

    # act_2_data
    # bloch_3_data
    act_col = '#333631'
    bloch_col ='#a57463'

    # # Visualise normalised force change
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    # ax.plot(medfilt(act_2_peak_force['Normalised_Force'],kernel_size=19), ls='-', lw = 2.0, c = act_col,
    #         label="Act'Pointe")
    # ax.plot(medfilt(bloch_3_peak_force['Normalised_Force'], kernel_size=19), ls='-', lw=2.0, c=bloch_col,
    #         label='Traditional Pointe')
    # # ax.plot(medfilt(peak_force['Normalised_Force'], kernel_size=19), 'r-',
    # #         label='Median Filtered (n=5) Normalised Peak Force', lw = 1.5)
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

    # Finalise and exit kernel
    # -------------------------------------------------------------------------

    # Doing this seems to avoid an error code when completing the script run
    os._exit(00)

# %% ---------- end of degradation_analysis.py ---------- %% #
