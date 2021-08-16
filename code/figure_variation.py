#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates variation due to cell-to-cell variation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import config


n_cycles = 1600
n_samples = 500
nominal_exponential_parameter = 1/150
rel_std_devs = [0.5, 2, 5, 20]

def get_retention(cycle_numbers, exp_param):
    retention = 101 - np.exp(exp_param * cycle_numbers)
    return retention

def generate_retention_from_rel_std_dev(rel_std_dev, cycle_numbers):
    
    # Generate distribution of exponential parameters
    std_dev = (rel_std_dev / 100) * nominal_exponential_parameter
    np.random.seed(0)
    exp_param_distribution = np.sort(np.random.normal(nominal_exponential_parameter, std_dev, n_samples))

    # Generate distribution of retention curves
    retention_array = np.zeros((n_samples, len(cycle_numbers)))
    cycle_life_to_80_percent = np.zeros((n_samples, ))
    
    for k, exp_param in enumerate(exp_param_distribution):
        retention_array[k, :] = get_retention(cycle_numbers, exp_param)
        cycle_life_to_80_percent[k] = cycle_numbers[np.where(retention_array[k, :] < 80)[0][0]]
        
    return retention_array, cycle_life_to_80_percent

# Define cycle numbers
cycle_numbers = np.arange(n_cycles)

# Get RSD trends for cycle life and retention
all_rsds = np.linspace(0, 30, 50)
rsd_cycle_lives = np.zeros((50, ))
rsd_retention = np.zeros((50, ))
for k, rel_std_dev in enumerate(all_rsds):
        
    # Get retention
    retention_array, \
        cycle_life_to_80_percent = generate_retention_from_rel_std_dev(rel_std_dev,
                                                                       np.arange(20000))
        
    # Calculate rsd of cycle lives
    rsd_cycle_lives[k] = 100 * np.std(cycle_life_to_80_percent) / np.mean(cycle_life_to_80_percent)
    
    # Calculate rsd of retention at 500 cycles
    rsd_retention[k] = 100 * np.std(retention_array[:, 500]) / np.mean(retention_array[:, 500])

# Generate figure handles
fig, ax = plt.subplots(
    figsize=(3 * config.FIG_WIDTH + 0.5, 2 * config.FIG_HEIGHT),
    nrows=2,
    ncols=3,
)
ax = ax.ravel()

colors = cm.get_cmap('RdBu_r')(np.linspace(0, 1, n_samples))

for k, rel_std_dev in enumerate(rel_std_devs):
    
    k = k + 1
    
    # Get retention
    retention_array, \
        cycle_life_to_80_percent = generate_retention_from_rel_std_dev(rel_std_dev,
                                                                       cycle_numbers)
    # remove negative retentions
    retention_array[retention_array < 0] = 0
        
    # Calculate rsd of cycle lives
    rsd_cyc_life = 100 * np.std(cycle_life_to_80_percent) / np.mean(cycle_life_to_80_percent)
    
    # Calculate rsd of retention at 500 cycles
    rsd_500_cyc = np.std(retention_array[:, 500]) / np.mean(retention_array[:, 500])*100
    
    # Plot
    ax[k].axhline(80, color="tab:gray")
    ax[k].axvline(500, color="tab:gray")
    for k2, retention_trend in enumerate(retention_array):
        ax[k].plot(cycle_numbers, retention_trend, color=colors[k2])

    
    ax[k].annotate("$c$ RSD = " + f"{rel_std_dev:.1f}%" + \
                    "\nCycles to 80% RSD = " + f"{rsd_cyc_life:.1f}%" + \
                    "\nRetention at 500 cycles RSD = " + f"{rsd_500_cyc:.1f}%",
                    xy = (1180, 94), ha='right')
    
    """
    Inset, remove for now
    ax_in = ax[k].inset_axes([100, 20, 250, 40], transform=ax[k].transData)
    ax_in.hist(cycle_life_to_80_percent, range=(400, 550), bins=20, edgecolor='k')
    ax_in.set_xlim([400, 550])
    ax_in.set_ylim([0, n_samples])
    ax_in.set_xlabel("Cycles to 80% retention")
    ax_in.set_ylabel("Count")
    """

    
for k in np.arange(5):
    
    ax[k].plot(cycle_numbers, get_retention(cycle_numbers, 
                                    nominal_exponential_parameter),
       color="k"
       )
    
    # Set axis title
    ax[k].set_title(chr(k + 97), loc="left", weight="bold")
    
    # Set axes labels
    ax[k].set_xlabel("Cycle number")
    ax[k].set_ylabel("Retention (%)")
    
    # Set axes limits
    ax[k].set_xlim([0, 1200])
    ax[k].set_ylim([70, 100])
    
ax[0].annotate("Retention = 100 - $\exp(cn)$\nNominal $c$ = 1/150",
               xy=(1180, 94), ha="right")

ax[5].plot([0, 30], [0, 30], ":", color="tab:gray")
ax[5].plot(all_rsds, rsd_retention, color="tab:purple", label="Retention at 500 cycles")
ax[5].plot(all_rsds, rsd_cycle_lives, color="tab:orange", label="Cycles to 80% retention")
ax[5].set_xlabel("Relative standard deviation of $c$ (%)")
ax[5].set_ylabel("Relative standard deviation of\nlifetime metric (%)")
ax[5].legend()
ax[5].set_xlim([0, 30])
ax[5].set_ylim(bottom=0)
ax[5].set_title("f", loc="left", weight="bold")

# Save figure as both .PNG and .EPS
plt.tight_layout()
fig.savefig(config.FIG_PATH / "variation.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "variation.eps", format="eps")

plt.show()