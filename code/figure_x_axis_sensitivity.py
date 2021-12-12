#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates knee sensitivity to x axis choice.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import config


# Generate fake data
cycle_numbers = np.arange(800)

deg1 = 100.5 - 0.5 * np.exp(cycle_numbers / 150)
deg2 = 100.5 - 0.5 * np.exp(cycle_numbers / 120)

deg1_cum_capacity = np.cumsum(deg1)
deg2_cum_capacity = np.cumsum(deg2)

# Load Wang data
file = Path().cwd() / "data" / "wang_dod-cycle_count-time.xlsx"
wang_data = pd.read_excel(file, sheet_name=None)
wang_labels = ["10% DOD", "20% DOD", "50% DOD", "80% DOD", "90% DOD"]

# Generate figure handles
fig, ax = plt.subplots(figsize=(2 * config.FIG_WIDTH, 2 * config.FIG_HEIGHT),
                       nrows=2, ncols=2)
ax = ax.ravel()

# Plot cycle number/throughput data
ax[0].plot(cycle_numbers, deg1, color="tab:blue", label="Cell 1")
ax[0].plot(cycle_numbers, deg2, color="tab:red", label="Cell 2")
ax[1].plot(deg1_cum_capacity / 100, deg1, color="tab:blue", label="Cell 1")
ax[1].plot(deg2_cum_capacity / 100, deg2, color="tab:red", label="Cell 2")

# Set axes labels
ax[0].set_xlabel("Cycle number")
ax[1].set_xlabel("Capacity throughput / nominal capacity")

# Set axes limits
for k in np.arange(2):
    ax[k].set_xlim([-5, 605])
    ax[k].set_ylim([50, 100.5])
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    ax[k].set_ylabel("Retention (%)")
    ax[k].legend(loc="lower left", title="Synthetic data")

# Plot Wang data (cycle number vs time)
columns = wang_data["Cycle Number"].columns
colors = cm.get_cmap('inferno')(np.linspace(0.8, 0.0, 5))

for k in np.arange(5):
    ax[2].plot(wang_data["Cycle Number"][columns[2 * k]].iloc[1:],
               wang_data["Cycle Number"][columns[2 * k + 1]].iloc[1:],
               '-o', label=wang_labels[k], color=colors[k]
              )
    ax[3].plot(wang_data["Days"][columns[2 * k]].iloc[1:],
           wang_data["Days"][columns[2 * k + 1]].iloc[1:],
           '-o', label=wang_labels[k], color=colors[k]
          )

# Settings
for k in [2, 3]:
    ax[k].legend(title="Wang et al.", loc="upper right")
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    ax[k].set_ylabel("Capacity retention (%)")
    
ax[3].annotate("\nLFP/graphite cylindrical cells\nC/2 charge/discharge, 60°C",
               xy = (5, 67))

ax[2].set_xlim([-1, 1e4])
ax[3].set_xlim([-1, 200])
ax[2].set_xlabel("Cycle number")
ax[3].set_xlabel("Time (days)")

# Save figure as both .PNG and .EPS
fig.tight_layout()
fig.savefig(config.FIG_PATH / "x_axis_sensitivity.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "x_axis_sensitivity.eps", format="eps")
