#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates how capacity and resistance knees are often correlated.

Obtained from Ecker et al.
10.1016/j.jpowsour.2013.09.143
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import config


# Identify and load data
file = Path().cwd() / "data" / "ecker_capacity_and_resistance_data.xlsx"

df_capacity = pd.read_excel(file, sheet_name="Capacity")
df_resistance = pd.read_excel(file, sheet_name="Resistance")

# Define colors
colors = cm.get_cmap('copper')(np.linspace(0.8, 0.0, 7))

# Plot vs. cycle number
# Generate figure handles
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT * 2),
                       nrows=2, ncols=1, sharex=True)
ax = ax.ravel()

# Plot
# Since some of the series are duplicate cells, keep track of labels used
unique_labels = set()
label_counter = 0

n_data_series = int(len(df_capacity.columns) / 2)

for k in np.arange(n_data_series):
    
    # Capacity/resistance
    efc_capacity = df_capacity[df_capacity.columns[2 * k]][1:]
    norm_capacity = df_capacity[df_capacity.columns[2 * k + 1]][1:]
    
    efc_resistance = df_resistance[df_resistance.columns[2 * k]][1:]
    norm_resistance = df_resistance[df_resistance.columns[2 * k + 1]][1:]
    
    # Labels and colors
    label = df_capacity.columns[2 * k].split("%")[0] + "% DOD"
    
    if label not in unique_labels:
        unique_labels.add(label)
        label_counter += 1
        temp_label = label
    else:
        temp_label = ""
    print(f"{label}: Group {label_counter}")
    
    # Plot
    ax[0].plot(efc_capacity, norm_capacity, "-", marker=".",
               color = colors[label_counter],
               label=temp_label)
    
    ax[1].plot(efc_resistance, norm_resistance, "-", marker=".",
               color = colors[label_counter],
               label=temp_label)
    

for k in range(len(ax)):
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    ax[k].set_xlim(left=-90)
    l = ax[k].legend(title="Ecker et al.\nNMC/graphite cylindrical cells\n1C/1C, 35°C")
    plt.setp(l.get_title(), multialignment='center')

ax[1].set_xlabel("Equivalent full cycles")
ax[0].set_ylabel('Capacity retention (%)')
ax[1].set_ylabel('Normalized resistance (%)')
ax[0].set_ylim([40, 100])
ax[1].set_ylim([100, 400])
    
fig.tight_layout()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "ecker_remade.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "ecker_remade.eps", format="eps")