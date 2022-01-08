#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates how temperature and pressure are best at intermediate values
on knees.

John Cannarella graciously provided data for panel b.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import config


# Identify and load data
path = Path().cwd() / "data" / "temperature_and_pressure"

waldmann_files = (path / "waldmann").glob("*.csv")

waldmann_dict = {}
for file in waldmann_files:
    
    temperature = int(file.stem.split("_")[2][:-1].replace("neg", "-"))
    
    df_temp = pd.read_csv(file, names=["time", "soh"])
    
    # Add t=0 points manually
    row0 = pd.DataFrame({"time": 0, "soh": 100}, index=[0])
    df_temp = pd.concat([row0, df_temp]).reset_index(drop=True)
    
    waldmann_dict[temperature] = df_temp
    
df_pressure = pd.read_excel(path / "cannarella_Figure5.xlsx")

# Define colors
colors_temp = cm.get_cmap('bwr')(np.linspace(0.0, 1.0, len(waldmann_dict)))
colors_temp[3] = [0, 0, 0, 1]
colors_pressure = cm.get_cmap('seismic')(np.linspace(0.0, 1.0, 4))
colors_pressure[0] = colors_pressure[1]
colors_pressure[1] = [0, 0, 0, 1]

# Plot vs. cycle number
# Generate figure handles
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT * 2),
                       nrows=2, ncols=1)
ax = ax.ravel()

# Plot
for k, (temperature, df) in enumerate(sorted(waldmann_dict.items())):
    ax[0].plot(df["time"], df["soh"], "-o",
               label=f"{temperature}°C",
               color=colors_temp[k])

max_cycle_numbers = {
    "0 MPa": 2000,
    "0.05 MPa": 2000,
    "0.5 MPa": 2000,
    "5 MPa": 640
}

for k, column in enumerate(df_pressure.columns[::-1]):
    if k % 2 == 1:
        
        # Get data for main trendline
        max_cycle_number = max_cycle_numbers[column]
        y_data = 100 * df_pressure[column].iloc[1:max_cycle_number].dropna()
        
        # Get data for errorbars
        y_err = 100 * df_pressure[df_pressure.columns[7 - k:9 - k]].iloc[1:max_cycle_number].dropna()
        y_data_errorbars = y_err[column]
        y_error_errorbars = y_err[y_err.columns[1]]
                
        ax[1].errorbar(y_err.index, y_data_errorbars, yerr=y_error_errorbars, 
                    color=colors_pressure[int(k / 2)], fmt='.', markersize=1)
        ax[1].plot(y_data.index, y_data,
                   label=column,
                   color=colors_pressure[int(k / 2)])

leg_titles = [
    "    Waldmann et al.\n  NMC:LMO/graphite\n     cylindrical cells\n            1C/1C",
    "  Cannarella and Arnold\nLCO/graphite pouch cells\n      0.5C/0.5C, ~25°C"
]
leg_locs = ["upper right", "lower right"]

for k in range(len(ax)):
    ax[k].set_ylabel('Capacity retention (%)')
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    leg = ax[k].legend(title=leg_titles[k], loc=leg_locs[k])
    leg._legend_box.align = "right" if k == 0 else "center"
    
ax[0].set_xlim([-1, 95])
ax[1].set_xlim([-10, 2005])
ax[1].set_ylim(top=100.5)
ax[0].set_xlabel("Time (days)")
ax[1].set_xlabel("Cycle number")

    
fig.tight_layout()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "temperature_and_pressure.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "temperature_and_pressure.eps", format="eps")