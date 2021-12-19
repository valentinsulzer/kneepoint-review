#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates how discharge rate and rest can have good and bad effects
on knees.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import config


# Identify and load data
path = Path().cwd() / "data" / "discharge_and_rest"

files = path.glob("*.xlsx")

data_dict = {}
for file in files:
    
    # Skip raw workbook
    if "raw" in file.stem:
        continue
    
    sheet_to_df_map = pd.read_excel(file, sheet_name=None)
    data_dict[file.stem] = sheet_to_df_map


# Define colors
colors_discharge = cm.get_cmap('plasma')(np.linspace(0.8, 0.0, 4))
colors_rest = cm.get_cmap('viridis')(np.linspace(0.8, 0.0, 3))

# Plot vs. cycle number
# Generate figure handles
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH * 2, config.FIG_HEIGHT * 2),
                       nrows=2, ncols=2)
ax = ax.ravel()

# Plot
for k, (key, value) in enumerate(reversed(list(data_dict["omar_dischargebad"].items()))):
    ax[0].plot(value["Cycle (EFC?)"], value["% Capacity"],
               label="1C/"+key.replace('It', 'C'),
               color=colors_discharge[k])
    
for k, (key, value) in enumerate(data_dict["keil_dischargegood"].items()):
    ax[1].plot(value["EFC"], value["% Capacity"],
               label=key.replace('-', '/'),
               color=colors_discharge[2*k])
    
for k, (key, value) in enumerate(reversed(list(data_dict["keil_restbad"].items()))):
    ax[2].plot(value["EFC"], value["% Capacity"],
               label=key.replace('-', '/').split(', ')[1],
               color=colors_rest[2*k])
    
for k, (key, value) in enumerate(data_dict["epding_restgood"].items()):
    ax[3].plot(value["Cycle (EFC?)"], value["% Capacity"],
               label=key.replace('-', '/').replace("2 day", "48h"),
               color=colors_rest[k])
    
titles = [
    "Omar et al.\nLFP/graphite cylindrical cells\n~25°C ($T_{surface}$ reached 55°C)",
    "Keil et al.\nNMC/graphite cylindrical cells\n~25°C",
    "Keil et al.\nNMC/graphite cylindrical cells\n1C/1C, ~25°C",
    "Epding et al.\nNMC/graphite\nprismatic cells\n2C/1C, 10°C"
]
leg_locations = [
    "upper right",
    "lower left",
    "lower left",
    "lower left"
]
leg_alignments = [
    "right",
    "left",
    "left",
    "left"
]

for k in range(len(ax)):
    ax[k].set_ylabel('Capacity retention (%)')
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    leg = ax[k].legend(title=titles[k], loc=leg_locations[k])
    leg._legend_box.align = leg_alignments[k]
    
ax[0].set_xlabel("Cycles")
ax[1].set_xlabel("Equivalent full cycles")
ax[2].set_xlabel("Equivalent full cycles")
ax[3].set_xlabel("Cycles")
    
fig.tight_layout()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "discharge_rate_rest_cycles.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "discharge_rate_rest_cycles.eps", format="eps")


## PLOT VS TIME
# Generate figure handles
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH * 2, config.FIG_HEIGHT * 2),
                       nrows=2, ncols=2)
ax = ax.ravel()

# Plot
for k, (key, value) in enumerate(reversed(list(data_dict["omar_dischargebad"].items()))):
    ax[0].plot(value["Time cycled (h)"], value["% Capacity"],
               label="1C/"+key.replace('It', 'C'),
               color=colors_discharge[k])
    
for k, (key, value) in enumerate(data_dict["keil_dischargegood"].items()):
    ax[1].plot(value["Time cycled (h)"], value["% Capacity"],
               label=key.replace('-', '/'),
               color=colors_discharge[2*k])
    
for k, (key, value) in enumerate(reversed(list(data_dict["keil_restbad"].items()))):
    ax[2].plot(value["Time cycled (h)"], value["% Capacity"],
               label=key.replace('-', '/').split(', ')[1],
               color=colors_rest[2*k])
    
for k, (key, value) in enumerate(data_dict["epding_restgood"].items()):
    ax[3].plot(value["Time Cycled (h)"], value["% Capacity"],
               label=key.replace('-', '/').replace("2 day", "48h"),
               color=colors_rest[k])

leg_locations = [
    "upper right",
    "lower left",
    "lower left",
    "lower left"
]
leg_alignments = [
    "right",
    "left",
    "left",
    "left"
]
for k in range(len(ax)):
    ax[k].set_xlabel("Estimated cycle time (h)")
    ax[k].set_ylabel('Capacity retention (%)')
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    leg = ax[k].legend(title=titles[k], loc=leg_locations[k])
    leg._legend_box.align = leg_alignments[k]
    
fig.tight_layout()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "discharge_rate_rest_time.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "discharge_rate_rest_time.eps", format="eps")