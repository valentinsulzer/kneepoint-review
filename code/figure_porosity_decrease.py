#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure remakes a plot from Yang et al.
10.1016/j.jpowsour.2017.05.110
"""

from pathlib import Path

import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import config


# Identify and load Yang data
path_yang = Path().cwd() / "data" / "yang"

files_yang = path_yang.glob("*.csv")

data_dict = {}
for file in files_yang:
    data_dict[file.stem] = pd.read_csv(file, header=None, names=["x", "y"])

# Load Frisco data
path_frisco = Path().cwd() / "data" / "frisco_pore_radius.xlsx"
data_frisco = pd.read_excel(path_frisco).dropna()
data_frisco.rename(columns={"Unnamed: 0": "bar",
                            "Unnamed: 1": "value",
                            "Unnamed: 2": "dataset"
                            }, inplace=True)
data_frisco_pristine = data_frisco[data_frisco.dataset == "Pristine"]
data_frisco_cycled = data_frisco[data_frisco.dataset == "Cycled"]

# Load Petzl data
path_petzl = Path().cwd() / "data" / "petzl_eis_data.xlsx"
data_petzl = pd.read_excel(path_petzl).iloc[1: , :]
data_petzl.rename(columns={"Cycle 0": "Z_re_0cycle",
                           "Unnamed: 1": "Z_im_0cycle",
                           "Cycle 40": "Z_re_40cycle",
                           "Unnamed: 3": "Z_im_40cycle",
                           "Cycle 80": "Z_re_80cycle",
                           "Unnamed: 5": "Z_im_80cycle",
                           "Cycle 120": "Z_re_120cycle",
                           "Unnamed: 7": "Z_im_120cycle"
                           }, inplace=True)

# Plot
# Generate figure handles
fig, ax = plt.subplots(figsize=(2 * config.FIG_WIDTH, 3 * config.FIG_HEIGHT),
                       nrows=3, ncols=2)
ax = ax.ravel()

# Placeholders -- ax[0] and ax[1]


# Plot Yang data
# Specify colors
sei_green = (190/255, 232/255, 170/255)
li_silver = (171/255, 171/255, 171/255)

# Load data for (a) from the integral of (b)
integral_b1 = cumtrapz(data_dict["yang_b1"].y, data_dict["yang_b1"].x)
integral_b2 = cumtrapz(data_dict["yang_b2"].y, data_dict["yang_b2"].x)

cycles = np.arange(0, 3300, 10)
y_a1 = np.interp(cycles, data_dict["yang_b1"].x[1:], integral_b1)
y_a2 = np.interp(cycles, data_dict["yang_b2"].x[1:], integral_b2)

# Plot
ax[2].stackplot(cycles, y_a1, y_a2,
                labels=["Lithium inventory loss due to SEI growth",
                        "Lithium inventory loss due to lithium plating"],
                colors=[sei_green, li_silver])
ax[3].plot(data_dict["yang_b1"].x, data_dict["yang_b1"].y,
           color=sei_green,
           label="Lithium inventory loss due to SEI growth\nper cycle")
ax[3].plot(data_dict["yang_b2"].x, data_dict["yang_b2"].y,
           color=li_silver,
           label="Lithium inventory loss due to lithium plating\nper cycle")


for k in range(2):
    ax[k + 2].set_xlabel("Cycle number")
    ax[k + 2].set_xlim([0, 3250])
    ax[k + 2].legend(loc="upper left", title="Yang et al.")

ax[2].set_ylabel("Total capacity loss (%)")
ax[3].set_ylabel("Capacity loss per cycle (%)")
ax[2].set_ylim([0, 30])
ax[3].set_ylim([0, 0.02])

# Plot Frisco data
x = np.linspace(80, 1450, len(data_frisco_cycled))
width = 25
ax[4].bar(x - width/2, data_frisco_pristine.value, width, label="Pristine", color="tab:blue")
ax[4].bar(x + width/2, data_frisco_cycled.value, width, label="Cycled (400 cycles, post-knee)", color="tab:red")
ax[4].set_xlabel("Pore radius (nm)")
ax[4].set_ylabel("Percent total volume (%)")
ax[4].set_xlim([0, 1495])
ax[4].set_ylim([0, 4.3])
ax[4].legend(title="Frisco et al.\nNMC/graphite cylindrical cell\n0.33C/0.33C, ~24°C")

for letter_idx, plot_index in enumerate([0, 2, 3, 4, 5]):
    ax[plot_index].set_title(chr(97 + letter_idx), loc="left", weight="bold")
ax[0].axis("off"), ax[1].axis("off"), ax[5].axis("off")
    
fig.tight_layout()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "porosity_to_plating.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "porosity_to_plating.eps", format="eps")