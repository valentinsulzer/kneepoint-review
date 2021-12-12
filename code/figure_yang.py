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


# Identify and load data
path = Path().cwd() / "data" / "yang"

files = path.glob("*.csv")

data_dict = {}
for file in files:
    data_dict[file.stem] = pd.read_csv(file, header=None, names=["x", "y"])

# Plot vs. cycle number
# Generate figure handles
fig, ax = plt.subplots(figsize=(2 * config.FIG_WIDTH, 2 * config.FIG_HEIGHT),
                       nrows=2, ncols=2)
ax = ax.ravel()

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
ax[0].stackplot(cycles, y_a1, y_a2,
                labels=["Lithium inventory loss due to SEI growth",
                        "Lithium inventory loss due to lithium plating"],
                colors=[sei_green, li_silver])
ax[1].plot(data_dict["yang_b1"].x, data_dict["yang_b1"].y,
           color=sei_green,
           label="Lithium inventory loss due to SEI growth\nper cycle")
ax[1].plot(data_dict["yang_b2"].x, data_dict["yang_b2"].y,
           color=li_silver,
           label="Lithium inventory loss due to lithium plating\nper cycle")


for k in range(2):
    ax[k].set_xlabel("Cycle number")
    ax[k].set_xlim([0, 3250])
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    ax[k].legend(loc="upper left")

ax[0].set_ylabel("Total capacity loss (%)")
ax[1].set_ylabel("Capacity loss per cycle (%)")
ax[0].set_ylim([0, 30])
ax[1].set_ylim([0, 0.02])

ax[2].set_title("c", loc="left", weight="bold")
ax[2].axis("off"), ax[3].axis("off")
    
fig.tight_layout()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "yang_remade.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "yang_remade.eps", format="eps")