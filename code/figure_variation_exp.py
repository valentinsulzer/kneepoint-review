#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates variation due to cell-to-cell variation.

Data/Figures:
    - Baumhofer et al, DOI: 
        Figure 6
        Data obtained from Philipp
    - Harris et al, DOI:
        Figure X 
        Data obtained from Steve
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as sio
import pandas as pd

import config

path = Path().cwd() / "data"
data_baumhofer = sio.loadmat(path / "variability" / "baumhofer.mat", 
                   simplify_cells=True)["lifetime"]
data_harris = pd.read_excel(path / "variability" / "harris2017.xlsx")

colors = cm.get_cmap('viridis_r')(np.linspace(0, 1, 48))

# Define Baumhofer color order
color_ordering_dict1 = {}
for cell_name, cell_data in data_baumhofer.items():
    color_ordering_dict1[cell_name] = cell_data["cap_aged"]
    
color_ordering_list = sorted(color_ordering_dict1.items(), key=lambda kv: kv[1])

color_ordering_dict2 = {}
for k, (cell_name, cell_data) in enumerate(color_ordering_list):
    color_ordering_dict2[cell_name] = k
    
# Define Harris color order
data_harris.sort_values(by=550, axis=1, inplace=True)


# Generate figure handles
fig, ax = plt.subplots(
    figsize=(config.FIG_WIDTH, 2 * config.FIG_HEIGHT),
    nrows=2,
    ncols=1,
)
ax = ax.ravel()

# Plot Baumhofer
for k, (cell_name, cell_data) in enumerate(data_baumhofer.items()):
    ax[0].plot(cell_data["cyc"], 100 * cell_data["cap"] / cell_data["cap"][0],
               color=colors[color_ordering_dict2[cell_name]])

# Plot Harris
for k, col in enumerate(data_harris.columns[:-1]):
    ax[1].plot(data_harris["Cycles"].iloc[:593],
               100 * data_harris[col].iloc[:593],
               color=colors[2 * k])

for k in np.arange(2):
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    ax[k].set_ylim([60, 100])
    ax[k].set_xlabel("Cycle number")
    ax[k].set_ylabel("Capacity retention (%)")

ax[0].set_xlim([0, 1800])
ax[1].set_xlim([0, 600])

ax[0].annotate("48 NMC/graphite cylindrical cells\n~1C/~1C between 3.5 V and 3.9 V\n25°C",
               xy=(90, 62))
ax[1].annotate("24 LCO/graphite pouch cells\n1C/10C between 3.0 V and 4.35 V\n25°C",
               xy=(30, 62))

# Save figure as both .PNG and .EPS
plt.tight_layout()
fig.savefig(config.FIG_PATH / "variation_exp.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "variation_exp.eps", format="eps")

plt.show()