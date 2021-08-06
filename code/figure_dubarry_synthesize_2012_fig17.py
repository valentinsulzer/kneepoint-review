#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reproduces Figure 17 of Dubarry et al.:
https://doi.org/10.1016/j.jpowsour.2012.07.016

The .mat file was obtained from Matthieu
"""

from pathlib import Path

import scipy.io as sio
import matplotlib.pyplot as plt

import config

path = Path().cwd() / "data"
data = sio.loadmat(path / "dubarry_synthesize_2012_fig17.mat")

# Generate figure handles
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT), nrows=1, ncols=1)

# Plots
ax.plot(
    data["x2"].flatten(),
    data["y2"].flatten(),
    "--",
    color="tab:blue",
    label="Loss of lithium inventory"
)

ax.plot(
    data["x3"].flatten(),
    data["y3"].flatten(),
    ":",
    color="tab:red",
    label="Loss of delithiated cathode active material"
)

ax.plot(
    data["x1"].flatten(),
    data["y1"].flatten(),
    color="k",
    label="Calculated capacity loss"
)


# Set axes labels
ax.set_xlabel("Cycle number")
ax.set_ylabel("Degradation or capacity loss (%)")

# Add axes limits
ax.set_xlim([0, 400])
ax.set_ylim([0, 100])

# Add legend
ax.legend()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "dubarry_cathode_saturation.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "dubarry_cathode_saturation.eps", format="eps")