#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates knee sensitivity to x axis choice.
"""

import numpy as np
import matplotlib.pyplot as plt

import config


# Generate fake data
cycle_numbers = np.arange(800)

deg1 = 100.5 - 0.5 * np.exp(cycle_numbers / 150)
deg2 = 100.5 - 0.5 * np.exp(cycle_numbers / 120)

deg1_cum_capacity = np.cumsum(deg1)
deg2_cum_capacity = np.cumsum(deg2)

# Generate figure handles
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, 2 * config.FIG_HEIGHT),
                       nrows=2, ncols=1)

# Plot
ax[0].plot(cycle_numbers, deg1, color="tab:blue")
ax[0].plot(cycle_numbers, deg2, color="tab:red")
ax[1].plot(deg1_cum_capacity / 100, deg1, color="tab:blue")
ax[1].plot(deg2_cum_capacity / 100, deg2, color="tab:red")

# Set axes labels
ax[0].set_xlabel("Cycle number")
ax[1].set_xlabel("Capacity throughput / nominal capacity")
ax[0].set_ylabel("Retention (%)")
ax[1].set_ylabel("Retention (%)")

# Set axes limits
for k in np.arange(2):
    ax[k].set_xlim([-5, 605])
    ax[k].set_ylim([50, 100.5])
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")

# Save figure as both .PNG and .EPS
fig.tight_layout()
fig.savefig(config.FIG_PATH / "x_axis_sensitivity.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "x_axis_sensitivity.eps", format="eps")
