#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates snowball vs hidden mechanisms.
"""

import numpy as np
import matplotlib.pyplot as plt

import config


# Generate fake data
cycle_numbers = np.arange(500)

snowball_degradation = np.exp(cycle_numbers / 150) - 1
linear_degradation1 = cycle_numbers / 100
linear_degradation2 = cycle_numbers / 8 - 37

knee_retention = 100 - snowball_degradation


# Generate figure handles
fig, ax = plt.subplots(
    figsize=(2 * config.FIG_WIDTH, 2 * config.FIG_HEIGHT),
    nrows=2,
    ncols=2,
    sharex=True,
    sharey="row",
)
ax = ax.ravel()

# Plot
ax[0].plot(cycle_numbers, knee_retention, color="tab:blue")
ax[1].plot(cycle_numbers, knee_retention, color="tab:purple")
ax[2].plot(cycle_numbers, snowball_degradation, "--", color="tab:blue")
ax[3].plot(cycle_numbers, linear_degradation1, ":", color="tab:purple")
ax[3].plot(cycle_numbers, linear_degradation2, "--", color="tab:purple")

# Set axes titles
ax[0].set_title("a", loc="left", weight="bold")
ax[1].set_title("b", loc="left", weight="bold")
ax[2].set_title("c", loc="left", weight="bold")
ax[3].set_title("d", loc="left", weight="bold")

# Set axes labels
ax[2].set_xlabel("Cycle number")
ax[3].set_xlabel("Cycle number")
ax[0].set_ylabel("Retention (%)")
ax[2].set_ylabel("State variable")

# Set axes limits
ax[0].set_xlim([-0.5, 500.5])
ax[0].set_ylim([72.5, 100.1])
ax[2].set_ylim([-0.5, 30])

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "snowball_vs_hidden_mechanism.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "snowball_vs_hidden_mechanism.eps", format="eps")
