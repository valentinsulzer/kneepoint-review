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
linear_degradation1 = cycle_numbers ** 0.5 / 1.2
linear_degradation2 = cycle_numbers / 20

cycle_numbers_subset1 = np.arange(322)
linear_degradation31 = cycle_numbers_subset1 / 100
cycle_numbers_subset2 = np.arange(322,500)
linear_degradation32 = cycle_numbers_subset2 / 8 - 37
threshold = 3.2

knee_retention = 100 - snowball_degradation


# Generate figure handles
fig, ax = plt.subplots(
    figsize=(2 * config.FIG_WIDTH, 1.5 * config.FIG_HEIGHT),
    nrows=2,
    ncols=3,
    sharex=True,
    sharey="row",
)
ax = ax.ravel()

# Plot
ax[0].plot(cycle_numbers, knee_retention, color="tab:blue")
ax[1].plot(cycle_numbers, knee_retention, color="tab:purple")
ax[2].plot(cycle_numbers, knee_retention, color="tab:red")
ax[3].plot(cycle_numbers, snowball_degradation, "--", color="tab:blue")
ax[4].plot(cycle_numbers, linear_degradation1, ":", color="tab:purple")
ax[4].plot(cycle_numbers, linear_degradation2, "--", color="tab:purple")
ax[5].plot(cycle_numbers_subset1, linear_degradation31, ":", color="tab:red")
ax[5].plot(cycle_numbers_subset2, linear_degradation32, "--", color="tab:red")
ax[5].axhline(threshold, linestyle="--", color="black")
ax[5].annotate("threshold", (10,3.7))

# Set axes titles
ax[0].set_title("a", loc="left", weight="bold")
ax[1].set_title("b", loc="left", weight="bold")
ax[2].set_title("c", loc="left", weight="bold")
ax[3].set_title("d", loc="left", weight="bold")
ax[4].set_title("e", loc="left", weight="bold")
ax[5].set_title("f", loc="left", weight="bold")

# Set axes labels
ax[3].set_xlabel("Cycle number")
ax[4].set_xlabel("Cycle number")
ax[5].set_xlabel("Cycle number")
ax[0].set_ylabel("Retention (%)")
ax[3].set_ylabel("State variable")

# Set axes limits
ax[0].set_xlim([-0.5, 500.5])
ax[0].set_ylim([72.5, 100.1])
ax[3].set_ylim([-0.5, 30])

# Set legends
ax[4].legend(["State 1", "State 2"], loc="upper left")
ax[5].legend(["State 1", "State 2"], loc="upper left")

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "snowball_vs_hidden_mechanism.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "snowball_vs_hidden_mechanism.eps", format="eps")

plt.show()