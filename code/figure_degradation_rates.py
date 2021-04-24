#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates degradation at different scalings.
"""

import numpy as np
import matplotlib.pyplot as plt

import config


# Generate fake data
cycle_numbers = np.arange(600)

sublinear_degradation = 100 - 0.8 * (cycle_numbers) ** 0.5
linear_degradation = 100 - 3 * cycle_numbers / 100
superlinear_degradation = 100.5 - 0.5 * np.exp(cycle_numbers / 160)

# Generate figure handles
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT), nrows=1, ncols=1,)

# Plot
ax.plot(
    cycle_numbers,
    sublinear_degradation,
    color="tab:blue",
    label="Sublinear degradation",
)
ax.plot(cycle_numbers, linear_degradation, color="k", label="Linear degradation")
ax.plot(
    cycle_numbers,
    superlinear_degradation,
    color="tab:red",
    label="Superlinear degradation",
)

# Set axes labels
ax.set_xlabel("Cycle number")
ax.set_ylabel("Retention (%)")

# Add legend
ax.legend()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "degradation_rates.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "degradation_rates.eps", format="eps")
