#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure adapts Figure 5 of Kupper et al.
https://iopscience.iop.org/article/10.1149/2.0941814jes/pdf
"""

import numpy as np
import matplotlib.pyplot as plt

import config


def relationship3(s):
    # a = 0.5·tanh(10·(s −0.5))+0.5
    return 0.5 * np.tanh(10 * (s - 0.5)) + 0.5

def relationship4(s):
    # a = (0.5·s +0.5)·(0.5·tanh(15·(s −0.5))+0.5)
    # Note a typo in Kupper Figure 5, which has the below relationship:
    # a = (0.5·s +0.5)·(0.5·tanh(15·(s −0.4))+0.5)
    return (0.5 * s + 0.5) * (0.5 * np.tanh(15 * (s - 0.5)) + 0.5)

# Generate fake data
saturation = np.arange(0, 1, 0.01)

activity3 = relationship3(saturation)
activity4 = relationship4(saturation)

# Generate figure handles
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT), nrows=1, ncols=1)

# Plot
ax.plot(
    saturation,
    saturation,
    ':',
    color='tab:gray'
)

ax.plot(
    saturation,
    activity3,
    color="gold",
    label="Relationship 3 (nonlinear, symmetric at $s = 0.5$):\n" + \
        r"$a = 0.5 \cdot \tanh(10(s - 0.5)) + 0.5$",
)

ax.plot(
    saturation,
    activity4,
    color="tab:purple",
    label="Relationship 4 (nonlinear, asymmetric):\n" + \
        r"$a = (0.5s + 0.5)\cdot(0.5\cdot\tanh(15(s - 0.5)) + 0.5)$",
)

# Set axes labels
ax.set_xlabel("Saturation (dimensionless)")
ax.set_ylabel("Activity (dimensionless)")

# Add axes limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Add legend
ax.legend(bbox_to_anchor=(0, 1.02), loc='lower left', title="Kupper et al.")

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "percolation.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "percolation.eps", format="eps")
