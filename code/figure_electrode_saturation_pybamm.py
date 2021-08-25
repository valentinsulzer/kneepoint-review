#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script illustrates a more complex model of electrode saturation.

The simulation is prepared in `simulation_electrode_saturation_pybamm.ipynb`.
"""

import pickle

import matplotlib.pyplot as plt
import pybamm

import config

# Load data
filehandler = open("saturation.pkl","rb")
sumvar = pickle.load(filehandler)


esoh_vars = [("Capacity [A.h]","Capacity (Ah)", "k"), 
             ("Loss of active material in negative electrode [%]", "LAM$_\mathrm{ne}$ (%)", "b"),
             ("Loss of active material in positive electrode [%]", "LAM$_\mathrm{pe}$ (%)", "r"),
             ("Loss of lithium inventory [%]", "LLI (%)","k")
             ]

# Plot
fig, axes = plt.subplots(2, 3,
                         figsize=(config.FIG_WIDTH * 2, config.FIG_HEIGHT * 2),
                         squeeze=False, sharex=True)
axes = axes.flatten()

for k, (long_name, short_name, c) in enumerate(esoh_vars):
    ax = axes[k]
    ax.plot(sumvar["Cycle number"], sumvar[long_name], c+"-")
    ax.set_ylabel(short_name)
    if k > 2:
        ax.set_xlabel("Cycle number")

ax = axes[-2]
for long_name in ["x_0","x_100"]:
    ax.plot(sumvar["Cycle number"], sumvar[long_name], "b-")
ax.set_ylabel("Negative electrode stoich. limits")

ax = axes[-1]
for long_name in ["y_0","y_100"]:
    ax.plot(sumvar["Cycle number"], sumvar[long_name], "r-")
ax.set_ylabel("Positive electrode stoich. limits")

for ax in [axes[-2], axes[-1]]:
    ax.set_xlabel("Cycle number")

for k, ax in enumerate(axes):
    ax.set_title(chr(97 + k), loc="left", weight="bold")
    ax.set_xlim([0, sumvar["Cycle number"][-1] + 30])
    #ax.grid(linestyle=":")

fig.tight_layout()
fig.savefig(config.FIG_PATH / "stoich_knee.eps")
fig.savefig(config.FIG_PATH / "stoich_knee.png", dpi=300, bbox_inches="tight")