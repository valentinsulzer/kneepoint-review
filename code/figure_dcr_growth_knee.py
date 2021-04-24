#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates DCR growth knees.
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd

import config


# URL of NMC capacity curve from BatteryArchive
url = "https://www.batteryarchive.org/data/SNL_18650_NMC_25C_0-100_0.5-0.5C_b_timeseries.csv"

# Cell capacity, from https://www.batteryarchive.org/list.html
capacity_Ah = 3.0

# DCR growth per cycle
dcr_growth_per_cycle_mohms = 0.2

# Lower cutoff voltage
lower_cutoff_voltage = 2.0

# Define C rates
C_rate1 = 1
C_rate2 = 2

## SCRIPT STARTS HERE

# Define currents
current1_A = capacity_Ah * C_rate1
current2_A = capacity_Ah * C_rate2

# Generate fake data
cycle_numbers = np.arange(1000)
cycle_numbers_short = np.arange(0, 1000, 100)
dcr_growth_mohms = dcr_growth_per_cycle_mohms * cycle_numbers

# Calculate overpotential from Ohm's law
overpotential_growth1_V = current1_A * dcr_growth_mohms / 1000
overpotential_growth2_V = current2_A * dcr_growth_mohms / 1000

# Read in dataset and get first discharge
df = pd.read_csv(url, nrows=2000)
df_discharge1 = df[(df["Cycle_Index"] == 1) & (df["Current (A)"] < 0)]
Qd = df_discharge1["Discharge_Capacity (Ah)"]
Ed = df_discharge1["Discharge_Energy (Wh)"]
Vd = df_discharge1["Voltage (V)"]

# Calculate voltage curves, capacity endpoints, and energy endpoints
Vd1_list, Vd2_list = [], []
Qd1_endpoints = np.zeros((len(cycle_numbers), ))
Qd2_endpoints = np.zeros((len(cycle_numbers), ))
Ed1_endpoints = np.zeros((len(cycle_numbers), ))
Ed2_endpoints = np.zeros((len(cycle_numbers), ))
for k, cycle_number in enumerate(cycle_numbers):
    
    overpotential1_V = current1_A * dcr_growth_per_cycle_mohms * cycle_number / 1000
    overpotential2_V = current2_A * dcr_growth_per_cycle_mohms * cycle_number / 1000
    
    idx_below_2V_1 = np.where(Vd - overpotential1_V < lower_cutoff_voltage)[0][0]
    idx_below_2V_2 = np.where(Vd - overpotential2_V < lower_cutoff_voltage)[0][0]
    
    Qd1_endpoints[k] = Qd.iloc[idx_below_2V_1]
    Qd2_endpoints[k] = Qd.iloc[idx_below_2V_2]
    
    # Calculate energy by getting endpoint energy - V*Q
    # Could also do something like this (yields similar result):
    # np.trapz(Vd[:idx_below_2V_1] - overpotential1_V, Qd[:idx_below_2V_1])
    Ed1_endpoints[k] = Ed.iloc[idx_below_2V_1] - overpotential1_V * Qd1_endpoints[k]
    Ed2_endpoints[k] = Ed.iloc[idx_below_2V_2] - overpotential2_V * Qd2_endpoints[k]
    
    if cycle_number in cycle_numbers_short:
        Vd1_list.append(Vd - overpotential1_V)
        Vd2_list.append(Vd - overpotential2_V)


# Generate figure handles
fig, ax = plt.subplots(
    figsize=(2 * config.FIG_WIDTH, 3 * config.FIG_HEIGHT),
    nrows=3, ncols=2,
)
ax = ax.ravel()

# Plot DCR growth vs. cycle number
ax[0].plot(cycle_numbers, dcr_growth_mohms, color="k")
ax[0].annotate(f"DCR growth rate = {dcr_growth_per_cycle_mohms} m$\Omega$/cycle",
               (15, 195), va="center")

# Plot overpotential growth vs. cycle number
ax[1].plot(cycle_numbers, overpotential_growth1_V, color="tab:blue",
           label=f"{C_rate1}C discharge ({current1_A} A)")
ax[1].plot(cycle_numbers, overpotential_growth2_V, color="tab:orange",
           label=f"{C_rate2}C discharge ({current2_A} A)")

ax[1].annotate(f"Cell capacity = {capacity_Ah} Ah",
               (15, 1.18), va="center")


# Plot voltage vs. capacity
ax[2].axhline(lower_cutoff_voltage, color="tab:gray")
ax[3].axhline(lower_cutoff_voltage, color="tab:gray")

colors = cm.viridis(np.linspace(0.9, 0.3, 10))[:,:10]

for k, Vd1 in enumerate(Vd1_list):
    idx_below_2V= np.where(Vd1 < lower_cutoff_voltage)[0][0]
    
    ax[2].plot(Qd, Vd1, color=colors[k],
               label=f"Cycle {cycle_numbers_short[k]}" if k % 2 == 0 else None)
    ax[2].plot(Qd.iloc[idx_below_2V], lower_cutoff_voltage, 'xk')
   
for k, Vd2 in enumerate(Vd2_list):
    idx_below_2V = np.where(Vd2 < lower_cutoff_voltage)[0][0]
    
    ax[3].plot(Qd, Vd2, color=colors[k],
               label=f"Cycle {cycle_numbers_short[k]}" if k % 2 == 0 else None)
    ax[3].plot(Qd.iloc[idx_below_2V], lower_cutoff_voltage, 'xk')

ax[2].annotate(f"Min voltage = {lower_cutoff_voltage} V",
               (1.2, lower_cutoff_voltage + 0.1), va="center")
ax[3].annotate(f"Min voltage = {lower_cutoff_voltage} V",
               (1.2, lower_cutoff_voltage + 0.1), va="center")
    
# Plot capacity retention
ax[4].plot(cycle_numbers, 100 * Qd1_endpoints / Qd1_endpoints[0], color="tab:blue",
           label=f"{C_rate1}C discharge")
ax[4].plot(cycle_numbers, 100 * Qd2_endpoints / Qd2_endpoints[0], color="tab:orange",
           label=f"{C_rate2}C discharge")

# Plot energy retention
ax[5].plot(cycle_numbers, 100 * Ed1_endpoints / Ed1_endpoints[0], color="tab:blue",
           label=f"{C_rate1}C discharge")
ax[5].plot(cycle_numbers, 100 * Ed2_endpoints / Ed2_endpoints[0], color="tab:orange",
           label=f"{C_rate2}C discharge")

# Set axes labels
ax[0].set_ylabel("DCR growth (m$\Omega$)")
ax[1].set_ylabel("Overpotential growth (V)")
ax[2].set_ylabel("Voltage (V)")
ax[3].set_ylabel("Voltage (V)")
ax[4].set_ylabel("Capacity retention (%)")
ax[5].set_ylabel("Energy retention (%)")

# Set axes limits
for k in [0, 1, 4, 5]:
    ax[k].set_xlabel("Cycle number")
    ax[k].set_xlim([-0.5, 1000.5])
    
for k in [2, 3]:
    ax[k].set_xlabel("Capacity (Ah)")
    ax[k].set_xlim([-0.01, 3.01])
    ax[k].set_ylim([0.2, 4.3])
  
ax[4].set_ylim([84.5, 100.5])
ax[5].set_ylim([59.5, 101])
    
# Set axes titles and legends
legend_title_dict = {2: f"{C_rate1}C discharge",
                     3: f"{C_rate2}C discharge"}
for k in np.arange(6):
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    ax[k].legend(loc="lower left" if k > 1 else "lower right",
                 title=legend_title_dict[k] if k in [2, 3] else None,
                 frameon=True if k in [2, 3] else None,
                 framealpha=1)


# Save figure as both .PNG and .EPS
plt.tight_layout()
fig.savefig(config.FIG_PATH / "dcr_growth_knee.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "dcr_growth_knee.eps", format="eps")
