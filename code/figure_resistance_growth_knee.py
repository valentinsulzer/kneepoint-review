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

# Define lower cutoff voltages
lower_cutoff_voltage1 = 2.0
lower_cutoff_voltage2 = 2.8
lower_cutoff_voltage3 = 2.0

# Define C rates
C_rate1 = 1
C_rate2 = 1
C_rate3 = 2

# Define colors
main_colors = ["tab:blue", "tab:blue", "tab:red"]

## SCRIPT STARTS HERE

# Define currents
current1_A = capacity_Ah * C_rate1
current2_A = capacity_Ah * C_rate2
current3_A = capacity_Ah * C_rate3

# Generate fake data
cycle_numbers = np.arange(1000)
cycle_numbers_short = np.arange(0, 1000, 100)
dcr_growth_mohms = dcr_growth_per_cycle_mohms * cycle_numbers

# Calculate overpotential from Ohm's law
overpotential_growth1_V = current1_A * dcr_growth_mohms / 1000
overpotential_growth2_V = current2_A * dcr_growth_mohms / 1000
overpotential_growth3_V = current3_A * dcr_growth_mohms / 1000

# Read in dataset and get first discharge
df = pd.read_csv(url, nrows=2000)
df_discharge1 = df[(df["Cycle_Index"] == 1) & (df["Current (A)"] < 0)]
Qd = df_discharge1["Discharge_Capacity (Ah)"]
Ed = df_discharge1["Discharge_Energy (Wh)"]
Vd = df_discharge1["Voltage (V)"]

# Calculate voltage curves, capacity endpoints, and energy endpoints
Vd1_list, Vd2_list, Vd3_list = [], [], []
Qd1_endpoints = np.zeros((len(cycle_numbers), ))
Qd2_endpoints = np.zeros((len(cycle_numbers), ))
Qd3_endpoints = np.zeros((len(cycle_numbers), ))
Ed1_endpoints = np.zeros((len(cycle_numbers), ))
Ed2_endpoints = np.zeros((len(cycle_numbers), ))
Ed3_endpoints = np.zeros((len(cycle_numbers), ))
Pd1_endpoints = np.zeros((len(cycle_numbers), ))
Pd2_endpoints = np.zeros((len(cycle_numbers), ))
Pd3_endpoints = np.zeros((len(cycle_numbers), ))

for k, cycle_number in enumerate(cycle_numbers):
    
    # Calculate overpotential
    overpotential1_V = current1_A * dcr_growth_per_cycle_mohms * cycle_number / 1000
    overpotential2_V = current2_A * dcr_growth_per_cycle_mohms * cycle_number / 1000
    overpotential3_V = current3_A * dcr_growth_per_cycle_mohms * cycle_number / 1000
    
    # Find index below MinV
    idx_below_MinV_1 = np.where(Vd - overpotential1_V < lower_cutoff_voltage1)[0][0]
    idx_below_MinV_2 = np.where(Vd - overpotential2_V < lower_cutoff_voltage2)[0][0]
    idx_below_MinV_3 = np.where(Vd - overpotential3_V < lower_cutoff_voltage3)[0][0]
    
    # Calculate capacity
    Qd1_endpoints[k] = Qd.iloc[idx_below_MinV_1]
    Qd2_endpoints[k] = Qd.iloc[idx_below_MinV_2]
    Qd3_endpoints[k] = Qd.iloc[idx_below_MinV_3]
    
    # Calculate energy by getting endpoint energy - V*Q
    # Could also do something like this (yields similar result):
    # np.trapz(Vd[:idx_below_MinV_1] - overpotential1_V, Qd[:idx_below_MinV_1])
    Ed1_endpoints[k] = Ed.iloc[idx_below_MinV_1] - overpotential1_V * Qd1_endpoints[k]
    Ed2_endpoints[k] = Ed.iloc[idx_below_MinV_2] - overpotential2_V * Qd2_endpoints[k]
    Ed3_endpoints[k] = Ed.iloc[idx_below_MinV_3] - overpotential3_V * Qd3_endpoints[k]
    
    # Calculate power by dividing energy by time (time = Q / I)
    Pd1_endpoints[k] = Ed1_endpoints[k] / (Qd1_endpoints[k] / current1_A)
    Pd2_endpoints[k] = Ed2_endpoints[k] / (Qd2_endpoints[k] / current2_A)
    Pd3_endpoints[k] = Ed3_endpoints[k] / (Qd3_endpoints[k] / current3_A)
    
    if cycle_number in cycle_numbers_short:
        Vd1_list.append(Vd - overpotential1_V)
        Vd2_list.append(Vd - overpotential2_V)
        Vd3_list.append(Vd - overpotential3_V)


# Generate figure handles
fig, ax = plt.subplots(
    figsize=(2 * config.FIG_WIDTH, 3 * config.FIG_HEIGHT),
    nrows=3, ncols=3,
)
ax = ax.ravel()

# Plot DCR growth vs. cycle number
ax[0].plot(cycle_numbers, dcr_growth_mohms, color="k")
ax[0].annotate(f"DCR growth rate =\n  {dcr_growth_per_cycle_mohms} m$\Omega$/cycle",
               (35, 190), va="center")

# Plot overpotential growth vs. cycle number
ax[1].plot(cycle_numbers, overpotential_growth1_V, color=main_colors[0],
           label=f"{C_rate1}C discharge ({current1_A} A)")
ax[1].plot(cycle_numbers, overpotential_growth3_V, color=main_colors[2],
           label=f"{C_rate3}C discharge ({current3_A} A)")

ax[1].annotate(f"Cell capacity = {capacity_Ah} Ah",
               (965, 0.02), va="center", ha="right")

# Plot V vs Q
ax2_twin = ax[2].twinx()  # instantiate a second axes that shares the same x-axis

ax[2].plot(Vd, Qd, color="tab:green")
ax2_twin.plot(Vd, Ed, color="tab:purple")

ax2_twin.set_ylabel('Discharge energy (Wh)', color="tab:purple")
ax[2].tick_params(axis='y', labelcolor="tab:green")
ax2_twin.tick_params(axis='y', labelcolor="tab:purple")

ax[2].annotate("NMC/graphite\n0.5C discharge\n25°C",
               (2.1, 0.25), va="center")

# Plot voltage vs. capacity
ax[3].axhline(lower_cutoff_voltage1, color="tab:gray")
ax[4].axhline(lower_cutoff_voltage2, color="tab:gray")
ax[5].axhline(lower_cutoff_voltage3, color="tab:gray")

colors = cm.viridis(np.linspace(0.9, 0.3, 10))[:,:10]

for k, Vd1 in enumerate(Vd1_list):
    idx_below_MinV= np.where(Vd1 < lower_cutoff_voltage1)[0][0]
    
    ax[3].plot(Qd, Vd1, color=colors[k],
               label=f"Cycle {cycle_numbers_short[k]}" if k % 2 == 0 else None)
    ax[3].plot(Qd.iloc[idx_below_MinV], lower_cutoff_voltage1, 'xk')
   
for k, Vd2 in enumerate(Vd2_list):
    idx_below_MinV = np.where(Vd2 < lower_cutoff_voltage2)[0][0]
    
    ax[4].plot(Qd, Vd2, color=colors[k],
               label=f"Cycle {cycle_numbers_short[k]}" if k % 2 == 0 else None)
    ax[4].plot(Qd.iloc[idx_below_MinV], lower_cutoff_voltage2, 'xk')
    
for k, Vd3 in enumerate(Vd3_list):
    idx_below_MinV = np.where(Vd3 < lower_cutoff_voltage3)[0][0]
    
    ax[5].plot(Qd, Vd3, color=colors[k],
               label=f"Cycle {cycle_numbers_short[k]}" if k % 2 == 0 else None)
    ax[5].plot(Qd.iloc[idx_below_MinV], lower_cutoff_voltage3, 'xk')
    
# Plot capacity retention
ax[6].plot(cycle_numbers, 100 * Qd1_endpoints / Qd1_endpoints[0], color=main_colors[0],
           label=f"{C_rate1}C discharge\nMinV = {lower_cutoff_voltage1} V")
ax[6].plot(cycle_numbers, 100 * Qd2_endpoints / Qd2_endpoints[0], "--", color=main_colors[1],
           label=f"{C_rate2}C discharge\nMinV = {lower_cutoff_voltage2} V")
ax[6].plot(cycle_numbers, 100 * Qd3_endpoints / Qd3_endpoints[0], color=main_colors[2],
           label=f"{C_rate3}C discharge\nMinV = {lower_cutoff_voltage3} V")

# Plot energy retention
ax[7].plot(cycle_numbers, 100 * Ed1_endpoints / Ed1_endpoints[0], color=main_colors[0],
           label=f"{C_rate1}C discharge\nMinV = {lower_cutoff_voltage1} V")
ax[7].plot(cycle_numbers, 100 * Ed2_endpoints / Ed2_endpoints[0], "--", color=main_colors[1],
           label=f"{C_rate2}C discharge\nMinV = {lower_cutoff_voltage2} V")
ax[7].plot(cycle_numbers, 100 * Ed3_endpoints / Ed3_endpoints[0], color=main_colors[2],
           label=f"{C_rate3}C discharge\nMinV = {lower_cutoff_voltage3} V")

# Plot power retention
ax[8].plot(cycle_numbers, 100 * Pd1_endpoints / Pd1_endpoints[0], color=main_colors[0],
           label=f"{C_rate1}C discharge\nMinV = {lower_cutoff_voltage1} V")
ax[8].plot(cycle_numbers, 100 * Pd2_endpoints / Pd2_endpoints[0], "--", color=main_colors[1],
           label=f"{C_rate2}C discharge\nMinV = {lower_cutoff_voltage2} V")
ax[8].plot(cycle_numbers, 100 * Pd3_endpoints / Pd3_endpoints[0], color=main_colors[2],
           label=f"{C_rate3}C discharge\nMinV = {lower_cutoff_voltage3} V")

# Set axes labels
ax[0].set_ylabel("DCR growth (m$\Omega$)")
ax[1].set_ylabel("Overpotential growth (V)")
ax[2].set_xlabel("Minimum voltage (V)")
ax[2].set_ylabel("Discharge capacity (Ah)", color="tab:green")
ax[3].set_ylabel("Voltage (V)")
ax[5].set_ylabel("Voltage (V)")
ax[6].set_ylabel("Capacity retention (%)")
ax[7].set_ylabel("Energy retention (%)")
ax[8].set_ylabel("Power retention (%)")

# Set axes limits
for k in [0, 1, 6, 7, 8]:
    ax[k].set_xlabel("Cycle number")
    ax[k].set_xlim([-0.5, 1000.5])
    
for k in [3, 4, 5]:
    ax[k].set_xlabel("Capacity (Ah)")
    ax[k].set_xlim([-0.01, 3.01])
    ax[k].set_ylim([0.2, 4.3])

ax[2].set_xlim([Vd.min(), Vd.max()])

ax[6].set_ylim([59.5, 101])
ax[7].set_ylim([59.5, 101])
ax[8].set_ylim([59.5, 101])
   
# Set axes titles and legends
legend_title_dict = {3: f"{C_rate1}C discharge\nMinV = {lower_cutoff_voltage1} V",
                     4: f"{C_rate2}C discharge\nMinV = {lower_cutoff_voltage2} V",
                     5: f"{C_rate3}C discharge\nMinV = {lower_cutoff_voltage3} V"}
legend_locs = {0: None, 1: "upper left"}
for k in np.arange(9):
    ax[k].set_title(chr(97 + k), loc="left", weight="bold")
    ax[k].legend(loc="lower left" if k > 2 else None,
                 title=legend_title_dict[k] if k in [3, 4, 5] else None,
                 frameon=True if k in [3, 4, 5] else None,
                 framealpha=1)


# Save figure as both .PNG and .EPS
plt.tight_layout()
fig.savefig(config.FIG_PATH / "dcr_growth_knee.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "dcr_growth_knee.eps", format="eps")
