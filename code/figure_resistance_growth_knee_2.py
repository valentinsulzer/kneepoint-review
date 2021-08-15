#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This figure illustrates resistance growth knees.
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

# Resistance growth per cycle
resistance_growth_per_cycle_mohms = 0.2

# Define lower cutoff voltages
lower_cutoff_voltage1 = 2.0
lower_cutoff_voltage2 = 2.8
lower_cutoff_voltage3 = 2.0
lower_cutoff_voltage4 = 2.8

# Define C rates
C_rate1 = 1
C_rate2 = 1
C_rate3 = 2
C_rate4 = 2

# Define colors
main_colors = ["tab:blue", "tab:blue", "tab:red", "tab:red"]

## SCRIPT STARTS HERE

# Define currents
current1_A = capacity_Ah * C_rate1
current2_A = capacity_Ah * C_rate2
current3_A = capacity_Ah * C_rate3
current4_A = capacity_Ah * C_rate4

# Generate fake data
cycle_numbers = np.arange(1000)
cycle_numbers_short = np.arange(0, 1000, 100)
resistance_growth_mohms = resistance_growth_per_cycle_mohms * cycle_numbers

# Calculate overpotential from Ohm's law
overpotential_growth1_V = current1_A * resistance_growth_mohms / 1000
overpotential_growth2_V = current2_A * resistance_growth_mohms / 1000
overpotential_growth3_V = current3_A * resistance_growth_mohms / 1000
overpotential_growth4_V = current4_A * resistance_growth_mohms / 1000


# Read in dataset and get first discharge
df = pd.read_csv(url, nrows=2000)
df_discharge1 = df[(df["Cycle_Index"] == 1) & (df["Current (A)"] < 0)]
Qd = df_discharge1["Discharge_Capacity (Ah)"]
Ed = df_discharge1["Discharge_Energy (Wh)"]
Vd = df_discharge1["Voltage (V)"]

# Calculate voltage curves, capacity endpoints, and energy endpoints
Vd1_list, Vd2_list, Vd3_list, Vd4_list = [], [], [], []
Qd1_endpoints = np.zeros((len(cycle_numbers), ))
Qd2_endpoints = np.zeros((len(cycle_numbers), ))
Qd3_endpoints = np.zeros((len(cycle_numbers), ))
Qd4_endpoints = np.zeros((len(cycle_numbers), ))

Ed1_endpoints = np.zeros((len(cycle_numbers), ))
Ed2_endpoints = np.zeros((len(cycle_numbers), ))
Ed3_endpoints = np.zeros((len(cycle_numbers), ))
Ed4_endpoints = np.zeros((len(cycle_numbers), ))

Pd1_endpoints = np.zeros((len(cycle_numbers), ))
Pd2_endpoints = np.zeros((len(cycle_numbers), ))
Pd3_endpoints = np.zeros((len(cycle_numbers), ))
Pd4_endpoints = np.zeros((len(cycle_numbers), ))

for k, cycle_number in enumerate(cycle_numbers):
    
    # Calculate overpotential
    overpotential1_V = current1_A * resistance_growth_per_cycle_mohms * cycle_number / 1000
    overpotential2_V = current2_A * resistance_growth_per_cycle_mohms * cycle_number / 1000
    overpotential3_V = current3_A * resistance_growth_per_cycle_mohms * cycle_number / 1000
    overpotential4_V = current4_A * resistance_growth_per_cycle_mohms * cycle_number / 1000

    
    # Find index below MinV
    idx_below_MinV_1 = np.where(Vd - overpotential1_V < lower_cutoff_voltage1)[0][0]
    idx_below_MinV_2 = np.where(Vd - overpotential2_V < lower_cutoff_voltage2)[0][0]
    idx_below_MinV_3 = np.where(Vd - overpotential3_V < lower_cutoff_voltage3)[0][0]
    idx_below_MinV_4 = np.where(Vd - overpotential4_V < lower_cutoff_voltage4)[0][0]

    # Calculate capacity
    Qd1_endpoints[k] = Qd.iloc[idx_below_MinV_1]
    Qd2_endpoints[k] = Qd.iloc[idx_below_MinV_2]
    Qd3_endpoints[k] = Qd.iloc[idx_below_MinV_3]
    Qd4_endpoints[k] = Qd.iloc[idx_below_MinV_4]

    # Calculate energy by getting endpoint energy - V*Q
    # Could also do something like this (yields similar result):
    # np.trapz(Vd[:idx_below_MinV_1] - overpotential1_V, Qd[:idx_below_MinV_1])
    Ed1_endpoints[k] = Ed.iloc[idx_below_MinV_1] - overpotential1_V * Qd1_endpoints[k]
    Ed2_endpoints[k] = Ed.iloc[idx_below_MinV_2] - overpotential2_V * Qd2_endpoints[k]
    Ed3_endpoints[k] = Ed.iloc[idx_below_MinV_3] - overpotential3_V * Qd3_endpoints[k]
    Ed4_endpoints[k] = Ed.iloc[idx_below_MinV_4] - overpotential4_V * Qd4_endpoints[k]
    
    # Calculate power by dividing energy by time (time = Q / I)
    Pd1_endpoints[k] = Ed1_endpoints[k] / (Qd1_endpoints[k] / current1_A)
    Pd2_endpoints[k] = Ed2_endpoints[k] / (Qd2_endpoints[k] / current2_A)
    Pd3_endpoints[k] = Ed3_endpoints[k] / (Qd3_endpoints[k] / current3_A)
    Pd4_endpoints[k] = Ed4_endpoints[k] / (Qd4_endpoints[k] / current4_A)

    if cycle_number in cycle_numbers_short:
        Vd1_list.append(Vd - overpotential1_V)
        Vd2_list.append(Vd - overpotential2_V)
        Vd3_list.append(Vd - overpotential3_V)
        Vd4_list.append(Vd - overpotential4_V)


# Generate figure / axis handles
fig = plt.figure(figsize=(config.FIG_WIDTH*2, config.FIG_HEIGHT*2.5))
ax0 = plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=1) # overpotential increase w/ resistance growth
ax1 = plt.subplot2grid((3,4), (0,2), colspan=2, rowspan=1) # discharge capacity, energy vs voltage @ BOL
ax2 = plt.subplot2grid((3,4), (1,0), colspan=2, rowspan=1) # discharge curves vs cycles, 1C
ax3 = plt.subplot2grid((3,4), (1,2), colspan=2, rowspan=1) # discharge curves vs cycles, 2C
ax4 = plt.subplot2grid((3,4), (2,0), colspan=1, rowspan=1) # capacity retention
ax5 = plt.subplot2grid((3,4), (2,1), colspan=1, rowspan=1) # energy retention
ax6 = plt.subplot2grid((3,4), (2,2), colspan=1, rowspan=1) # power retention
ax7 = plt.subplot2grid((3,4), (2,3), colspan=1, rowspan=1) # power retention

# Plot overpotential growth vs. cycle number
ax0.plot(cycle_numbers, overpotential_growth1_V, color=main_colors[0],
           label=f"{C_rate1}C discharge ({current1_A} A)")
ax0.plot(cycle_numbers, overpotential_growth3_V, color=main_colors[2],
           label=f"{C_rate3}C discharge ({current3_A} A)")

ax0.annotate(f"Cell capacity = {capacity_Ah} Ah",
               (985, 0.02), va="center", ha="right")
ax0.annotate(f"Resistance growth rate = {resistance_growth_per_cycle_mohms} m$\Omega$/cycle",
               (985, 0.11), va="center", ha='right')

# Plot V vs Q
ax1_twin = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax1.plot(Vd, Qd, color="tab:green")
ax1_twin.plot(Vd, Ed, color="tab:purple")

ax1_twin.set_ylabel('Discharge energy (Wh)', color="tab:purple")
ax1.tick_params(axis='y', labelcolor="tab:green")
ax1_twin.tick_params(axis='y', labelcolor="tab:purple")

ax1.axvline(lower_cutoff_voltage2, linestyle="--", color='tab:gray')

ax1.annotate("NMC/graphite\n0.5C discharge\n25°C",
               (2.05, 0.3), va="center")

# Plot voltage vs. capacity
ax2.axhline(lower_cutoff_voltage1, color="tab:gray")
ax2.axhline(lower_cutoff_voltage2, color="tab:gray", linestyle='--')
ax3.axhline(lower_cutoff_voltage3, color="tab:gray")
ax3.axhline(lower_cutoff_voltage4, color="tab:gray", linestyle='--')

colors = cm.viridis(np.linspace(0.9, 0.3, 10))[:,:10]

for k, Vd1 in enumerate(Vd1_list):
    idx_below_MinV= np.where(Vd1 < lower_cutoff_voltage1)[0][0]
    
    ax2.plot(Qd, Vd1, color=colors[k],
               label=f"Cycle {cycle_numbers_short[k]}" if k % 2 == 0 else None)
    ax2.plot(Qd.iloc[idx_below_MinV], lower_cutoff_voltage1, 'xk')
   
for k, Vd2 in enumerate(Vd2_list):
    idx_below_MinV = np.where(Vd2 < lower_cutoff_voltage2)[0][0]
    ax2.plot(Qd.iloc[idx_below_MinV], lower_cutoff_voltage2, 'xk')
    
for k, Vd3 in enumerate(Vd3_list):
    idx_below_MinV = np.where(Vd3 < lower_cutoff_voltage3)[0][0]
    
    ax3.plot(Qd, Vd3, color=colors[k],
               label=f"Cycle {cycle_numbers_short[k]}" if k % 2 == 0 else None)
    ax3.plot(Qd.iloc[idx_below_MinV], lower_cutoff_voltage3, 'xk')

for k, Vd4 in enumerate(Vd4_list):
    idx_below_MinV = np.where(Vd4 < lower_cutoff_voltage4)[0][0]
    ax3.plot(Qd.iloc[idx_below_MinV], lower_cutoff_voltage4, 'xk')

ax2.annotate(f"{C_rate1}C discharge",
               (2.95, 4), va="center", ha='right')
ax3.annotate(f"{C_rate3}C discharge",
               (2.95, 4), va="center", ha='right')

# Plot capacity retention
ax4.plot(cycle_numbers, 100 * Qd1_endpoints / Qd1_endpoints[0], color=main_colors[0],
           label=f"{C_rate1}C discharge\nMinV = {lower_cutoff_voltage1} V")
ax4.plot(cycle_numbers, 100 * Qd2_endpoints / Qd2_endpoints[0], "--", color=main_colors[1],
           label=f"{C_rate2}C discharge\nMinV = {lower_cutoff_voltage2} V")
ax4.plot(cycle_numbers, 100 * Qd3_endpoints / Qd3_endpoints[0], color=main_colors[2],
           label=f"{C_rate3}C discharge\nMinV = {lower_cutoff_voltage3} V")
ax4.plot(cycle_numbers, 100 * Qd4_endpoints / Qd4_endpoints[0], "--", color=main_colors[3],
           label=f"{C_rate4}C discharge\nMinV = {lower_cutoff_voltage4} V")

# Plot energy retention
ax5.plot(cycle_numbers, 100 * Ed1_endpoints / Ed1_endpoints[0], color=main_colors[0],
           label=f"{C_rate1}C discharge\nMinV = {lower_cutoff_voltage1} V")
ax5.plot(cycle_numbers, 100 * Ed2_endpoints / Ed2_endpoints[0], "--", color=main_colors[1],
           label=f"{C_rate2}C discharge\nMinV = {lower_cutoff_voltage2} V")
ax5.plot(cycle_numbers, 100 * Ed3_endpoints / Ed3_endpoints[0], color=main_colors[2],
           label=f"{C_rate3}C discharge\nMinV = {lower_cutoff_voltage3} V")
ax5.plot(cycle_numbers, 100 * Ed4_endpoints / Ed4_endpoints[0], "--", color=main_colors[3],
           label=f"{C_rate4}C discharge\nMinV = {lower_cutoff_voltage4} V")

# Plot power retention
ax6.plot(cycle_numbers, 100 * Pd1_endpoints / Pd1_endpoints[0], color=main_colors[0],
           label=f"{C_rate1}C discharge\nMinV = {lower_cutoff_voltage1} V")
ax6.plot(cycle_numbers, 100 * Pd2_endpoints / Pd2_endpoints[0], "--", color=main_colors[1],
           label=f"{C_rate2}C discharge\nMinV = {lower_cutoff_voltage2} V")
ax6.plot(cycle_numbers, 100 * Pd3_endpoints / Pd3_endpoints[0], color=main_colors[2],
           label=f"{C_rate3}C discharge\nMinV = {lower_cutoff_voltage3} V")
ax6.plot(cycle_numbers, 100 * Pd4_endpoints / Pd4_endpoints[0], "--", color=main_colors[3],
           label=f"{C_rate4}C discharge\nMinV = {lower_cutoff_voltage4} V")

#### ONLY ONE LEGEND ON RIGHT OF ALL 3 PLOTS

# Set axes labels
ax0.set_title(chr(97 + 0), loc="left", weight="bold")
ax0.set_ylabel("Overpotential growth (V)")
ax0.set_xlabel("Cycle number")
ax0.set_xlim([-0.5, 1000.5])
ax0.legend(loc='upper left')

ax1.set_title(chr(97 + 1), loc="left", weight="bold")
ax1.set_ylabel("Discharge capacity (Ah)", color="tab:green")
ax1.set_xlabel("Minimum voltage (V)")
ax1.set_xlim([Vd.min(), Vd.max()])

ax2.set_title(chr(97 + 2), loc="left", weight="bold")
ax2.set_ylabel("Voltage (V)")
ax2.set_xlabel("Capacity (Ah)")
ax2.set_ylim([0.2, 4.3])
ax2.set_xlim([-0.01, 3.01])
ax2.legend(loc="lower left", frameon=True, framealpha=1)

ax3.set_title(chr(97 + 3), loc="left", weight="bold")
ax3.set_ylabel("Voltage (V)")
ax3.set_xlabel("Capacity (Ah)")
ax3.set_ylim([0.2, 4.3])
ax3.set_xlim([-0.01, 3.01])
ax3.legend(loc="lower left", frameon=True, framealpha=1)

ax4.set_title(chr(97 + 4), loc="left", weight="bold")
ax4.set_ylabel("Capacity retention (%)")
ax4.set_xlabel("Cycle number")
ax4.set_ylim([59.5, 101])
ax4.set_xlim([-0.5, 1000.5])

ax5.set_title(chr(97 + 5), loc="left", weight="bold")
ax5.set_ylabel("Energy retention (%)")
ax5.set_xlabel("Cycle number")
ax5.set_ylim([59.5, 101])
ax5.set_xlim([-0.5, 1000.5])

ax6.set_title(chr(97 + 6), loc="left", weight="bold")
ax6.set_ylabel("Power retention (%)")
ax6.set_xlabel("Cycle number")
ax6.set_ylim([59.5, 101])
ax6.set_xlim([-0.5, 1000.5])
handles, labels = ax6.get_legend_handles_labels()
ax7.set_axis_off()
ax7.legend(handles, labels, loc="center left", frameon=True, framealpha=1)

# Set axes titles and legends
legend_title_dict = {3: f"{C_rate1}C discharge\nMinV = {lower_cutoff_voltage1} V",
                     4: f"{C_rate2}C discharge\nMinV = {lower_cutoff_voltage2} V",
                     5: f"{C_rate3}C discharge\nMinV = {lower_cutoff_voltage3} V"}


# Save figure as both .PNG and .EPS
plt.tight_layout()
fig.savefig(config.FIG_PATH / "resistance_growth_knee_2.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "resistance_growth_knee_2.eps", format="eps")
