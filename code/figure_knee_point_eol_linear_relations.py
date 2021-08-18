from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import config
from sklearn.linear_model import LinearRegression


# Load the data to be plotted from CSV file
data = pd.read_csv("./data/knee_point_EOL_linreg_data.csv", header=0, names=['src', 'knee_point', 'EOL'])
# Load the plotting configuration dict from pkl file
with open("./data/knee_point_eol_plot_dict.pkl", "rb") as a_file:
    plot_dict = pickle.load(a_file)
del a_file
    
# Find out how many data points there are for each source
# Add this to the legend for each source we will use on the plot
for source in data['src'].unique():
    num_cells = len(data[data['src']==source])
    if num_cells == 1:
        plot_dict[source]['legend'] += f" ({num_cells} cell)"
    else:
        plot_dict[source]['legend'] += f" ({num_cells} cells)"


# Linear regression
# Extract all data
all_xdata = data['knee_point'].to_numpy().reshape((-1,1))
all_ydata = data['EOL'].to_numpy().reshape((-1,1))

# Do the linear regression for all the data
linreg = LinearRegression()
reg = linreg.fit(all_xdata, all_ydata)
r2 = reg.score(all_xdata, all_ydata)

# Set up the regression line
x_cont = np.arange(0, ax_lim).reshape((-1,1))
y_pred = linreg.predict(x_cont)


# Plotting data
fig, ax = plt.subplots()
for source in data['src'].unique():
    # Get the plotting configuration for each source
    label, color, marker = plot_dict[source].values()
    # Extract the data for each source
    x_vals = data[data['src']==source]['knee_point']
    y_vals = data[data['src']==source]['EOL']
    # Add data to plot with pre-defined labels, colours and markers
    ax.scatter(x_vals, y_vals, label=label, color=color, marker=marker)

# Plot configuration

# Define an axis limit value, based on the data plotted. Same for x and y axes.
ax_lim = int(1.1*max(max(all_xdata), max(all_ydata)))
ax.set_xlim([0, ax_lim])
ax.set_ylim([0, ax_lim])
ax.set_xlabel("Knee Point (Cycles)")
ax.set_ylabel("End of Life (Cycles)")

# Plot the regression line
ax.plot(x_cont, y_pred, color='black', linewidth=1)

# Add a dashed line y = x
current_ax = plt.gca()
plot_xvals = current_ax.get_xlim()
x_dashed = np.arange(plot_xvals[0], plot_xvals[1]+1)
y_dashed = x_dashed
plt.plot(x_dashed, y_dashed, linestyle='--', color='gray')

# Place equation on plot
slope = linreg.coef_[0][0]
intercept = linreg.intercept_[0]

if intercept < 0:
    # Omit the "plus" sign if the intercept sign is -ve
    plot_intercept = np.abs(intercept)
else:
    plot_intercept = intercept

# Place the equation on the plot
ax.annotate(f'y = {slope:.3f}x - {plot_intercept:.3f}', xy=(3,1), xytext=(0.05, 0.9), textcoords='axes fraction')
# Add the R2 value
ax.annotate(f'$R^2$ = {r2:.3f}', xy=(3,1), xytext=(0.05, 0.85), textcoords='axes fraction')

ax.legend() 
#plt.show() 
   
# Save figure to PNG and EPS, once configured correctly
fig.savefig(config.FIG_PATH / "knee_point_eol_linear_relations.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "knee_point_eol_linear_relations.eps", format="eps")
    