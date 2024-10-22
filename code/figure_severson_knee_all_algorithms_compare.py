from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config
from sklearn.linear_model import LinearRegression


def linear_regression_with_plot(arr1, arr2, x_cont, method1, method2, ax=None, unicode_title=None):
    '''
    Given two arrays, fit an sklearn LinearRegression instance and generate
    the line of best fit over the range of x value
    
    Parameters
    ----------
    arr1 (type: numpy.ndarray)
        Array of cycle numbers at which the knee point was identified
    
    arr2 (type: numpy.ndarray)
        Array of cycle numbers at which the knee point was identified
    
    x_cont (type: numpy.ndarray)
        Array of x values to be used to generate the line of best fit,
        for plotting.
    
    method1 (type: str)
        Name of first knee point method.
    
    method2 (type: str)
        Name of second knee point method.
    
    ax (type: AxesSubplot)
        An AxesSubplot instance, onto which the data will be plotted.
        e.g. axes[0] where axes comes from the code:
            fig, axes = plt.subplots(2, 2)
    
    unicode_title (type: int)
        Value to be passed to built-in function chr, to return a unicode string.
        Used to set the titles for ax in a subplot setting.
    
    '''

    if ax is None:
        ax = plt.gca()

    # Assert the inputs are 2 dimensional
    assert(len(arr1.shape) == 2)    
    assert(len(arr2.shape) == 2)
    # Check for correct shape for LinearRegression class
    if arr1.shape[-1] != 1:
        arr1_vals = arr1.reshape(-1, 1)
    else:
        arr1_vals = arr1
    if arr2.shape[-1] != 1:
        arr2_vals = arr2.reshape(-1, 1)
    else:
        arr2_vals = arr2

    # Create an instance of LinearRegression and fit it to the data
    linreg = LinearRegression().fit(arr1_vals, arr2_vals)
    # Get the R2 score for the linear model applied to the data
    r2_val = linreg.score(arr1_vals, arr2_vals)
    
    # Check for correct shape for LinearRegression instance
    if x_cont.shape[-1] != 1:
        x_cont = x_cont.reshape(-1, 1)
    
    # Get the slope and intercept values for printing the equation on the plot
    slope = linreg.coef_[0][0]
    intercept = linreg.intercept_[0]

    # Make the strings for the labels to go on the plot
    if linreg.intercept_ < 0:
        # Omit the "plus" sign if the intercept sign is -ve
        eqn_label = f"y = {slope:.3f}x - {np.abs(intercept):.3f}"
    else:
        eqn_label = f"y = {slope:.3f}x + {intercept:.3f}"

    # Add the R2 value
    r2_label = f"$R^2$ = {r2_val:.3f}"
    
    # Generate a line of best fit to be plotted
    linear_fit = linreg.predict(x_cont)

    # Add the data points, linear fit and y=x line
    ax.plot(arr1_vals, arr2_vals, 'x', linewidth=2)
    ax.plot(x_cont, linear_fit, color='black')
    ax.plot(x_cont, x_cont, linestyle='dashed', color='gray')
    
    # Plot configuration
    ax.set_xlabel(f"Cycles to knee point: {method1}")
    ax.set_ylabel(f"Cycles to knee point: {method2}")
    ax.set_xlim([0, np.max(x_cont)])
    ax.set_ylim([0, np.max(x_cont)])
    ax.set_aspect('equal', 'box')
    if unicode_title != None:
        ax.set_title(chr(unicode_title), loc="left", weight="bold")
    #ax.grid(alpha=0.4)
    
    # Annotate the plot, using the axes fraction argument for text coordinates
    ax.annotate(eqn_label, xy=(3,1), xycoords='data', xytext=(0.98, 0.1),
                textcoords='axes fraction', ha="right")
    ax.annotate(r2_label, xy=(3,1), xycoords='data', xytext=(0.98, 0.03),
                textcoords='axes fraction', ha="right")
    



# Load the data from CSV
data = pd.read_csv("./data/severson2019_EOL-2-knee-All-algorithms.csv",
                   names=["Cell", "BW_onset", "BW_point", "diao_point",
                          "kneedle_point", "bisector_point", "EOL"],
                   header=0)

# Remove onset information, since it's not needed here
data.drop(columns=['BW_onset'], inplace=True)


# Extract the identified knee points for each method
bw_points = data['BW_point'].to_numpy().reshape(-1, 1)
kneedle_points = data['kneedle_point'].to_numpy().reshape(-1, 1)
diao_points = data['diao_point'].to_numpy().reshape(-1, 1)
bisector_points = data['bisector_point'].to_numpy().reshape(-1, 1)

# Create a range of x values to use for line of best fit
# Find the largest knee point across all results
max_val = np.max(np.max(data.iloc[:,1:])) + 75
x_values = np.arange(max_val)


# Create the figure
fig, ax = plt.subplots(2, 3, figsize=(2 * config.FIG_WIDTH, 2 * config.FIG_HEIGHT))
ax = ax.ravel()

# For each knee identification method, call this function to do the linear regression.
# Notice one subplot axis is passed to this function, for each method
linear_regression_with_plot(kneedle_points, bw_points, x_values, "Kneedle", "Bacon-Watts", ax=ax[0], unicode_title=97)
linear_regression_with_plot(kneedle_points, diao_points, x_values, "Kneedle", "Tangent-ratio", ax=ax[1], unicode_title=98)
linear_regression_with_plot(kneedle_points, bisector_points, x_values, "Kneedle", "Bisector", ax=ax[2], unicode_title=99)
linear_regression_with_plot(bw_points, diao_points, x_values, "Bacon-Watts", "Tangent-ratio", ax=ax[3], unicode_title=100)
linear_regression_with_plot(bw_points, bisector_points, x_values, "Bacon-Watts", "Bisector", ax=ax[4], unicode_title=101)
linear_regression_with_plot(diao_points, bisector_points, x_values, "Tangent-ratio", "Bisector", ax=ax[5], unicode_title=102)

plt.tight_layout()

# Save figure to PNG and EPS, once configured correctly
fig.savefig(config.FIG_PATH / "severson_knee_compare_all_algorithms.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "severson_knee_compare_all_algorithms.eps", format="eps")
