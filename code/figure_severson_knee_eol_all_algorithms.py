from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config
from sklearn.linear_model import LinearRegression


def linear_regression_with_plot(
    point_arr,
    eol_arr, 
    x_cont, 
    ax=None, 
    unicode_title=None, 
    x_axis_appendix=None
):
    '''
    Given two arrays, fit an sklearn LinearRegression instance and generate
    the line of best fit over the range of x value
    
    Parameters
    ----------
    point_arr (type: numpy.ndarray)
        Array of cycle numbers at which the knee point was identified
    
    eol_arr (type: numpy.ndarray)
        Array of cycle numbers at which end of life occurred
    
    x_cont (type: numpy.ndarray)
        Array of x values to be used to generate the line of best fit,
        for plotting.
    
    ax (type: AxesSubplot)
        An AxesSubplot instance, onto which the data will be plotted.
        e.g. axes[0] where axes comes from the code:
            fig, axes = plt.subplots(2, 2)
    
    unicode_title (type: int)
        Value to be passed to built-in function chr, to return a unicode string.
        Used to set the titles for ax in a subplot setting.
        
    x_axis_appendix (type: str)
        X axis appendix
    
    '''

    if ax is None:
        ax = plt.gca()

    # Assert the inputs are 2 dimensional
    assert(len(point_arr.shape) == 2)
    # Check for correct shape for LinearRegression class
    if point_arr.shape[-1] != 1:
        point_vals = point_arr.reshape(-1, 1)
    else:
        point_vals = point_arr

    # Create an instance of LinearRegression and fit it to the data
    linreg = LinearRegression().fit(point_vals, eol_vals)
    # Get the R2 score for the linear model applied to the data
    r2_val = linreg.score(point_vals, eol_vals)
    
    # Check for correct shape for LinearRegression instance
    if x_cont.shape[-1] != 1:
        x_cont = x_cont.reshape(-1, 1)
    
    # Get the slope and intercept values for printing the equation on the plot
    slope = linreg.coef_[0]
    intercept = linreg.intercept_

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
    ax.plot(point_arr, eol_vals, 'x', linewidth=2)
    ax.plot(x_cont, linear_fit, color='black')
    ax.plot(x_cont, x_cont, linestyle='dashed', color='gray')
    
    # Plot configuration
    if x_axis_appendix is not None:
        ax.set_xlabel(f"Cycles to knee point: {x_axis_appendix}")
    else:
        ax.set_xlabel("Cycles to knee point")
    ax.set_ylabel("Cycles to end-of-life")
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

# Get the EOL cycle numbers. These are independent of the knee finding method
eol_vals = data['EOL'].to_numpy()

# Create a range of x values to use for line of best fit
# Find the largest knee point or EOL value across all results
max_val = np.max(np.max(data.iloc[:,1:])) + 75
x_values = np.arange(max_val)


# Create the figure
fig, ax = plt.subplots(2,2, figsize=(2 * config.FIG_WIDTH, 2 * config.FIG_WIDTH))
ax = ax.ravel()

# For each knee identification method, call this function to do the linear regression.
# Notice one subplot axis is passed to this function, for each method
linear_regression_with_plot(bw_points, eol_vals, x_cont=x_values, ax=ax[0], unicode_title=97, x_axis_appendix="Bacon-Watts")
linear_regression_with_plot(kneedle_points, eol_vals, x_cont=x_values, ax=ax[1], unicode_title=98, x_axis_appendix="Kneedle")
linear_regression_with_plot(diao_points, eol_vals, x_cont=x_values, ax=ax[2], unicode_title=99, x_axis_appendix="Tangent-ratio")
linear_regression_with_plot(bisector_points, eol_vals, x_cont=x_values, ax=ax[3], unicode_title=100, x_axis_appendix="Bisector")

plt.tight_layout()

# Save figure to PNG and EPS, once configured correctly
fig.savefig(config.FIG_PATH / "severson_knee_eol_all_algorithms.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "severson_knee_eol_all_algorithms.eps", format="eps")
