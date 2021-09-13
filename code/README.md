# Contributing programmatically-generated figures

This README has a few brief instructions to help create standardized programmatically-generated figures using `matplotlib`.

## Loading standard settings

`config.py` contains some globals and some modifications to `rcParams` to help create publication-ready figures. Having a centralized file for these configuration settings will enable us to readily make changes down the road if needed. Begin your script with: 
````
import config
````

## Creating figures

Access JECS-standard figure widths and heights from `config`:
````
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, config.FIG_HEIGHT), nrows=1, ncols=1)
````
To make a larger figure:
````
fig, ax = plt.subplots(figsize=(2 * config.FIG_WIDTH, config.FIG_HEIGHT), nrows=1, ncols=1)
````

## Saving figures

Save figures to `config.FIG_PATH` in both `.png` and `.eps` format (`.png` for easy viewing, `.eps` for vector graphics for the final paper):

````
fig.savefig(config.FIG_PATH / "[FIGURE_NAME].png", format="png")
fig.savefig(config.FIG_PATH / "[FIGURE_NAME].eps", format="eps")
````

## Requirements
A simple `requirements.txt` file is present for creating a virtual environment.
