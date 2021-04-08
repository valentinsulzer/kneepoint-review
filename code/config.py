#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This config file should be imported for all scripts generating figures.
It defines some globals and sets some rcParams for consistent plotting.
"""

from pathlib import Path

from matplotlib import rcParams

# Globals
FIG_WIDTH = 3.25  # ECS spec is 3.25" width
FIG_HEIGHT = (3 / 4) * FIG_WIDTH  # standard ratio
FIG_PATH = Path.cwd().parent / "figures"

# Set default rcParams for all programmatically-generated figures
rcParams["lines.markersize"] = 5
rcParams["lines.linewidth"] = 1.0
rcParams["font.size"] = 7
rcParams["legend.fontsize"] = 7
rcParams["legend.frameon"] = False
rcParams["font.sans-serif"] = "Arial"
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Arial"
rcParams["savefig.bbox"] = "tight"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
