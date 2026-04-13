# py_files/setup.py
from __future__ import annotations

# 1. stdlib
import math
from dataclasses import dataclass
from io import StringIO
from functools import reduce

# 2. third-party
import numpy as np
from numpy.linalg import inv

try:
    import pandas as pd
except Exception:  # optional dependency
    pd = None

try:
    import requests
except Exception:  # optional dependency
    requests = None

try:
    from IPython.display import display
except Exception:  # optional dependency
    display = None

try:
    from dstapi import DstApi
except Exception:  # optional dependency
    DstApi = None

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch

from scipy.interpolate import interp1d

# 3. your project
# from old.capinc import CapIncModel
import py_files.shocks as shocks
# import old.build_output_v2 as build_output
import py_files.var_groups as var_groups
import py_files.build_output_single as build_output_single
import py_files.investment_shares as il
import py_files.LS_aggregator as sls
import py_files.wage_employment as we

from py_files.capinc_single import CapIncModel_single
import py_files.elas as elas
from py_files.QP_diagram import make_figure
from py_files.IRF import plot_irf
from py_files.sweep import run_sweeps, plot_sweep, print_elas, load_or_compute_sweeps
from py_files.LS_aggregator import load_or_compute_ls_timeseries


# ==================== ==================== ==================== ====================
# 1. enable autoreload

def enable_autoreload(mode: int = 2) -> None:
    """
    Enable autoreload in IPython/Jupyter if available.
    mode=2 -> reload all modules (except those excluded by autoreload itself).
    Safe no-op outside notebooks.
    """
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return

    ip = get_ipython()
    if ip is None:
        return

    # 1. load extension if needed, then set mode
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", str(int(mode)))


# ==================== ==================== ==================== ====================
# 2. set matplotlib style for AEJ-style figures

def set_aej(**kwargs):
    """Set matplotlib style for AEJ-style figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 15,

        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,
        "legend.frameon": False,

        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.spines.top": True,
        "axes.spines.right": True,

        # legend
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.borderaxespad": 0.4,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,

        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# ==================== ==================== ==================== ====================
# 3. setup notebook environment

def setup_notebook(*, autoreload: int = 2, aej: bool = True, **aej_kwargs) -> None:
    """
    One call in a notebook to get the usual environment:
      - autoreload
      - matplotlib style
    """
    enable_autoreload(autoreload)
    if aej:
        set_aej(**aej_kwargs)


# 1. What you get with: from py_files.setup import *
__all__ = [
    # functions
    "enable_autoreload", "set_aej", "setup_notebook",
    # common modules/objects
    "np", "pd", "plt", "mticker", "math", "StringIO", "requests",
    "dataclass", "display", "DstApi", "reduce", "inv",
    "Line2D", "Patch", "interp1d",
    # project imports, "CapIncModel", "build_output" 
    "shocks", "var_groups",
    "build_output_single", "CapIncModel_single", "il", "sls", "we",
    "elas", "make_figure", "plot_irf",
    "run_sweeps", "plot_sweep", "print_elas",
    "load_or_compute_ls_timeseries", 'load_or_compute_sweeps'
]
