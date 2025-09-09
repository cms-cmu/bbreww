"""
Interactive plotting utility for HH4b analysis.

This module provides functions for creating and customizing 1D and 2D plots
from histogram data, with support for multiple variables, regions, and processes.
"""

import os
import sys
from typing import Optional, Union, List, Tuple, Dict, Any

# Third-party imports
import hist
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, os.getcwd())
from bbreww.plots.plots import load_config_bbWW
from src.plotting.plots import (
    makePlot, make2DPlot, load_hists,
    read_axes_and_cuts, parse_args, print_cfg
)
import src.plotting.iPlot_config as cfg

# Constants
DEFAULT_OUTPUT_FILE = "test.pdf"


def ls(option: str = "var", var_match: Optional[str] = None) -> None:
    """List available variables in the configuration.

    Args:
        option: The type of labels to list (default: "var")
        var_match: Optional string to filter variables by
    """
    for k in cfg.axisLabels[option]:
        if var_match:
            if var_match in k:
                print(k)
        else:
            print(k)


def info() -> None:
    """Print the current configuration."""
    print_cfg(cfg)


def examples() -> None:
    """Print example usage of the plotting functions."""
    examples_text = """
examples:

# Nominal plot of data and background in a region passing a cut
plot("mbb", flavor=sum, channel=sum, cut="preselection")

# Can get a print out of the variables
ls()
plot("*", flavor=sum, channel=sum, cut="preselection")
plot("m*", flavor=sum, channel=sum, cut="preselection")

# Can add ratio
plot("mbb", flavor=sum, channel=sum, cut="preselection", doRatio=1)

# Can rebin
plot("mbb", flavor=sum, channel=sum, cut="preselection", doRatio=1, rebin=4)

# Can normalize
plot("mbb", flavor=sum, channel=sum, cut="preselection", doRatio=1, rebin=4, norm=1)

# Can set logy
plot("mbb", flavor=sum, channel=sum, cut="preselection", doRatio=1, rebin=4, norm=1, yscale="log")

# Can set ranges
plot("mbb", flavor=sum, channel=sum, cut="preselection", doRatio=1, rebin=4, norm=1, rlim=[0.5,1.5])
plot("mbb", flavor=sum, channel=sum, cut="preselection", doRatio=1, rebin=4, norm=1, xlim=[50,200])
plot("mbb", flavor=sum, channel=sum, cut="preselection", doRatio=1, rebin=4, norm=1, ylim=[0,0.01])

# Can overlay different flavors or channels or cuts
plot("mbb", flavor=["e","mu"], channel=sum, cut="preselection", process="HHbbWW", doRatio=1, rebin=4, histtype="step")
plot("mbb", flavor=sum,  channel=["hadronic_W","leptonic_W"], cut="preselection", process="TTbar", doRatio=1, rebin=4, histtype="step")
plot("mbb", flavor=sum, channel=sum,  cut=["preselection","nominal_4j2b"], process="HHbbWW", doRatio=1, rebin=4, norm=1)

# Can overlay different variables
plot(["mbb","qq_mass"], flavor=sum, channel=sum,  cut="preselection", doRatio=1, process="HHbbWW")
plot(["mbb","qq_mass","bjets_genjets_mass"], flavor=sum, channel=sum,  cut="preselection", doRatio=1, process="TTbar")

# Can plot a single process
plot("mbb", flavor=sum, channel=sum,  cut="preselection", process="HHbbWW")

# Can overlay processes
plot("mbb", flavor=sum, channel=sum,  cut="preselection", norm=1, process=["HHbbWW","TTbar"],doRatio=1)

# Can overlay years
plot("mbb", flavor=sum, channel=sum,  cut="preselection", doRatio=1, process="HHbbWW", year=["2022","2022"])

# Plot 2d hists
plot2d("mbb_vs_bb_dr", process="HHbbWW", flavor=sum, channel=sum,  cut="preselection")
plot2d("mbb_vs_bb_dr", process="HHbbWW", flavor=sum, channel=sum,  cut="preselection", full=True)

"""
    print(examples_text)


def save_and_open_plot(fig: plt.Figure, output_file: str) -> bool:
    """Save the figure to a file and open it.

    Args:
        fig: The matplotlib figure to save
        output_file: Path where to save the figure

    Returns:
        True if successful, False otherwise
    """
    try:
        fig.savefig(output_file)
        plt.close()
        os.system(f"open {output_file}")
        return True
    except Exception as e:
        print(f"Error saving plot: {e}")
        return False


def handle_wildcards(var: Union[str, List[str]]) -> bool:
    """Handle wildcard matching in variable names.

    Args:
        var: Variable(s) to check for wildcards

    Returns:
        True if wildcards were found and handled, False otherwise
    """
    if isinstance(var, str) and "*" in var:
        ls(var_match=var.replace("*", ""))
        return True
    if isinstance(var, list) and var[0].find("*") != -1:
        ls(var_match=var[0].replace("*", ""))
        return True
    return False


def plot(var: Union[str, List[str]] = 'selJets.pt', *,
         cut: Union[str, List[str]] = "passPreSel",
         flavor: Union[str, List[str]] = None,
         channel: Union[str, List[str]] = None,
         axis_opts: Dict = {},
         output_file: str = DEFAULT_OUTPUT_FILE,
         **kwargs) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Create a 1D plot of the specified variable.

    Args:
        var: Variable(s) to plot. Can be a string or list of strings.
        cut: Selection cut to apply (default: "passPreSel")
        region: Region to plot (default: "SR")
        axis_opts: Additional axis options as a dictionary
        output_file: Name of the output file (default: "test.pdf")
        **kwargs: Additional plotting options

    Returns:
        Optional tuple of (figure, axes) if debug mode is enabled
    """
    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    # Handle wildcard matching
    if handle_wildcards(var):
        return

    # Add channel to axis_opts
    if flavor:
        axis_opts["flavor"] = flavor
    if channel:
        axis_opts["channel"] = channel

    opts = {"var": var,
            "cut": cut,
            "axis_opts": axis_opts,
            "outputFolder": cfg.outputFolder
            }
    opts.update(kwargs)

    if len(cfg.hists) > 1:
        opts["fileLabels"] = cfg.fileLabels

    try:
        fig, ax = makePlot(cfg, **opts)

    except ValueError as e:
        print(f"Error creating plot: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    # Save and display the plot
    if not save_and_open_plot(fig, output_file):
        return

    if kwargs.get("debug", False):
        return fig, ax


def plot2d(var: str = 'quadJet_selected.lead_vs_subl_m',
           process: Union[str, List[str]] = "HH4b",
           *,
           cut: Union[str, List[str]] = "passPreSel",
           flavor: Union[str, List[str]] = None,
           channel: Union[str, List[str]] = None,
           axis_opts: Dict = {},
           output_file: str = DEFAULT_OUTPUT_FILE,
           **kwargs) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Create a 2D plot of the specified variable.

    Args:
        var: Variable to plot
        process: Process to plot (default: "HH4b")
        cut: Selection cut to apply (default: "passPreSel")
        region: Region to plot (default: "SR")
        output_file: Name of the output file (default: "test.pdf")
        **kwargs: Additional plotting options

    Returns:
        Optional tuple of (figure, axes) if debug mode is enabled
    """
    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    if handle_wildcards(var):
        return

    # Add channel to axis_opts
    if flavor:
        axis_opts["flavor"] = flavor
    if channel:
        axis_opts["channel"] = channel

    try:
        fig, ax = make2DPlot(cfg, process, var=var, cut=cut,
                             axis_opts=axis_opts, outputFolder=cfg.outputFolder, **kwargs)
    except Exception as e:
        print(f"Error creating 2D plot: {e}")
        return

    if not save_and_open_plot(fig, output_file):
        return

    if kwargs.get("debug", False):
        return fig, ax


def initialize_config() -> None:
    """Initialize the configuration from command line arguments."""
    args = parse_args()
    cfg.plotConfig = load_config_bbWW(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files

    if cfg.outputFolder and not os.path.exists(cfg.outputFolder):
        os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)


if __name__ == '__main__':
    initialize_config()
    print_cfg(cfg)
