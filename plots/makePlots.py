import os
# import time
import sys
import yaml
import hist
import argparse
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np

sys.path.insert(0, os.getcwd())
from bbreww.plots.plots import load_config_bbWW
from src.plotting.plots import makePlot, make2DPlot, load_hists, read_axes_and_cuts, parse_args
import src.plotting.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')

def doPlots(varList, debug=False):

    if args.doTest:
        varList = ["qq_mass", "Hbb.mass","mbb_vs_bb_dr"]

    cut = "preselection"

    #
    #  Nominal 1D Plots
    #
    for v in varList:
        if debug: print(f"plotting 1D ...{v}")

        vDict = cfg.plotModifiers.get(v, {})
        if debug: print(v, vDict, vDict.get("2d", False))
        if vDict.get("2d", False):
            continue

        vDict["ylabel"] = "Entries"

        vDict["legend"] = True
        vDict["year"] = "Run3"
        vDict["doRatio"] = False
        #vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)

        if args.doTest:
            vDict["write_yaml"] = True

        for flavor in ["e", "mu", sum]:

            for channel in ["hadronic_W", "leptonic_W", sum]:

                if debug: print(f"plotting 1D ...{v}")
                plot_args  = {}
                plot_args["var"] = v
                plot_args["cut"] = cut
                plot_args["outputFolder"] = args.outputFolder
                plot_args["axis_opts"] = {"flavor":flavor, "channel":channel}
                plot_args = plot_args | vDict
                if debug: print(plot_args)
                try:
                    fig = makePlot(cfg, **plot_args)
                except ValueError:
                    print(f"ValueError: {v} {flavor} {channel} {cut}")
                    pass

                plt.close()

    #
    #  2D Plots
    #
    for v in varList:
        if debug: print(v)

        vDict = cfg.plotModifiers.get(v, {})

        if not vDict.get("2d", False):
            continue

        vDict["ylabel"] = "Entries"
        vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
        vDict["legend"] = True
        vDict["year"] = "Run3"

        if args.doTest:
            vDict["write_yaml"] = True

        for process in ["HHbbWW","TTbar"]:

            for flavor in ["e", "mu", sum]:

                for channel in ["hadronic_W", "leptonic_W", sum]:

                    plot_args  = {}
                    plot_args["var"] = v
                    plot_args["cut"] = cut
                    plot_args["axis_opts"] = {"flavor":flavor, "channel":channel}
                    plot_args["outputFolder"] = args.outputFolder
                    plot_args = plot_args | vDict

                    if debug: print("process is ",process)
                    if debug: print(plot_args)

                    fig = make2DPlot(cfg, process,
                                     **plot_args)
                    plt.close()

    #
    #  Comparison Plots
    #
    varListComp = []
    if args.doTest:
        varListComp = ["Hbb.mass"]

        for v in varListComp:
            if debug: print(v)

            vDict = cfg.plotModifiers.get(v, {})

            vDict["ylabel"] = "Entries"
            vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
            vDict["legend"] = True
            vDict["year"] = "Run3"

            if args.doTest:
                vDict["write_yaml"] = True

            for process in ["HHbbWW", "TTbar"]:

                #
                # Comp Cuts
                #
                for channel in ["hadronic_W", "leptonic_W", sum]:

                    flavor = sum

                    plot_args  = {}
                    plot_args["var"] = v
                    plot_args["cut"] = ["preselection", "nominal_4j2b"]
                    plot_args["axis_opts"] = {"flavor":flavor, "channel":channel}
                    plot_args["outputFolder"] = args.outputFolder
                    plot_args["process"] = process
                    plot_args["norm"] = True
                    plot_args = plot_args | vDict

                    if debug: print("comp Cuts ")
                    if debug: print(plot_args)

                    fig = makePlot(cfg, **plot_args)


                    plt.close()

                #
                # Comp channels
                #
                plot_args  = {}
                plot_args["var"] = v
                plot_args["cut"] = "preselection"
                plot_args["axis_opts"] = {"flavor":sum, "channel": ["hadronic_W", "leptonic_W", sum]}
                plot_args["outputFolder"] = args.outputFolder
                plot_args["process"] = process
                plot_args["norm"] = True
                plot_args = plot_args | vDict

                if debug: print("comp channels")
                if debug: print(plot_args)

                fig = makePlot(cfg,
                               **plot_args,
                               )

                plt.close()


if __name__ == '__main__':

    args = parse_args()

    cfg.plotConfig = load_config_bbWW(args.metadata)
    cfg.outputFolder = args.outputFolder

    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    if args.list_of_hists:
        varList = args.list_of_hists
    else:
        varList = [h for h in cfg.hists[0]['hists'].keys() if not any(skip in h for skip in args.skip_hists)]

    doPlots(varList, debug=args.debug)
