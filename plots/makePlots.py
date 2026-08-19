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
from src.plotting.plots import makePlot, make2DPlot, load_hists, read_axes_and_cuts, init_arg_parser
from src.plotting.iPlot_config import plot_config

cfg = plot_config()

np.seterr(divide='ignore', invalid='ignore')

# --category selects the hist collection and the cut in one place.
CATEGORIES = {
    "nominal": ("hists_4j2b", "nominal_4j2b"),
    "lowpt":   ("hists",      "lowpt_4j2b"),
    "3j2b":    ("hists",      "incl_3j2b"),
}


def _cut_for(hist_key):
    if args.cut:
        return args.cut
    if args.category in CATEGORIES:
        return CATEGORIES[args.category][1]
    # No category given: historical per-collection defaults.
    return "incl_3j2b" if hist_key == "hists" else "nominal_4j2b"


def doPlots(varList, debug=False):

    if args.doTest:
        varList = [("Hbb.mass", "hists"), ("mbb_vs_bb_dr", "hists")]

    # Derive the years to plot from the year axis actually present in the
    # loaded hists, so we never request a year missing from the coffea
    # (e.g. CI test data is a single year). Falls back to the full Run3 list.
    years = []
    for hk in ('hists', 'hists_4j2b'):
        for y in (cfg.axisLabelsDict.get(hk, {}).get('year') or []):
            if y is not None and y not in years:
                years.append(y)
    if not years:
        years = ["2022", "2023"]
    # Combined category: "Run3" sums over the year axis (see _normalize_year).
    if "Run3" not in years:
        years.append("Run3")

    for year in years:
        if debug: print(f"=== plotting year {year} ===")

        #
        #  Nominal 1D Plots
        #
        for v, hist_key in varList:
            if debug: print(f"plotting 1D ...{v} from {hist_key}")
            cfg.set_hist_key(hist_key)

            cut = _cut_for(hist_key)


            vDict = cfg.plotModifiers.get(v, {})
            if debug: print(v, vDict, vDict.get("2d", False))
            if vDict.get("2d", False):
                continue

            vDict["ylabel"] = "Entries"
            vDict["legend"] = True
            vDict["year"] = year
            vDict["yscale"] = "log"
            vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)

            if args.doTest:
                vDict["write_yaml"] = True

            for flavor in ["e", "mu", sum]:
            #for channel in ["hadronic_W", "leptonic_W", sum]:
                for region in ["SR", "CR", sum]:

                    if debug: print(f"plotting 1D ...{v}")
                    plot_args  = {}
                    plot_args["var"] = v
                    plot_args["cut"] = cut
                    plot_args["outputFolder"] = args.outputFolder
                    plot_args["axis_opts"] = {"flavor":flavor, "region": region}  #"channel":channel,}
                    plot_args = plot_args | vDict
                    if debug: print(plot_args)
                    try:
                        fig = makePlot(cfg, **plot_args)
                    except ValueError as e:
                        print(f"ValueError: {v} {flavor} {region} {cut} {year}: {e}")
                        pass

                    plt.close()

        #
        #  2D Plots
        #
        for v, hist_key in varList:
            if debug: print(v)
            cfg.set_hist_key(hist_key)
            cut = _cut_for(hist_key)

            vDict = cfg.plotModifiers.get(v, {})

            if not vDict.get("2d", False):
                continue

            vDict["ylabel"] = "Entries"
            vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
            vDict["legend"] = True
            vDict["year"] = year

            if args.doTest:
                vDict["write_yaml"] = True

            for process in ["HHbbWW","TTbar"]:
                for flavor in ["e", "mu", sum]:
                    #for channel in ["hadronic_W", "leptonic_W", sum]:
                    for region in ["SR", "CR", sum]:

                        plot_args  = {}
                        plot_args["var"] = v
                        plot_args["cut"] = cut
                        plot_args["axis_opts"] = {"flavor":flavor, "region" :region} #"channel":channel}
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
            varListComp = ["Hbb.mass", "Elec.pt", "Muon.pt"]

            for v in varListComp:
                if debug: print(v)

                vDict = cfg.plotModifiers.get(v, {})

                vDict["ylabel"] = "Entries"
                vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
                vDict["legend"] = True
                vDict["year"] = year

                if args.doTest:
                    vDict["write_yaml"] = True

                for process in ["HHbbWW", "TTbar"]:

                    #
                    # Comp channels
                    #
                    #for channel in ["hadronic_W", "leptonic_W", sum]:
                    for region in ["SR", "CR", sum]:

                        flavor = sum

                        plot_args  = {}
                        plot_args["var"] = v
                        plot_args["cut"] = ["nominal_4j2b"]
                        plot_args["hist_key_list"] = ["hists_4j2b"]
                        plot_args["axis_opts"] = {"flavor":flavor, "region": region} # "channel":channel}
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
                    plot_args["cut"] = "nominal_4j2b"
                    plot_args["axis_opts"] = {"flavor":sum, "region" : ["SR", "CR"]}#"channel": ["hadronic_W", "leptonic_W", sum]}
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

    parser = init_arg_parser()
    parser.add_argument('--cut', default=None,
                        help='Cut category to plot (e.g. nominal_4j2b, lowpt_4j2b, incl_3j2b). '
                             'Default None keeps the per-collection defaults.')
    args = parser.parse_args()

    cfg.plotConfig = load_config_bbWW(args.metadata)
    cfg.outputFolder = args.outputFolder

    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists, cfg.plotConfig, hist_keys=['hists','hists_4j2b'])

    # --category routes hists to the right collection and cut (see CATEGORIES):
    #   nominal -> hists_4j2b (cut: nominal_4j2b)
    #   lowpt   -> hists      (cut: lowpt_4j2b)
    #   3j2b    -> hists      (cut: incl_3j2b)
    if args.category is not None and args.category not in CATEGORIES:
        parser.error(f"--category must be one of {sorted(CATEGORIES)} (got '{args.category}')")

    only_hist_key = CATEGORIES[args.category][0] if args.category else 'hists_4j2b'

    if args.list_of_hists:
        varList = [(v, only_hist_key) for v in args.list_of_hists]
    else:
        varList = []
        if args.category is None or CATEGORIES[args.category][0] == 'hists':
            for h in cfg.hists[0]['hists'].keys():
                if not any(skip in h for skip in args.skip_hists):
                    varList.append((h, 'hists'))

        if args.category is None or CATEGORIES[args.category][0] == 'hists_4j2b':
            for h in cfg.hists[0].get('hists_4j2b', {}).keys():
                if not any(skip in h for skip in args.skip_hists):
                    varList.append((h, 'hists_4j2b'))

    doPlots(varList, debug=args.debug)
