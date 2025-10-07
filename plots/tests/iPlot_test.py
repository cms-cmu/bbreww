import unittest
import argparse
#from coffea.util import load
import yaml
import sys


import os
sys.path.insert(0, os.getcwd())

from bbreww.plots.iPlot import plot, plot2d, cfg
from bbreww.plots.plots import load_config_bbWW
from src.plotting.plots import load_hists, read_axes_and_cuts
from bbreww.tests.parser import wrapper


class iPlotTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.inputFile = wrapper.args["inputFile"]


    def do_plots(self):

        args     = {"var": "Hbb.*", "flavor": sum, "channel":sum, "cut": "preselection",}
        doRatio  = {"doRatio": 1}
        norm     = {"norm": 1}
        logy     = {"yscale": "log"}
        rlim     = {"rlim": [0, 2]}
        rebin    = {"rebin": 4}
        add_flow = {"add_flow": True}

        print(f"plot with {args}")
        plot(**args)
        args["var"] = "Hbb.mass"

        print(f"plot with {args}")
        plot(**args)

        print(f"plot with {args | doRatio}")
        plot(**(args | doRatio))

        print(f"plot with {args | doRatio| add_flow}")
        plot(**(args | doRatio| add_flow))

        print(f"plot with {args | norm}")
        plot(**(args | norm))

        print(f"plot with {args | logy}")
        plot(**(args | logy))

        print(f"plot with {args | rebin}")
        plot(**(args | rebin))

        print(f"plot with {args | doRatio | norm}")
        plot(**(args | doRatio | norm))

        print(f"plot with {args | doRatio | norm | rlim}")
        plot(**(args | doRatio | norm | rlim))

        print(f"plot with {args | doRatio | norm | rlim | rebin}")
        plot(**(args | doRatio | norm | rlim | rebin))

        manyCuts = {"cut": ["nominal_4j2b", "preselection"],
                    "process": "HHbbWW"}
        print(f"plot with {args | doRatio | norm | rlim | manyCuts}")
        plot(**(args | doRatio | norm | rlim | manyCuts))


        args2d = {"var": "mbb_vs_bb_dr", "channel": sum, "flavor": sum,
                  "cut": "preselection", "process": "HHbbWW"}
        full = {"full": True}
        print(f"plot with {args2d}")
        plot2d(**args2d)

        print(f"plot with {args2d | full}")
        plot2d(**(args2d | full))

        manyProcs = {"cut": "preselection",
                     "process": ["HHbbWW","TTbar"]}
        print(f"plot with {args | doRatio | norm | rlim | manyProcs}")
        plot(**(args | doRatio | norm | rlim | manyProcs))

        manyVars = {"cut": "nominal_4j2b",
                    "var": ["Hbb.lead.pt","Hbb.subl.pt","Wqq.lead.pt","Wqq.subl.pt"],
                    "process": "HHbbWW"}
        print(f"plot with {args | doRatio | norm | rlim | manyVars}")
        plot(**(args | doRatio | norm | rlim | manyVars))

        manyYears = {"cut": "preselection",
                     "var": "Hbb.mass",
                     "year":  ["2022","2022"],
                     "process": "HHbbWW"}
        print(f"plot with {args | doRatio | norm | rlim | manyYears}")
        plot(**(args | doRatio | norm | rlim | manyYears))

        args["var"] = "Hbb.mass"
        invalid_region = {"region": "InvalidRegion"}
        print(f"plot with {args | invalid_region}")
        self.assertIsNone(plot(**(args | invalid_region)))

        invalid_cut    = {"cut": "InvalidCut"}
        print(f"plot with {args | invalid_cut}")
        self.assertIsNone(plot(**(args | invalid_cut)))


    def test_singleFile(self):

        metadata = "bbreww/plots/metadata/plotsAll.yml"
        cfg.plotConfig = load_config_bbWW(metadata)

        input_files = [self.inputFile]
        cfg.hists = load_hists(input_files)

        cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists,
                                                                 cfg.plotConfig,
                                                                 hist_keys=['hists','hists_4j2b'])

        cfg.set_hist_key("hists")

        self.do_plots()



#    def test_multipleFiles(self):
#
#        metadata = "bbreww/plots/metadata/plotsAll.yml"
#        cfg.plotConfig = load_config_bbWW(metadata)
#
#        input_files = [self.inputFile, self.inputFile]
#        cfg.hists = load_hists(input_files)
#        cfg.fileLabels = ["file1", "file2"]
#
#        cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists,
#                                                                 cfg.plotConfig)
#        cfg.set_hist_key("hists")
#
#        args    = {"var": "v4j.*", "region": "SR",
#                   "cut": "preselection", "process": "HHbbWW"}
#
#        doRatio = {"doRatio": 1}
#        norm    = {"norm": 1}
#        logy    = {"yscale": "log"}
#        rlim    = {"rlim": [0, 2]}
#
#        print(f"plot with {args}")
#        plot(**args)
#        args["var"] = "v4j.mass"
#
#        print(f"plot with {args}")
#        plot(**args)
#
#        print(f"plot with {args | doRatio}")
#        plot(**(args | doRatio))
#
#        print(f"plot with {args | norm}")
#        plot(**(args | norm))
#
#        print(f"plot with {args | logy}")
#        plot(**(args | logy))
#
#        print(f"plot with {args | doRatio | norm}")
#        plot(**(args | doRatio | norm))
#
#        print(f"plot with {args | doRatio | norm | rlim}")
#        plot(**(args | doRatio | norm | rlim))
#




if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)
