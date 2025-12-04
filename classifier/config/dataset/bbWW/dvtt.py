from __future__ import annotations
from typing import TYPE_CHECKING
from src.classifier.task import ArgParser
from src.classifier.config.dataset.HCR import _group
from bbreww.classifier.config.dataset.bbWW._common import CommonEval, CommonTrain
from bbreww.classifier.config.dataset.bbWW import _picoAOD

if TYPE_CHECKING:
    import pandas as pd

def _data_selection(df: pd.DataFrame):
    """Select control region events excluding signal region events"""
    return df[df["CR"] & (~df["SR"])]

class Train(CommonTrain):
    """Training dataset configuration for HH→bbWW classifier"""
    
    argparser = ArgParser()
    argparser.add_argument(
        "--no-SR",
        action="store_true",
        help="remove SR events from training",
    )

    def preprocess_by_group(self):
        from src.classifier.df.tools import add_label_index, prescale

        ps = []
        ps.append(
            _group.fullmatch(
                ("label:data",),
                processors=[
                    lambda: _data_selection,
                    lambda: add_label_index("data"),
                ],
                name="data selection",
            ),
        )
        ps.append(
            _group.fullmatch(
                ("label:ttbar",),
                processors=[
                    lambda: _data_selection,
                    lambda: add_label_index("ttbar"),
                ],
                name="ttbar selection",
            ),
        )
        minor_bkgs =  ["WplusJets", "tW", "singleTop"]
        for bkg in minor_bkgs:
            if hasattr(self, 'mc_processes') and bkg in self.mc_processes:
                ps.append(
                    _group.fullmatch(
                        (f"label:{bkg}",),
                        processors=[
                            lambda: _data_selection,
                            lambda: add_label_index("other"),
                        ],
                        name="minor background selection",
                    ),
                )
        ps.append(_group.add_year())

        return list(super().preprocess_by_group()) + ps

class TrainData(_picoAOD.Data, Train): 
    """Baseline training with data"""
    ...
    
class TrainBaseline(_picoAOD.Background, Train): 
    """Baseline training with background processes"""
    ...
class Eval(_picoAOD.Background, CommonEval): 
    """Evaluation dataset for HH→bbWW classifier"""
    ...