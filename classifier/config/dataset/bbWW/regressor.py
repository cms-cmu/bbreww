from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial

from src.classifier.task import ArgParser
from src.classifier.config.dataset.HCR import _group
from src.classifier.task import ArgParser, converter, parse
from src.classifier.config.setting.df import Columns
from src.classifier.config.setting.cms import CollisionData
from bbreww.classifier.config.dataset.bbWW._common import CommonEval, CommonTrain
from bbreww.classifier.config.dataset.bbWW import _picoAOD
from bbreww.classifier.config.setting.METRegressor import Input, InputBranch


if TYPE_CHECKING:
    import pandas as pd

class Train(CommonTrain):
    """Training dataset configuration for MET pz regressor"""

    argparser = ArgParser()
    argparser.add_argument(
        "--no-SR",
        action="store_true",
        help="remove SR events from training",
    )
    argparser.add_argument(
        "--ttbar-prescale",
        default=1,
        type=int,
        help="prescale factor for ttbar events (e.g., 10 keeps 1/10 of events)",
    )

    def __init__(self):
        super().__init__()
        # Remove the region mapping preprocessor since regressor doesn't filter by SR/CR
        from src.classifier.df.tools import map_selection_to_flag
        self.preprocessors[:] = [
            p for p in self.preprocessors 
            if not isinstance(p, map_selection_to_flag)
        ]

        # Remove the debug postprocessor that depends on region_index
        from bbreww.classifier.config.dataset.bbWW._common import _debug_print_weight
        self.postprocessors[:] = [
            p for p in self.postprocessors 
            if p is not _debug_print_weight
        ]

        # add gen-level regression targets
        (
            self.to_tensor
            .add(Input.genNu, "float32").columns(*InputBranch.feature_genNu)
            .add(Input.genLepW, "float32").columns(*InputBranch.feature_genLepW)
            .add(Input.true_nbjet_flat, "float32").columns("true_nbjet_flat", target=4, pad_value=0)
        )

    def other_branches(self):
        return (super().other_branches() - {"SR", "CR"}) | set(InputBranch.feature_genNu) | set(InputBranch.feature_genLepW) | {"true_nbjet_flat"}

    def preprocess_by_group(self):
        from src.classifier.df.tools import add_label_index, prescale

        ps = []
        ps.append(
            _group.fullmatch(
                ("label:signal",),
                processors=[
                    lambda: add_label_index("signal"),
                ],
                name="HH signal selection",
            ),
        )
        if "ttbar" in self.mc_processes:
            # reduce no. of ttbar events for validation
            ttbar_processors = [
                lambda: prescale(scale=10, seed=("validation", "subsample", 0)),
                lambda: add_label_index("ttbar"),
            ]

            if self.opts.ttbar_prescale > 1:
                ttbar_processors.insert(
                    0,
                    lambda: prescale(
                        scale=self.opts.ttbar_prescale,
                        seed=("ttbar", "prescale", 0),
                    ),
                )
            ps.append(
                _group.fullmatch(
                    ("label:ttbar",),
                    processors=ttbar_processors,
                    name="ttbar selection",
                ),
            )
        

        return list(super().preprocess_by_group()) + ps

# use only semileptonic ttbar for regression
class _ttbar(_picoAOD._MCDataset):
    processes = ("ttbar",)

    def __new__(cls, self: _picoAOD.MC, metadata: str):
        filelists = []
        if "ttbar" in self.mc_processes:
            for year in CollisionData.eras:
                filelists.append(
                    [
                        f"label:ttbar,year:{year}",
                        metadata + f".TTToSemiLeptonic.{year}.picoAOD.files",
                    ]
                )
        return filelists
        
class RegressorBackground(_picoAOD.MC):
    pico_filelists = (_ttbar,)

class Background(RegressorBackground, Train):
    argparser = ArgParser()
    argparser.add_argument(
        "--norm",
        default=1.0,
        type=converter.float_pos,
        help="normalization factor",
    )

    def __init__(self):
        super().__init__()
        self.postprocessors.insert(0, partial(self.normalize, norm=self.opts.norm))

    @staticmethod
    def normalize(df: pd.DataFrame, norm: float):
        df.loc[:, "weight"] /= df["weight"].sum() / norm
        return df


class Signal(_picoAOD.Signal, Train):
    ...


class TrainBaseline(_picoAOD.Signal, Background, Train):
    """Baseline training with signal and background processes"""
    ...


class Eval(_picoAOD.Signal, _picoAOD.Background, CommonEval):
    """MC Evaluation for MET regressor"""
    ...


class DataEval(_picoAOD.Data, CommonEval):
    """Data Evaluation for MET regressor"""
    ...

