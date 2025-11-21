from __future__ import annotations
from typing import TYPE_CHECKING

from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser
from bbreww.classifier.config.model.bbWW.HCR._HCR import ROC_BIN, HCREval, HCRTrain
from bbreww.classifier.config.setting.bbWWHCR import Input, Output

if TYPE_CHECKING:
    from src.classifier.ml import BatchType

_BKG = ("ttbar", "other",)

class _roc_signal_selection:
    def __init__(self, sig: str):
        self.sig = sig

    def __call__(self, batch: BatchType):
        selected = self._select(batch)
        result = {
            "y_pred": batch[Output.hh_prob][selected],  # Signal probability
            "y_true": batch[Input.label][selected],
            "weight": batch[Input.weight][selected],
        }

        return result

    def _select(self, batch: BatchType):
        import torch

        label = batch[Input.label]
        return torch.isin(label, label.new_tensor(MultiClass.indices(*_BKG, self.sig)))


class Train(HCRTrain):
    argparser = ArgParser(description="Train bbWW Model")
    model = "svb"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F
        
        # Simple binary classification
        logits = batch[Output.hh_raw]
        labels = batch[Input.label]
        weight = batch[Input.weight]
        weight[weight < 0] = 0

        cross_entropy = F.cross_entropy(logits, labels, reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss

    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC
        
        return [
            # this ROC is for plotting ROC and AUC of signal vs background
            ROC(
                name="Signal vs Background",
                selection=_roc_signal_selection("signal"),
                bins=ROC_BIN,
                pos=("signal",),  # Signal class
            ),
            ROC(
                name="TTbar vs Others",
                selection=_roc_signal_selection("signal"),
                bins=ROC_BIN,
                pos=("ttbar",), 
            ),
            ROC(
                name="Minor backgrounds vs others",
                selection=_roc_signal_selection("signal"),
                bins=ROC_BIN,
                pos=("other",), 
            ),
        ]


class Eval(HCREval):
    model = "svb"

    @staticmethod
    def output_definition(batch: BatchType):
        return {
            "phh":       batch["p_signal"], 
            "ptt":       batch["p_ttbar"],
            "poth":      batch["p_other"],  
            "tt_b1Whad": batch["tt_b1Whad"],
            "tt_b2Whad": batch["tt_b2Whad"],
            "hh_vs_tt":  batch["p_signal"]/(batch["p_signal"] + batch["p_ttbar"]),
            "hh_vs_oth": batch["p_signal"]/(batch["p_signal"] + batch["p_other"]),
            "tt_vs_oth": batch["p_ttbar"]/(batch["p_ttbar"] + batch["p_other"]),
        }