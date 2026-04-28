from __future__ import annotations
from typing import TYPE_CHECKING

from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser
from bbreww.classifier.config.model.bbWW.bbWW_lowpt._bbWWBase import ROC_BIN, bbWWBaseEval, bbWWBaseTrain
from bbreww.classifier.config.setting.bbWW import Input, Output

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


class Train(bbWWBaseTrain):
    argparser = ArgParser(description="Train bbWW Model")
    model = "svb"

    @staticmethod
    def loss(batch: BatchType):
        import torch.nn.functional as F

        logits = batch[Output.hh_raw]
        labels = batch[Input.label]
        weight = batch[Input.weight]
        weight[weight < 0] = 0

        cross_entropy = F.cross_entropy(logits, labels, reduction="none")
        return (cross_entropy * weight).sum() / weight.sum().clamp(min=1e-8)

    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC

        return [
            # Signal vs all background (score = p_signal).
            # Z_A on this = total sensitivity vs everything else.
            ROC(
                name="Signal vs Background",
                selection=_roc_signal_selection("signal"),
                bins=ROC_BIN,
                pos=("signal",),
            ),

            # Signal vs ttbar only (score = linear_differ(p_signal, p_ttbar)).
            # Z_A here = sensitivity against the dominant background. If this is
            # much worse than Z_A[Signal vs Others], ttbar is the bottleneck.
            ROC(
                name="Signal vs TTbar",
                selection=_roc_signal_selection("signal"),
                bins=ROC_BIN,
                pos=("signal",),
                neg=("ttbar",),
                score="differ",
            ),

            # Signal vs minor-bkg only (score = linear_differ(p_signal, p_other)).
            # Z_A here = sensitivity against WJets/tW/singleTop aggregate.
            ROC(
                name="Signal vs Others",
                selection=_roc_signal_selection("signal"),
                bins=ROC_BIN,
                pos=("signal",),
                neg=("other",),
                score="differ",
            ),

            # Legacy diagnostic ROCs (not sensitivity-relevant — left for backward
            # compat with existing plotting code).
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


class Eval(bbWWBaseEval):
    model = "svb"

    @staticmethod
    def output_definition(batch: BatchType):
        return {
            "phh":       batch["p_signal"], 
            "ptt":       batch["p_ttbar"],
            "poth":      batch["p_other"],  
            "tt_b1Whad": batch["tt_b1Whad"],
            "tt_b2Whad": batch["tt_b2Whad"],
            "WW_score1":  batch["WW_score1"],
            "WW_score2":  batch["WW_score2"],
            "WW_score3":  batch["WW_score3"],
            "hh_vs_tt":  batch["p_signal"]/(batch["p_signal"] + batch["p_ttbar"]),
            "hh_vs_oth": batch["p_signal"]/(batch["p_signal"] + batch["p_other"]),
            "tt_vs_oth": batch["p_ttbar"]/(batch["p_ttbar"] + batch["p_other"]),
        }
