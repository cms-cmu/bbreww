from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser
from bbreww.classifier.config.setting.bbWWHCR import Input, MassRegion, Output
from bbreww.classifier.config.model.bbWW.base._HCR import ROC_BIN, HCREval, HCRTrain

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


def _roc_data_selection(batch: BatchType):
    
    def __call__(self, batch: BatchType):
        selected = self._select(batch)
        return {
            "y_pred": batch[Output.tt_prob][selected],  # Signal probability
            "y_true": batch[Input.label][selected],
            "weight": batch[Input.weight][selected],
        }

    def _select(self, batch: BatchType):
        import torch
        label = batch[Input.label]
        return torch.isin(label, label.new_tensor(MultiClass.indices("ttbar", "data")))


class Train(HCRTrain):
    argparser = ArgParser(description="Train dvtt")
    model = "dvtt"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F

        # get tensors
        ## use hh score below, as it already captures TTbar info and excludes signal in reweighting
        tt_score = batch[Output.hh_raw]
        weight = batch[Input.weight]
        weight[weight < 0] = 0
        is_SR = MassRegion.SR

        # calculate loss
        cross_entropy = torch.zeros_like(weight)
        cross_entropy[~is_SR] = F.cross_entropy(
            tt_score[~is_SR], batch[Input.label][~is_SR], reduction="none"
        )
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss
        
    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC
        
        return[
            ROC(
                name="ttbar vs data",
                selection=_roc_data_selection,
                bins=ROC_BIN,
                pos=["ttbar"],
                neg=["data"]
            )
        ]

class Eval(HCREval):
    model = "dvtt"

    @staticmethod
    def output_definition(batch: BatchType):
            ttbar_idx = MultiClass.indices("ttbar")[0]
            output = {
                "p_score": batch[Output.class_prob][:, ttbar_idx]  # ttbar probability
            }
            return output