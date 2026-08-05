from __future__ import annotations
from typing import TYPE_CHECKING
from bbreww.classifier.config.model.bbWW.bbWW_lowpt._bbWWBase import ROC_BIN, bbWWBaseEval, bbWWBaseTrain
from bbreww.classifier.config.setting.bbWW import Input, Output

if TYPE_CHECKING:
    from src.classifier.ml import BatchType

class SparseDenseTrain(bbWWBaseTrain):
    model = "test-SvD"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F
        
        logits = batch[Output.hh_raw]
        labels = batch[Input.label]
        weight = batch[Input.weight]

        cross_entropy = F.cross_entropy(logits, labels, reduction="none")
        loss = (cross_entropy * weight).sum() / (weight.sum() + 1e-8)
        return loss

    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC
        from bbreww.classifier.config.model.bbWW.bbWW_lowpt._bbWWBase import roc_nominal_selection
        
        return [
            ROC(
                name="sparse vs dense",
                selection=roc_nominal_selection,
                bins=ROC_BIN,
                pos=["sparse"],
            )
        ]

class SparseDenseEval(bbWWBaseEval):
    model = "test-SvD"

    @staticmethod
    def output_definition(batch: BatchType):
        output = {
            "hh_prob": batch["p_sparse"],
            "tt_prob": batch["p_dense"],
            "ww_prob": batch["WW_score"].mean(dim=list(range(1, batch["WW_score"].dim()))),
        }
        return output
