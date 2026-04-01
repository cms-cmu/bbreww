from __future__ import annotations
from typing import TYPE_CHECKING

from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser
from bbreww.classifier.config.model.bbWW.HCR_lowpt._HCR import ROC_BIN, HCREval, HCRTrain
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


class Train(HCRTrain):
    argparser = ArgParser(description="Train bbWW Model")
    model = "svb"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F

        # Classification loss
        logits = batch[Output.hh_raw]
        labels = batch[Input.label]
        weight = batch[Input.weight]
        weight[weight < 0] = 0

        cross_entropy = F.cross_entropy(logits, labels, reduction="none")
        clf_loss = (cross_entropy * weight).sum() / weight.sum()

        # Jet attention loss: supervise WW attention with truth W jet labels
        true_nbjet = batch[Input.true_nbjet_flat]  # (n, wsl) binary: 1 if true q from W
        has_true_jets = (true_nbjet.sum(dim=-1) > 0)  # only events with labeled jets

        if has_true_jets.any():
            ww_weights = batch["ww_weights"]  # (n, heads, 1, wsl)
            # Average across heads and squeeze: (n, heads, 1, wsl) -> (n, wsl)
            ww_weights = ww_weights.squeeze(2).mean(dim=1)

            # Normalize truth to a probability distribution
            target_dist = true_nbjet[has_true_jets]
            target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp(min=1)

            # Cross-entropy between attention weights and truth distribution
            jet_attn_loss = -(target_dist * torch.log(ww_weights[has_true_jets] + 1e-8)).sum(dim=-1).mean()
        else:
            jet_attn_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return clf_loss + 0.1 * jet_attn_loss

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
            "WW_score1":  batch["WW_score1"],
            "WW_score2":  batch["WW_score2"],
            "WW_score3":  batch["WW_score3"],
            "hh_vs_tt":  batch["p_signal"]/(batch["p_signal"] + batch["p_ttbar"]),
            "hh_vs_oth": batch["p_signal"]/(batch["p_signal"] + batch["p_other"]),
            "tt_vs_oth": batch["p_ttbar"]/(batch["p_ttbar"] + batch["p_other"]),
        }
