from __future__ import annotations
from typing import TYPE_CHECKING

from src.classifier.task import ArgParser
from bbreww.classifier.config.model.bbWW.METRegressor._METRegressor import (
    METRegressorTrain,
    METRegressorEval,
)
from bbreww.classifier.config.setting.bbWW import Input

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


class Train(METRegressorTrain):
    argparser = ArgParser(description="Train MET pz Regressor")
    model = "met_regressor"

    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F

        pred = batch["pred_nu"]          # (n, 3): predicted pT, eta, phi
        target = batch[Input.genNu]      # (n, 3): true pT, eta, phi
        weight = batch[Input.weight]     # (n,)
        weight = weight.clamp(min=0)

        # Mask out events with no gen neutrino (filled with -1)
        valid = (target[:, 0] >= 0)
        pred = pred[valid]
        target = target[valid]
        weight = weight[valid]

        if weight.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Huber loss on each coordinate
        loss = F.smooth_l1_loss(pred, target, reduction="none")  # (n, 3)
        loss = loss.mean(dim=1)  # (n,) average over coordinates
        loss = (loss * weight).sum() / weight.sum()
        return loss


class Eval(METRegressorEval):
    model = "met_regressor"

    @staticmethod
    def output_definition(batch: BatchType):
        return {
            "nu_pt":  batch["nu_pt"],
            "nu_eta": batch["nu_eta"],
            "nu_phi": batch["nu_phi"],
            "estimated_mW": batch["estimated_mW"],
            "regime_prob":  batch["regime_prob"],
        }
