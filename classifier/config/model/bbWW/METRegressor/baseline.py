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

        # Per-coordinate Huber loss with independent beta and normalization
        # (beta is chosen around the mean of MET resolution for pT and phi)
        loss_pt  = F.smooth_l1_loss(pred[:, 0], target[:, 0], beta=25.0, reduction="none")  / 25.0
        loss_eta = F.smooth_l1_loss(pred[:, 1], target[:, 1], beta=1.0,  reduction="none")  / 1.0
        loss_phi = F.smooth_l1_loss(pred[:, 2], target[:, 2], beta=1.0,  reduction="none")  / 1.0
        kinematic_loss = loss_pt + loss_eta + loss_phi  # (n,)
        kinematic_loss = (kinematic_loss * weight).sum() / weight.sum()

        # W mass loss: penalize deviation from true leptonic W mass
        lep = batch["_leadingLep"][valid]  # (n, 6): pt, eta, phi, mass, isE, isM
        genLepW = batch[Input.genLepW][valid]  # (n, 2): onShell, genLepWmass
        target_mW = genLepW[:, 1]  # true W mass

        # Build neutrino and lepton 4-vectors in Cartesian coordinates
        nu_pt, nu_eta, nu_phi = pred[:, 0], pred[:, 1], pred[:, 2]
        lep_pt, lep_eta, lep_phi, lep_mass = lep[:, 0], lep[:, 1], lep[:, 2], lep[:, 3]

        # neutrino (massless)
        nu_px = nu_pt * torch.cos(nu_phi)
        nu_py = nu_pt * torch.sin(nu_phi)
        nu_pz = nu_pt * torch.sinh(nu_eta)
        nu_e  = nu_pt * torch.cosh(nu_eta)

        # lepton
        lep_px = lep_pt * torch.cos(lep_phi)
        lep_py = lep_pt * torch.sin(lep_phi)
        lep_pz = lep_pt * torch.sinh(lep_eta)
        lep_e  = torch.sqrt(lep_pt**2 * torch.cosh(lep_eta)**2 + lep_mass**2)

        # invariant mass of (lepton + neutrino) system
        wlnu_e  = nu_e + lep_e
        wlnu_px = nu_px + lep_px
        wlnu_py = nu_py + lep_py
        wlnu_pz = nu_pz + lep_pz
        mW_sq = wlnu_e**2 - wlnu_px**2 - wlnu_py**2 - wlnu_pz**2
        pred_mW = torch.sqrt(torch.clamp(mW_sq, min=1e-6))

        mass_loss = F.smooth_l1_loss(pred_mW, target_mW, beta=25.0, reduction="none")  / 25.0
        mass_loss = (mass_loss * weight).sum() / weight.sum()

        return kinematic_loss + mass_loss


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
