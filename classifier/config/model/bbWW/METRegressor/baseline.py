from __future__ import annotations
from typing import TYPE_CHECKING

from src.classifier.task import ArgParser
from bbreww.classifier.config.model.bbWW.METRegressor._METRegressor import (
    METRegressorTrain,
    METRegressorEval,
)
from bbreww.classifier.config.setting.bbWW import Input
from bbreww.classifier.nn.blocks.bbWW_models import get_nu_pz_cartesian

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


class Train(METRegressorTrain):
    argparser = ArgParser(description="Train MET pz Regressor")
    argparser.set_defaults(architecture={"n_features": 16})
    model = "met_regressor"

    @staticmethod
    def loss(batch: BatchType):
        """Returns three INDEPENDENT losses: (backbone_loss, onshell_nll, offshell_nll).

        Each is minimized by its own optimizer — they are never summed.
        - backbone_loss:  classifier BCE → trains shared backbone
        - onshell_nll:    Gaussian NLL on isLepW==1 events (pz from analytic constraint)
        - offshell_nll:   Gaussian NLL on isLepW==0 events + W mass reco penalty
        """
        import torch
        import torch.nn.functional as F
        import math

        target_ptep = batch[Input.genNu]              # (n, 3): true pT, eta, phi
        weight = batch[Input.weight]                   # (n,)
        pred_on = batch["pred_nu_on"]                  # (n, 3): on-shell hypothesis px, py, pz
        pred_off = batch["pred_nu_off"]                # (n, 3): off-shell hypothesis px, py, pz
        cholesky_L_on = batch["cholesky_L_on"]         # (n, 3, 3)
        cholesky_L_off = batch["cholesky_L_off"]       # (n, 3, 3)
        weight = weight.clamp(min=0)

        # Mask out events with no gen neutrino (filled with -1)
        valid = (target_ptep[:, 0] >= 0)
        target_ptep = target_ptep[valid]
        weight = weight[valid]
        pred_on = pred_on[valid]
        pred_off = pred_off[valid]
        cholesky_L_on = cholesky_L_on[valid]
        cholesky_L_off = cholesky_L_off[valid]

        zero = torch.tensor(0.0, device=target_ptep.device, requires_grad=True)
        if weight.sum() == 0:
            return zero, zero, zero

        # Convert target from (pT, eta, phi) to (px, py, pz)
        t_pt, t_eta, t_phi = target_ptep[:, 0], target_ptep[:, 1], target_ptep[:, 2]
        target = torch.stack([
            t_pt * torch.cos(t_phi),
            t_pt * torch.sin(t_phi),
            t_pt * torch.sinh(t_eta),
        ], dim=1)  # (n, 3): px, py, pz

        # --- Label-based masks ---
        genLepW = batch[Input.genLepW][valid]  # (n, 2): isLepW, genLepWmass
        isLepW = genLepW[:, 0]                 # (n,): 1=on-shell, 0=off-shell, -1=unknown
        target_mW = genLepW[:, 1]              # (n,)
        is_on = (isLepW == 1)
        is_off = (isLepW == 0)
        has_label = (isLepW >= 0)

        # Gaussian NLL with full Cholesky covariance
        def _nll(pred, cholesky_L, target, mask, w):
            if mask.sum() == 0 or w[mask].sum() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            p = pred[mask]
            residual = (target[mask] - p).unsqueeze(-1)

            L = cholesky_L[mask]
            z = torch.linalg.solve_triangular(L, residual, upper=False).squeeze(-1)
            z = z.clamp(-100, 100)
            
            log_det = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            nll = 0.5 * (z ** 2).sum(dim=1) + log_det + 1.5 * math.log(2 * math.pi)
            
            return (nll * w[mask]).sum() / w[mask].sum()

        # Gaussian NLL + W mass reconstruction penalty
        def _nll_and_reco(pred, cholesky_L, target, mask, w, lep_px, lep_py, lep_pz, lep_E, target_mW):
            if mask.sum() == 0 or w[mask].sum() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            p = pred[mask]
            residual = (target[mask] - p).unsqueeze(-1)
            L = cholesky_L[mask]

            z = torch.linalg.solve_triangular(L, residual, upper=False).squeeze(-1)
            z = z.clamp(-100, 100)
            
            log_det = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            nll = 0.5 * (z ** 2).sum(dim=1) + log_det + 1.5 * math.log(2 * math.pi)
            nll_loss = (nll * w[mask]).sum() / w[mask].sum()
            
            # W mass reconstruction loss: log-normal NLL.
            # The Jacobian term log(mW_reco) breaks symmetry in linear mass space,
            # penalizing overestimates more than underestimates and pulling predictions
            # toward the mode of the right-skewed off-shell distribution.
            nu_E = torch.sqrt(p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2)
            mW_sq = (lep_E[mask] + nu_E)**2 - (lep_px[mask] + p[:, 0])**2 \
                    - (lep_py[mask] + p[:, 1])**2 - (lep_pz[mask] + p[:, 2])**2
            mW = torch.sqrt(F.softplus(mW_sq, beta=1.0, threshold=20.0).clamp(min=1.0))
            log_mW = torch.log(mW)
            log_target_mW = torch.log(target_mW[mask].clamp(min=1.0))
            
            reco = 0.5 * (log_mW - log_target_mW)**2 - log_mW
            reco_loss = (reco * w[mask]).sum() / w[mask].sum()
            
            return nll_loss + 1 * reco_loss

        # Precompute lepton kinematics (needed for off-shell W mass reco)
        lep = batch["_leadingLep"][valid]
        lep_px = lep[:, 0] * torch.cos(lep[:, 2])
        lep_py = lep[:, 0] * torch.sin(lep[:, 2])
        lep_pz = lep[:, 0] * torch.sinh(lep[:, 1])
        lep_E = torch.sqrt(lep_px**2 + lep_py**2 + lep_pz**2 + lep[:, 3]**2)

        # ---- Loss 1: on-shell proper mixture NLL + auxiliary selector BCE ----
        logit_sol_on = batch["pz_hint_on"][valid]  # carries logit_sol from regressor
        if is_on.sum() > 0 and weight[is_on].sum() > 0:
            # Re-solve quadratic with corrected MET to get both pz solutions
            pz_s1, pz_s2, _, _ = get_nu_pz_cartesian(
                lep[:, 0], lep[:, 1], lep[:, 2], lep[:, 3],
                pred_on[:, 0], pred_on[:, 1], mW=80.379,
            )
            # Build full neutrino 3-vectors for each solution
            pred_sol1 = torch.stack([pred_on[:, 0], pred_on[:, 1], pz_s1], dim=1)
            pred_sol2 = torch.stack([pred_on[:, 0], pred_on[:, 1], pz_s2], dim=1)

            # Per-event NLL for each solution (no reduction)
            def _nll_per_event(pred, cholesky_L, target, mask):
                p = pred[mask]
                residual = (target[mask] - p).unsqueeze(-1)
                L = cholesky_L[mask]

                z = torch.linalg.solve_triangular(L, residual, upper=False).squeeze(-1)
                z = z.clamp(-100, 100)
                log_det = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)

                return 0.5 * (z ** 2).sum(dim=1) + log_det + 1.5 * math.log(2 * math.pi)

            nll1 = _nll_per_event(pred_sol1, cholesky_L_on, target, is_on)  # (n_on,)
            nll2 = _nll_per_event(pred_sol2, cholesky_L_on, target, is_on)  # (n_on,)

            # Proper mixture NLL: -log(p1 * exp(-nll1) + p2 * exp(-nll2))
            # Uses log-sum-exp for numerical stability
            log_p1 = F.logsigmoid(logit_sol_on[is_on])    # log(sigmoid(x))
            log_p2 = F.logsigmoid(-logit_sol_on[is_on])   # log(1 - sigmoid(x))
            mixture_nll = -torch.logaddexp(log_p1 - nll1, log_p2 - nll2)

            w_on = weight[is_on]
            loss_mixture = (mixture_nll * w_on).sum() / w_on.sum()

            # Auxiliary BCE: truth-match selector to the closer pz root
            true_pz = target[is_on, 2]
            dist1 = (pz_s1[is_on] - true_pz).abs()
            dist2 = (pz_s2[is_on] - true_pz).abs()
            
            target_sol = (dist1 < dist2).float()  # 1 if sol1 closer, 0 if sol2
            # Downweight near-degenerate cases where both roots are similar
            margin = (dist1 - dist2).abs()
            sol_weight = w_on * (margin > 5.0).float().clamp(min=0.1)
            sol_bce = F.binary_cross_entropy_with_logits(
                logit_sol_on[is_on], target_sol,
                weight=sol_weight, reduction="sum"
            ) / sol_weight.sum()

            loss_onshell = loss_mixture + 1.0 * sol_bce
        else:
            loss_onshell = torch.tensor(0.0, device=pred_on.device, requires_grad=True)

        # ---- Loss 2: off-shell NLL + W mass reco penalty ----
        loss_offshell = _nll_and_reco(pred_off, cholesky_L_off, target, is_off, weight,
                                      lep_px, lep_py, lep_pz, lep_E, target_mW)

        # ---- Loss 3: backbone (classifier only) ----
        logit_onshell = batch["logit_onshell"][valid]

        if has_label.sum() > 0 and weight[has_label].sum() > 0:
            clf_loss = F.binary_cross_entropy_with_logits(
                logit_onshell[has_label], isLepW[has_label].clamp(0.0, 1.0),
                weight=weight[has_label], reduction="sum"
            ) / weight[has_label].sum()
        else:
            clf_loss = torch.tensor(0.0, device=pred_on.device, requires_grad=True)

        loss_backbone = clf_loss

        return loss_backbone, loss_onshell, loss_offshell


class Eval(METRegressorEval):
    model = "met_regressor"

    @staticmethod
    def output_definition(batch: BatchType):
        return {
            # Selected (best hypothesis) neutrino
            "nu_px":        batch["nu_px"],
            "nu_py":        batch["nu_py"],
            "nu_pz":        batch["nu_pz"],
            "nu_sigma_px":  batch["nu_sigma_px"],
            "nu_sigma_py":  batch["nu_sigma_py"],
            "nu_sigma_pz":  batch["nu_sigma_pz"],
            # On-shell hypothesis
            "nu_px_on":     batch["nu_px_on"],
            "nu_py_on":     batch["nu_py_on"],
            "nu_pz_on":     batch["nu_pz_on"],
            # Off-shell hypothesis
            "nu_px_off":    batch["nu_px_off"],
            "nu_py_off":    batch["nu_py_off"],
            "nu_pz_off":    batch["nu_pz_off"],
            # Classifier
            "p_onshell":    batch["p_onshell"],
            "sigma_pz_on":  batch["nu_sigma_pz_on"],
            "sigma_pz_off": batch["nu_sigma_pz_off"],
            # Per-jet attention weights (2 heads × 3 jets)
            "jet_weight_0": batch["jet_weight_0"],
            "jet_weight_1": batch["jet_weight_1"],
            "jet_weight_2": batch["jet_weight_2"],
            "jet_weight_3": batch["jet_weight_3"],
            "jet_weight_4": batch["jet_weight_4"],
            "jet_weight_5": batch["jet_weight_5"],
        }
