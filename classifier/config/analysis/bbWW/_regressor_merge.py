import numpy as np

from src.classifier.config.analysis.kfold import Merge

# Must match the threshold used in the processor (hh_bbww_processor.py) and in
# RegressorModelEval (METRegressor.py): p_onshell > 0.55 selects the on-shell hypothesis.
P_ONSHELL_THRESHOLD = 0.55


def _select_hypothesis(merged: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Re-select the neutrino hypothesis from the ensemble-averaged p_onshell.
    # The per-model nu_* branches are selected before merging, so their mean mixes
    # on- and off-shell kinematics for events where models disagree; the per-hypothesis
    # branches average cleanly and are re-selected here with the averaged classifier.
    if "p_onshell" not in merged:
        return {}
    use_on = merged["p_onshell"] > P_ONSHELL_THRESHOLD
    out = {}
    for q in ("px", "py", "pz"):
        on, off = f"nu_{q}_on", f"nu_{q}_off"
        if on in merged and off in merged:
            out[f"nu_{q}"] = np.where(use_on, merged[on], merged[off])
    if "sigma_pz_on" in merged and "sigma_pz_off" in merged:
        out["nu_sigma_pz"] = np.where(
            use_on, merged["sigma_pz_on"], merged["sigma_pz_off"]
        )
    return out


class RegressorMerge(Merge):
    def post_methods(self):
        return [_select_hypothesis]
