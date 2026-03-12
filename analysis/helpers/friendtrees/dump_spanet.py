import os
import tempfile

import awkward as ak
import numpy as np
import h5py
from src.friendtrees.dump_friend import _build_cutflow
from src.storage.eos import EOS


def dump_spanet_h5(
    events: ak.Array,
    output_dir: str,
    bcand: str = "b_cands",
    nonbcand: str = "q_cands_soft",
    lepton: str = "leading_lep",
    met: str = "MET",
    genNu: str = "genNu",
    max_nonbjets: int = 3,
    weight: str = "weight",
):
    meta = events.metadata
    chunk_id = f"{meta['fileuuid']}_{meta['entrystart']}_{meta['entrystop']}"
    filename = f"spanet_{chunk_id}.h5"
    eos_output = EOS(output_dir) / filename
    is_local = eos_output.is_local

    if is_local:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        output_path = tmp.name
        tmp.close()

    # --- filter out events with NaN regression targets ---
    # (e.g. dileptonic / fully hadronic ttbar with no gen-matched leptonic W->lnu)
    _nu_pt_np = ak.to_numpy(ak.fill_none(events[f"{genNu}_pt"], np.nan))
    _wmass_np = ak.to_numpy(ak.fill_none(events["gen_lepW_mass"], np.nan))
    valid_reg = (
        ~ak.is_none(events[f"{genNu}_pt"], axis=0)
        & ~np.isnan(_nu_pt_np)
        & ~np.isnan(_wmass_np)
        & (_wmass_np < 1e6)  # filter out int64-max sentinel values
    )
    events = events[valid_reg]

    n_events = len(events[bcand])
    if n_events == 0:
        return

    # --- b-jets ---
    bjets = events[bcand]
    bjet_pt   = ak.to_numpy(ak.fill_none(ak.pad_none(bjets.pt,   2), 0))[:, :2].astype(np.float32)
    bjet_eta  = ak.to_numpy(ak.fill_none(ak.pad_none(bjets.eta,  2), 0))[:, :2].astype(np.float32)
    bjet_phi  = ak.to_numpy(ak.fill_none(ak.pad_none(bjets.phi,  2), 0))[:, :2].astype(np.float32)
    bjet_mass = ak.to_numpy(ak.fill_none(ak.pad_none(bjets.mass, 2), 0))[:, :2].astype(np.float32)
    bjet_btag = ak.to_numpy(ak.fill_none(ak.pad_none(bjets.btagScore, 2), 0))[:, :2].astype(np.float32)
    bjet_mask = np.ones((n_events, 2), dtype=bool)

    # --- non-b-jets ---
    nonbjets = events[nonbcand]
    nonbjets_padded = ak.pad_none(nonbjets, max_nonbjets)[:, :max_nonbjets]
    nonbjet_mask = ak.to_numpy(~ak.is_none(nonbjets_padded, axis=1)).astype(bool)
    nonbjet_pt   = ak.to_numpy(ak.fill_none(nonbjets_padded.pt,   0)).astype(np.float32)
    nonbjet_eta  = ak.to_numpy(ak.fill_none(nonbjets_padded.eta,  0)).astype(np.float32)
    nonbjet_phi  = ak.to_numpy(ak.fill_none(nonbjets_padded.phi,  0)).astype(np.float32)
    nonbjet_mass = ak.to_numpy(ak.fill_none(nonbjets_padded.mass, 0)).astype(np.float32)

    # --- lepton ---
    lep = events[lepton]
    lep_pt   = ak.to_numpy(lep.pt).flatten().astype(np.float32)
    lep_eta  = ak.to_numpy(lep.eta).flatten().astype(np.float32)
    lep_phi  = ak.to_numpy(lep.phi).flatten().astype(np.float32)
    lep_mass = ak.to_numpy(lep.mass).flatten().astype(np.float32)
    lep_isE  = ak.to_numpy(events.flavor.e).flatten().astype(np.float32)
    lep_isM  = ak.to_numpy(events.flavor.mu).flatten().astype(np.float32)
    lep_mask = np.ones(n_events, dtype=bool)

    # --- MET ---
    met_obj = events[met]
    _met_pt  = ak.to_numpy(met_obj.pt).astype(np.float32)
    _met_phi = ak.to_numpy(met_obj.phi).astype(np.float32)
    met_px   = _met_pt * np.cos(_met_phi)
    met_py   = _met_pt * np.sin(_met_phi)

    # --- event-level ---
    evt_njets     = ak.to_numpy(events.njets).astype(np.float32)
    evt_nsoftjets = ak.to_numpy(events.nsoftjets).astype(np.float32)
    evt_HT        = ak.to_numpy(events.HT).astype(np.float32)
    evt_SR        = ak.to_numpy(events.region.SR).astype(np.float32)
    evt_CR        = ak.to_numpy(events.region.CR).astype(np.float32)

    # --- event weight ---
    evt_weight = ak.to_numpy(events[weight]).astype(np.float32)

    # --- TARGETS: higgs_bb ---
    # For HH signal: find indices within b_cands that are gen-matched to H->bb
    # For everything else: -1 (no valid H->bb assignment)
    is_hh = 'HH' in events.metadata.get('dataset', '')
    if is_hh and hasattr(bjets, 'isbFromH'):
        local_idx = ak.local_index(bjets, axis=1)
        hbb_indices = ak.to_numpy(
            ak.fill_none(ak.pad_none(local_idx[bjets.isbFromH], 2), -1)
        )[:, :2].astype(np.int64)
        hbb_b1 = hbb_indices[:, 0]
        hbb_b2 = hbb_indices[:, 1]
    else:
        hbb_b1 = np.full(n_events, -1, dtype=np.int64)
        hbb_b2 = np.full(n_events, -1, dtype=np.int64)

    # --- TARGETS: higgs_WW -> q1, q2 from nonb_jets, l always index 0 ---
    try:
        isQfromW = nonbjets.isQfromW
        isQfromW_padded = ak.fill_none(ak.pad_none(isQfromW, max_nonbjets)[:, :max_nonbjets], False)
        q_indices = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(ak.local_index(isQfromW_padded)[isQfromW_padded], 2),
                -1,
            )
        )[:, :2].astype(np.int64)
    except Exception:
        q_indices = np.full((n_events, 2), -1, dtype=np.int64)

    hww_l = np.zeros(n_events, dtype=np.int64)  # lepton always index 0

    # --- REGRESSIONS: neutrino ---
    _nu_pt  = ak.to_numpy(events[f"{genNu}_pt"]).astype(np.float32)
    _nu_eta = ak.to_numpy(events[f"{genNu}_eta"]).astype(np.float32)
    _nu_phi = ak.to_numpy(events[f"{genNu}_phi"]).astype(np.float32)
    nu_px   = _nu_pt * np.cos(_nu_phi)
    nu_py   = _nu_pt * np.sin(_nu_phi)
    nu_pz   = _nu_pt * np.sinh(_nu_eta)

    gen_lepW_mass = ak.to_numpy(events["gen_lepW_mass"]).astype(np.float32)

    # --- Write HDF5 ---
    with h5py.File(output_path, "w") as f:
        # INPUTS
        inputs = f.create_group("INPUTS")

        g_b = inputs.create_group("b_jets")
        g_b.create_dataset("MASK", data=bjet_mask)
        g_b.create_dataset("pt",   data=bjet_pt)
        g_b.create_dataset("eta",  data=bjet_eta)
        g_b.create_dataset("phi",  data=bjet_phi)
        g_b.create_dataset("mass", data=bjet_mass)
        g_b.create_dataset("btag", data=bjet_btag)

        g_q = inputs.create_group("nonb_jets")
        g_q.create_dataset("MASK", data=nonbjet_mask)
        g_q.create_dataset("pt",   data=nonbjet_pt)
        g_q.create_dataset("eta",  data=nonbjet_eta)
        g_q.create_dataset("phi",  data=nonbjet_phi)
        g_q.create_dataset("mass", data=nonbjet_mass)

        g_l = inputs.create_group("lepton")
        g_l.create_dataset("MASK", data=lep_mask)
        g_l.create_dataset("pt",   data=lep_pt)
        g_l.create_dataset("eta",  data=lep_eta)
        g_l.create_dataset("phi",  data=lep_phi)
        g_l.create_dataset("mass", data=lep_mass)
        g_l.create_dataset("isE",  data=lep_isE)
        g_l.create_dataset("isM",  data=lep_isM)

        g_met = inputs.create_group("met")
        g_met.create_dataset("px", data=met_px)
        g_met.create_dataset("py", data=met_py)

        g_evt = inputs.create_group("event")
        g_evt.create_dataset("njets",     data=evt_njets)
        g_evt.create_dataset("nsoftjets", data=evt_nsoftjets)
        g_evt.create_dataset("HT",        data=evt_HT)
        g_evt.create_dataset("SR",        data=evt_SR)
        g_evt.create_dataset("CR",        data=evt_CR)

        # TARGETS
        targets = f.create_group("TARGETS")

        g_hbb = targets.create_group("higgs_bb")
        g_hbb.create_dataset("b1", data=hbb_b1)
        g_hbb.create_dataset("b2", data=hbb_b2)
        g_hbb.create_dataset("WEIGHT", data=evt_weight)

        g_hww = targets.create_group("higgs_WW")
        g_hww.create_dataset("q1", data=q_indices[:, 0])
        g_hww.create_dataset("q2", data=q_indices[:, 1])
        g_hww.create_dataset("l",  data=hww_l)
        g_hww.create_dataset("WEIGHT", data=evt_weight)
        f.create_dataset("EVENT_WEIGHT", data=evt_weight)

        # REGRESSIONS
        regressions = f.create_group("REGRESSIONS")
        g_evt = regressions.create_group("EVENT")
        g_evt.create_dataset("neutrino_px",     data=nu_px)
        g_evt.create_dataset("neutrino_py",     data=nu_py)
        g_evt.create_dataset("neutrino_pz",     data=nu_pz)
        g_evt.create_dataset("leptonic_w_mass", data=gen_lepW_mass)


    if not is_local:
        try:
            EOS(output_path).copy_to(eos_output, parents=True, overwrite=True)
        finally:
            os.remove(output_path)

