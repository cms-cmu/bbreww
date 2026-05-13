import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector
from bbreww.analysis.helpers.common import met_reconstr, distance, elliptical_region

def Hbb_candidate_selection(events):

    # this funciton called assuming we have applied basic preseleciton
    #   >= 2 bjets

    Hbb_cand = events.b_cands[:,0] + events.b_cands[:,1]
    Hbb_cand["lead"] = events.b_cands[:,0]
    Hbb_cand["subl"] = events.b_cands[:,1]
    Hbb_cand["st"]   = Hbb_cand["lead"].pt + Hbb_cand["subl"].pt
    Hbb_cand["dr"]   = Hbb_cand["lead"].delta_r  (Hbb_cand["subl"])
    Hbb_cand["dphi"] = Hbb_cand["lead"].delta_phi(Hbb_cand["subl"])

    events['Hbb_cand'] = Hbb_cand


    return events


def Wlnu_candidate_selection(events):

    # calculate MET pz requiring (lepton + nu).mass == W_mass and take the smaller solution
    nu, pz_1, pz_2 = met_reconstr(events, events.leading_lep)

    Wlnu_cand = events.leading_lep + nu
    Wlnu_cand["lep"] = events.leading_lep
    Wlnu_cand["pz_1"] = pz_1
    Wlnu_cand["pz_2"] = pz_2
    Wlnu_cand["nu"]  = nu
    Wlnu_cand["dr"]   = Wlnu_cand["lep"].delta_r(Wlnu_cand["nu"])
    Wlnu_cand["deta"]   = Wlnu_cand["lep"].eta - Wlnu_cand["nu"].eta

    Wlnu_cand["dphi"] = Wlnu_cand["lep"].delta_phi(Wlnu_cand["nu"])
    Wlnu_cand["mT"]   = np.sqrt(2 * Wlnu_cand.lep.pt * Wlnu_cand.nu.pt * (1 - np.cos(Wlnu_cand.dphi)))
    
    events['Wlnu_cand'] = Wlnu_cand
    return events


def Wqq_candidate_selection(events):
    Wqq_cand = events.q_cands_nom[:,0] + events.q_cands_nom[:,1]
    Wqq_cand["lead"] = events.q_cands_nom[:,0]
    Wqq_cand["subl"] = events.q_cands_nom[:,1]
    Wqq_cand["st"]   = Wqq_cand["lead"].pt + Wqq_cand["subl"].pt
    Wqq_cand["dr"]   = Wqq_cand["lead"].delta_r  (Wqq_cand["subl"])
    Wqq_cand["dphi"] = Wqq_cand["lead"].delta_phi(Wqq_cand["subl"])

    events['Wqq_cand'] = Wqq_cand
    return events

def Hww_candidate_selection(events):
    Hww_cand = events.Wlnu_cand + events.Wqq_cand
    Hww_cand["dr"]   = events.Wlnu_cand.delta_r(events.Wqq_cand)
    Hww_cand["dphi"] = events.Wlnu_cand.delta_phi(events.Wqq_cand)
    Hww_cand["lqq_dr"] = events.Wlnu_cand.lep.delta_r(events.Wqq_cand)
    Hww_cand["lqq_mass"] = (events.Wlnu_cand.lep+events.Wqq_cand).mass

    events['Hww_cand'] = Hww_cand
    return events

def ttbar_candidate_selection(events, run_SvB: bool = True):

    lepTop_1 = (events.b_cands[:,0] + events.Wlnu_cand)
    hadTop_1 = (events.b_cands[:,1] + events.Wqq_cand)
    tt_1 = lepTop_1 + hadTop_1
    tt_1["lepTop"] = lepTop_1
    tt_1["lepTop", "dr"]   = events.b_cands[:,0].delta_r  (events.Wlnu_cand)
    tt_1["lepTop", "dphi"] = events.b_cands[:,0].delta_phi(events.Wlnu_cand)

    tt_1["hadTop"] = hadTop_1
    tt_1["hadTop", "dr"]   = events.b_cands[:,1].delta_r  (events.Wqq_cand)
    tt_1["hadTop", "dphi"] = events.b_cands[:,1].delta_phi(events.Wqq_cand)

    tt_1["mass_distance"] = distance(lepTop_1.mass,  hadTop_1.mass,  172.5, 172.5)

    lepTop_2 = (events.b_cands[:,1] + events.Wlnu_cand)
    hadTop_2 = (events.b_cands[:,0] + events.Wqq_cand)
    tt_2 = lepTop_2 + hadTop_2
    tt_2["lepTop"] = lepTop_2
    tt_2["lepTop", "dr"]   = events.b_cands[:,1].delta_r  (events.Wlnu_cand)
    tt_2["lepTop", "dphi"] = events.b_cands[:,1].delta_phi(events.Wlnu_cand)

    tt_2["hadTop"] = hadTop_2
    tt_2["hadTop", "dr"]   = events.b_cands[:,0].delta_r  (events.Wqq_cand)
    tt_2["hadTop", "dphi"] = events.b_cands[:,0].delta_phi(events.Wqq_cand)

    tt_2["mass_distance"] = distance(lepTop_2.mass,  hadTop_2.mass,  172.5, 172.5)

    events['tt_cands'] = ak.zip({"b1Whad": tt_2,
                                 "b2Whad": tt_1,
                                 }) # save the two ttbar candidates (order correctly matches classifier)
    
    if run_SvB:
        try:
            #### select candidate based on ML classifier score
            events["tt_cands", "b1Whad", "cls_score"] = events.SvB.tt_b1Whad  # corresponds to tt_2 in ML classifier
            events["tt_cands", "b2Whad", "cls_score"] = events.SvB.tt_b2Whad  # corresponds to tt_1 in ML classifier
            tt_best = ak.where(events.SvB.tt_b1Whad > events.SvB.tt_b2Whad, tt_2, tt_1)
        except:
            tt_best = tt_1
            print(f"classifier scores not available for {events.metadata['dataset']}, selecting a default value")
    else:
        # select ttbar candidates based on mass
        b_sel_nom =  tt_1.mass_distance < tt_2.mass_distance #pick pair closest to ttbar mass
        tt_best  = ak.where(b_sel_nom,  tt_1 ,  tt_2)

    tt_sel = ak.zip({"p": tt_best.lepTop + tt_best.hadTop,
                     "lepTop": tt_best.lepTop,
                     "hadTop": tt_best.hadTop,
                     })

    tt_sel["p","dr"]   = tt_best.lepTop.delta_r(tt_best.hadTop)
    tt_sel["p","dphi"] = tt_best.lepTop.delta_phi(tt_best.hadTop)

    events['tt_sel'] = tt_sel
    return events

def Wqq_soft_candidate_selection(events, year):
    QvG_key = 'btagPNetQvG' if '202' in year else 'particleNetAK4_QvsG' # use particleNET for quark vs. gluon tagging

    #q_cands_soft = events.q_cands_soft_init[ak.argsort(getattr(events.q_cands_soft_init,QvG_key), axis=1, ascending=False)] #particleNetAK4_QvsG btagPNetQvG
    #q_cands_soft = q_cands_soft[:,:4] #top 4 quark vs gluon non b-jets
    #q_cands_soft = q_cands_soft[ak.argsort(q_cands_soft.pt, axis=1, ascending=False)] #pt sort the jets
    q_cands_soft = events.q_cands_soft_init[ak.argsort(events.q_cands_soft_init.pt, axis=1, ascending=False)]
    q_cands_soft = q_cands_soft[:, :4]
    events['q_cands_soft'] = q_cands_soft

    ## pt sorting soft + nominal candidates
    q_cands_pt_sorted = events.q_cands_soft_init[ak.argsort(events.q_cands_soft_init.pt, axis=1, ascending=False)]
    events['q_cands_pt_sorted'] = ak.pad_none(q_cands_pt_sorted[:,:2], 2, axis=1)
    ####
    
    jj_i = ak.argcombinations(q_cands_soft, 2, replacement = False, fields=["j1","j2"]) #take dijet combinations
    #jj_i = jj_i[(q_cands_soft[jj_i.j1] - q_cands_soft[jj_i.j2]).eta<2.0]
    #jj_i = jj_i[(q_cands_soft[jj_i.j1] + q_cands_soft[jj_i.j2]).mass<120.0] #dijet cuts
    events['dijet_combs_new'] = jj_i

    events['j_lead_new'] =  q_cands_soft[jj_i.j1] # leading jet
    events['j_sublead_new'] =  q_cands_soft[jj_i.j2] # subleading jet

    events['qq_mass'] = ak.fill_none((q_cands_soft[jj_i.j1] + q_cands_soft[jj_i.j2]).mass,np.nan) # plotting gives issues with None values
    events['qq_soft'] = ak.pad_none(q_cands_soft[jj_i.j1] + q_cands_soft[jj_i.j2], 3, axis=1)
    return events


def Hww_soft_candidate_selection(events):
    Hww_cand_soft = events.Wlnu_cand + events.qq_soft
    Hww_cand_soft["dr"]   = events.Wlnu_cand.delta_r  (events.qq_soft)
    Hww_cand_soft["dphi"] = events.Wlnu_cand.delta_phi(events.qq_soft)

    events['Hww_cand_soft'] = Hww_cand_soft
    return events


def ttbar_soft_candidate_selection(events):

    lepTop_soft_1 = (events.Wlnu_cand + events.b_cands[:,1])
    hadTop_soft_1 = (events.b_cands[:,0] + events.qq_soft) #hadronic candidate 1

    tt_soft_1 = lepTop_soft_1 + hadTop_soft_1
    tt_soft_1["lepTop"] = lepTop_soft_1
    tt_soft_1["lepTop", "dr"]   = events.b_cands[:,1].delta_r  (events.Wlnu_cand)
    tt_soft_1["lepTop", "dphi"] = events.b_cands[:,1].delta_phi(events.Wlnu_cand)

    tt_soft_1["hadTop"] = hadTop_soft_1
    tt_soft_1["hadTop", "dr"]   = events.b_cands[:,0].delta_r  (events.qq_soft)
    tt_soft_1["hadTop", "dphi"] = events.b_cands[:,0].delta_phi(events.qq_soft)

    tt_soft_1["mass_distance"] = distance(lepTop_soft_1.mass,  hadTop_soft_1.mass,  172.5, 172.5)

    lepTop_soft_2 = (events.Wlnu_cand + events.b_cands[:,0])
    hadTop_soft_2 = (events.b_cands[:,1] + events.qq_soft) #hadronic candidate 2

    tt_soft_2 = lepTop_soft_2 + hadTop_soft_2
    tt_soft_2["lepTop"] = lepTop_soft_2
    tt_soft_2["lepTop", "dr"]   = events.b_cands[:,0].delta_r  (events.Wlnu_cand)
    tt_soft_2["lepTop", "dphi"] = events.b_cands[:,0].delta_phi(events.Wlnu_cand)

    tt_soft_2["hadTop"] = hadTop_soft_2
    tt_soft_2["hadTop", "dr"]   = events.b_cands[:,1].delta_r  (events.qq_soft)
    tt_soft_2["hadTop", "dphi"] = events.b_cands[:,1].delta_phi(events.qq_soft)

    tt_soft_2["mass_distance"] = distance(lepTop_soft_2.mass,  hadTop_soft_2.mass,  172.5, 172.5)

    b_sel_soft =  tt_soft_1.mass_distance < tt_soft_2.mass_distance

    #final ttbar candidates
    tt_best_soft = ak.where(b_sel_soft, tt_soft_1 , tt_soft_2)

    tt_soft = ak.zip({"p": tt_best_soft.lepTop + tt_best_soft.hadTop,
                      "lepTop": tt_best_soft.lepTop,
                      "hadTop": tt_best_soft.hadTop,
                      })

    tt_soft["p","dr"]   = tt_best_soft.lepTop.delta_r(tt_best_soft.hadTop)
    tt_soft["p","dphi"] = tt_best_soft.lepTop.delta_r(tt_best_soft.hadTop)

    events['tt_soft'] = tt_soft

    return events

def regressed_nu(events, met_regression: bool = False):
    if met_regression:
        is_3jet = events.incl_3j2b
        def _pick(field):
            return ak.where(is_3jet, events.met_regressor_3jet[field], events.met_regressor[field])
        nu_px = _pick("nu_px")
        nu_py = _pick("nu_py")
        nu_pz = _pick("nu_pz")
        events["reg_nu"] = ak.zip({
            "x": nu_px,
            "y": nu_py,
            "z": nu_pz,
            "t": np.sqrt((nu_px**2 + nu_py**2 + nu_pz**2)),
            "charge": ak.zeros_like(pt, dtype=int),
        },
        with_name="PtEtaPhiMCandidate",
        behavior=vector.behavior,
        )
        events["reg_mW"] =  ak.fill_none((events.reg_nu + events.leading_lep).mass, np.nan) # regressed leptonic W mass

        #check how well regressor is selecting jets
        ml_jet_scores_full = ak.concatenate(
            [ak.singletons(0.5 * (_pick("jet_weight_0") + _pick("jet_weight_4"))),  # jet 0 (avg of two attention heads)
             ak.singletons(0.5 * (_pick("jet_weight_1") + _pick("jet_weight_5"))),  # jet 1
             ak.singletons(0.5 * (_pick("jet_weight_2") + _pick("jet_weight_6"))),  # jet 2
             ak.singletons(0.5 * (_pick("jet_weight_3") + _pick("jet_weight_7")))], # jet 3
            axis=1)

        # ak.local_index to build a per-entry boolean mask and filter axis=1.
        n_jets = ak.num(events.q_cands_soft)
        events["q_cands_soft", "ml_jet_scores"] = ml_jet_scores_full[ak.local_index(ml_jet_scores_full, axis=1) < n_jets]

        has_two_jets = ak.num(events.q_cands_soft) >= 2
        valid_nu = ~np.isnan(nu_pz)
        mask_all = has_two_jets & valid_nu
        
        # Sort jets by attention weight descending, keep only indices pointing to real jets
        masked_scores = ak.mask(events.q_cands_soft.ml_jet_scores, mask_all)
        sorted_indices = ak.argsort(masked_scores, ascending=False)
        sorted_indices = sorted_indices[sorted_indices < n_jets]
        
        # Top 2 jets by attention weight
        events['sel_qq_l']  = events.q_cands_soft[sorted_indices[:, 0:1]]
        events['sel_qq_sl'] = events.q_cands_soft[sorted_indices[:, 1:2]]

        events['HWW_mass'] = ak.fill_none((events.sel_qq_l + events.sel_qq_sl + events.leading_lep + events.reg_nu).mass, np.nan)

        # mlvq mass for 3jet events: single non-b jet + lepton + regressed neutrino
        has_one_jet = ak.num(events.q_cands_soft) >= 1
        mlvq_mask = has_one_jet & valid_nu
        masked_scores_1j = ak.mask(events.q_cands_soft.ml_jet_scores, mlvq_mask)
        sorted_indices_1j = ak.argsort(masked_scores_1j, ascending=False)
        sorted_indices_1j = sorted_indices_1j[sorted_indices_1j < n_jets]
        events['sel_q_mlvq'] = events.q_cands_soft[sorted_indices_1j[:, 0:1]]
        events['mlvq_mass'] = ak.fill_none((events.sel_q_mlvq + events.leading_lep + events.reg_nu).mass, np.nan)

    return events

def candidate_selection(events, params, year, run_SvB, run_MET_regression, classifier_SvB = None):

    #
    # Common
    #
    events = Hbb_candidate_selection(events)
    events = Wlnu_candidate_selection(events)

    # compute ML classifier output from within the processor (currently directly evaluating in the classifier framework)
    if classifier_SvB is not None:
        from bbreww.analysis.helpers.classifier.SvB_helpers import compute_SvB
        compute_SvB(events,
            mask = events.nominal_4j2b, # apply nominal analysis mask
            SvB=classifier_SvB,
            doCheck=False)
    #
    #  Nomninal Candidate selection
    #
    events = Wqq_candidate_selection(events)
    events = Hww_candidate_selection(events)
    events = ttbar_candidate_selection(events, run_SvB)
    #
    # soft jets analysis
    #
    events = Wqq_soft_candidate_selection(events, year)
    events = Hww_soft_candidate_selection(events)
    events = ttbar_soft_candidate_selection(events)
    events = regressed_nu(events, run_MET_regression)

    # Define the SR and CR based on H ->> bb candidate mass and HWW_mass using regressed neutrino
    if run_MET_regression:
       signal_region_4jet = elliptical_region(events.Hbb_cand.mass, events.HWW_mass,
                                         115, 135, 100, 100 ) # elliptical signal 4jet region
       signal_region_3jet = elliptical_region(events.Hbb_cand.mass, events.mlvq_mass,
                                         120, 117, 62, 62 ) # elliptical signal region 3jet region
    else:
        signal_region_4jet = ak.singletons(ak.ones_like(events.event, dtype = bool))
        signal_region_3jet = ak.singletons(ak.ones_like(events.event, dtype = bool))

    sr_flat = ak.fill_none(ak.firsts(signal_region_4jet), False)
    sr_flat_3jet = ak.fill_none(ak.firsts(signal_region_3jet), False)

    sr_flat = ak.where(events.incl_3j2b, sr_flat_3jet, sr_flat)
    cr_flat = ak.where(events.incl_3j2b, ~sr_flat_3jet, ~sr_flat)

    events['region'] = ak.zip({
        'SR': sr_flat,
        'CR': cr_flat
    })
    
    return events


## function only for skimmer
def bjet_flag(events,params,year):
    j_clean = events.Jet[events.Jet.isclean]
    j_soft = j_clean[j_clean.preselected]
    events['j_init'] = j_soft # initial preselected jets

    # QvG_key = 'btagPNetQvG' if '202' in year else 'particleNetAK4_QvsG' # use particleNET for quark vs. gluon tagging
    bTag_key = 'btagPNetB' if '202' in year else 'particleNetAK4_B' # use particleNET b-tagging
    btag_threshold = params[year].btagWP.L # using loose working point

    j_candidates = j_soft[ak.argsort(j_soft.pt, axis=1, ascending=False)]
    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,bTag_key), axis=1, ascending=False)]#particleNetAK4_B btagPNetB
    j_bcand_pool = j_candidates[j_candidates.pt > 25.0]  # Only jets > 25 GeV for b-jets
    j_bcand_pool = j_bcand_pool[getattr(j_bcand_pool,bTag_key) > btag_threshold]

    events['has_2_bjets'] = ak.num(j_bcand_pool, axis=1) >= 2

    return events
    
