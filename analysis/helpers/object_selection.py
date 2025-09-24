import numpy as np
import awkward as ak
from src.physics.objects.jet_corrections import apply_jet_veto_maps
from bbreww.analysis.helpers.ids import lepton_preselection, jet_preselection, tau_preselection, ak8_jet_preselection
from bbreww.analysis.helpers.corrections import get_met_xy_correction

## this file contains object preselection for MET, electrons, muons, taus, photons, and jets


def muon_selection(events,params):
    events['Muon','isloose'] = lepton_preselection(events, "Muon", params, "loose")
    events['Muon','istight'] = lepton_preselection(events, "Muon", params, "tight")

    events['mu_nloose'] = ak.num(events.Muon[events.Muon.isloose], axis=1)
    events['mu_ntight'] = ak.num(events.Muon[events.Muon.istight], axis=1)

    events['sel_muon'] = events.Electron[events.Electron.istight]

    return events

def electron_selection(events, params):
    events['Electron', 'isloose'] = lepton_preselection(events, "Electron", params, "loose")
    events['Electron', 'istight'] = lepton_preselection(events, "Electron", params, "tight")

    events['e_nloose'] = ak.num(events.Electron[events.Electron.isloose], axis=1)
    events['e_ntight'] = ak.num(events.Electron[events.Electron.istight], axis=1)

    events['sel_elec'] = events.Electron[events.Electron.istight]

    return events

def tau_selection(events,params):
    events['Tau','ismedium']= tau_preselection(events, params, "medium")
    events['tau_nmedium']=ak.num(events.Tau[events.Tau.ismedium], axis=1)

    return events

def jet_selection(events, params, year):
    # jet veto maps for detector issues
    if '202' in year:
        events['Jet', 'jet_veto_maps'] = apply_jet_veto_maps(params[year]['jet_veto_maps'], events.Jet, event_veto = True)
        # events['Jet'] = events.Jet[events.Jet.jet_veto_maps] uncomment to apply on individual jets

    events['Jet','isclean'] = (
        ak.all(events.Jet.metric_table(events.Muon[events.Muon.istight]) > 0.4, axis=2)
        & ak.all(events.Jet.metric_table(events.Electron[events.Electron.istight]) > 0.4, axis=2)
    )
    events['Jet', 'isnominal'], events['Jet', 'issoft'],  events['Jet', 'preselected'] = jet_preselection(events, params)

    j_clean = events.Jet[events.Jet.isclean]
    j_init = j_clean[j_clean.preselected] # initial preselected jets

    # pt sort to take higher pT when b-tag scores are tied
    j_candidates = j_init[ak.argsort(j_init.pt, axis=1, ascending=False)]


    j_soft = j_clean[j_clean.issoft]
    events['j_nsoft']= ak.num(j_soft, axis=1)
    events['njets'] = ak.fill_none(ak.num(j_clean[j_clean.isnominal],axis=1),np.nan)
    events['has_3_presel_jets'] = (ak.num(j_init[j_init.preselected],axis=1)>2)
    events['has_exactly_3_presel_jets'] = (ak.num(j_init[j_init.preselected],axis=1)==3)
    events['has_4_presel_jets'] = (ak.num(j_init[j_init.preselected],axis=1)>3)

    #
    #  b-jet selection
    #
    bTag_key = 'btagPNetB' if '202' in year else 'particleNetAK4_B' # use particleNET b-tagging
    btag_threshold = params[year].btagWP.M # using medium working point

    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,bTag_key), axis=1, ascending=False)]#particleNetAK4_B btagPNetB
    j_candidates["btagScore"] = getattr(j_candidates,bTag_key)

    j_candidates_nom = j_candidates[j_candidates.isnominal]
    events['nom_njets4'] = (ak.num(j_candidates_nom, axis=1) > 3)
    events['nom_njets3'] = (ak.num(j_candidates_nom, axis=1) == 3)

    j_btagged = j_candidates_nom[getattr(j_candidates_nom,bTag_key) > btag_threshold]
    j_btagged = j_btagged[ak.argsort(j_btagged.pt, axis=1, ascending=False)] #particleNetAK4_B btagPNetB
    events['b_cands'] = j_btagged[:,:2]

    events['has_2_bjets'] = ak.num(j_btagged, axis=1) >= 2
    events['has_1_bjet']  = ak.num(j_btagged, axis=1) >= 1 #add for cutflow plot


    #
    # nominal non-bjet selection
    #
    q_cands_nom = ak.mask(j_candidates_nom[:,2:], ak.num(j_candidates_nom[:,2:],axis=1)>=2) # require 2 or more q-jet candidates
    q_cands_nom = q_cands_nom[ak.argsort(q_cands_nom.pt, axis=1, ascending=False)] # pT sort the jets
    events["q_cands_nom"] = q_cands_nom[:,:2]

    #
    # Soft Jet Selection
    #
    j_candidates_soft = j_candidates[j_candidates.issoft]

    events['q_cands_soft'] = ak.concatenate([q_cands_nom, j_candidates_soft], axis=1)


    return events




def ak8_jet_selection(events,params):
    #### AK-8 jets selection
    is_clean_ak8 = (
        ak.all(events.FatJet.metric_table(events.Muon[events.Muon.istight]) > 0.8, axis=2)
        & ak.all(events.FatJet.metric_table(events.Electron[events.Electron.istight]) > 0.8, axis=2)
    )
    ak8_selected = ak8_jet_preselection(events.FatJet[is_clean_ak8], params)
    events['n_ak8_jets'] = ak.num(ak8_selected,  axis=1)
    return events

def apply_mll_cut(events):
    # electrons
    loose_e = events.Electron[events.Electron.isloose]
    loose_e = ak.mask(loose_e, events.e_nloose > 1) # only keep events with two leptons of same flavour
    e_pairs = ak.argcombinations(loose_e, 2, replacement = False, fields=["e1","e2"])
    e_pairs_mass = (loose_e[e_pairs.e1] + loose_e[e_pairs.e2]).mass

    is_same_charge_e = (loose_e[e_pairs.e1].charge == loose_e[e_pairs.e2].charge) # pairs with same charge
    passes_mass_cut_e = (abs(e_pairs_mass - 91.19) > 10) # & (e_pairs_mass > 12.0)
    is_good_pair_e = is_same_charge_e | passes_mass_cut_e # either same charge electrons or pass m_ll cuts
    pass_cut_e = ak.fill_none(ak.all(is_good_pair_e,axis=1), True) # pass cut for None values

    # muons
    loose_mu = events.Muon[events.Muon.isloose]
    loose_mu = ak.mask(loose_mu, events.mu_nloose > 1) # only keep events with two leptons of same flavour
    mu_pairs = ak.argcombinations(loose_mu, 2, replacement = False, fields=["mu1","mu2"])
    mu_pairs_mass = (loose_mu[mu_pairs.mu1] + loose_mu[mu_pairs.mu2]).mass

    is_same_charge_mu = (loose_mu[mu_pairs.mu1].charge == loose_mu[mu_pairs.mu2].charge)  # pairs with same charge
    passes_mass_cut_mu = (abs(mu_pairs_mass - 91.19) > 10)# &  mu_pairs_mass > 12.0)
    is_good_pair_mu = is_same_charge_mu | passes_mass_cut_mu # either same charge muons or pass m_ll cuts
    pass_cut_mu = ak.fill_none(ak.all(is_good_pair_mu,axis=1), True) # pass cut for None values

    events['pass_mll_cut'] = pass_cut_e & pass_cut_mu # m_Z window cut on opposite charged leptons

    return events

def apply_bbWW_preselection(events, year,params, isMC):
    events = muon_selection(events, params) #muons
    events = electron_selection(events, params) #electrons
    events = tau_selection(events,params)
    events = jet_selection(events,params, year)
    events = ak8_jet_selection(events, params)

    # require exactly one tight electron(muon) with no loose muon(electron)
    events['e_region'] = (events.e_ntight==1) & (events.mu_ntight==0)
    events['mu_region'] = (events.mu_ntight==1) & (events.e_ntight==0)
    return events
