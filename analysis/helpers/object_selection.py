import numpy as np 
import awkward as ak
from bbww.analysis.helpers.ids import lepton_preselection, tau_preselection, photon_preselection, jet_preselection, HEMjet_preselection    
from bbww.analysis.helpers.corrections import(
    get_met_xy_correction,
    get_ele_loose_id_sf,
    get_mu_loose_id_sf,
    get_mu_loose_iso_sf,
    get_mu_tight_id_sf,
    get_mu_tight_iso_sf,
    get_ele_reco_sf_above20,
    get_ele_reco_sf_below20,
    get_ele_tight_id_sf
)

## this file contains object preselection for MET, electrons, muons, taus, photons, and jets

def met_selection(events, year, isData):
    npv = events.PV.npvsGood 
    run = events.run
    events['met'] = events.MET # keep corrected met and uncorrected separate
    events['met','pt'] , events['met','phi'] = get_met_xy_correction(year, npv, run, events.MET.pt, events.MET.phi, isData)
    return events


def muon_selection(events,year,params):       
    mu = events.Muon   
    events['Muon','isloose'] = lepton_preselection(events, "Muon", params, "loose")
    events['Muon','id_sf'] = ak.where(
        events.Muon.isloose, 
        get_mu_loose_id_sf(year, abs(mu.eta), mu.pt), 
        ak.ones_like(mu.pt)
    )
    events['Muon','iso_sf'] = ak.where(
        events.Muon.isloose, 
        get_mu_loose_iso_sf(year, abs(mu.eta), mu.pt), 
        ak.ones_like(mu.pt)
    )
    events['Muon','istight'] = lepton_preselection(events, "Muon", params, "tight")
    events['Muon','id_sf'] = ak.where(
        events.Muon.istight, 
        get_mu_tight_id_sf(year, abs(mu.eta), mu.pt), 
        events.Muon.id_sf
    )
    events['Muon','iso_sf'] = ak.where(
        events.Muon.istight, 
        get_mu_tight_iso_sf(year, abs(mu.eta), mu.pt), 
        events.Muon.iso_sf
    )
    events['mu_nloose'] = ak.num(events.Muon[events.Muon.isloose], axis=1)
    events['mu_ntight'] = ak.num(events.Muon[events.Muon.istight], axis=1)

    return events

def electron_selection(events,year, params):       
    e = events.Electron
    events['Electron', 'reco_sf'] = ak.where(
        (e.pt<20),
        get_ele_reco_sf_below20(year, e.eta+e.deltaEtaSC, e.pt), 
        get_ele_reco_sf_above20(year, e.eta+e.deltaEtaSC, e.pt)
    )
    events['Electron', 'isloose'] = lepton_preselection(events, "Electron", params, "loose")
    events['Electron', 'id_sf'] = ak.where(
        events.Electron.isloose,
        get_ele_loose_id_sf(year, e.eta+e.deltaEtaSC, e.pt),
        ak.ones_like(e.pt)
    )

    events['Electron', 'istight'] = lepton_preselection(events, "Electron", params, "tight")
    events['Electron', 'id_sf'] = ak.where(
        events.Electron.istight,
        get_ele_tight_id_sf(year, e.eta+e.deltaEtaSC, e.pt),
        events.Electron.id_sf
    )
    events['Electron','isclean'] = ak.all(e.metric_table(events.Muon[events.Muon.isloose]) > 0.3, axis=2)

    e_clean = events.Electron[events.Electron.isclean]
    events['e_nloose'] = ak.num(e_clean[e_clean.isloose], axis=1) #use clean electons for loose/tight selection
    events['e_ntight'] = ak.num(e_clean[e_clean.istight], axis=1)
    
    return events

def tau_selection(events,params): 
    e_clean = events.Electron[events.Electron.isclean]

    events['Tau','isclean']=(
        ak.all(events.Tau.metric_table(events.Muon[events.Muon.isloose]) > 0.4, axis=2) 
        & ak.all(events.Tau.metric_table(e_clean[e_clean.isloose]) > 0.4, axis=2)
    )
    events['Tau','isloose']= tau_preselection(events, params, "loose")

    tau_clean=events.Tau[events.Tau.isclean]
    tau_loose=tau_clean[tau_clean.isloose]
    events['tau_nloose']=ak.num(tau_loose, axis=1)

    return events

def photon_selection(events, params):
    e_clean = events.Electron[events.Electron.isclean]
    tau_clean = events.Tau[events.Tau.isclean]

    events['Photon','isclean']=(
        ak.all(events.Photon.metric_table(events.Muon[events.Muon.isloose]) > 0.5, axis=2)
        & ak.all(events.Photon.metric_table(e_clean[e_clean.isloose]) > 0.5, axis=2)
        & ak.all(events.Photon.metric_table(tau_clean[tau_clean.isloose]) > 0.5, axis=2)
    )
    events['Photon','isloose']= photon_preselection(events, params, "loose")

    pho_clean=events.Photon[events.Photon.isclean]
    pho_loose= pho_clean[pho_clean.isloose]
    events['pho_nloose']=ak.num(pho_loose, axis=1)

    return events

def jet_selection(events, params, year):
    e_clean = events.Electron[events.Electron.isclean]
    tau_clean = events.Tau[events.Tau.isclean]
    pho_clean = events.Photon[events.Photon.isclean]

    events['Jet','isclean'] = (
        ak.all(events.Jet.metric_table(events.Muon[events.Muon.isloose]) > 0.4, axis=2)
        & ak.all(events.Jet.metric_table(e_clean[e_clean.isloose]) > 0.4, axis=2)
        & ak.all(events.Jet.metric_table(tau_clean[tau_clean.isloose]) > 0.4, axis=2)
        & ak.all(events.Jet.metric_table(pho_clean[pho_clean.isloose]) > 0.4, axis=2)
    )
    events['Jet','issoft'] = jet_preselection(events, params, year)
    events['Jet','isHEM'] = HEMjet_preselection(events)
    j_clean = events.Jet[events.Jet.isclean]
    j_soft = j_clean[j_clean.issoft]
    j_HEM = events.Jet[events.Jet.isHEM]
    events['j_nsoft']= ak.num(j_soft, axis=1)
    events['j_nHEM'] = ak.num(j_HEM, axis=1)

    return events




