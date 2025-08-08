#! /usr/bin/env python
import correctionlib
import awkward as ak
import numpy as np
import json
from omegaconf import OmegaConf

path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/"

### weights, scale factors keys taken from https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/

### Electron and Muon Scale Factors

# retrieve electron scale factors for id = Loose, Tight, RecoBelow20, Reco20to75, etc
def get_ele_sf (params,year, eta, pt, id):
    evaluator = list(correctionlib.CorrectionSet.from_file(params[year].ele_sf.file).values())[0]
    year_label = params[year].ele_sf.tag

    if 'Reco' in id:
        eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
        eta = ak.where((eta<2.399), ak.full_like(eta,-2.399), eta)
        pt = ak.where((pt<10.), ak.full_like(pt,10.), pt)        
        
        if 'RecoBelow20' in id:
            pt = ak.where((pt>19.99), ak.full_like(pt,19.99), pt)
        elif 'Reco20to75'in id: 
            pt = ak.where((pt<20), ak.full_like(pt,20), pt)
            pt = ak.where((pt>74.99), ak.full_like(pt,74.99), pt)   
    else:
        pt = ak.where((pt<10.), ak.full_like(pt,10.), pt)
        
    flateta, counts = ak.flatten(eta), ak.num(eta)
    flatpt = ak.flatten(pt)

    if '202' in year: 
        # retrieve the year_label from the json file itself
        weight = evaluator.evaluate(year_label, "sf", id, flateta, flatpt) # run3
    else: 
        weight = evaluator.evaluate(year_label,"sf", id, flateta, flatpt) #run2

    return ak.unflatten(weight, counts=counts)


def get_mu_id_sf (params,year, eta, pt, id):
    evaluator = correctionlib.CorrectionSet.from_file(params[year].mu_sf)

    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)

    pt  = ak.where((pt<15.),ak.full_like(pt,15.),pt)
    flatpt = ak.flatten(pt)

    if (year == '2018') or ('202' in year):
        weight = evaluator[f'NUM_{id}ID_DEN_TrackerMuons'].evaluate(flateta, flatpt, "nominal")
    else:
        weight = evaluator[f'NUM_{id}ID_DEN_genTracks'].evaluate(flateta, flatpt, "nominal")

    return ak.unflatten(weight, counts=counts)

def get_mu_iso_sf (params,year, eta, pt, id):
    evaluator = correctionlib.CorrectionSet.from_file(params[year].mu_sf)

    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)

    pt  = ak.where((pt<15.),ak.full_like(pt,15.),pt)
    flatpt = ak.flatten(pt)

    if '202' in year:
        weight = evaluator[f'NUM_LoosePFIso_DEN_{id}ID'].evaluate(flateta, flatpt, "nominal") #run3
    else: 
        suffix = 'IDandIPCut' if id == 'Tight' else 'ID'
        weight = evaluator[f'NUM_{id}RelIso_DEN_{id}{suffix}'].evaluate(flateta, flatpt, "nominal") #run2

    return ak.unflatten(weight, counts=counts)

####
# XY MET Correction
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#xy_Shift_Correction_MET_phi_modu
####

# https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/
# correction_labels = ["metphicorr_pfmet_mc", "metphicorr_puppimet_mc", "metphicorr_pfmet_data", "metphicorr_puppimet_data"]

## only used for run2
def get_met_xy_correction(year, npv, run, pt, phi, isMC):
    npv = ak.where((npv>200),ak.full_like(npv,200),npv)
    pt  = ak.where((pt>1000.),ak.full_like(pt,1000.),pt)

    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/JME/{year}/met.json.gz')

    if not isMC:
        corrected_pt = evaluator['pt_metphicorr_pfmet_data'].evaluate(pt,phi,npv,run)
        corrected_phi = evaluator['phi_metphicorr_pfmet_data'].evaluate(pt,phi,npv,run)

    if isMC:
        corrected_pt = evaluator['pt_metphicorr_pfmet_mc'].evaluate(pt,phi,npv,run)
        corrected_phi = evaluator['phi_metphicorr_pfmet_mc'].evaluate(pt,phi,npv,run)

    return corrected_pt, corrected_phi

def add_ele_sf(events, year):
    e = events.Electron

    events['Electron', 'reco_sf'] = ak.where(
            (e.pt<20),
            get_ele_sf(year, e.eta+e.deltaEtaSC, e.pt, "RecoBelow20"), 
            get_ele_sf(year, e.eta+e.deltaEtaSC, e.pt, "Reco20to75")
        )
    events['Electron', 'id_sf'] = ak.where(
            e.isloose,
            get_ele_sf(year, e.eta+e.deltaEtaSC, e.pt, "Loose"),
            ak.ones_like(e.pt)
        )
    events['Electron','id_sf'] = ak.where(
        e.istight,
        get_ele_sf(year, e.eta+e.deltaEtaSC, e.pt, "Tight"),
        events.Electron.id_sf
        )

def add_mu_sf(events,year):
    mu = events.Muon

    events['Muon', 'id_sf'] = ak.where(
        mu.isloose, 
        get_mu_id_sf(year, abs(mu.eta), mu.pt, "Loose"), 
        ak.ones_like(mu.pt)
    )
    events['Muon', 'id_sf'] = ak.where(
        mu.istight, 
        get_mu_id_sf(year, abs(mu.eta), mu.pt, "Tight"), 
        events.Muon.id_sf
    )
    events['Muon', 'iso_sf'] = ak.where(
        mu.isloose, 
        get_mu_iso_sf(year, abs(mu.eta), mu.pt, "Loose"), 
        ak.ones_like(mu.pt)
        )
    events['Muon', 'iso_sf'] = ak.where(
        mu.istight, 
        get_mu_iso_sf(year, abs(mu.eta), mu.pt, "Tight"), 
        events.Muon.iso_sf
        )
#####################################################
    
def get_ttbar_weight(pt):
    return np.exp(0.0615 - 0.0005 * np.clip(pt, 0, 800))

### might move this to base_class
def apply_met_corrections_after_jec(events, jets):
    from coffea.jetmet_tools import CorrectedMETFactory
    jec_name_map = {
        'JetPt': 'pt',
        'JetMass': 'mass',
        'JetEta': 'eta',
        'JetA': 'area',
        'ptGenJet': 'pt_gen',
        'ptRaw': 'pt_raw',
        'massRaw': 'mass_raw',
        'Rho': 'event_rho',
        'METpt': 'pt',
        'METphi': 'phi',
        'JetPhi': 'phi',
        'UnClusteredEnergyDeltaX': 'MetUnclustEnUpDeltaX',
        'UnClusteredEnergyDeltaY': 'MetUnclustEnUpDeltaY',
    }

    met_factory = CorrectedMETFactory(jec_name_map)
    met_variations = met_factory.build(events.MET, jets, {})
    return met_variations