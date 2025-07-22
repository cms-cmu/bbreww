#! /usr/bin/env python
import correctionlib
import awkward as ak
import numpy as np
import json
from omegaconf import OmegaConf

path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/"
corrections_metadata = "analysis/metadata/corrections.yml"
corrections = OmegaConf.load(corrections_metadata)

####
# Electron ID scale factor
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaSFJSON
# jsonPOG: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM
# /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration
####

# change year to match naming schemes used by object POGs
year_keys = {
    '2022_preEE' :  '2022Re-recoBCD',
    '2022_EE':      '2022Re-recoE+PromptFG',
    '2023_preBPix': '2023PromptC',
    '2023_BPix':    '2023PromptD'
}

# retrieve electron scale factors for id = Loose, Tight, RecoBelow20, Reco20to75, etc
def get_ele_id_sf (year, eta, pt, id):
    sf_file = corrections[year]['ele_sf']
    year = year_keys[year]
    evaluator = correctionlib.CorrectionSet.from_file(sf_file)

    if 'Reco' in id:
        eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
        eta = ak.where((eta<2.399), ak.full_like(eta,-2.399), eta)
        flateta, counts = ak.flatten(eta), ak.num(eta)
        
        pt = ak.where((pt<10.), ak.full_like(pt,10.), pt)
        pt = ak.where((pt>19.99), ak.full_like(pt,19.99), pt)
        flatpt = ak.flatten(pt)
    else:
        flateta, counts = ak.flatten(eta), ak.num(eta)
        pt = ak.where((pt<10.), ak.full_like(pt,10.), pt)
        flatpt = ak.flatten(pt)
    
    if '202' in year: 
        weight = evaluator["Electron-ID-SF"].evaluate(year, "sf", id, flateta, flatpt) # run3
    else: 
        weight = evaluator["UL-Electron-ID-SF"].evaluate(year.split('_')[0], "sf", id, flateta, flatpt) #run2

    return ak.unflatten(weight, counts=counts)


def get_mu_id_sf (year, eta, pt):
    sf_file = corrections[year]['mu_sf']
    year = year_keys[year]
    evaluator = correctionlib.CorrectionSet.from_file(sf_file)

    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)

    pt  = ak.where((pt<15.),ak.full_like(pt,15.),pt)
    flatpt = ak.flatten(pt)

    if (year == '2018') or ('202' in year):
        weight = evaluator[f'NUM_{id}ID_DEN_TrackerMuons'].evaluate(flateta, flatpt, "nominal")
    else:
        weight = evaluator[f'NUM_{id}ID_DEN_genTracks'].evaluate(flateta, flatpt, "nominal")

    return ak.unflatten(weight, counts=counts)

def get_mu_iso_sf (year, eta, pt):
    sf_file = corrections[year]['mu_sf']
    year = year_keys[year]
    evaluator = correctionlib.CorrectionSet.from_file(sf_file)

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
    year = year_keys[year]
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

def get_ttbar_weight(pt):
    return np.exp(0.0615 - 0.0005 * np.clip(pt, 0, 800))
