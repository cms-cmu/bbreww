#! /usr/bin/env python
import correctionlib
import awkward as ak
import numpy as np
import json

path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/"

####
# Electron ID scale factor
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaSFJSON
# jsonPOG: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM
# /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration
####

def get_ele_loose_id_sf (year, eta, pt):
    '''
    We need to modify this year.split('_')[0].
    '''
    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/EGM/{year}/electron.json.gz')

    flateta, counts = ak.flatten(eta), ak.num(eta)
    
    pt = ak.where((pt<10.), ak.full_like(pt,10.), pt)
    flatpt = ak.flatten(pt)
    
    weight = evaluator["UL-Electron-ID-SF"].evaluate(year.split('_')[0], "sf", "Loose", flateta, flatpt)

    return ak.unflatten(weight, counts=counts)

def get_ele_tight_id_sf (year, eta, pt):
    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/EGM/{year}/electron.json.gz')

    flateta, counts = ak.flatten(eta), ak.num(eta)
    
    pt = ak.where((pt<10.), ak.full_like(pt,10.), pt)
    flatpt = ak.flatten(pt)
    
    weight = evaluator["UL-Electron-ID-SF"].evaluate(year.split('_')[0], "sf", "Tight", flateta, flatpt)
    
    return ak.unflatten(weight, counts=counts)

####
# Electron Reco scale factor
# root files: https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018
# Code Copy from previous correctionsUL.py file
####

def get_ele_reco_sf_below20(year, eta, pt):
    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/EGM/{year}/electron.json.gz')
    
    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    eta = ak.where((eta<2.399), ak.full_like(eta,-2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)
    
    pt = ak.where((pt<10.), ak.full_like(pt,10.), pt)
    pt = ak.where((pt>19.99), ak.full_like(pt,19.99), pt)
    flatpt = ak.flatten(pt)
    
    weight = evaluator["UL-Electron-ID-SF"].evaluate(year.split('_')[0], "sf", "RecoBelow20", flateta, flatpt)

    return ak.unflatten(weight, counts=counts)


def get_ele_reco_sf_above20(year, eta, pt):
    
    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/EGM/{year}/electron.json.gz')
    
    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    eta = ak.where((eta<2.399), ak.full_like(eta,-2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)
    
    pt = ak.where((pt<20.), ak.full_like(pt,20.), pt)
    pt = ak.where((pt>499.99), ak.full_like(pt,499.99), pt)
    flatpt = ak.flatten(pt)

    weight = evaluator["UL-Electron-ID-SF"].evaluate(year.split('_')[0], "sf", "RecoAbove20", flateta, flatpt)
    return ak.unflatten(weight, counts=counts)


####
# Photon ID scale factor
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaSFJSON
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM
# /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration
####


# https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/tree/master/Run2/UL

####
# Muon ID scale factor
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018n?topic=MuonUL2018
# jsonPOG: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/MUO
# /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration
####

def get_mu_loose_id_sf (year, eta, pt):
    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/MUO/{year}/muon_Z.json.gz')

    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)

    pt  = ak.where((pt<15.),ak.full_like(pt,15.),pt)
    flatpt = ak.flatten(pt)
    
    if year == '2018':
        weight = evaluator["NUM_LooseID_DEN_TrackerMuons"].evaluate(flateta, flatpt, "nominal")
    else:
        weight = evaluator["NUM_LooseID_DEN_genTracks"].evaluate(flateta, flatpt, "nominal")

    return ak.unflatten(weight, counts=counts)

def get_mu_tight_id_sf (year, eta, pt):
    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/MUO/{year}/muon_Z.json.gz')
    
    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)

    pt  = ak.where((pt<15.),ak.full_like(pt,15.),pt)
    flatpt = ak.flatten(pt)
    
    if year == '2018':
        weight = evaluator["NUM_TightID_DEN_TrackerMuons"].evaluate(flateta, flatpt, "nominal")
    else:
        weight = evaluator["NUM_TightID_DEN_genTracks"].evaluate(flateta, flatpt, "nominal")

    return ak.unflatten(weight, counts=counts)




####
# Muon Iso scale factor
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018n?topic=MuonUL2018
# jsonPOG: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/MUO
# /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration
####

def get_mu_loose_iso_sf (year, eta, pt):
    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/MUO/{year}/muon_Z.json.gz')

    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)

    pt  = ak.where((pt<15.),ak.full_like(pt,15.),pt)
    flatpt = ak.flatten(pt)
    
    weight = evaluator["NUM_LooseRelIso_DEN_LooseID"].evaluate(flateta, flatpt, "nominal")

    return ak.unflatten(weight, counts=counts)

def get_mu_tight_iso_sf (year, eta, pt):
    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/MUO/{year}/muon_Z.json.gz')

    eta = ak.where((eta>2.399), ak.full_like(eta,2.399), eta)
    flateta, counts = ak.flatten(eta), ak.num(eta)

    pt  = ak.where((pt<15.),ak.full_like(pt,15.),pt)
    flatpt = ak.flatten(pt)

    weight = evaluator["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(flateta, flatpt, "nominal")

    return ak.unflatten(weight, counts=counts)

####
# XY MET Correction
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#xy_Shift_Correction_MET_phi_modu
####

# https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/
# correction_labels = ["metphicorr_pfmet_mc", "metphicorr_puppimet_mc", "metphicorr_pfmet_data", "metphicorr_puppimet_data"]

def get_met_xy_correction(year, npv, run, pt, phi, isData):
    
    npv = ak.where((npv>200),ak.full_like(npv,200),npv)
    pt  = ak.where((pt>1000.),ak.full_like(pt,1000.),pt)

    evaluator = correctionlib.CorrectionSet.from_file(f'{path}/JME/{year}/met.json.gz')

    if isData:
        corrected_pt = evaluator['pt_metphicorr_pfmet_data'].evaluate(pt,phi,npv,run)
        corrected_phi = evaluator['phi_metphicorr_pfmet_data'].evaluate(pt,phi,npv,run)

    if not isData:
        corrected_pt = evaluator['pt_metphicorr_pfmet_mc'].evaluate(pt,phi,npv,run)
        corrected_phi = evaluator['phi_metphicorr_pfmet_mc'].evaluate(pt,phi,npv,run)

    return corrected_pt, corrected_phi
