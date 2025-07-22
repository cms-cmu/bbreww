import numpy as np
import awkward as ak

######
## Electron
## Electron_cutBased Int_t cut-based ID Fall17 V2
## (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
## https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
######

def lepton_preselection(events, lepton_flavour, params, id):

    leptons = events[lepton_flavour]
    cuts = params.object_preselection[lepton_flavour][id]
    # Requirements on pT
    passes_pt = leptons.pt > cuts["pt"]

    if lepton_flavour == "Electron":
        # Requirements on SuperCluster eta, dxy, dz for barrel and endcap regions
        etaSC = abs(leptons.deltaEtaSC + leptons.eta)
        passes_SC = (
            # barrel cuts
            (etaSC < 1.4442) 
            & (abs(leptons.dxy) < 0.05) 
            & (abs(leptons.dz) < 0.1)
        ) | (
            # endcap cuts
            (etaSC < 2.5) & (etaSC > 1.5660)
            & (abs(leptons.dxy) < 0.1)
            & (abs(leptons.dz) < 0.2)
        )

        passes_iso = True
        if "iso" in cuts.keys():
            passes_iso = leptons.pfRelIso03_all < cuts["iso"]
        if id == "loose":
            passes_cutbased = leptons.cutBased >= cuts["cutBased"]
        elif id == "tight":
            passes_cutbased = leptons.cutBased = cuts["cutBased"]

        good_leptons = passes_pt & passes_SC & passes_cutbased

    elif lepton_flavour == "Muon":
        # Requirements on isolation and id
        passes_eta = abs(leptons.eta) < cuts["eta"]
        passes_iso = leptons.pfRelIso04_all < cuts["iso"]
        passes_id = leptons[cuts['id']] == True
        
        good_leptons = passes_pt & passes_eta & passes_iso & passes_id
    
    return good_leptons


def tau_preselection(events, params, id):

    taus = events["Tau"]
    cuts = params.object_preselection["Tau"][id]

    try:
        passes_decayModeDMs=taus.decayModeFindingNewDMs
    except:
        passes_decayModeDMs=~np.isnan(ak.ones_like(taus.pt))

    passes_pt = taus.pt > cuts["pt"]
    passes_eta = abs(taus.eta) < cuts["eta"]
    passes_dz = abs(taus.dz) < cuts["dz"]
    try:
        passes_deeptauid = (taus.idDeepTau2018v2p5VSjet & 5) == 5 #medium working point
    except:
        passes_deeptauid = (taus.idDeepTau2017v2p1VSjet & 4) == 4 # fall back to older collection if newer not available

    good_taus = passes_pt & passes_eta &  passes_dz & passes_deeptauid & passes_decayModeDMs

    return good_taus

######
## Photon
## https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedPhotonIdentificationRun2
## Photon_cutBased Int_t cut-based ID bitmap, Fall17V2,
## (0:fail, 1:loose, 2:medium, 3:tight)
## Note: Photon IDs are integers, not bit masks
######

def photon_preselection(events, params, id):

    photons = events["Photon"]
    cuts = params.object_preselection["Photon"][id]

    passes_pt = photons.pt > cuts["pt"]
    passes_eta = ~((abs(photons.eta) > 1.4442) & (abs(photons.eta) < 1.5660))& (abs(photons.eta) < 2.5)
    passes_cutbased = photons.cutBased >= cuts["cutBased"]

    good_photons = passes_pt & passes_eta & passes_cutbased & photons.electronVeto

    return good_photons

######
## Jet
## https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL
## Tight working point including lepton veto (TightLepVeto)
##
## For Jet ID flags, bit1 is Loose (always false in 2017 since it does not
## exist), bit2 is Tight, bit3 is TightLepVeto. The POG recommendation is to
## use Tight Jet ID as the standard Jet ID.
######
## PileupJetID
## https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL
## Using Loose Pileup ID
##
## Note: There is a bug in 2016 UL in which bit values for Loose and Tight Jet
## Pileup IDs are accidentally flipped relative to 2017 UL and 2018 UL.
##
## For 2016 UL,
## Jet_puId = (passtightID*4 + passmediumID*2 + passlooseID*1).
##
## For 2017 UL and 2018 UL,
## Jet_puId = (passlooseID*4 + passmediumID*2 + passtightID*1).
######


def jet_preselection(events, params, year):

    jets = events["Jet"]
    cuts = params.object_preselection["Jet"]

    # pileup ID is only needed for run 2 (CHS jets)
    if '201' in year:
        ## custom low pT pileup Id (only valid for pT < 30 GeV)
        def puId_cut_low_pt(jet_pt):
            puId = (0.85-0.7)*(jet_pt-30)/(30-8) + 0.85
            return puId
        
        puId_value = 4
        if '2016' in year:
            puId_value =1

        passes_puId  = ak.where(
            jets.pt > 30,
            (jets.pt >= 50) | ((jets.puId & puId_value) == puId_value),
            (jets.puIdDisc > puId_cut_low_pt(jets.pt)) # soft jets: pT < 30 GeV
            )
    else:
        passes_puId = True
    
    passes_pt = jets.pt > cuts["pt"]
    passes_eta = abs(jets.eta) < cuts["eta"]

    good_jets = passes_eta & passes_pt & passes_puId

    return good_jets

def HEMjet_preselection(events):
    jets = events["Jet"]
    passes_pt = jets.pt > 30
    passes_eta = (jets.eta > -3.0) & (jets.eta < -1.3) 
    passes_phi = (jets.phi > -1.57) & (jets.phi < -0.87)

    good_jets = passes_pt & passes_eta & passes_phi

    return good_jets