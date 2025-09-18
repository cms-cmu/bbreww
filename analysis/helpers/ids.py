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
    passes_pt = leptons.pt > cuts.pt
    passes_eta = abs(leptons.eta) < cuts.eta

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
            passes_iso = leptons.pfRelIso03_all < cuts.iso
        if id == "loose":
            passes_cutbased = leptons.cutBased >= cuts.cutBased
        elif id == "tight":
            passes_cutbased = leptons.cutBased == cuts.cutBased
        good_leptons = passes_pt & passes_eta & passes_SC & passes_cutbased

    elif lepton_flavour == "Muon":
        # Requirements on isolation and id
        passes_iso = leptons.pfRelIso04_all < cuts.iso
        passes_id = leptons[cuts.id] == True

        good_leptons = passes_pt & passes_eta & passes_iso & passes_id

    return good_leptons


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


def jet_preselection(events, params):
    jets = events.Jet
    nominal_cuts = params.object_preselection.Jet.nominal
    soft_cuts = params.object_preselection.Jet.soft

    nominal_pt = jets.pt > nominal_cuts.pt
    soft_pt =  (jets.pt > soft_cuts.pt) & (jets.pt < nominal_cuts.pt)
    presel_pt = jets.pt > soft_cuts.pt

    passes_eta = abs(jets.eta) < soft_cuts.eta
    passes_jetId  = (jets.jetId & soft_cuts.jetId) == 2

    nominal_jets = passes_eta & nominal_pt & passes_jetId
    soft_jets = passes_eta & soft_pt & passes_jetId
    preselected_jets = passes_eta & presel_pt & passes_jetId

    return nominal_jets, soft_jets, preselected_jets

def ak8_jet_preselection(events, fat_jets, params):
    cuts = params.object_preselection.fatJet

    passes_pt = fat_jets.pt > cuts.pt
    passes_eta = abs(fat_jets.eta) < cuts.eta
    passes_msoftdrop = (fat_jets.msoftdrop >= cuts.msoftdrop_lower) & (fat_jets.msoftdrop <= cuts.msoftdrop_upper)
    passes_nsubjettines_ratio = (fat_jets.tau2/fat_jets.tau1) <= cuts.nsubjettiness_ratio
    passes_btag_WP = fat_jets.particleNetWithMass_HbbvsQCD > cuts.btagWP

    good_jets = passes_pt & passes_eta & passes_msoftdrop & passes_nsubjettines_ratio & passes_btag_WP

    return good_jets


def tau_preselection(events, params, id):

    taus = events.Tau
    cuts = params.object_preselection.Tau[id]

    try:
        passes_decayModeDMs=taus.decayModeFindingNewDMs
    except:
        passes_decayModeDMs=~np.isnan(ak.ones_like(taus.pt))

    passes_pt = taus.pt > cuts.pt
    passes_eta = abs(taus.eta) < cuts.eta
    passes_dz = abs(taus.dz) < cuts.dz
    passes_deeptauid = (taus.idDeepTau2018v2p5VSjet == cuts.wp)


    good_taus = passes_pt & passes_eta &  passes_dz & passes_deeptauid & passes_decayModeDMs

    return good_taus
