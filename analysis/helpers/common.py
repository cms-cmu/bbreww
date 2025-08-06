import numpy as np
import awkward as ak
import logging
import correctionlib
from coffea.nanoevents.methods import vector
from bbww.analysis.helpers.corrections import get_ele_sf, get_mu_id_sf, get_mu_iso_sf

def match(a, b, val):
    combinations = a.cross(b, nested=True)
    return (combinations.i0.delta_r(combinations.i1)<val).any()

def sigmoid(x,a,b,c,d):
    """
    Sigmoid function for trigger turn-on fits.
    f(x) = c + (d-c) / (1 + np.exp(-a * (x-b)))
    """
    return c + (d-c) / (1 + np.exp(-a * (x-b)))


def update_events(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

def nu_pz(l,v):
    #ttbar and hadronic W neutrino pz reconstruction
    m_w = 80.379
    m_l = l.mass            
    A = (l.px*v.pt * np.cos(v.phi)+l.py*v.pt * np.sin(v.phi)) + (m_w**2 - m_l**2)/2
    B = l.energy**2*((v.pt * np.cos(v.phi))**2+(v.pt * np.sin(v.phi))**2)
    C = l.energy**2 - l.pz**2
    discriminant = (2 * A * l.pz)**2 - 4 * (B - A**2) * C

    # avoiding imaginary solutions
    sqrt_discriminant = ak.where(discriminant >= 0.0, 
                                 np.sqrt(discriminant), 
                                 0.0)
    
    pz_1 = np.real(ak.fill_none((2*A*l.pz + sqrt_discriminant)/(2*C), np.nan))
    pz_2 = np.real(ak.fill_none((2*A*l.pz - sqrt_discriminant)/(2*C), np.nan))
    return ak.where(abs(pz_1) <= abs(pz_2), pz_1, pz_2)

def chi_square(data,mean,std):
    chi2 = ((data - mean)/std)**2
    return chi2

def met_reconstr(events, e, mu):
    met = events.met    
    pz_e = nu_pz(e, met)
    pz_mu = nu_pz(mu,met)
    v_e = ak.zip({
            "x": met.pt * np.cos(met.phi),
            "y": met.pt * np.sin(met.phi),
            "z": pz_e,
            "t": np.sqrt(met.pt**2 + pz_e**2),
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    v_mu = ak.zip(
        {
            "x": met.pt * np.cos(met.phi),
            "y": met.pt * np.sin(met.phi),
            "z": pz_mu,
            "t": np.sqrt(met.pt**2+pz_mu**2) ,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    v_mu = ak.mask(v_mu, ~np.isnan(v_mu.pz))
    v_e = ak.mask(v_e, ~np.isnan(v_e.pz)) #avoid calculations for imaginary solutions

    return v_mu, v_e

def get_ele_sfs(electron, year):
    reco_sf =  ak.where(
            (electron.pt<20),
            get_ele_sf(year, electron.eta+electron.deltaEtaSC, electron.pt, "RecoBelow20"), 
            get_ele_sf(year, electron.eta+electron.deltaEtaSC, electron.pt, "Reco20to75")
        )
    id_sf = ak.where(
            electron.isloose,
            get_ele_sf(year, electron.eta+electron.deltaEtaSC, electron.pt, "Loose"),
            ak.ones_like(electron.pt)
        )
    id_sf = ak.where(
            electron.istight,
            get_ele_sf(year, electron.eta+electron.deltaEtaSC, electron.pt, "Tight"),
            id_sf
        )

    return reco_sf, id_sf

def get_mu_sfs(muon, year):
    id_sf = ak.where(
            muon.isloose, 
            get_mu_id_sf(year, abs(muon.eta), muon.pt, "Loose"), 
            ak.ones_like(muon.pt)
        )
    id_sf = ak.where(
            muon.istight, 
            get_mu_id_sf(year, abs(muon.eta), muon.pt, "Tight"), 
            id_sf
        )
    iso_sf = ak.where(
            muon.isloose, 
            get_mu_iso_sf(year, abs(muon.eta), muon.pt, "Loose"), 
            ak.ones_like(muon.pt)
        )
    iso_sf = ak.where(
        muon.istight, 
        get_mu_iso_sf(year, abs(muon.eta), muon.pt, "Tight"), 
        iso_sf
    )

    return iso_sf, id_sf 

#combined electron and muon scale factors
# 0: electron, 1: muon
def add_lepton_sfs(events, electron, muon, weights, year):
    ele_reco_sf, ele_id_sf = get_ele_sfs(electron, year)
    mu_reco_sf = ak.ones_like(events.Muon.pt, dtype = float)
    mu_iso_sf, mu_id_sf = get_mu_sfs(muon, year)
    ele_iso_sf = ak.ones_like(events.Electron.pt, dtype = float)
    leading_lep = events.lepton_choice == 0 # electron: 0, muon : 1

    reco_sf = ak.where(leading_lep, # select leading lepton out of leading electrons and leading muons
                       ak.firsts(ele_reco_sf[electron.istight]),# leading electrons
                       ak.firsts(mu_reco_sf[muon.istight])) # leading muons
    id_sf = ak.where(leading_lep, 
                    ak.firsts(ele_id_sf[electron.istight]), 
                    ak.firsts(mu_id_sf[muon.istight]))
    iso_sf = ak.where(leading_lep, 
                      ak.firsts(ele_iso_sf[electron.istight]), 
                      ak.firsts(mu_iso_sf[muon.istight]))

    weights.add('reco_sf', reco_sf)
    weights.add('id_sf', id_sf)
    weights.add('iso_sf', iso_sf)
    return weights

### placeholder: wanna use apply_jet_veto_maps from the base framework common.py, not bbww
def apply_jet_veto_maps( corrections_metadata, jets ):
    '''
    taken from https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/lib/cut_functions.py#L65
    modified to veto jets not events
    '''

    mask_for_VetoMap = (
        ((jets.jetId & 2)==2) # Must fulfill tight jetId
        & (abs(jets.eta) < 5.19) # Must be within HCal acceptance
        & ((jets.neEmEF + jets.chEmEF) < 0.9) # Energy fraction not dominated by ECal
    )
    if 'muonSubtrFactor' in jets.fields:  ### AGE: this should be temporary for old picos. New skims should have this field
        mask_for_VetoMap = mask_for_VetoMap & (jets.muonSubtrFactor < 0.8) # May no be Muons misreconstructed as jets
    else: logging.warning("muonSubtrFactor NOT in jets fields. This is correct only for mixeddata and old picos.")

    corr = correctionlib.CorrectionSet.from_file(corrections_metadata['file'])[corrections_metadata['tag']]

    etaFlat, phiFlat, etaCounts = ak.flatten(jets.eta), ak.flatten(jets.phi), ak.num(jets.eta)
    phiFlat = np.clip(phiFlat, -3.14159, 3.14159) # Needed since no overflow included in phi binning
    weight = ak.unflatten(
        corr.evaluate("jetvetomap", etaFlat, phiFlat),
        counts=etaCounts,
    )
    jetMask = ak.where( weight == 0, True, False, axis=1 )  # if 0 is not vetoed, then True

    return jetMask & mask_for_VetoMap

def get_sequential_cutflow(selection, events, selection_list, channels=['hadronic_W', 'leptonic_W', 'null']):
    """
    Create a sequential cutflow dictionary by progressively applying cuts from preselection list.
    
    Parameters:
    -----------
    selection : PackedSelection object
        The selection object containing all cuts
    events : awkward array
        Events array with weights
    selection_list : dict
        Dictionary containing the list of cuts
    channels : list
        List of channel names to track
    """
    
    sequential_cutflow = {}
    preselection_cuts = selection_list['preselection']

    for channel in channels:
        sequential_cutflow[channel] = {
            'events': {},
            'weights': {}
        }    

        isoneEorM_mask = selection.all('isoneEorM') & selection.all(channel)
        sequential_cutflow[channel]['events']['isoneEorM'] = np.sum(isoneEorM_mask)
        sequential_cutflow[channel]['weights']['isoneEorM'] = np.sum(events.weight[isoneEorM_mask])
        
        # Build cumulative selections
        for i in range(len(preselection_cuts)):
            # Get cuts up to current index
            cuts_so_far = preselection_cuts[:i+1]
            cuts_with_lepton = ['isoneEorM'] + cuts_so_far
            cumulative_mask = selection.all(*cuts_with_lepton)
            final_mask = cumulative_mask & selection.all(channel)
            
            # Store results with the name of the last cut applied
            cut_name = preselection_cuts[i]
            sequential_cutflow[channel]['events'][cut_name] = np.sum(final_mask)
            sequential_cutflow[channel]['weights'][cut_name] = np.sum(events.weight[final_mask])
    
    return sequential_cutflow
