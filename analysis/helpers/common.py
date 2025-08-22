import numpy as np
import awkward as ak
import logging
import correctionlib
from coffea.nanoevents.methods import vector
from bbww.analysis.helpers.corrections import get_ele_sf, get_mu_id_sf, get_mu_iso_sf, get_mu_trig_sf, get_ele_trig_sf

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
    
    pz_1 = ak.fill_none((2*A*l.pz + sqrt_discriminant)/(2*C), np.nan)
    pz_2 = ak.fill_none((2*A*l.pz - sqrt_discriminant)/(2*C), np.nan)
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

def get_ele_sfs(params, electron, year):
    reco_sf =  ak.where(
            (electron.pt<20),
            get_ele_sf(params, year, electron.eta+electron.deltaEtaSC, electron.pt, electron.phi, "RecoBelow20"), 
            get_ele_sf(params, year, electron.eta+electron.deltaEtaSC, electron.pt, electron.phi, "Reco20to75")
        )

    id_sf = ak.where(
            electron.istight,
            get_ele_sf(params, year, electron.eta+electron.deltaEtaSC, electron.pt, electron.phi, "Tight"),
            ak.ones_like(electron.pt)
        )
    
    trig_sf = ak.where(
        electron.istight,
        get_ele_trig_sf(params, year, electron.eta+electron.deltaEtaSC, electron.pt, "Tight"),
        ak.ones_like(electron.pt)
    )

    return reco_sf, id_sf, trig_sf

def get_mu_sfs(params, muon, year):
    id_sf = ak.where(
            muon.istight, 
            get_mu_id_sf(params, year, abs(muon.eta), muon.pt, "Tight"), 
            ak.ones_like(muon.pt)
        )
    iso_sf = ak.where(
        muon.istight, 
        get_mu_iso_sf(params, year, abs(muon.eta), muon.pt, "Tight"), 
        ak.ones_like(muon.pt)
    )
    trig_sf = ak.where(
        muon.istight, 
        get_mu_trig_sf(params, year, abs(muon.eta), muon.pt, "Tight"), 
        ak.ones_like(muon.pt)
    )

    return iso_sf, id_sf, trig_sf

#combined electron and muon scale factors
# 0: electron, 1: muon
def add_lepton_sfs(params, events, electron, muon, weights, year, is_mc):
    if is_mc:
        e_clean = electron[electron.isclean]        
        ele_reco_sf, ele_id_sf, ele_trig_sf = get_ele_sfs(params, e_clean, year)
        mu_reco_sf = ak.ones_like(events.Muon.pt, dtype = float)
        mu_iso_sf, mu_id_sf, mu_trig_sf = get_mu_sfs(params, muon, year)
        ele_iso_sf = ak.ones_like(events.Electron.pt, dtype = float)
        
        reco_sf = ak.where(events.e_region, # select leading lepton out of leading electrons and leading muons
                        ak.firsts(ele_reco_sf[e_clean.istight]),# leading electrons
                        ak.firsts(mu_reco_sf[muon.istight])) # leading muons
        id_sf = ak.where(events.e_region, 
                        ak.firsts(ele_id_sf[e_clean.istight]), 
                        ak.firsts(mu_id_sf[muon.istight]))
        iso_sf = ak.where(events.e_region, 
                        ak.firsts(ele_iso_sf[electron.istight]),
                        ak.firsts(mu_iso_sf[muon.istight]))
        trig_sf = ak.where(events.e_region, 
                        ak.firsts(ele_trig_sf[e_clean.istight]),
                        ak.firsts(mu_trig_sf[muon.istight]))

        weights.add('reco_sf', reco_sf)
        weights.add('id_sf', id_sf)
        weights.add('iso_sf', iso_sf)
        weights.add('trig_sf', trig_sf)
    return weights

def get_sequential_cutflow(selection, events, selection_list):
    sequential_cutflow = {
        'events': {},
        'weights': {}
    }
    cumulative_cuts = []

    for cut_name in selection_list:
        # Add the cuts in sequence
        cumulative_cuts.append(cut_name)
        current_mask = selection.all(*cumulative_cuts)
        
        sequential_cutflow['events'][cut_name] = np.sum(current_mask)
        sequential_cutflow['weights'][cut_name] = np.sum(events.weight[current_mask])

    return sequential_cutflow

def add_output_cutflow(events, output):
    region_map = {
    'hadronic_W': events.channel.hadronic_W,
    'leptonic_W': events.channel.leptonic_W,
    'mu_region':  events.region.mu_region,
    'e_region':   events.region.e_region
    }
    output['cutflow_weights'][events.metadata['dataset']] = {
        name: {
            'events': {
                'preselection':      np.sum(events.selection.preselection[selector]),
                'nominal_4j2b':      np.sum(events.selection.nominal_4j2b[selector]),
                'nominal_3j1b':      np.sum(events.selection.nominal_3j2b[selector]),
                'lowpt_4j2b':        np.sum(events.selection.lowpt_4j2b[selector]),
                'chi_sq_nom_4j2b':   np.sum(events.selection.chi_sq_nom_4j2b[selector]),
                'chi_sq_nom_3j2b':   np.sum(events.selection.chi_sq_nom_3j2b[selector]),
                'chi_sq_lowpt_4j2b': np.sum(events.selection.chi_sq_lowpt_4j2b[selector])
            },
            'weights': {
                'preselection':      np.sum(events.weight[events.selection.preselection[selector]]),
                'nominal_4j2b':      np.sum(events.weight[events.selection.nominal_4j2b[selector]]),
                'nominal_3j1b':      np.sum(events.weight[events.selection.nominal_3j2b[selector]]),
                'lowpt_4j2b':        np.sum(events.weight[events.selection.lowpt_4j2b[selector]]),
                'chi_sq_nom_4j2b':   np.sum(events.weight[events.selection.chi_sq_nom_4j2b[selector]]),
                'chi_sq_nom_3j2b':   np.sum(events.weight[events.selection.chi_sq_nom_3j2b[selector]]),
                'chi_sq_lowpt_4j2b': np.sum(events.weight[events.selection.chi_sq_lowpt_4j2b[selector]])
            }
        }
        for name, selector in region_map.items()
    }
    return output