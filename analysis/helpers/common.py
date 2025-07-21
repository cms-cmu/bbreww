import numpy as np
import awkward as ak
from omegaconf import OmegaConf
import logging
import correctionlib

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
    sqrt_discriminant = ak.where(discriminant >= 0, np.sqrt(discriminant), np.nan)
    pz_1 = (2*A*l.pz + sqrt_discriminant)/(2*C)
    pz_2 = (2*A*l.pz - sqrt_discriminant)/(2*C)
    return ak.where(abs(pz_1) < abs(pz_2), pz_1, pz_2)

def chi_square(data,mean,std):
    x_2 = ak.sum(data**2)
    n = ak.count(data[~ak.is_none(data)])
    chi2 = ((data - mean)/std)**2
    return chi2, mean, std

def met_reconstr(events, e, mu):
    met = events.met    
    # need "charge" here so we can add them with electrons/muons four vectors
    v_e = ak.zip(
        {
            "x": met.pt * np.cos(met.phi),
            "y": met.pt * np.sin(met.phi),
            "z": nu_pz(e, met),
            "t": np.sqrt(met.pt**2 + nu_pz(e, met)**2),
            "charge" : met.pt * 0 
        },
        with_name="Candidate"
    )

    v_mu = ak.zip(
        {
            "x": met.pt * np.cos(met.phi),
            "y": met.pt * np.sin(met.phi),
            "z": nu_pz(mu, met),
            "t": np.sqrt(met.pt**2+nu_pz(mu, met)**2) ,
            "charge" : met.pt * 0 
        },
        with_name="Candidate"
    )

    v_mu = ak.mask(v_mu, ~np.isnan(v_mu.pz))
    v_e = ak.mask(v_e, ~np.isnan(v_e.pz)) #avoid calculations for imaginary solutions

    return v_mu, v_e

def apply_jet_veto_maps( corrections_metadata, jets ):
    '''
    taken from https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/lib/cut_functions.py#L65
    modified to veto jets not events
    '''

    mask_for_VetoMap = (
        ((jets.jetId & 2)==2) # Must fulfill tight jetId
        & (abs(jets.eta) < 5.19) # Must be within HCal acceptance
        & (jets.neEmEF + jets.chEmEF < 0.9) # Energy fraction not dominated by ECal
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

    