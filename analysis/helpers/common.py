import numpy as np
import awkward as ak
import pickle
import tempfile
import awkward as ak
from coffea.nanoevents.methods import vector
from bbreww.analysis.helpers.corrections import get_ele_sf, get_mu_id_sf, get_mu_iso_sf, get_mu_trig_sf, get_ele_trig_sf
from src.storage.eos import EOS

def distance(x1,y1,x2,y2):
    return abs(ak.fill_none(np.sqrt((x2-x1)**2+(y2-y1)**2),np.nan))

def match(a, b, val):
    combinations = a.cross(b, nested=True)
    return (combinations.i0.delta_r(combinations.i1)<val).any()

def sigmoid(x,a,b,c,d):
    """
    Sigmoid function for trigger turn-on fits.
    f(x) = c + (d-c) / (1 + np.exp(-a * (x-b)))
    """
    return c + (d-c) / (1 + np.exp(-a * (x-b)))

def elliptical_region(x, y, center_x, center_y, width, height):
    """
    Check if point (x, y) is inside an ellipse (used for signal and control region)

    Parameters:
    - x, y: coordinates of the point to check
    - center_x, center_y: center of the ellipse
    - width, height: full width and height of the ellipse

    Returns:
    - True if point is inside the ellipse, False otherwise
    """
    # Semi-axes (width and height are full dimensions, so divide by 2)
    a = width / 2
    b = height / 2

    # Standard ellipse equation: ((x-h)/a)^2 + ((y-k)/b)^2 <= 1
    result = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2

    return result <= 1

def dump_phh_to_pickle(selected_events, dataset, output_path, year=None):
    """Dump signal SvB.phh data to a pickle file for quantile rebinning.

    Also saves per-event fit-channel (channel, flavor) so quantile regression
    can validate bin populations at the actual fit granularity
    (channel, flavor, year).

    Supports writing to EOS by first writing to a local temp file,
    then copying to the remote destination.
    """
    # Per-event channel/flavor as string arrays (small, easy to filter on)
    ch_hadW = ak.to_numpy(selected_events.channel.hadronic_W)
    ch_lepW = ak.to_numpy(selected_events.channel.leptonic_W)
    channel = np.where(ch_hadW, 'hadW', np.where(ch_lepW, 'lepW', 'none'))

    fl_e  = ak.to_numpy(selected_events.flavor.e)
    fl_mu = ak.to_numpy(selected_events.flavor.mu)
    flavor = np.where(fl_e, 'e', np.where(fl_mu, 'mu', 'none'))

    def _region_block(region_mask):
        m = ak.to_numpy(region_mask)
        return {
            'phh':     ak.to_numpy(selected_events.SvB.phh[region_mask]),
            'weight':  ak.to_numpy(selected_events.weight[region_mask]),
            'channel': channel[m],
            'flavor':  flavor[m],
        }

    phh_data = {
        'dataset': dataset,
        'year':    year,
        'nominal_4j2b': _region_block(selected_events.nominal_4j2b & selected_events.region.SR),
        'lowpt_4j2b':   _region_block(selected_events.lowpt_4j2b   & selected_events.region.SR),
        'incl_3j2b':    _region_block(selected_events.incl_3j2b    & selected_events.region.SR),
    }

    eos_path = EOS(output_path)
    if eos_path.is_local:
        with open(output_path, 'wb') as f:
            pickle.dump(phh_data, f)
    else:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
            pickle.dump(phh_data, f)
        try:
            EOS(temp_path).copy_to(eos_path, parents=True, overwrite=True)
        finally:
            import os
            os.remove(temp_path)


def update_events(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

def nu_pz(l,v):
    """Return calculated MET pz with W mass constraint on (lepton + MET)"""
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
    #return ak.where(abs(pz_1) <= abs(pz_2), pz_1, pz_2)
    return pz_1, pz_2

def chi_square(data, mean, std, power=1):
    chi2 = ((data - mean)/std)**power
    return chi2

def met_reconstr(events, lep):
    met = events.MET
    pz_1, pz_2 = nu_pz(lep, met)
    pz =  ak.where(abs(pz_1) <= abs(pz_2), pz_1, pz_2)
    nu = ak.zip({
            "pt": met.pt,
            "eta": np.arcsinh(pz / met.pt),
            "phi": met.phi,
            "mass": ak.zeros_like(met.pt),
            "charge": ak.zeros_like(met.pt, dtype=int),
        },
        with_name="PtEtaPhiMCandidate",
        behavior=vector.behavior,
    )

    return nu, pz_1, pz_2

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
def add_lepton_sfs(params, events, electron, muon, weights, list_weight_names, year):
    ele_reco_sf, ele_id_sf, ele_trig_sf = get_ele_sfs(params, electron, year)
    mu_reco_sf = ak.ones_like(events.Muon.pt, dtype = float)
    mu_iso_sf, mu_id_sf, mu_trig_sf = get_mu_sfs(params, muon, year)
    ele_iso_sf = ak.ones_like(events.Electron.pt, dtype = float)

    reco_sf = ak.where(events.e_region, # select leading lepton out of leading electrons and leading muons
                    ak.firsts(ele_reco_sf[electron.istight]),# leading electrons
                    ak.firsts(mu_reco_sf[muon.istight])) # leading muons
    id_sf = ak.where(events.e_region,
                    ak.firsts(ele_id_sf[electron.istight]),
                    ak.firsts(mu_id_sf[muon.istight]))
    iso_sf = ak.where(events.e_region,
                    ak.firsts(ele_iso_sf[electron.istight]),
                    ak.firsts(mu_iso_sf[muon.istight]))
    trig_sf = ak.where(events.e_region,
                    ak.firsts(ele_trig_sf[electron.istight]),
                    ak.firsts(mu_trig_sf[muon.istight]))

    weights.add('reco_sf', reco_sf)
    weights.add('id_sf', id_sf)
    weights.add('iso_sf', iso_sf)
    weights.add('trig_sf', trig_sf)

    list_weight_names.append(f"Ele_reco_SF")
    list_weight_names.append(f"Muon_iso_SF")
    list_weight_names.append(f"Ele_id_SF")
    list_weight_names.append(f"Muon_id_SF")
    list_weight_names.append(f"Ele_trig_SF")
    list_weight_names.append(f"Muon_trig_SF")
    
    return weights, list_weight_names
