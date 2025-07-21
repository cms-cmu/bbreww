import awkward as ak
import numpy as np
from coffea.lumi_tools import LumiMask
from analysis.helpers.common import mask_event_decision

def apply_event_selection(events, corrections_metadata:dict, isMC):

    # Apply luminosity mask
    if 'goldenJSON' not in corrections_metadata:
        raise KeyError("Missing 'goldenJSON' in corrections_metadata.")
    lumimask = LumiMask(corrections_metadata['goldenJSON'])
    events['lumimask'] = (np.full(len(events), True) if isMC else
        np.array(lumimask(events.run, events.luminosityBlock))
    )
    
    # Apply HLT triggers mask
    events['passHLT'] = (
    np.full(len(events), True)
    if 'HLT' not in events.fields else mask_event_decision(
        events, decision="OR", branch="HLT", list_to_mask=events.metadata.get('trigger', [])
        )   
    )

    # Apply noise filters mask
    noise_filters = corrections_metadata.get('NoiseFilter', [])
    events['passNoiseFilter'] = (
        np.full(len(events), True)
        if 'Flag' not in events.fields else mask_event_decision(
            events, decision="AND", branch="Flag", list_to_mask=noise_filters,
            list_to_skip=['BadPFMuonDzFilter', 'hfNoisyHitsFilter']
        )
    )

    ######## need to add single electron and single muon trigger masks here 
    ###### (need to add run3 triggers to datasets_run3.yml first)
    return events