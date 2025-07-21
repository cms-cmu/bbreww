import os
import sys
import json
import yaml
import warnings
import logging

import numpy as np
import awkward as ak
import vector

from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents.methods import candidate
from coffea.util import load, save
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from analysis.helpers.processor_config import processor_config

import hist
from optparse import OptionParser
from omegaconf import OmegaConf

from base_class.hist import Fill
from base_class.physics.event_selection import apply_event_selection

from bbww.analysis.helpers.common import update_events
from bbww.analysis.helpers.object_selection import apply_bbWW_selection
from bbww.analysis.helpers.event_selection import apply_event_selection
from bbww.analysis.helpers.candidate_selection import candidate_selection
from bbww.analysis.helpers.chi_square import chi_sq
from bbww.analysis.helpers.gen_process import gen_process
from bbww.analysis.helpers import common
from bbww.analysis.helpers.fill_histograms import fill_histograms
import bbww.analysis.helpers.corrections as corrections


warnings.filterwarnings("ignore", "Missing cross-reference index for")
warnings.filterwarnings("ignore", "Please ensure")
warnings.filterwarnings("ignore", "invalid value")

vector.register_awkward()

class analysis(processor.ProcessorABC):
    def __init__(
        self,
        path: str = "bbww/analysis/data",
        parameters: str = "bbww/analysis/metadata/object_preselection.yaml",
        corrections_metadata: str = "analysis/metadata/corrections.yml",
    ):
        self.path = path
        self.parameters = parameters
        corrections= OmegaConf.load(corrections_metadata)
        parameters = OmegaConf.load(self.parameters)
        self.params = OmegaConf.merge(corrections, parameters)

    def process(self, events):

        logging.debug(f"Metadata: {events.metadata}\n")
        self.dataset = events.metadata['dataset']
        self.year = events.metadata['year']
        self.year_label = self.params[self.year].year_label
        self.processName = events.metadata['processName']
        self.isData = not hasattr(events, "genWeight")
        self.isMC = not self.isData
        self.n_events = len(events)

        events = apply_event_selection(
            events, 
            self.corrections_metadata[self.year], 
            cut_on_lumimask=True if self.is_data else False
        )

        # jets = apply_jerc_corrections(
        #     events.Jet,
        #     corrections_metadata=self.corrections_metadata[self.year],
        #     is_mc=self.is_mc,
        #     run_systematics=False, ###self.run_systematics,
        #     dataset=self.dataset
        # )

        shifts = [({"Jet": events.Jet}, None)]
        #need to work on a new way of loading corrections centrally from CMS
        '''corrections = load(f'{path}/corrections.coffea')

        jet_factory              = corrections['jet_factory']
        met_factory              = corrections['met_factory']

        nojer = "NOJER" if skipJER else ""
        if 'year' in events.metadata:
            year = events.metadata['year'].replace('UL','20').replace("_", "")
        thekey = f"{year}mc{nojer}"

        def add_jec_variables(jets, event_rho):
            jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
            jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
            jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
            return jets
        
        jets = jet_factory[thekey].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll))
        met = met_factory.build(events.MET, jets)

        shifts = [({"Jet": jets,"MET": met}, None)]
        if systematics:
            shifts.extend([
                ({"Jet": jets.JES_jes.up, "MET": met.JES_jes.up}, "JESUp"),
                ({"Jet": jets.JES_jes.down, "MET": met.JES_jes.down}, "JESDown"),
                ({"Jet": jets, "MET": met.MET_UnclusteredEnergy.up}, "UESUp"),
                ({"Jet": jets, "MET": met.MET_UnclusteredEnergy.down}, "UESDown"),
            ])
            if not skipJER:
                shifts.extend([
                    ({"Jet": jets.JER.up, "MET": met.JER.up}, "JERUp"),
                    ({"Jet": jets.JER.down, "MET": met.JER.down}, "JERDown"),
                ])

        # fill histograms separately for each systematic key
        for collections, name in shifts:
            selection(update(events, collections), output, name)'''

        weights = Weights(None, storeIndividual=True)
        list_weight_names = []

        return processor.accumulate( self.process_shift(update_events(events, collections), name, weights, list_weight_names) for collections, name in shifts )

    def process_shift(self, events, shift_name, weights, list_weight_names):

        output = {}
        selection = PackedSelection(dtype="uint64")
        year = events.metadata['year']

        if 'UL' in year:
            year_number = year.replace('UL', '')
            year = f'20{year_number}_UL' #make name compatible with corrections file

        scale = 1 if self.is_data else 1000.*float(events.metadata['lumi'])*events.metadata['xs']

        events.metadata['genEventSumw'] = events.metadata.get('genEventSumw', 1.0)
        if self.is_MC: weights.add('xsec', scale*events.genWeight/events.metadata['genEventSumw'])

        ### object preselection   
        events = apply_bbWW_selection( events, year = year, params = self.params,isMC=self.isMC,corrections_metadata=self.params[year])

        ### candidate selection and chi square computation
        events = candidate_selection(events)
        events = chi_sq(events) # chi square selection and calculation

        ### apply event selections
        events = apply_event_selection(events, self.params[year], isMC = self.isMC)

        if self.isMC:
            weights = gen_process(events, weights) ## genweights for MC

        oneE =(events.e_ntight==1) & (events.mu_nloose==0) & (events.pho_nloose==0) & (events.tau_nloose==0)
        oneM = (events.mu_ntight==1) & (events.e_nloose==0) & (events.pho_nloose==0) & (events.tau_nloose==0)

        # these selections are only required for 2018 samples
        noHEMj = ak.ones_like(events.MET.pt, dtype=bool)
        if year=='2018': noHEMj = (events.j_nHEM==0)
        noHEMmet = ak.ones_like(events.MET.pt, dtype=bool)
        if year=='2018': noHEMmet = (events.MET.pt>470)|(events.MET.phi>-0.62)|(events.MET.phi<-1.62)
        
        selection.add( "lumimask", events.lumimask)
        selection.add( "passNoiseFilter", events.passNoiseFilter)
        selection.add('isoneEorM', oneE|oneM )
        selection.add('njets',  (events.j_nsoft>2))
        selection.add('noHEMj', noHEMj)
        selection.add('noHEMmet', noHEMmet)
        selection.add('njets',  (events.j_nsoft>2))
        selection.add('leptonic_W', events.sr_boolean == 0)
        selection.add('hadronic_W', events.sr_boolean == 1)

        events['weight'] = weights.weight()  

        #### AGE comment: I am starting to think that we dont need to separarate channels and selection. We can just use the selection and then filter the events by channel
        selection_list = {
            'basic_selection': ['lumimask', 'met_filters', 'trigger'],
            'preselection': ['lumimask', 'met_filters', 'trigger', 'noHEMj', 'noHEMmet', 'njets'],
        }
        events['selection'] = ak.zip({
            'basic_selection': selection.all(*selection_list['basic_selection']),
            'preselection': selection.all(*selection_list['preselection']),
        })

        events['channel'] = ak.zip({
            'hadronic_W': selection.all('isoneEorM') & selection.all('hadronic_W'),
            'leptonic_W': selection.all('isoneEorM') & selection.all('leptonic_W'),
        }) 

        output = {}
        if not shift_name:
            output['events_processed'] = {}
            output['events_processed'][events.metadata['dataset']] = {
                'n_events' : self.n_events,
                'sum_genweights': np.sum(events.genWeight) if self.is_mc else self.n_events
            }
            
            output['cutflow'] = {}
            output['cutflow'][events.metadata['dataset']] = {
                'hadronic_W': {
                    'basic_selection': np.sum(events.selection.basic_selection[events.channel.hadronic_W]),
                    'preselection': np.sum(events.selection.preselection[events.channel.hadronic_W]),
                },
                'leptonic_W': {
                    'basic_selection': np.sum(events.selection.basic_selection[events.channel.leptonic_W]),
                    'preselection': np.sum(events.selection.preselection[events.channel.leptonic_W]),
                },
            }

        hists = fill_histograms(
            events,
            processName=self.processName,
            year=self.year_label,
            is_mc=self.is_mc,
            selection_list=['basic_selection', 'preselection']
        )

        return hists | output
    
    def postprocess(self, accumulator):
        return accumulator