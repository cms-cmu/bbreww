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

import hist
from optparse import OptionParser
from omegaconf import OmegaConf

from base_class.hist import Fill
from base_class.physics.event_selection import apply_event_selection

from bbww.analysis.helpers.common import update_events
from bbww.analysis.helpers.object_selection import met_selection, electron_selection, muon_selection, jet_selection, tau_selection, photon_selection
from bbww.analysis.helpers.candidate_selection import candidate_selection
from bbww.analysis.helpers.chi_square import chi_sq
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
        with open(corrections_metadata, "r") as f:
            self.corrections_metadata = yaml.safe_load(f)

    def process(self, events):

        logging.debug(f"Metadata: {events.metadata}\n")
        self.dataset = events.metadata['dataset']
        self.year = events.metadata['year']
        self.year_label = self.corrections_metadata[self.year]['year_label']
        self.processName = events.metadata['processName']
        self.is_data = not hasattr(events, "genWeight")
        self.is_mc = not self.is_data
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
        if self.is_mc: weights.add('xsec', scale*events.genWeight/events.metadata['genEventSumw'])

        params = OmegaConf.load(self.parameters)

        ### object preselection

        events = met_selection(events, year, is_data = False) #MET ##isData is placeholder
        events = muon_selection(events, year, params) #muons
        events = electron_selection(events, year, params) #electrons
        events = tau_selection(events,params) #taus
        events = photon_selection(events,params) #photon
        events = jet_selection(events,params,year) #jets

        ### candidate selection and chi square computation
        events = candidate_selection(events)
        events = chi_sq(events) # chi square selection and calculation

        if self.is_mc:
                
            gen = events.GenPart

            gen['isTop'] = (abs(gen.pdgId)==6)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            genTops = gen[gen.isTop]
            nlo = ak.ones_like(events.MET.pt, dtype=float)
            if('TT' in self.dataset): 
                nlo = np.sqrt(corrections.get_ttbar_weight(genTops[:,0].pt) * corrections.get_ttbar_weight(genTops[:,1].pt))
                
            gen['isW'] = (abs(gen.pdgId)==24)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isZ'] = (abs(gen.pdgId)==23)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            
            genWs = gen[gen.isW] 
            genZs = gen[gen.isZ]
            genDYs = gen[gen.isZ&(gen.mass>30)]
            
            ###
            # Isolation weights for muons
            ###

            if hasattr(events, "L1PreFiringWeight"): 
                weights.add('prefiring', events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)
            weights.add('genw',events.genWeight)
            
            #### AGE: not sure we need this section. I commented it out for now
            # nnlo_nlo = {}
            # nlo_qcd = ak.ones_like(events.MET.pt, dtype=float)
            # nlo_ewk = ak.ones_like(events.MET.pt, dtype=float)
            # weights.add('nlo_ewk',nlo_ewk)
            # #weights.add('nlo',nlo) 
            # if 'cen' in nnlo_nlo:
            #     #weights.add('nnlo_nlo',nnlo_nlo['cen'])
            #     weights.add('qcd1',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd1up']/nnlo_nlo['cen'], nnlo_nlo['qcd1do']/nnlo_nlo['cen'])
            #     weights.add('qcd2',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd2up']/nnlo_nlo['cen'], nnlo_nlo['qcd2do']/nnlo_nlo['cen'])
            #     weights.add('qcd3',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd3up']/nnlo_nlo['cen'], nnlo_nlo['qcd3do']/nnlo_nlo['cen'])
            #     weights.add('ew1',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew1up']/nnlo_nlo['cen'], nnlo_nlo['ew1do']/nnlo_nlo['cen'])
            #     weights.add('ew2G',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Gup']/nnlo_nlo['cen'], nnlo_nlo['ew2Gdo']/nnlo_nlo['cen'])
            #     weights.add('ew3G',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Gup']/nnlo_nlo['cen'], nnlo_nlo['ew3Gdo']/nnlo_nlo['cen'])
            #     weights.add('ew2W',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Wup']/nnlo_nlo['cen'], nnlo_nlo['ew2Wdo']/nnlo_nlo['cen'])
            #     weights.add('ew3W',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Wup']/nnlo_nlo['cen'], nnlo_nlo['ew3Wdo']/nnlo_nlo['cen'])
            #     weights.add('ew2Z',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Zup']/nnlo_nlo['cen'], nnlo_nlo['ew2Zdo']/nnlo_nlo['cen'])
            #     weights.add('ew3Z',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Zup']/nnlo_nlo['cen'], nnlo_nlo['ew3Zdo']/nnlo_nlo['cen'])
            #     weights.add('mix',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['mixup']/nnlo_nlo['cen'], nnlo_nlo['mixdo']/nnlo_nlo['cen'])
            #     #weights.add('muF',ak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muFup']/nnlo_nlo['cen'], nnlo_nlo['muFdo']/nnlo_nlo['cen'])
            #     #weights.add('muR',ak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muRup']/nnlo_nlo['cen'], nnlo_nlo['muRdo']/nnlo_nlo['cen'])
            
        selection.add('lumimask', events.lumimask)
        selection.add('met_filters', events.passNoiseFilter)
        selection.add('trigger', events.passHLT)

        noHEMj = ak.ones_like(events.MET.pt, dtype=bool)
        if year=='2018': noHEMj = (events.j_nHEM==0)
        noHEMmet = ak.ones_like(events.MET.pt, dtype=bool)
        if year=='2018': noHEMmet = (events.MET.pt>470)|(events.MET.phi>-0.62)|(events.MET.phi<-1.62)
        
        selection.add('isoneE', (events.e_ntight==1) & (events.mu_nloose==0) & (events.pho_nloose==0) & (events.tau_nloose==0))
        selection.add('isoneM', (events.mu_ntight==1) & (events.e_nloose==0) & (events.pho_nloose==0) & (events.tau_nloose==0))
        selection.add('noHEMj', noHEMj)
        selection.add('noHEMmet', noHEMmet)
        selection.add('njets',  (events.j_nsoft>2))


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
            'e_channel': selection.all('isoneE') & ~selection.all('isoneM'),
            'm_channel': selection.all('isoneM') & ~selection.all('isoneE'),
            'other': ~selection.all('isoneE') & ~selection.all('isoneM')
        }) 
        
        # def normalize(val,cut):
        #     if cut is None:
        #         return ak.fill_none(val, np.nan)
        #     else:
        #         return ak.fill_none(val[cut], np.nan)

        
        # def fill(systematic):
        #     cut = selection.all(*regions[region])
        #     if systematic in weights.variations:
        #         weight = weights.weight(modifier=systematic)[cut]
        #     else:
        #         weight = weights.weight()[cut]
        #     sname = 'nominal' if systematic is None else systematic
        #     variables = {
        #         'met':                         met.pt,
        #         'chi_hadW':                    ak.firsts(chi_sq_hadW),
        #         'chi_hadWs':                   ak.firsts(chi_sq_hadWs),
        #         'chi_tt':                      ak.firsts(chi_sq_tt),
        #     }

        #     for variable in output['hists']:
        #         if variable not in variables:
        #             continue
        #         normalized_variable = {variable: normalize(variables[variable],cut)}
        #         output['hists'][variable].fill(
        #             dataset = self.dataset,
        #             systematic = sname,
        #             signal_region = signal_region_boolean[cut],
        #             **normalized_variable,
        #             weight= weight
        #         )

        ### AGE comment: weigfhts needs to be reviewed
        ### We need to add weight to the events before filling the histograms
        events['weight'] = weights.weight()  

        output = {}
        if not shift_name:
            output['events_processed'] = {}
            output['events_processed'][events.metadata['dataset']] = {
                'n_events' : self.n_events,
                'sum_genweights': np.sum(events.genWeight) if self.is_mc else self.n_events
            }

            
            output['cutflow'] = {}
            output['cutflow'][events.metadata['dataset']] = {
                'e_channel': {
                    'events': {
                        'basic_selection': np.sum(events.selection.basic_selection[events.channel.e_channel]),
                        'preselection': np.sum(events.selection.preselection[events.channel.e_channel]),
                    },
                    'weights': {
                        'basic_selection': np.sum(events.weight[events.selection.basic_selection[events.channel.e_channel]]),
                        'preselection': np.sum(events.weight[events.selection.preselection[events.channel.e_channel]]),
                    },
                },
                'm_channel': {
                    'events': {
                        'basic_selection': np.sum(events.selection.basic_selection[events.channel.m_channel]),
                        'preselection': np.sum(events.selection.preselection[events.channel.m_channel]),
                    },
                    'weights': {
                        'basic_selection': np.sum(events.weight[events.selection.basic_selection[events.channel.m_channel]]),
                        'preselection': np.sum(events.weight[events.selection.preselection[events.channel.m_channel]]),
                    },
                },
                'other': {
                    'events': {
                        'basic_selection': np.sum(events.selection.basic_selection[events.channel.other]),
                        'preselection': np.sum(events.selection.preselection[events.channel.other]),
                    },
                    'weights': {
                        'basic_selection': np.sum(events.weight[events.selection.basic_selection[events.channel.other]]),
                        'preselection': np.sum(events.weight[events.selection.preselection[events.channel.other]]),
                    },
                },
            }

        hists = fill_histograms(
            events,
            processName=self.processName,
            year=self.year_label,
            is_mc=self.is_mc,
            selection_list=['basic_selection', 'preselection'],
            channel_list=['e_channel', 'm_channel', 'other'],
        )

        return hists | output
    
    def postprocess(self, accumulator):
        return accumulator