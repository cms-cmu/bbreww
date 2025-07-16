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

from bbww.analysis.helpers.common import update_events
from bbww.analysis.helpers.object_selection import met_selection, electron_selection, muon_selection, jet_selection, tau_selection, photon_selection
from bbww.analysis.helpers.candidate_selection import candidate_selection
from bbww.analysis.helpers.chi_square import chi_sq
from bbww.analysis.helpers import common
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

    # def make_output():
    #     return {
    #         'nEvents': {},
    #         'cutflow': {},
    #         # 'test': {},
    #         # 'hists' : {
    #         #     "met": (
    #         #         hda.Hist.new
    #         #         .Reg(50, 0, 300, name="met", label = 'pT [GeV]')  
    #         #         .StrCat([], name="systematic", growth = True)      
    #         #         .StrCat([], name="dataset", growth = True)     
    #         #         .Int(0,2, name="signal_region", overflow  = False, underflow = False) 
    #         #         .Weight()                               
    #         #     ),
    #         #     "chi_hadW": (
    #         #         hda.Hist.new
    #         #         .Reg(50, 0, 5, name="chi_hadW", label=r'$\chi^2$')
    #         #         .StrCat([], name="dataset", growth = True)
    #         #         .StrCat([], name="systematic", growth = True)  
    #         #         .Int(0,2, name="signal_region", overflow  = False, underflow = False)
    #         #         .Weight()
    #         #     ),
    #         #     "chi_hadWs": (
    #         #         hda.Hist.new
    #         #         .Reg(50, 0, 5, name="chi_hadWs", label=r'$\chi^2$')
    #         #         .StrCat([], name="systematic", growth = True)  
    #         #         .StrCat([], name="dataset", growth = True)
    #         #         .Int(0,2, name="signal_region", overflow  = False, underflow = False)
    #         #         .Weight()
    #         #     ),
    #         #     "chi_tt": (
    #         #         hda.Hist.new
    #         #         .Reg(50, 0, 5, name="chi_tt", label=r'$\chi^2$')
    #         #         .StrCat([], name="systematic", growth = True)  
    #         #         .StrCat([], name="dataset", growth = True)
    #         #         .Int(0,2, name="signal_region", overflow  = False, underflow = False)
    #         #         .Weight()
    #         #     ),
    #         # }
    #     }

    def process(self, events):

        logging.debug(f"Metadata: {events.metadata}\n")
        self.dataset = events.metadata['dataset']
        self.year = events.metadata['year']
        self.year_label = self.corrections_metadata[self.year]['year_label']
        self.processName = events.metadata['processName']
        self.isData = not hasattr(events, "genWeight")
        self.isMC = not self.isData
        

        # jets = apply_jerc_corrections(
        #     events.Jet,
        #     corrections_metadata=self.corrections_metadata[self.year],
        #     isMC=self.isMC,
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
        scale = 1 if self.isData else 1000.*float(events.metadata['lumi'])*events.metadata['xs']

        events.metadata['genEventSumw'] = events.metadata.get('genEventSumw', 1.0)
        if self.isMC: weights.add('xsec', scale*events.genWeight/events.metadata['genEventSumw'])

        params = OmegaConf.load(self.parameters)
                
        deepflavWPs = common.common['btagWPs']['deepflav'][self.year_label]
        deepcsvWPs =  common.common['btagWPs']['deepcsv'][self.year_label]

        ### object preselection

        events = met_selection(events, year, isData = False) #MET ##isData is placeholder
        events = muon_selection(events, year, params) #muons
        events = electron_selection(events, year, params) #electrons
        events = tau_selection(events,params) #taus
        events = photon_selection(events,params) #photon
        events = jet_selection(events,params,year) #jets

        ### candidate selection and chi square computation
        events = candidate_selection(events)
        events = chi_sq(events) # chi square selection and calculation

        if self.isMC:
                
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
            
            nnlo_nlo = {}
            nlo_qcd = ak.ones_like(events.MET.pt, dtype=float)
            nlo_ewk = ak.ones_like(events.MET.pt, dtype=float)
                                            

            ###
            # Isolation weights for muons
            ###

            if hasattr(events, "L1PreFiringWeight"): 
                weights.add('prefiring', events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)
            weights.add('genw',events.genWeight)
            weights.add('nlo_ewk',nlo_ewk)
            #weights.add('nlo',nlo) 
            if 'cen' in nnlo_nlo:
                #weights.add('nnlo_nlo',nnlo_nlo['cen'])
                weights.add('qcd1',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd1up']/nnlo_nlo['cen'], nnlo_nlo['qcd1do']/nnlo_nlo['cen'])
                weights.add('qcd2',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd2up']/nnlo_nlo['cen'], nnlo_nlo['qcd2do']/nnlo_nlo['cen'])
                weights.add('qcd3',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['qcd3up']/nnlo_nlo['cen'], nnlo_nlo['qcd3do']/nnlo_nlo['cen'])
                weights.add('ew1',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew1up']/nnlo_nlo['cen'], nnlo_nlo['ew1do']/nnlo_nlo['cen'])
                weights.add('ew2G',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Gup']/nnlo_nlo['cen'], nnlo_nlo['ew2Gdo']/nnlo_nlo['cen'])
                weights.add('ew3G',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Gup']/nnlo_nlo['cen'], nnlo_nlo['ew3Gdo']/nnlo_nlo['cen'])
                weights.add('ew2W',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Wup']/nnlo_nlo['cen'], nnlo_nlo['ew2Wdo']/nnlo_nlo['cen'])
                weights.add('ew3W',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Wup']/nnlo_nlo['cen'], nnlo_nlo['ew3Wdo']/nnlo_nlo['cen'])
                weights.add('ew2Z',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew2Zup']/nnlo_nlo['cen'], nnlo_nlo['ew2Zdo']/nnlo_nlo['cen'])
                weights.add('ew3Z',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['ew3Zup']/nnlo_nlo['cen'], nnlo_nlo['ew3Zdo']/nnlo_nlo['cen'])
                weights.add('mix',ak.ones_like(events.MET.pt, dtype=float), nnlo_nlo['mixup']/nnlo_nlo['cen'], nnlo_nlo['mixdo']/nnlo_nlo['cen'])
                #weights.add('muF',ak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muFup']/nnlo_nlo['cen'], nnlo_nlo['muFdo']/nnlo_nlo['cen'])
                #weights.add('muR',ak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muRup']/nnlo_nlo['cen'], nnlo_nlo['muRdo']/nnlo_nlo['cen'])
            
        lumimask = ak.ones_like(events.MET.pt, dtype=bool) #using events.MET.pt to get 1d array with len(events)
        #if isData:
            #lumimask = lumiMasks[year](events.run, events.luminosityBlock)
        selection.add('lumimask', lumimask)

        # met_filters =  ak.ones_like(events.MET.pt, dtype=bool)
        # #if isData: met_filters = met_filters & events.Flag['eeBadScFilter'] #this filter is recommended for data only
        # for flag in met_filters_names[year]:
        #     met_filters = met_filters & events.Flag[flag]
        # selection.add('met_filters',met_filters)

        # triggers = dak.zeros_like(events.MET.pt, dtype=bool)
        # for trigger_path in singleelectron_triggers[year]:
        #     if not hasattr(events.HLT, trigger_path): continue
        #     triggers = triggers | events.HLT[trigger_path]
        # selection.add('singleelectron_triggers', triggers)
        
        # triggers = dak.zeros_like(events.MET.pt, dtype=bool)
        # for trigger_path in singlemuon_triggers[year]:
        #     if not hasattr(events.HLT, trigger_path): continue
        #     triggers = triggers | events.HLT[trigger_path]
        # selection.add('singlemuon_triggers', triggers)

        noHEMj = ak.ones_like(events.MET.pt, dtype=bool)
        if year=='2018': noHEMj = (events.j_nHEM==0)
        noHEMmet = ak.ones_like(events.MET.pt, dtype=bool)
        if year=='2018': noHEMmet = (events.MET.pt>470)|(events.MET.phi>-0.62)|(events.MET.phi<-1.62)    
        
        selection.add('isoneE', (events.e_ntight==1) & (events.mu_nloose==0) & (events.pho_nloose==0) & (events.tau_nloose==0))
        selection.add('isoneM', (events.mu_ntight==1) & (events.e_nloose==0) & (events.pho_nloose==0) & (events.tau_nloose==0))
        selection.add('njets',  (events.j_nsoft>2))
        selection.add('noHEMj', noHEMj)
        selection.add('noHEMmet', noHEMmet)

        # regions = {
        #     'esr': ['isoneE', 'noHEMj', 'njets', 'met_filters', 'noHEMmet'],
        #     'msr': ['isoneM', 'noHEMj', 'njets', 'met_filters', 'noHEMmet']
        #     }
        
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


        #     if systematic is None and self.isMC:
        #         output['nEvents'] = {
        #             'events': len(events),
        #             'genWeights': ak.sum(events.genWeight),
        #             'weights': ak.sum(weights.weight())
        #         }

        #         wgtcutflow = selection.cutflow(*regions[region], weights = weights) #, weightsmodifier = systematic) #weights  
        #         wgtcutflow_result = wgtcutflow.result()
        #         output['cutflow'] = {
        #             'selections': wgtcutflow_result.labels,
        #             'events': wgtcutflow_result.nevcutflow,
        #             'events_min_one': wgtcutflow_result.nevonecut,
        #             'weights': wgtcutflow_result.wgtevcutflow,
        #             'weights_min_one': wgtcutflow_result.wgtevonecut
        #         }
        #         ### AGE: maybe we can save the histogram from wgtcutflow.yieldhist in hists

        # fill(shift_name)

        return output
    
    def postprocess(self, accumulator):
        return accumulator