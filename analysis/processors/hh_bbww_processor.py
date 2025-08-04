import warnings
import logging

import numpy as np
import awkward as ak
import vector

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection

from omegaconf import OmegaConf

from base_class.physics.event_selection import apply_event_selection
from base_class.physics.objects.jet_corrections import apply_jerc_corrections
from base_class.physics.event_weights import add_weights

from bbww.analysis.helpers.common import update_events, add_lepton_sfs
from bbww.analysis.helpers.object_selection import apply_bbWW_selection
from bbww.analysis.helpers.candidate_selection import candidate_selection
from bbww.analysis.helpers.chi_square import chi_sq, chi_sq_cut
from bbww.analysis.helpers.gen_process import gen_process, add_gen_info, gen_studies
from bbww.analysis.helpers.fill_histograms import fill_histograms

warnings.filterwarnings("ignore", "Missing cross-reference index for")
warnings.filterwarnings("ignore", "Please ensure")
warnings.filterwarnings("ignore", "invalid value")

vector.register_awkward()

class analysis(processor.ProcessorABC):
    def __init__(
        self,
        path: str = "bbww/analysis/data",
        parameters: str = "bbww/analysis/metadata/object_preselection_run3.yaml",
        corrections_metadata: str = "analysis/metadata/corrections.yml",
    ):
        self.path = path
        self.parameters = parameters
        corrections= OmegaConf.load(corrections_metadata)
        parameters = OmegaConf.load(self.parameters)
        btagWPs = OmegaConf.load("bbww/analysis/metadata/btag_WPs.yaml")
        self.params = OmegaConf.merge(corrections, parameters, btagWPs)

    def process(self, events):

        logging.debug(f"Metadata: {events.metadata}\n")
        self.dataset = events.metadata['dataset']
        self.year = events.metadata['year']
        self.year_label = self.params[self.year].year_label
        self.processName = events.metadata['processName']
        self.is_mc = hasattr(events, "genWeight")
        self.n_events = len(events)

        events = apply_event_selection(
            events, 
            self.params[self.year], 
            self.is_mc
        )

        jets = apply_jerc_corrections(
            events,
            corrections_metadata=self.params[self.year],
            isMC=self.is_mc,
            run_systematics=False, ###self.run_systematics,
            dataset=self.dataset,
            jet_type='AK4PFPuppi.txt',
        )

        shifts = [({"Jet": events.Jet}, None)]
        '''
        ## AGE comment: we need to add MET corrections 
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
        '''

        weights = Weights(None, storeIndividual=True)
        list_weight_names = []

        return processor.accumulate( self.process_shift(update_events(events, collections), name, weights, list_weight_names) for collections, name in shifts )

    def process_shift(self, events, shift_name, weights, list_weight_names):

        output = {}
        selection = PackedSelection(dtype="uint64")

        scale = 1 if (not self.is_mc) else 1000.*float(events.metadata['lumi'])*events.metadata['xs']

        events.metadata['genEventSumw'] = events.metadata.get('genEventSumw', 1.0)
        if self.is_mc: weights.add('xsec', scale*events.genWeight/events.metadata['genEventSumw'])

        ### object preselection   
        events = apply_bbWW_selection( events, year = self.year, params = self.params, isMC=self.is_mc,corrections_metadata=self.params[self.year])
        events = add_gen_info(events) if self.is_mc else events # add gen level info before candidate seleciton
        ### candidate selection and chi square computation
        events = candidate_selection(events, self.params, self.year, self.is_mc)
        
        if self.is_mc:
            events = gen_studies(events)
        events = chi_sq(events) # chi square selection and calculation

        ### apply event selections
        events = apply_event_selection(events, self.params[self.year], cut_on_lumimask = not self.is_mc)

        oneE =(events.e_ntight==1) & (events.mu_nloose==0) & (events.pho_nloose==0) & (events.tau_nloose==0)
        oneM = (events.mu_ntight==1) & (events.e_nloose==0) & (events.pho_nloose==0) & (events.tau_nloose==0)

        # these selections are only required for 2018 samples
        noHEMj = ak.ones_like(events.MET.pt, dtype=bool)
        if '18' in self.year: noHEMj = (events.j_nHEM==0)
        noHEMmet = ak.ones_like(events.MET.pt, dtype=bool)
        if '18' in self.year: noHEMmet = (events.MET.pt>470)|(events.MET.phi>-0.62)|(events.MET.phi<-1.62)
        
        selection.add( "lumimask", events.lumimask)
        selection.add( "passNoiseFilter", events.passNoiseFilter)
        selection.add("trigger", events.passHLT)
        selection.add('isoneEorM', oneE|oneM )
        selection.add('njets',  (events.j_nsoft>2))
        selection.add('noHEMj', noHEMj)
        selection.add('noHEMmet', noHEMmet)
        selection.add('njets',  (events.j_nsoft>3))
        selection.add('twoBjets', events.has_2_bjets)

        jet_veto_maps = (ak.any(events.Jet.jet_veto_maps,axis=1) if '202' in self.year 
                         else ak.ones_like(events.MET.pt,dtype=bool))
        
        selection.add('jet_veto_mask', jet_veto_maps)
        selection.add('leptonic_W', events.sr_boolean == 0)
        selection.add('hadronic_W', events.sr_boolean == 1)
        selection.add('null_region', events.sr_boolean==5) # events where the selected two W jets don't have a matching genjet

        weights, list_weight_names = add_weights(
            events,
            do_MC_weights=self.is_mc,
            dataset=self.dataset,
            year_label=self.year_label,
            friend_trigWeight=None,
            corrections_metadata=self.params[self.year],
            apply_trigWeight=False,
            isTTForMixed=False
        )

        weights = add_lepton_sfs(events, events.Electron, events.Muon, weights, self.year)
        events['weight'] = weights.weight() 

        selection_list = {
            'basic_selection': ['lumimask', 'passNoiseFilter', 'trigger'],
            'preselection': ['lumimask', 'passNoiseFilter', 'trigger', 'noHEMj', 'noHEMmet', 'njets', 'jet_veto_mask'],
        }
        events['selection'] = ak.zip({
            'basic_selection': selection.all(*selection_list['basic_selection']),
            'preselection': selection.all(*selection_list['preselection']),
        })

        events['channel'] = ak.zip({
            'hadronic_W': selection.all('isoneEorM') & selection.all('hadronic_W'),
            'leptonic_W': selection.all('isoneEorM') & selection.all('leptonic_W'),
            'null' : selection.all('isoneEorM') & selection.all('null_region')
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
                    'events': {
                        'basic_selection': np.sum(events.selection.basic_selection[events.channel.hadronic_W]),
                        'preselection': np.sum(events.selection.preselection[events.channel.hadronic_W]),
                    },
                    'weights': {
                        'basic_selection': np.sum(events.weight[events.selection.basic_selection[events.channel.hadronic_W]]),
                        'preselection': np.sum(events.weight[events.selection.preselection[events.channel.hadronic_W]]),
                    },
                },
                'leptonic_W': {
                    'events': {
                        'basic_selection': np.sum(events.selection.basic_selection[events.channel.leptonic_W]),
                        'preselection': np.sum(events.selection.preselection[events.channel.leptonic_W]),
                    },
                    'weights': {
                        'basic_selection': np.sum(events.weight[events.selection.basic_selection[events.channel.leptonic_W]]),
                        'preselection': np.sum(events.weight[events.selection.preselection[events.channel.leptonic_W]]),
                    },
                },
                'null': {
                    'events': {
                        'basic_selection': np.sum(events.selection.basic_selection[events.channel.null]),
                        'preselection': np.sum(events.selection.preselection[events.channel.null]),
                    },
                    'weights': {
                        'basic_selection': np.sum(events.weight[events.selection.basic_selection[events.channel.null]]),
                        'preselection': np.sum(events.weight[events.selection.preselection[events.channel.null]]),
                    },
                }
            }

        hists = fill_histograms(
            events,
            processName=self.processName,
            year=self.year_label,
            is_mc=self.is_mc,
            selection_list=['basic_selection', 'preselection'],
            channel_list=['hadronic_W', 'leptonic_W', 'null']
        )

        return hists | output
    
    def postprocess(self, accumulator):
        return accumulator