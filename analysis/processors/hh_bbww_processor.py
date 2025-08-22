import warnings
import logging

import numpy as np
import awkward as ak
import vector

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection

from omegaconf import OmegaConf

from src.physics.event_selection import apply_event_selection
from src.physics.objects.jet_corrections import apply_jerc_corrections
from src.physics.event_weights import add_weights

from bbww.analysis.helpers.common import update_events, add_lepton_sfs, get_sequential_cutflow, add_output_cutflow
from bbww.analysis.helpers.corrections import apply_met_corrections_after_jec
from bbww.analysis.helpers.object_selection import apply_bbWW_preselection
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
        parameters: str = "bbww/analysis/metadata/object_preselection_run3.yaml",
        corrections_metadata: str = "src/physics/corrections.yml",
    ):
        self.parameters = parameters
        loaded_parameters = OmegaConf.load(self.parameters)
        self.params = OmegaConf.merge(corrections_metadata, loaded_parameters)

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
            not self.is_mc
        )         

        jets = apply_jerc_corrections(
            events,
            corrections_metadata=self.params[self.year],
            isMC=self.is_mc,
            run_systematics=False, ###self.run_systematics,
            dataset=self.dataset,
            jet_type='AK4PFPuppi.txt',
        )
        met = apply_met_corrections_after_jec(events, jets)

        shifts = [({"Jet": jets, "MET":met}, None)] 
        
        '''if systematics:
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

        events = add_gen_info(events, self.is_mc) # add gen particle information
        events = apply_bbWW_preselection(events, self.year, self.params, self.is_mc) #preselection
        events = candidate_selection(events, self.params, self.year) # select HH->bbWW candidates

        # apply selections before computing chi_square
        selection.add( "lumimask", events.lumimask) # apply lumimask on data
        selection.add( "passNoiseFilter", events.passNoiseFilter) # apply various noise filters
        selection.add("trigger", events.passHLT) # apply trigger selection
        selection.add('twoBjets', events.has_2_bjets) # require 2 b-tagged jets
        selection.add('njets',  ak.num(events.j_init[events.j_init.preselected],axis=1)>3) # at least 4 ak4 jets
        selection.add('oneBjet', events.has_1_bjet)
        selection.add('isoneEorM', events.e_region | events.mu_region )
        selection.add('tau_veto', (events.tau_nmedium==0))
        selection.add('nom_njets4',  ak.num(events.j_init[events.j_init.isnominal],axis=1)>3) # nominal pT region
        selection.add('nom_njets3',  ak.num(events.j_init[events.j_init.isnominal],axis=1)==3) # exact 3 jets region
        selection.add('lowpt_njets4', ~selection.all('nom_njets4') & (ak.num(events.j_init[events.j_init.preselected],axis=1)>3) )
        # veto events with jets affected by EE water leak (2022) and hole in Pixel L3/L4 (2023)  
        jet_veto_maps = (ak.all(events.Jet.jet_veto_maps,axis=1) if '202' in self.year 
                         else ak.ones_like(events.MET.pt,dtype=bool))
        selection.add('jet_veto_mask', jet_veto_maps)

        selection_list = {
            'basic_selection': ['lumimask', 'passNoiseFilter', 'trigger'],
            'preselection': ['lumimask', 'passNoiseFilter', 'trigger', 'njets','jet_veto_mask', 'isoneEorM', 'tau_veto','twoBjets'],
        }
        events['selection'] = ak.zip({
            'preselection': selection.all(*selection_list['preselection']),
            'nominal_4j2b': selection.all('nom_njets4'),
            'nominal_3j2b': selection.all('nom_njets3'),
            'lowpt_4j2b' :  selection.all('lowpt_njets4')
        })

        ## add weights
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
        weights = add_lepton_sfs(self.params, events, events.Electron, events.Muon, weights, self.year, self.is_mc)
        events['weight'] = weights.weight() 
        ##

        #study sequential cutflow (get weights and events after each cut)
        output = {}
        if not shift_name:
            # list below contains individual selections that we might wanna study
            full_sel_list = ['lumimask', 'passNoiseFilter', 'trigger', 'oneBjet', 'twoBjets', 'isoneEorM', 'tau_veto','jet_veto_mask']
            output['sequential_cutflow'] = {}
            output['sequential_cutflow'][events.metadata['dataset']] = get_sequential_cutflow(
                selection,
                events,
                full_sel_list
            )

        events = events[events.selection.preselection]
        events = chi_sq(events) # chi square selection and calculation
        events = chi_sq_cut(events)

        #add regions separated by chi square calculation
        selection = PackedSelection(dtype="uint64") # reset selection to match sliced events shapes
        selection.add('leptonic_W',  ak.firsts(events.sr_boolean) == 0)
        selection.add('hadronic_W',  ak.firsts(events.sr_boolean) == 1)
        selection.add('chi_sq', events.passChiSqTT & events.passChiSqLepW) # chi square cuts

        # add chi square cuts selection in each analysis region
        events['selection'] = ak.with_field(events['selection'], 
                                        events.selection.nominal_4j2b & selection.all('chi_sq'), 'chi_sq_nom_4j2b')
        events['selection'] = ak.with_field(events['selection'], 
                                        events.selection.nominal_3j2b & selection.all('chi_sq'), 'chi_sq_nom_3j2b')
        events['selection'] = ak.with_field(events['selection'], 
                                        events.selection.lowpt_4j2b & selection.all('chi_sq'), 'chi_sq_lowpt_4j2b')
         
        selection.add('isoneE', events.e_region) # no. of tight electrons = 1, loose muons = 0
        selection.add('isoneM', events.mu_region) # no. of tight muons =1, loose electrons = 0     

        events['channel'] = ak.zip({
            'hadronic_W': selection.all('hadronic_W'),
            'leptonic_W': selection.all('leptonic_W')
        }) 
        events['region'] = ak.zip({
            'e_region':  selection.all('isoneE'),
            'mu_region': selection.all('isoneM')
        }) # separate electron and muon regions

        events = gen_studies(events, self.is_mc) # gen particle studies for MC

        if not shift_name:
            output['events_processed'] = {}
            output['events_processed'][events.metadata['dataset']] = {
                'n_events' : self.n_events,
                'sum_genweights': np.sum(events.genWeight) if self.is_mc else self.n_events,
            }
            output['cutflow_weights'] = {}
            output = add_output_cutflow(events, output)

        hists = fill_histograms(
            events,
            processName=self.processName,
            year=self.year_label,
            is_mc=self.is_mc,
            selection_list=['preselection','chi_sq_nom_4j2b','chi_sq_nom_3j2b',
                            'chi_sq_lowpt_4j2b','nominal_4j2b','nominal_3j2b','lowpt_4j2b'],
            channel_list=['hadronic_W', 'leptonic_W'],
            region_list = ['e_region', 'mu_region']
        )

        return hists | output
    
    def postprocess(self, accumulator):
        return accumulator