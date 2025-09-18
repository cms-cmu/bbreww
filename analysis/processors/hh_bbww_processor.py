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

from bbreww.analysis.helpers.common import update_events, add_lepton_sfs
from bbreww.analysis.helpers.chi_square import chi_sq, chi_sq_cut
from bbreww.analysis.helpers.cutflow import cutflow_bbWW
from bbreww.analysis.helpers.dump_friendtrees import dump_input_friend
from bbreww.analysis.helpers.corrections import apply_met_corrections_after_jec
from bbreww.analysis.helpers.object_selection import apply_bbWW_preselection, apply_mll_cut
from bbreww.analysis.helpers.candidate_selection import candidate_selection, hbb_candidate_selection
from bbreww.analysis.helpers.gen_process import gen_process, add_gen_info, gen_studies
from bbreww.analysis.helpers.fill_histograms import fill_histograms

warnings.filterwarnings("ignore", "Missing cross-reference index for")
warnings.filterwarnings("ignore", "Please ensure")
warnings.filterwarnings("ignore", "invalid value")

vector.register_awkward()

def add_to_selection(cut_name, cut, selections, mask):
    presel_mask = selections.all(*mask)
    # Convert to numpy and replace None with False
    presel_mask_np = ak.to_numpy(ak.fill_none(presel_mask, False))
    cut_np = ak.to_numpy(ak.fill_none(cut, False))
    full_mask = np.zeros(len(presel_mask_np), dtype=bool)
    full_mask[presel_mask_np] = cut_np
    selections.add(cut_name, full_mask)


class analysis(processor.ProcessorABC):
    def __init__(
        self,
        parameters: str = "bbreww/analysis/metadata/object_preselection_run3.yaml",
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
        self.make_classifier_input = False

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

        # Instantiate cutflow_bbWW
        cutflow = cutflow_bbWW(selections=selection)

        events = add_gen_info(events, self.is_mc) # add gen particle information
        events = apply_bbWW_preselection(events, self.year, self.params, self.is_mc) #preselection
        events = apply_mll_cut(events)

        # apply selections before computing chi_square
        selection.add( "lumimask", events.lumimask) # apply lumimask on data
        selection.add( "passNoiseFilter", events.passNoiseFilter) # apply various noise filters
        selection.add("trigger", events.passHLT) # apply trigger selection
        selection.add('twoBjets', events.has_2_bjets) # require 2 b-tagged jets
        selection.add('njets',  events.has_3_presel_jets) # at least 3 ak4 jets
        selection.add('oneBjet', events.has_1_bjet)
        selection.add('oneE', events.e_region) # no. of tight electrons = 1, loose muons = 0
        selection.add('oneM', events.mu_region) # no. of tight muons =1, loose electrons = 0
        selection.add('oneEorM', events.e_region | events.mu_region )
        selection.add('tau_veto', (events.tau_nmedium==0))
        selection.add('mll_cut', events.pass_mll_cut)
        selection.add('njets_ak8', (events.n_ak8_jets == 0))
        selection.add('nom_njets4',  events.nom_njets4) # nominal pT region
        selection.add('nom_njets3',  events.nom_njets3) # exact 3 jets region
        selection.add('lowpt_njets4', ~selection.all('nom_njets4') & (events.has_4_presel_jets) )
        selection.add('lowpt_njets3', ~(selection.all('nom_njets4')) & (ak.num(events.j_init[events.j_init.preselected],axis=1)==3) )
        # veto events with jets affected by EE water leak (2022) and hole in Pixel L3/L4 (2023)
        jet_veto_maps = (ak.all(events.Jet.jet_veto_maps,axis=1) if '202' in self.year
                         else ak.ones_like(events.run,dtype=bool))
        selection.add('jet_veto_mask', jet_veto_maps)

        selection_list = {
            'preselection': ['lumimask', 'passNoiseFilter', 'trigger', 'njets','jet_veto_mask', 'oneEorM', 'tau_veto', 'mll_cut', 'njets_ak8', 'twoBjets' ],
        }
        selection_list['nominal_4j2b'] = selection_list['preselection'] + ['nom_njets4', 'twoBjets']
        selection_list['nominal_3j2b'] = selection_list['preselection'] + ['nom_njets3', 'twoBjets']
        selection_list['lowpt_4j2b'] = selection_list['preselection'] + ['lowpt_njets4', 'twoBjets']
        selection_list['lowpt_3j2b'] = selection_list['preselection'] + ['lowpt_njets3', 'twoBjets']

        events['preselection'] = selection.all(*selection_list['preselection'])
        events['nominal_4j2b'] = selection.all(*selection_list['nominal_4j2b'])
        events['nominal_3j2b'] = selection.all(*selection_list['nominal_3j2b'])
        events['lowpt_4j2b'] = selection.all('lowpt_njets4') & selection.all('twoBjets')
        events['lowpt_3j2b'] =  selection.all(*selection_list['lowpt_3j2b'])


        events['flavor'] = ak.zip({
            'e':  selection.all('oneE') & selection.all(*selection_list['preselection']),
            'mu': selection.all('oneM') & selection.all(*selection_list['preselection'])
        }) # separate electron and muon regions

        ## add weights
        weights, list_weight_names = add_weights(
            events,
            do_MC_weights=self.is_mc,
            dataset=self.dataset,
            year_label=self.year_label,
            apply_trigWeight=False,
            corrections_metadata=self.params[self.year],
        )
        weights = add_lepton_sfs(self.params, events, events.Electron, events.Muon, weights, self.year, self.is_mc)
        events['weight'] = weights.weight()
        ##
        signal_region = ((events.mbb > 75) & (events.mbb < 135)
                        & (events.bb_dr > 0.85) & (events.bb_dr < 2.15)) # elliptical signal region
        control_region = ((events.mbb > 55) & (events.mbb < 155)
                        & (events.bb_dr > 0.42) & (events.bb_dr < 2.58)
                        & ~signal_region) # sideband TTbar control region

        events['region'] = ak.zip({
            'SR': ak.fill_none(signal_region, False),
            'CR': ak.fill_none(control_region, False)
        })

        #study sequential cutflow (get weights and events after each cut)
        if not shift_name:
            # list below contains individual selections that we might wanna study
            full_sel_list = ['lumimask', 'passNoiseFilter', 'trigger','jet_veto_mask', 'oneEorM', 'oneBjet', 'twoBjets', 'tau_veto', 'njets_ak8', 'mll_cut']
            cumulative_cuts = []
            for cut_name in full_sel_list:
                cumulative_cuts.append(cut_name)
                cutflow.fill(events, cut_name,cumulative_cuts, weights.weight())

<<<<<<< HEAD
        selected_events = events[events.selection.preselection]

        selected_events = hbb_candidate_selection(selected_events, self.params, self.year) # select HH->bbWW candidates
        selected_events = candidate_selection(selected_events, self.params, self.year) # select HH->bbWW candidates



=======
        selected_events = events[events.preselection]
>>>>>>> origin/master
        del events
        selected_events = chi_sq(selected_events) # chi square selection and calculation
        selected_events = chi_sq_cut(selected_events) # add chi square cuts booleans


        #add regions separated by chi square calculation
        add_to_selection(
            'leptonic_W',
            (ak.firsts(selected_events.sr_boolean) == 0),
            selection,
            selection_list['preselection']
        )

        # hadronic_W = ak.zeros_like(presel_mask,dtype=bool)
        add_to_selection(
            'hadronic_W',
            ak.firsts(selected_events.sr_boolean) == 1,
            selection,
            selection_list['preselection']
        )

        # chi_sq = ak.zeros_like(presel_mask,dtype=bool)
        add_to_selection(
            'chi_sq',
            selected_events.passChiSqTT & selected_events.passChiSqLepW,
            selection,
            selection_list['preselection']
        )

        # add chi square cuts selection in each analysis region
        selected_events['chi_sq_nom_4j2b'] = selected_events.nominal_4j2b & selection.all('chi_sq')[selection.all(*selection_list['preselection'])]
        selected_events['chi_sq_nom_3j2b'] = selected_events.nominal_3j2b & selection.all('chi_sq')[selection.all(*selection_list['preselection'])]
        selected_events['chi_sq_lowpt_4j2b'] = selected_events.lowpt_4j2b & selection.all('chi_sq')[selection.all(*selection_list['preselection'])]

        selected_events['chi_sq_nom_4j2b']   = selected_events.selection.chi_sq_nom_4j2b
        selected_events['chi_sq_nom_3j2b']   = selected_events.selection.chi_sq_nom_3j2b
        selected_events['chi_sq_lowpt_4j2b'] = selected_events.selection.chi_sq_lowpt_4j2b

        selected_events['channel'] = ak.zip({
            'hadronic_W': selection.all('hadronic_W')[selection.all(*selection_list['preselection'])],
            'leptonic_W': selection.all('leptonic_W')[selection.all(*selection_list['preselection'])]
        })
        selected_events = gen_studies(selected_events, self.is_mc) # gen particle studies for MC
        analysis_selections = selection.all(*selection_list['nominal_4j2b']) & selection.all(*selection_list['preselection'])

        if self.make_classifier_input is not None:
            selev = selected_events[selected_events.nominal_4j2b]
            friends = { 'friends': {} }
            friends["friends"] = ( friends["friends"]
                | dump_input_friend(
                    selev,
                    'output/friendtrees',
                    "classifier_input",
                    analysis_selections,
                    weight = "weight"
                )
            )
        if not shift_name:
            output['events_processed'] = {}
            output['events_processed'][self.dataset] = {
                'n_events' : self.n_events,
                'sum_genweights': np.sum(selected_events.genWeight) if self.is_mc else self.n_events,
            }
            # add cuts for different regions
            cutflow_list = ['nominal_4j2b','nominal_3j2b', 'lowpt_4j2b', 'lowpt_3j2b', 'chi_sq_nom_4j2b', 'chi_sq_nom_3j2b', 'chi_sq_lowpt_4j2b']
            for cuts in cutflow_list:
                cutflow.fill(selected_events,cuts, [], selected_events.weight, fill_region = True)
            cutflow.add_output(output['events_processed'], self.dataset)


        # Hack for debuggging
        # selected_events = selected_events[selected_events.nominal_4j2b]


        hists = fill_histograms(
            selected_events,
            processName=self.processName,
            year=self.year_label,
            is_mc=self.is_mc,
            histCuts=['preselection',
                      'nominal_4j2b',    'nominal_3j2b',    'lowpt_4j2b', 'lowpt_3j2b'
                      'chi_sq_nom_4j2b', 'chi_sq_nom_3j2b', 'chi_sq_lowpt_4j2b',
                      ],
            channel_list=['hadronic_W', 'leptonic_W'],
            flavor_list=['e', 'mu']
        )

        return hists | output

    def postprocess(self, accumulator):
        return accumulator
