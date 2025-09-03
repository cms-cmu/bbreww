import logging

import awkward as ak
import numpy as np
import yaml
from omegaconf import OmegaConf
from src.skimmer.mc_weight_outliers import OutlierByMedian
from src.physics.event_selection import apply_event_selection
from src.skimmer.picoaod import PicoAOD
from bbreww.analysis.helpers.object_selection import electron_selection, muon_selection, jet_selection
from bbreww.analysis.helpers.candidate_selection import bjet_flag
from bbreww.analysis.helpers.cutflow import cutflow_bbWW
from coffea.analysis_tools import PackedSelection, Weights


class Skimmer(PicoAOD):
    def __init__(
            self, 
            corrections_metadata: dict = {},
            params_file: str = "bbreww/analysis/metadata/object_preselection_run3.yaml",
            mc_outlier_threshold:int|None=200, 
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        parameters = OmegaConf.load(params_file)
        self.params = OmegaConf.merge(corrections_metadata, parameters)
        self.mc_outlier_threshold = mc_outlier_threshold
        self._cutFlow = cutflow_bbWW()


    def select(self, event):

        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        self.is_mc = hasattr(event, "genWeight")

        #
        # Set process and datset dependent flags
        #
        event = muon_selection(event, self.params) #muons
        event = electron_selection(event, self.params) #electrons
        event = jet_selection(event, self.params, year)
        event = bjet_flag(event, self.params, year)  
        event = apply_event_selection( event, self.params[year], cut_on_lumimask= not self.is_mc )

        oneE =(event.e_ntight==1) & (event.mu_ntight==0)
        oneM = (event.mu_ntight==1) & (event.e_ntight==0)
        
        selections = PackedSelection()
        self._cutFlow.selections = selections
        selections.add('all', np.ones(len(event), dtype=bool))
        selections.add('lumimask', event.lumimask)
        selections.add('passNoiseFilter', event.passNoiseFilter)
        selections.add('trigger', event.passHLT)
        selections.add('oneE', oneE )
        selections.add('oneM', oneM )
        selections.add('isoneEorM', oneE|oneM )
        selections.add('oneBjet', event.has_1_bjet)
        selections.add('njets', ak.num(event.j_init, axis=1) > 2)
        final_selection = selections.require(
            lumimask=True,
            passNoiseFilter=True,
            trigger=True,
            njets=True,
            oneBjet=True,
            isoneEorM=True
        )

        weights = Weights(len(event), storeIndividual=True)
        self._cutFlow.weights = weights
        if self.is_mc:
            weights.add( "genweight_", event.genWeight )

        self._cutFlow.fill( "all", ['all'], weights.weight() )
        cumulative_cuts = []
        for cut in selections.names:
            if ('oneE' in cut) or ('oneM' in cut): continue
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, cumulative_cuts, weights.weight() )

        processOutput = {}

        return final_selection, None, processOutput


    def preselect(self, event):
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        if self.mc_outlier_threshold is not None and "genWeight" in event.fields:
            return OutlierByMedian(self.mc_outlier_threshold)(event.genWeight)
