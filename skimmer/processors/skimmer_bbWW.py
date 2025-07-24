import logging

import numpy as np
import yaml
from omegaconf import OmegaConf
from analysis.helpers.mc_weight_outliers import OutlierByMedian
from bbww.analysis.helpers.object_selection import muon_selection, electron_selection
from base_class.physics.event_selection import apply_event_selection
from coffea.analysis_tools import PackedSelection, Weights
from skimmer.processor.picoaod import PicoAOD


class Skimmer(PicoAOD):
    def __init__(
        self, 
        corrections_file: str = "analysis/metadata/corrections.yml",
        params_file: str = "bbww/analysis/metadata/object_preselection.yaml",
        mc_outlier_threshold:int|None=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        corrections = yaml.safe_load(open(corrections_file, 'r'))
        parameters = OmegaConf.load(params_file)
        self.params = OmegaConf.merge(corrections, parameters)
        self.mc_outlier_threshold = mc_outlier_threshold


    def select(self, event):

        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        self.is_mc = hasattr(event, "genWeight")

        #
        # Set process and datset dependent flags
        #
        event = muon_selection(event, year, self.params)
        event = electron_selection(event, year, self.params)
        event = apply_event_selection( event, self.params[year], cut_on_lumimask= not self.is_mc )

        oneE =(event.e_nloose==1) & (event.mu_nloose==0)
        oneM = (event.mu_nloose==1) & (event.e_nloose==0)
        
        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add("trigger", event.passHLT)
        selections.add('isoneEorM', oneE|oneM )
        final_selection = selections.require(lumimask=True, passNoiseFilter=True, trigger = True, isoneEorM = True)

        weights = Weights(len(event), storeIndividual=True)
        if self.is_mc:
            weights.add( "genweight_", event.genWeight )
        event["weight"] = weights.weight()

        self._cutFlow.fill( "all", event, allTag=True )
        cumulative_cuts = []
        for cut in selections.names:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        processOutput = {}

        return final_selection, None, processOutput


    def preselect(self, event):
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        if self.mc_outlier_threshold is not None and "genWeight" in event.fields:
            return OutlierByMedian(self.mc_outlier_threshold)(event.genWeight)
