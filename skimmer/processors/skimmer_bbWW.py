import logging

import numpy as np
import yaml
from omegaconf import OmegaConf
from analysis.helpers.mc_weight_outliers import OutlierByMedian
from analysis.helpers.processor_config import processor_config
from bbww.analysis.helpers.object_selection import apply_bbWW_selection
from bbww.analysis.helpers.event_selection import apply_event_selection
from coffea.analysis_tools import PackedSelection, Weights
from skimmer.processor.picoaod import PicoAOD


class Skimmer(PicoAOD):
    def __init__(self, loosePtForSkim=False, skim4b=False, mc_outlier_threshold:int|None=200, *args, **kwargs):
        if skim4b:
            kwargs["pico_base_name"] = f'picoAOD_bbWW'
        super().__init__(*args, **kwargs)
        self.loosePtForSkim = loosePtForSkim
        self.corrections_metadata = yaml.safe_load(open('analysis/metadata/corrections.yml', 'r'))
        self.params = OmegaConf.load("bbww/analysis/metadata/object_preselection.yaml")
        self.mc_outlier_threshold = mc_outlier_threshold


    def select(self, event):

        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']

        #
        # Set process and datset dependent flags
        #
        config = processor_config(processName, dataset, event)
        logging.debug(f'config={config}\n')

        event = apply_bbWW_selection( event, year = year, params = self.params,isMC=config["isMC"],corrections_metadata=self.corrections_metadata[year])
        event = apply_event_selection( event, self.corrections_metadata[year], isMC = config["isMC"] )
  
        weights = Weights(len(event), storeIndividual=True)
        
        # general event weights
        
        if config["isMC"]:
            weights.add( "genweight_", event.genWeight )

        oneE =(event.e_ntight==1) & (event.mu_nloose==0) & (event.pho_nloose==0) & (event.tau_nloose==0)
        oneM = (event.mu_ntight==1) & (event.e_nloose==0) & (event.pho_nloose==0) & (event.tau_nloose==0)

        
        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add('isoneEorM', oneE|oneM )
        selections.add('njets',  (event.j_nsoft>2))
        final_selection = selections.require( lumimask=True, passNoiseFilter=True, isoneEorM = True, njets = True)

        event["weight"] = weights.weight()

        self._cutFlow.fill( "all",             event, allTag=True )
        cumulative_cuts = []
        for cut in selections.names:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        # debug_mask = ((event.event == 110614) & (event.run == 275890) & (event.luminosityBlock == 1))
        # debug_event = event[debug_mask]
        # print(f"debug {debug_event.fourTag} {debug_event.threeTag} {debug_event.nJet_tagged} {debug_event.nJet_tagged_loose} {debug_event.nJet_selected} {debug_event.Jet.tagged} {debug_event.Jet.selected} {debug_event.Jet.btagScore}")
        # print(f"debug {debug_event.passHLT} {debug_event.passJetMult} {debug_event.passPreSel} {debug_event.Jet.pt} {debug_event.Jet.pt_raw} \n\n\n")

        processOutput = {}
        #from analysis.helpers.write_debug_info import add_debug_Run3_data_skim
        #add_debug_Run3_data_skim(event, processOutput, selection)

        return final_selection, None, processOutput


    def preselect(self, event):
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        config = processor_config(processName, dataset, event)
        if config["isMC"] and self.mc_outlier_threshold is not None and "genWeight" in event.fields:
            return OutlierByMedian(self.mc_outlier_threshold)(event.genWeight)
