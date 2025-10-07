# Specialized subclass for 4b analysis
import numpy as np
import awkward as ak
import logging
from src.skimmer.cutflow import cutflow

class cutflow_bbWW(cutflow):
    def __init__(self, selections=None, weights=None):
        self._cutflow_ele = {}
        self._cutflow_mu = {}
        self._cutflow_SR = {}
        self._cutflow_CR= {}
        self.selections = selections

    def fill(self, events, cut_name, cut_list, weight, fill_flavour: bool = False, fill_region: bool = False, skim: bool = False):

        if 'oneE' in cut_list: cut_list.remove('oneE')
        if 'oneM' in cut_list: cut_list.remove('oneM')
        if self.selections is None or not hasattr(self.selections, 'names') or 'oneE' not in self.selections.names or 'oneM' not in self.selections.names:
            logging.error('Selections MUST have a oneE (electron) and oneM (muon) selection')

        if cut_name not in self._cutflow_ele:
            self._cutflow_ele[cut_name] = (0, 0)    # weighted, raw
            self._cutflow_mu[cut_name] = (0, 0)    # weighted, raw
            if fill_region:
                self._cutflow_SR[cut_name] = (0, 0)    # weighted, raw
                self._cutflow_CR[cut_name] = (0, 0)    # weighted, raw

        # fill with regions
        if fill_flavour:
            ele_cut = events[cut_name] & events.flavor.e
            mu_cut =  events[cut_name] & events.flavor.mu
            SR_cut =  events[cut_name] & events.region.SR
            CR_cut =  events[cut_name] & events.region.CR
        # fill with individual cuts
        else:
            ele_cut = self.selections.all(*cut_list) & (self.selections.require(oneE=True))
            mu_cut = self.selections.all(*cut_list) & (self.selections.require(oneM=True))
            #if not skim:
                #if fill_region:
                #    SR_cut = self.selections.all(*cut_list) #& (events.region.SR) # currently not separating CR and SR before these cuts 
                #    CR_cut = self.selections.all(*cut_list) #& (events.region.CR)

        self._cutflow_ele[cut_name] = (np.sum(ele_cut), np.sum(weight[ele_cut]))
        self._cutflow_mu[cut_name] = (np.sum(mu_cut), np.sum(weight[mu_cut]))
        if not skim:
            if fill_region:
                self._cutflow_SR[cut_name] = (np.sum(SR_cut), np.sum(weight[SR_cut]))
                self._cutflow_CR[cut_name] = (np.sum(CR_cut), np.sum(weight[CR_cut]))

        logging.debug(f"Cutflow {cut_name}: Ele: {self._cutflow_ele[cut_name]}, Mu: {self._cutflow_mu[cut_name]}")


    def add_output(self, o, dataset):
        o[dataset]['cutflow'] = {
        "e_region": { "events": {}, "weights": {} },
        "mu_region": { "events": {}, "weights": {} },
        "SR": { "events": {}, "weights": {} },
        "CR": { "events": {}, "weights": {} }
        }
        for k, v in self._cutflow_ele.items():
            o[dataset]["cutflow"]["e_region"]["events"][k] = float(v[0])
            o[dataset]["cutflow"]["e_region"]["weights"][k] = float(v[1])
        for k, v in self._cutflow_mu.items():
            o[dataset]["cutflow"]["mu_region"]["events"][k] = float(v[0])
            o[dataset]["cutflow"]["mu_region"]["weights"][k] = float(v[1])
        for k, v in self._cutflow_SR.items():
            o[dataset]["cutflow"]["SR"]["events"][k] = float(v[0])
            o[dataset]["cutflow"]["SR"]["weights"][k] = float(v[1])
        for k, v in self._cutflow_CR.items():
            o[dataset]["cutflow"]["CR"]["events"][k] = float(v[0])
            o[dataset]["cutflow"]["CR"]["weights"][k] = float(v[1])

    def addOutputSkim(self, o, dataset):

        o[dataset]['cutflow'] = {
            "e_region": { "events": {}, "weights": {} },
            "mu_region": { "events": {}, "weights": {} },
        }
        for k, v in self._cutflow_ele.items():
            o[dataset]["cutflow"]["e_region"]["events"][k] = float(v[0])
            o[dataset]["cutflow"]["e_region"]["weights"][k] = float(v[1])
        for k, v in self._cutflow_mu.items():
            o[dataset]["cutflow"]["mu_region"]["events"][k] = float(v[0])
            o[dataset]["cutflow"]["mu_region"]["weights"][k] = float(v[1])
