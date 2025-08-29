# Specialized subclass for 4b analysis
import numpy as np
import awkward as ak
import logging
from src.skimmer.cutflow import cutflow

########### IT IS NOT READY
########### NEEDS WORK

class cutflow_bbWW(cutflow):
    def __init__(self, selections=None, weights=None):
        self._cutflow_ele = {}
        self._cutflow_mu = {}
        self.selections = selections

    def get_sequential_cutflow(self, selection, events, selection_list):
        sequential_cutflow = {
            'events': {},
            'weights': {}
        }
        cumulative_cuts = []

        for cut_name in selection_list:
            cumulative_cuts.append(cut_name)
            current_mask = selection.all(*cumulative_cuts)
            sequential_cutflow['events'][cut_name] = np.sum(current_mask)
            sequential_cutflow['weights'][cut_name] = np.sum(events.weight[current_mask])

        return sequential_cutflow

    def add_output_cutflow(self, events, output):
        region_map = {
            'hadronic_W': events.channel.hadronic_W,
            'leptonic_W': events.channel.leptonic_W,
            'mu_region':  events.region.mu_region,
            'e_region':   events.region.e_region
        }
        output['cutflow'][events.metadata['dataset']] = {
            name: {
                'events': {
                    'preselection':      np.sum(events.selection.preselection[selector]),
                    'nominal_4j2b':      np.sum(events.selection.nominal_4j2b[selector]),
                    'nominal_3j2b':      np.sum(events.selection.nominal_3j2b[selector]),
                    'lowpt_4j2b':        np.sum(events.selection.lowpt_4j2b[selector]),
                    'chi_sq_nom_4j2b':   np.sum(events.selection.chi_sq_nom_4j2b[selector]),
                    'chi_sq_nom_3j2b':   np.sum(events.selection.chi_sq_nom_3j2b[selector]),
                    'chi_sq_lowpt_4j2b': np.sum(events.selection.chi_sq_lowpt_4j2b[selector])
                },
                'weights': {
                    'preselection':      np.sum(events.weight[events.selection.preselection[selector]]),
                    'nominal_4j2b':      np.sum(events.weight[events.selection.nominal_4j2b[selector]]),
                    'nominal_3j2b':      np.sum(events.weight[events.selection.nominal_3j2b[selector]]),
                    'lowpt_4j2b':        np.sum(events.weight[events.selection.lowpt_4j2b[selector]]),
                    'chi_sq_nom_4j2b':   np.sum(events.weight[events.selection.chi_sq_nom_4j2b[selector]]),
                    'chi_sq_nom_3j2b':   np.sum(events.weight[events.selection.chi_sq_nom_3j2b[selector]]),
                    'chi_sq_lowpt_4j2b': np.sum(events.weight[events.selection.chi_sq_lowpt_4j2b[selector]])
                }
            }
            for name, selector in region_map.items()
        }
        return output

    def fill(self, cut_name, cut_list, weight ):

        if 'oneE' in cut_list: cut_list.remove('oneE')
        if 'oneM' in cut_list: cut_list.remove('oneM')
        if self.selections is None or not hasattr(self.selections, 'names') or 'oneE' not in self.selections.names or 'oneM' not in self.selections.names:
            logging.error('Selections MUST have a oneE (electron) and oneM (muon) selection')

        if cut_name not in self._cutflow_ele:
            self._cutflow_ele[cut_name] = (0, 0)    # weighted, raw
            self._cutflow_mu[cut_name] = (0, 0)    # weighted, raw

        ele_cut = self.selections.all(*cut_list) & (self.selections.require(oneE=True))
        mu_cut  = self.selections.all(*cut_list) & (self.selections.require(oneM=True))

        self._cutflow_ele[cut_name] = (np.sum(ele_cut), np.sum(weight[ele_cut]))
        self._cutflow_mu[cut_name] = (np.sum(mu_cut), np.sum(weight[mu_cut]))

        logging.debug(f"Cutflow {cut_name}: Ele: {self._cutflow_ele[cut_name]}, Mu: {self._cutflow_mu[cut_name]}")


    def addOutput(self, *args, **kwargs):
        pass  # Stub or update as needed

    def addOutputSkim(self, o, dataset):

        o[dataset]['cutflow'] = {
            "e_region": { "events": {}, "weights": {} },
            "mu_region": { "events": {}, "weights": {} }
        }
        for k, v in self._cutflow_ele.items():
            o[dataset]["cutflow"]["e_region"]["events"][k] = float(v[0])
            o[dataset]["cutflow"]["e_region"]["weights"][k] = float(v[1])
        for k, v in self._cutflow_mu.items():
            o[dataset]["cutflow"]["mu_region"]["events"][k] = float(v[0])
            o[dataset]["cutflow"]["mu_region"]["weights"][k] = float(v[1])
