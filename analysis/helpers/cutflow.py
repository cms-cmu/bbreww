# Specialized subclass for 4b analysis
import numpy as np
import awkward as ak
from src.skimmer.cutflow import cutflow

########### IT IS NOT READY
########### NEEDS WORK

class cutflow_bbWW(cutflow):
    def __init__(self):
        self._cutflow_ele = {}
        self._cutflow_mu = {}

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

    def fill(self, *args, **kwargs):
        pass  # Stub or update as needed

    def addOutput(self, *args, **kwargs):
        pass  # Stub or update as needed

    def addOutputSkim(self, *args, **kwargs):
        pass  # Stub or update as needed
