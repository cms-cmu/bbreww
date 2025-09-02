import argparse
from coffea.util import load
import yaml
import numpy as np

def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dump cutflow to YAML')
    parser.add_argument('-i', '--inputFile', default='hists.pkl', help='Input coffea file')
    parser.add_argument('-o', '--outputFile', default='cutflow.yml', help='Output YAML file')
    args = parser.parse_args()

    with open(args.inputFile, 'rb') as infile:
        hists = load(infile)
    cutflow = hists['events_processed']

    cutflow_clean = convert_types(cutflow)

    with open(args.outputFile, 'w') as outfile:
        yaml.safe_dump(cutflow_clean, outfile, sort_keys=False, default_flow_style=False)