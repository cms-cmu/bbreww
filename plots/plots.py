import yaml

def load_config_bbWW(metadata):
    """  Load meta data
    """
    plotConfig = yaml.safe_load(open(metadata, 'r'))

    return plotConfig
