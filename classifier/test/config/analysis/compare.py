from src.classifier.task import Analysis, ArgParser

def _verify_dataset(result: str, nevents: int):
    # Trigger rerun 2
    import uproot
    import os
    
    import glob
    
    print(f"Verifying result file: {result}")
    parent = os.path.dirname(result)
    if os.path.exists(parent):
        print(f"Files in {parent}: {glob.glob(os.path.join(parent, '*'))}")
    else:
        print(f"Directory {parent} does not exist!")
        
    if not os.path.exists(result):
        raise FileNotFoundError(f"Result file {result} does not exist!")
        
    with uproot.open(result) as f:
        tree = f["Events"]
        n = len(tree)
        print(f"Found {n} events in tree.")
        if n != nevents:
            raise ValueError(f"Expected {nevents} events, but found {n}!")
        
        keys = tree.keys()
        print(f"Found branches: {keys}")
        expected = ["hh_prob", "tt_prob", "ww_prob"]
        for branch in expected:
            if branch not in keys:
                raise KeyError(f"Expected branch '{branch}' is missing from the tree!")
                
    print("Result file successfully verified!")


class _Runner:
    def __init__(self, result: str, nevents: int):
        self.result = result
        self.nevents = nevents

    def __call__(self):
        _verify_dataset(self.result, self.nevents)


class VerifyOutput(Analysis):
    argparser = ArgParser()
    argparser.add_argument("--result", type=str, required=True)
    argparser.add_argument("--nevents", type=int, default=8192)

    def analyze(self, results):
        return [_Runner(self.opts.result, self.opts.nevents)]
