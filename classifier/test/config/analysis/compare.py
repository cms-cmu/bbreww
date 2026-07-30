from src.classifier.task import Analysis, ArgParser

class VerifyOutput(Analysis):
    argparser = ArgParser()
    argparser.add_argument("--result", type=str, required=True)
    argparser.add_argument("--nevents", type=int, default=8192)

    def analyze(self, results):
        return [self._verify]

    def _verify(self):
        import uproot
        import os
        
        print(f"Verifying result file: {self.opts.result}")
        if not os.path.exists(self.opts.result):
            raise FileNotFoundError(f"Result file {self.opts.result} does not exist!")
            
        with uproot.open(self.opts.result) as f:
            tree = f["Events"]
            n = len(tree)
            print(f"Found {n} events in tree.")
            if n != self.opts.nevents:
                raise ValueError(f"Expected {self.opts.nevents} events, but found {n}!")
            
            keys = tree.keys()
            print(f"Found branches: {keys}")
            expected = ["hh_prob", "tt_prob", "ww_prob"]
            for branch in expected:
                if branch not in keys:
                    raise KeyError(f"Expected branch '{branch}' is missing from the tree!")
                    
        print("Result file successfully verified!")
