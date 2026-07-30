from src.classifier.task import Main, EntryPoint

class Main(Main):
    def run(self, parser: EntryPoint):
        import uproot
        import os
        
        # Check result
        result = "output/main_cache/chunks/label_sparse_dense.root"
        print(f"Verifying result file: {result}")
        if not os.path.exists(result):
            raise FileNotFoundError(f"Result file {result} does not exist!")
            
        with uproot.open(result) as f:
            tree = f["Events"]
            n = len(tree)
            print(f"Found {n} events in tree.")
            if n != 16384:
                raise ValueError(f"Expected 16384 events, but found {n}!")
            
            keys = tree.keys()
            print(f"Found branches: {keys}")
            expected = ["hh_prob", "tt_prob", "ww_prob"]
            for branch in expected:
                if branch not in keys:
                    raise KeyError(f"Expected branch '{branch}' is missing from the tree!")
                    
        print("Result file successfully verified!")
