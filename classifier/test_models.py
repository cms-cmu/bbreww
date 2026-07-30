import sys
from pathlib import Path
import torch

# Add parent directory to sys.path so bbreww can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append("/srv/")

def test_models():
    print("Testing bbreww models...")
    
    ancillary = ["ancillary_" + str(i) for i in range(10)]
    
    # 1. Test GCN & bbWWBase
    from bbreww.classifier.nn.blocks.bbWW_models import GCN, bbWWBase
    print("Initializing bbWWBase model...")
    base_model = bbWWBase(dijetFeatures=16, ancillaryFeatures=ancillary, device="cpu")
    print(f"Initialized {base_model.name} successfully.")
    
    try:
        from bbreww.classifier.nn.blocks.bbWW_models import GCN
        print("Initializing GCN model...")
        gcn_model = GCN(dijetFeatures=16, ancillaryFeatures=ancillary, device="cpu")
        print(f"Initialized {gcn_model.name} successfully.")
    except ModuleNotFoundError as e:
        if "torch_geometric" in str(e):
            print("Skipping GCN model testing (torch_geometric is not installed).")
        else:
            raise
    
    # 2. Test METRegressor
    from bbreww.classifier.nn.blocks.met_pz_regressor import METRegressor
    print("Initializing METRegressor model...")
    met_model = METRegressor(dijetFeatures=16, ancillaryFeatures=ancillary, device="cpu")
    print(f"Initialized {met_model.name} successfully.")
    
    # 3. Test bbWW_3jet
    from bbreww.classifier.nn.blocks.bbWW_3jet import bbWW_3jet
    print("Initializing bbWW_3jet model...")
    model_3jet = bbWW_3jet(dijetFeatures=16, ancillaryFeatures=ancillary, device="cpu")
    print(f"Initialized {model_3jet.name} successfully.")
    
    # 4. Test bbWW_lowpt
    from bbreww.classifier.nn.blocks.bbWW_lowpt import bbWW_lowpt
    print("Initializing bbWW_lowpt model...")
    model_lowpt = bbWW_lowpt(dijetFeatures=16, ancillaryFeatures=ancillary, device="cpu")
    print(f"Initialized {model_lowpt.name} successfully.")
    
    print("All models initialized successfully!")

if __name__ == "__main__":
    test_models()
