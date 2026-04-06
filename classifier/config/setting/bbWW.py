from enum import IntEnum
from src.classifier.task import GlobalSetting

class InputBranch(GlobalSetting):
    "Name of branches in the input root file"
    feature_bJetCand: list[str] = ["pt", "eta", "phi", "mass", "btagScore"]  
    feature_nonbJetCand: list[str] = ["pt", "eta", "phi", "mass", "attn_score"]
    feature_leadingLep: list[str] = ["pt", "eta", "phi", "mass", "isE", "isM"]
    feature_ancillary: list[str] = ["HT", "njets", "nsoftjets", "year"]
    feature_true_nbjet_flat: list[str] = ["0", "1", "2", "3"]
    feature_regressed_nu: list[str] = ["px", "py", "pz", "E"]
    nbJetCand: int = 2
    nnonbJetCand: int = 4 

    @classmethod
    def get__feature_bJetCand(cls, var: list[str]):
        return [f"bJetCand_{f}" for f in var]

    @classmethod
    def get__feature_nonbJetCand(cls, var: list[str]):
        return [f"nonbJetCand_{f}" for f in var]

    @classmethod
    def get__feature_leadingLep(cls, var: list[str]):
        return [f"leadingLep_{f}" for f in var]

    @classmethod
    def get__feature_ancillary(cls, var: list[str]):
        return var.copy()

    @classmethod
    def get__feature_true_nbjet_flat(cls, var: list[str]):
        return [f"true_nbjet_flat_{f}" for f in var]

    @classmethod
    def get__feature_regressed_nu(cls, var: list[str]):
        return [f"regressed_nu_{f}" for f in var]


class Input(GlobalSetting):
    "Name of the keys in the input batch."
    label: str = "label"
    weight: str = "weight"
    true_nbjet_flat: str = "true_nbjet_flat"
    bJetCand: str = "bJetCand"
    nonbJetCand: str = "nonbJetCand" 
    leadingLep: str = "leadingLep"
    ancillary: str = "ancillary"
    regressed_nu: str = "regressed_nu"

class Output(GlobalSetting):
    "Name of the keys in the output batch."
    hh_raw: str = "hh_raw"
    tt_raw: str = "tt_raw"
    ww_raw: str = "ww_raw"
    hh_prob: str = "hh_prob"
    tt_prob: str = "tt_prob"
    ww_prob: str = "ww_prob"
    
# create indeces map to different regions
class MassRegion(IntEnum):
    ALL = 0b00 # neither signal nor CR
    SR = 0b01
    CR = 0b10
