from enum import IntEnum
from src.classifier.task import GlobalSetting

class InputBranch(GlobalSetting):
    "Name of branches in the input root file"
    feature_bJetCand: list[str] = ["pt", "eta", "phi", "mass", "btagScore"]  
    feature_nonbJetCand: list[str] = ["pt", "eta", "phi", "mass"]
    feature_leadingLep: list[str] = ["pt", "eta", "phi", "mass", "isE", "isM"]
    feature_MET: list[str] = ["pt", "phi"]
    feature_ancillary: list[str] = ["HT", "njets", "nsoftjets", "year"]
    feature_genNu: list[str] = ["pt", "eta", "phi"]
    feature_genLepW: list[str] = ["onShell", "genLepWmass"]
    nbJetCand: int = 2
    nnonbJetCand: int = 4 # nominal: 2, lowpt: 3

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
    def get__feature_MET(cls, var: list[str]):
        return [f"MET_{f}" for f in var]
    
    @classmethod
    def get__feature_ancillary(cls, var: list[str]):
        return var.copy()

    @classmethod
    def get__feature_genNu(cls, var: list[str]):
        return [f"genNu_{f}" for f in var]

    @classmethod
    def get__feature_genLepW(cls, var: list[str]):
        return [f"genLepW_{f}" for f in var]


class Input(GlobalSetting):
    "Name of the keys in the input batch."
    label: str = "label"
    weight: str = "weight"
    bJetCand: str = "bJetCand"
    nonbJetCand: str = "nonbJetCand" 
    leadingLep: str = "leadingLep"
    MET: str = "MET"
    ancillary: str = "ancillary"
    genNu: str = "genNu"
    genLepW: str = "genLepW"

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
