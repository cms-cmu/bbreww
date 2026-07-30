from functools import cache, cached_property
from src.classifier.config.setting.df import Columns
from bbreww.classifier.config.setting.bbWW import Input, InputBranch
from src.classifier.config.setting.ml import KFold
from src.classifier.task import ArgParser
from bbreww.classifier.test.config.dataset._root import LoadGroupedRootForTest

class SimplifiedTrain(LoadGroupedRootForTest):
    trainable = True
    argparser = ArgParser()

    def __init__(self):
        super().__init__()
        from src.classifier.df.io import ToTensor

        self._to_tensor = ToTensor()
        (
            self._to_tensor
            .add(KFold.offset, KFold.offset_dtype).columns(Columns.event)
            .add(Input.label, Columns.index_dtype).columns(Columns.label_index)
            .add(Input.weight, "float32").columns(Columns.weight)
            .add(Input.ancillary, "float32").columns(*InputBranch.feature_ancillary)
            .add(Input.bJetCand, "float32").columns(*InputBranch.feature_bJetCand, target=InputBranch.nbJetCand)
            .add(Input.nonbJetCand, "float32").columns(*InputBranch.feature_nonbJetCand, target=InputBranch.nnonbJetCand, pad_value=-1)
            .add(Input.leadingLep, "float32").columns(*InputBranch.feature_leadingLep)
            .add(Input.regressed_nu, "float32").columns(*InputBranch.feature_regressed_nu)
            .add(Input.true_nbjet_flat, "float32").columns("true_nbjet_flat", target=4, pad_value=0)
        )

    @cached_property
    def _branches(self):
        # Flatten all feature lists to get the full branch list
        branches = set().union(
            InputBranch.feature_ancillary,
            [f"bJetCand_{f}_{i}" for f in InputBranch.feature_bJetCand for i in range(InputBranch.nbJetCand)],
            [f"nonbJetCand_{f}_{i}" for f in InputBranch.feature_nonbJetCand for i in range(InputBranch.nnonbJetCand)],
            [f"leadingLep_{f}" for f in InputBranch.feature_leadingLep],
            [f"regressed_nu_{f}" for f in InputBranch.feature_regressed_nu],
            [f"true_nbjet_flat_{f}" for f in InputBranch.feature_true_nbjet_flat],
            [Columns.weight, Columns.event, "label"],
        )
        return branches

    @cache
    def from_root(self, groups: frozenset[str]):
        from src.classifier.df.io import FromRoot
        from src.classifier.df.tools import rename_columns
        
        # Add label_index column mapping
        pres = [
            rename_columns(label=Columns.label_index)
        ]
        pres.extend(self.preprocessors)

        return FromRoot(
            branches=self._branches.intersection,
            preprocessors=pres,
        )

class SimplifiedEval(SimplifiedTrain):
    trainable = False
    evaluable = True

    def __init__(self):
        super().__init__()
        self.preprocessors.clear()
        self.to_tensor.remove(Input.label).remove(Input.weight)

    @cached_property
    def _branches(self):
        return super()._branches - {Columns.weight}
