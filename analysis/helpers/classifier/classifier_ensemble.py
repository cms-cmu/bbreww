import re
import fsspec
from typing import TypedDict

import torch
import torch.nn.functional as F
import awkward as ak
import numpy.typing as npt
from src.classifier.config.model._kfold import _find_models
from bbreww.classifier.config.setting.bbWWHCR import Input
from src.classifier.config.setting.ml import KFold, SplitterKeys
from src.classifier.ml import BatchType
from src.classifier.ml.skimmer import Splitter
from bbreww.classifier.nn.blocks.bbWW_models import HCR

class RECModelMetadata(TypedDict):
    path: str
    name: str

class _RECKFoldModel:
    def __init__(self, model: str, splitter: Splitter, **_):
        self.splitter = splitter
        with fsspec.open(model, "rb") as f:
            states = torch.load(f, map_location=torch.device("cpu"))
        self.ancillary = states["input"]["feature_ancillary"]

        self._classes: list[str] = states["label"]
        self._reindex: list[int] = None
        self._model = HCR(
            dijetFeatures=states["arch"]["n_features"],
            ancillaryFeatures=self.ancillary,
            nClasses=len(self._classes),
            device="cpu",
        )
        self._model.to("cpu")
        self._model.load_state_dict(states["model"])
        self._model.eval()

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        if set(value) <= set(self._classes):
            if value != self._classes:
                self._reindex = [self._classes.index(c) for c in value]
        else:
            raise ValueError(
                f"HCR evaluation: classes mismatch, unknown classes: {set(value) - set(self._classes)}"
            )

    @property
    def eval(self):
        return self

    def __call__(self, b, nb, l, nu, a):
        hh_logits, tt_logits = self._model(b, nb, l, nu, a)
        if self._reindex is not None:
            hh_logits = hh_logits[:, self._reindex]
        return hh_logits, tt_logits

class RECEnsemble:
    _year_pattern = re.compile(r"\w*(?P<year>\d{2}).*")

    def __init__(self, paths: list[RECModelMetadata]):
        self.models = [
            _RECKFoldModel(**metadata)
            for metadata in _find_models((path["name"], path["path"]) for path in paths)
        ]
        self.classes = self.models[0].classes
        self.ancillary = self.models[0].ancillary
        for model in self.models:
            for k in ("ancillary",):
                if getattr(self, k) != getattr(model, k):
                    raise ValueError(
                        f"HCR evaluation: {k} mismatch, expected {getattr(self, k)} got {getattr(model, k)}"
                    )
            model.classes = self.classes

    @classmethod
    def get_year(cls, year: str):
        if match := cls._year_pattern.fullmatch(year):
            return float(match.group("year"))
        else:
            raise ValueError(f"Invalid year: {year}")

    @torch.no_grad()
    def __call__(self, events: ak.Array) -> tuple[npt.NDArray, npt.NDArray]:
        n = len(events)
        batch: BatchType = {
            Input.bJetCand: torch.zeros(n, 5, 2, dtype=torch.float32),
            Input.nonbJetCand: torch.zeros(n, 4, 2, dtype=torch.float32),
            Input.leadingLep: torch.zeros(n, 6, 1, dtype=torch.float32),
            Input.MET: torch.zeros(n, 2, 1, dtype=torch.float32),
            Input.ancillary: torch.zeros(n, len(self.ancillary), dtype=torch.float32),
        }

        b = batch[Input.bJetCand] # bjet (b) features
        for i, k in enumerate(("pt", "eta", "phi", "mass", "btagScore")):
            b[:, i, :] = torch.tensor(events.b_cands[k])

        nb = batch[Input.nonbJetCand] # non-bjet (nb) features
        for i, k in enumerate(("pt", "eta", "phi", "mass")):
            # q_cands_nom are nominal analysis jets, for low_pt analysis need q_cands_soft in low_pt selection region
            nb[:, i, :] = torch.tensor(events.q_cands_nom[k])

        l = batch[Input.leadingLep]
        for i, k in enumerate(("pt", "eta", "phi", "mass", "is_e", "is_mu" )): # to do: add is_M here with new classifier inputs
            if 'is' in k:
                k = k.split('_')[1]
                l[:, i, :] = torch.tensor(ak.singletons(events.flavor[k]))
            else:
                l[:, i, :] = torch.tensor(ak.singletons(events.leading_lep[k]))

        nu = batch[Input.MET]
        for i, k in enumerate(("pt", "phi",)):
            nu[:, i, :] = torch.tensor(ak.singletons(events.MET[k]))

        # ancillary features
        a = batch[Input.ancillary]
        for i, k in enumerate(self.ancillary):
            match k:
                case "year":
                    a[:, i] = self.get_year(events.metadata["year"])
                case "njets":
                    a[:, i] = torch.tensor(events.njets)
                case "nsoftjets":
                    a[:, i] = torch.tensor(events.nsoftjets)
                case "HT":
                    a[:, i] = torch.tensor(events.HT)
        # event offset
        batch[KFold.offset] = torch.from_numpy(events.event.to_numpy().view("int64"))

        hh_logits = torch.zeros(n, len(self.classes), dtype=torch.float32)
        tt_logits = torch.zeros(n, 2, dtype=torch.float32)
        for model in self.models:
            mask = model.splitter.split(batch)[SplitterKeys.validation] #splitter knows which model to evaluate on which file
            if mask.sum() > 0:
                hh_logits[mask], tt_logits[mask] = model(b[mask], nb[mask], l[mask], nu[mask], a[mask])

        return F.softmax(hh_logits, dim=-1).numpy(), F.softmax(tt_logits, dim=-1).numpy()
