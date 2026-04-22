from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

import fsspec
import torch
import torch.nn.functional as F
import torch.types as tt
from src.classifier.config.scheduler import SkimStep
from src.classifier.config.setting.ml import SplitterKeys
from src.classifier.config.state.label import MultiClass
from torch import Tensor

from bbreww.classifier.nn.blocks.bbWW_lowpt import HCR_lowpt
from bbreww.classifier.config.setting.bbWW import Input, InputBranch, Output
from src.classifier.algorithm.utils import Selector, map_batch, to_num
from src.classifier.nn.schedule import MilestoneStep, Schedule
from src.classifier.utils import MemoryViewIO
from src.classifier.ml import BatchType
from src.classifier.ml.benchmarks.multiclass import ROC
from src.classifier.ml.evaluation import Evaluation, EvaluationStage
from src.classifier.ml.skimmer import Skimmer, Splitter
from src.classifier.ml.training import (
    BenchmarkStage,
    Model,
    MultiStageTraining,
    OutputStage,
    TrainingStage,
)

if TYPE_CHECKING:
    from src.storage.eos import PathLike


@dataclass
class HCRArch:
    __skip_save = frozenset(("loss",))

    loss: Callable[[BatchType], Tensor] = None
    n_features: int = 12
    attention: bool = True

    @classmethod
    def load(cls, saved: dict[str]):
        obj = cls()
        for k, v in saved.items():
            if k in cls.__annotations__:
                setattr(obj, k, v)
        return obj

    def save(self):
        return {
            k: getattr(self, k)
            for k in self.__annotations__
            if k not in self.__skip_save
        }


@dataclass
class GBNSchedule(MilestoneStep):
    n_batches: int = 64
    milestones: list[int] = (1, 3, 6, 10, 21, 22, 23)
    gamma: float = 0.5

    def __post_init__(self):
        self.milestones = sorted(self.milestones)
        self._last_bs = self.n_batches
        self.reset()

    def get_bs(self):
        self._last_bs = max(1, int(self.n_batches * (self.gamma**self.milestone)))
        return self._last_bs

    def get_last_bs(self):
        return self._last_bs


@dataclass
class HCRBenchmarks:
    rocs: Iterable[ROC]
    scalars: Iterable[Callable[[BatchType], dict[str, Tensor]]] = None


def _HCRInput(batch: BatchType, device: tt.Device, selection: Tensor = None):
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=True)
    inputs = [batch.pop(k) for k in (Input.bJetCand, Input.nonbJetCand, Input.leadingLep, Input.ancillary, Input.regressed_nu)]
    if selection is not None:
        selection = selection.to(device, non_blocking=True)
        inputs = [i[selection] for i in inputs]
    return inputs


class _HCRSkim(Skimmer):
    def __init__(
        self,
        nn: HCR,
        device: tt.Device,
        splitter: Splitter,
    ):
        self._nn = nn
        self._device = device
        self._splitter = splitter

    @torch.no_grad()
    def train(self, batch: BatchType):
        selections = self._splitter.step(batch)
        if self._nn is not None and selections[SplitterKeys.training].sum() > 0:
            self._nn.updateMeanStd(
                *_HCRInput(batch, self._device, selections[SplitterKeys.training])
            )
        return super().train(batch)


class HCRModel(Model):
    def __init__(
        self,
        device: tt.Device,
        arch: HCRArch,
        benchmarks: HCRBenchmarks,
    ):
        self._loss = arch.loss
        self._device = device
        self._gbn = None
        self._arch = arch
        self._nn = HCR_lowpt(
            dijetFeatures=arch.n_features,
            ancillaryFeatures=InputBranch.feature_ancillary,
            device=device,
            nClasses=MultiClass.n_trainable(),
        )
        self._benchmarks = benchmarks
        self._shap_results = None
    
    @property
    def ghost_batch(self):
        return self._gbn

    @ghost_batch.setter
    def ghost_batch(self, gbn: GBNSchedule):
        self._gbn = gbn
        if gbn is None:
            self._nn.setGhostBatches(0, False)
        else:
            self._gbn.reset()
            self._nn.setGhostBatches(self._gbn.n_batches, False)

    @property
    def hyperparameters(self) -> dict[str]:
        return {
            "n ghost batch": (
                self.ghost_batch.get_last_bs() if self.ghost_batch is not None else 0
            )
        }

    @property
    def nn(self):
        return self._nn

    def train(self, batch: BatchType, compute_shap: bool = False) -> Tensor:
        hh, tt, ww = self._nn(*_HCRInput(batch, self._device))
        batch[Output.hh_raw] = hh
        batch[Output.tt_raw] = tt
        batch[Output.ww_raw] = ww
        batch[Output.ww_weights] = self._nn._jet_weights  # (n, heads, 1, wsl) per-jet attention weights

        loss = self._loss(batch)

        return loss
    
    def validate(self, batches: Iterable[BatchType]) -> dict[str]:
        weight = 0.0
        scalars = defaultdict(float)
        scalar_funcs = self._benchmarks.scalars
        rocs = [r.copy() for r in self._benchmarks.rocs]

        for batch in batches:
            hh, tt, ww = self._nn(*_HCRInput(batch, self._device))
            batch |= {
                Output.hh_raw: hh,
                Output.tt_raw: tt,
                Output.ww_raw: ww,
                Output.hh_prob: F.softmax(hh, dim=1),
                Output.tt_prob: F.softmax(tt, dim=1),
                Output.ww_prob: F.softmax(ww, dim=1),
                Output.ww_weights: self._nn._jet_weights,
            }
            sumw = to_num(batch[Input.weight].sum())
            if scalar_funcs is None:
                if MultiClass.n_nontrainable() == 0:
                    scalars["loss"] += to_num(self._loss(batch)) * sumw
            else:
                for func in scalar_funcs:
                    measured = func(batch)
                    for name, value in measured.items():
                        scalars[name] += to_num(value) * sumw
            weight += sumw
            for roc in rocs:
                roc.update(batch)
        for k in scalars:
            scalars[k] /= weight

        # Binned Asimov significance from the accumulated ROC histograms.
        # Each ROC's __TP / __FP store per-bin weighted signal / background yields
        # (weights = batch[Input.weight]). For the "Signal vs Background" ROC
        # (pos=signal, neg=~signal) this gives Z_A for signal vs everything-else.
        # Must run BEFORE to_json(), which calls .roc() and resets the accumulators.
        asimov = {}
        for roc in rocs:
            tp_buf = getattr(roc, "_FixedThresholdROC__TP", None)
            fp_buf = getattr(roc, "_FixedThresholdROC__FP", None)
            if tp_buf is None or fp_buf is None:
                continue  # ROC never saw any events
            tp, _ = tp_buf.hist()
            fp, _ = fp_buf.hist()
            s = tp.detach().to(dtype=torch.float64, device="cpu")
            b = fp.detach().to(dtype=torch.float64, device="cpu")
            mask = b > 1e-6                      # drop empty / ultra-low-stat bins
            if not bool(mask.any()):
                continue
            s_m, b_m = s[mask], b[mask]
            z_sq = torch.sum(2.0 * ((s_m + b_m) * torch.log1p(s_m / b_m) - s_m))
            z_a = float(torch.sqrt(torch.clamp(z_sq, min=0.0)).item())
            z_naive = float(torch.sqrt(torch.sum(s_m ** 2 / b_m)).item())
            s_tot = float(s_m.sum().item())
            b_tot = float(b_m.sum().item())
            asimov[roc._name] = {
                "Z_A": z_a,
                "S_over_sqrtB": z_naive,
                "S_total": s_tot,
                "B_total": b_tot,
            }
            logging.info(
                f"  Z_A[{roc._name}] = {z_a:.4f}   "
                f"S/sqrt(B) = {z_naive:.4f}   "
                f"(S={s_tot:.3g}, B={b_tot:.3g})"
            )

        return {"scalars": scalars,
                "roc": [r.to_json() for r in rocs],
                "asimov": asimov,
                "shap": self._shap_results
                }

    ## work in progress feature
    def _compute_shap_gradient(self, batch):
        """Compute SHAP using gradients during training"""
        import shap
        import numpy as np
        
        all_feature_names = (
            [f"bJet_{f}" for f in InputBranch.feature_bJetCand] +
            [f"nonbJet_{f}" for f in InputBranch.feature_nonbJetCand] +
            [f"lep_{f}" for f in InputBranch.feature_leadingLep] +
            list(InputBranch.feature_ancillary) +
            list(InputBranch.feature_regressed_nu)
        )
        
        # Wrapper model for SHAP
        class ConcatenatedModel(torch.nn.Module):
            def __init__(self, original_model, n_features=[10, 8, 5, 2, 3]):
                super().__init__()
                self.model = original_model
                self.n_features = n_features
            
            def forward(self, X):
                splits = torch.split(X, self.n_features, dim=1)
                hh, *_ = self.model(*splits)
                signal_idx = MultiClass.trainable_labels.index("signal")
                probs = F.softmax(hh, dim=1)[:, signal_idx]
                return probs.unsqueeze(1)
        
        wrapped_model = ConcatenatedModel(self._nn)
        
        # Use current batch as test data
        inputs = _HCRInput(batch, self._device)
        test_data = torch.cat(inputs, dim=1)[:500]  # Sample subset
        
        # Use subset as background
        background = test_data[:100]
        explainer = shap.GradientExplainer(wrapped_model, background)
        shap_values = explainer.shap_values(test_data)
        
        # Aggregate importance
        importance = {}
        feature_importance = np.abs(shap_values).mean(axis=0)
        for i, name in enumerate(all_feature_names):
            importance[name] = float(feature_importance[i])
        
        return importance

    def step(self, epoch: int = None):
        if self.ghost_batch is not None and self.ghost_batch.step(epoch):
            self._nn.setGhostBatches(self.ghost_batch.get_bs(), False)


class HCRTraining(MultiStageTraining):
    def __init__(
        self,
        arch: HCRArch,
        ghost_batch: GBNSchedule,
        cross_validation: Splitter,
        training_schedule: Schedule,
        finetuning_schedule: Schedule = None,
        benchmarks: HCRBenchmarks = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._arch = arch
        self._ghost_batch = ghost_batch
        self._splitter = cross_validation
        self._training = training_schedule
        self._finetuning = finetuning_schedule
        self._benchmarks = benchmarks or HCRBenchmarks()
        self._HCR: HCRModel = None

    def stages(self):
        self._HCR = HCRModel(
            device=self.device,
            arch=self._arch,
            benchmarks=self._benchmarks,
        )
        self._HCR.ghost_batch = self._ghost_batch
        self._HCR.to(self.device)
        self._splitter.setup(self.dataset)
        skim = _HCRSkim(self._HCR._nn, self.device, self._splitter)
        yield TrainingStage(
            name="Initialization",
            model=skim,
            schedule=SkimStep(),
            training=self.dataset,
        )
        self._HCR.nn.initMeanStd()
        validation_sets = self._splitter.get()
        training_set = validation_sets[SplitterKeys.training]
        yield BenchmarkStage(
            name="Baseline",
            model=self._HCR,
            validation=validation_sets,
        )
        yield TrainingStage(
            name="Training",
            model=self._HCR,
            schedule=self._training,
            training=training_set,
            validation=validation_sets,
        )
        self._HCR.ghost_batch = None
        if self._finetuning is not None:
            layers = self._HCR._nn.layers
            layers.setLayerRequiresGrad(
                requires_grad=False, index=self._HCR._nn.embedding_layers()
            )
            yield TrainingStage(
                name="Finetuning",
                model=self._HCR,
                schedule=self._finetuning,
                training=training_set,
                validation=validation_sets,
            )
            self._HCR.ghost_batch = self._ghost_batch
            layers.setLayerRequiresGrad(requires_grad=True)
        output_stage = OutputStage(name="Final", path=f"{self.name}__{self.uuid}.pkl")
        output_path = output_stage.absolute_path
        if not output_path.is_null:
            logging.info(f"Saving model to {output_path}")
            with fsspec.open(output_path, "wb") as f:
                torch.save(
                    {
                        "model": self._HCR.nn.state_dict(),
                        "metadata": self.metadata,
                        "uuid": self.uuid,
                        "label": MultiClass.trainable_labels,
                        "arch": self._arch.save(),
                        "input": {
                            k: getattr(InputBranch, k)
                            for k in (
                            "feature_ancillary",
                            "feature_bJetCand",
                            "feature_nonbJetCand",
                            "feature_leadingLep",
                            "feature_regressed_nu",
                            )
                        },
                    },
                    MemoryViewIO(f),
                )
            yield output_stage


class HCRModelEval(Model):
    def __init__(
        self,
        device: tt.Device,
        saved: dict[str],
        splitter: Splitter,
        mapping: Callable[[BatchType], BatchType],
    ):
        self._device = device
        self._splitter = splitter
        self._mapping = mapping
        self._classes = saved["label"]
        for k in saved["input"].keys():
            if getattr(InputBranch, k) != saved["input"][k]:
                raise ValueError(
                    f'Input features "{k}" mismatch: training={saved["input"][k]}, evaluation={getattr(InputBranch, k)}'
                )
        self._arch = HCRArch.load(saved["arch"])
        self._nn = HCR_lowpt(
            dijetFeatures=self._arch.n_features,
            ancillaryFeatures=InputBranch.feature_ancillary,
            device=device,
            nClasses=len(self._classes),
        )
        self._nn.load_state_dict(saved["model"])

    @property
    def nn(self):
        return self._nn


    def evaluate(self, batch: BatchType) -> BatchType:
        selection = self._splitter.split(batch)[SplitterKeys.validation]
        selector = Selector(selection)

        HH, *_ = self._nn(*_HCRInput(batch, self._device, selection))
        TT_cands = self._nn._last_tt_logits #TTbar candidates scores
        WW_score = self.nn._jet_weights
        
        HH = F.softmax(HH, dim=1).cpu()
        TT_cands = F.softmax(TT_cands, dim=-1).cpu()
        
        output = {}
        output["tt_b1Whad"] = TT_cands[:, 0]
        output["tt_b2Whad"] = TT_cands[:, 1]
        jet_scores = WW_score.squeeze(2).squeeze(2).mean(dim=1)  # (batch, 4)
        output["WW_score1"] = jet_scores[:, 0]
        output["WW_score2"] = jet_scores[:, 1]
        output["WW_score3"] = jet_scores[:, 2]
        output["WW_score4"] = jet_scores[:, 3]
        for i, label in enumerate(self._classes):   
            output[f"p_{label}"] = HH[:, i]
        return selector.pad(map_batch(self._mapping, output))

class HCREvaluation(Evaluation):
    def __init__(
        self,
        saved_model: PathLike,
        cross_validation: Splitter,
        output_definition: Callable[[BatchType], BatchType],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = saved_model
        self._splitter = cross_validation
        self._mapping = output_definition

    def stages(self):
        with fsspec.open(self._model, "rb") as f:
            load_kw = {}
            if self.device.type == "cpu":
                load_kw["map_location"] = torch.device("cpu")
            saved = torch.load(f, weights_only=False, **load_kw)
        self._HCR = HCRModelEval(
            device=self.device,
            saved=saved,
            splitter=self._splitter,
            mapping=self._mapping,
        )
        self._HCR.to(self.device)
        yield EvaluationStage(
            name="Evaluation",
            model=self._HCR,
            dataset=self.dataset,
            dumper_kwargs={"name": self.name},
        )

