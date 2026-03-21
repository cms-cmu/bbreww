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
from torch import Tensor

from bbreww.classifier.nn.blocks.met_pz_regressor import METRegressor
from bbreww.classifier.config.setting.bbWW import Input, InputBranch
from src.classifier.algorithm.utils import Selector, map_batch, to_num
from src.classifier.nn.schedule import MilestoneStep, Schedule
from src.classifier.utils import MemoryViewIO
from src.classifier.ml import BatchType
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
class RegressorArch:
    __skip_save = frozenset(("loss",))

    loss: Callable[[BatchType], Tensor] = None
    n_features: int = 8

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
    milestones: list[int] = (6, 15)
    gamma: float = 0.25

    def __post_init__(self):
        self.milestones = sorted(self.milestones)
        self._last_bs = self.n_batches
        self.reset()

    def get_bs(self):
        self._last_bs = max(4, int(self.n_batches * (self.gamma**self.milestone)))
        return self._last_bs

    def get_last_bs(self):
        return self._last_bs


@dataclass
class RegressorBenchmarks:
    scalars: Iterable[Callable[[BatchType], dict[str, Tensor]]] = None


def _RegressorInput(batch: BatchType, device: tt.Device, selection: Tensor = None):
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=True)
    inputs = [batch.pop(k) for k in (Input.bJetCand, Input.nonbJetCand, Input.leadingLep, 
                                     Input.MET, Input.ancillary)]

    # keep lepton tensor accessible for W mass loss computation
    batch["_leadingLep"] = inputs[2]
    if selection is not None:
        selection = selection.to(device, non_blocking=True)
        inputs = [i[selection] for i in inputs]
        batch["_leadingLep"] = inputs[2]
    return inputs

class _RegressorSkim(Skimmer):
    def __init__(
        self,
        nn: METRegressor,
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
                *_RegressorInput(batch, self._device, selections[SplitterKeys.training])
            )
        return super().train(batch)


class RegressorModel(Model):
    def __init__(
        self,
        device: tt.Device,
        arch: RegressorArch,
        benchmarks: RegressorBenchmarks,
    ):
        self._loss = arch.loss
        self._device = device
        self._gbn = None
        self._arch = arch
        self._nn = METRegressor(
            dijetFeatures=arch.n_features,
            ancillaryFeatures=InputBranch.feature_ancillary,
            device=device,
        )
        self._benchmarks = benchmarks
        n_params = sum(p.numel() for p in self._nn.parameters() if p.requires_grad)
        logging.info(f"METRegressor: {n_params:,} trainable parameters")
        # Three independent optimizers — created lazily after model is on device
        self._opt_backbone = None
        self._opt_onshell = None
        self._opt_offshell = None
        self._classifier_frozen = False

    def _ensure_optimizers(self):
        """Create three independent optimizers if not yet initialized."""
        if self._opt_backbone is not None:
            return
        import torch.optim as optim
        from torch.optim.lr_scheduler import MultiStepLR
        nn = self._nn

        # Collect parameter id sets for the two regressor heads
        onshell_params = (
            list(nn.nu_regressor_onshell.parameters()) +
            list(nn.nu_cholesky_onshell.parameters())
        )
        offshell_params = (
            list(nn.nu_regressor_offshell.parameters()) +
            list(nn.nu_cholesky_offshell.parameters())
        )
        head_ids = {id(p) for p in onshell_params + offshell_params}

        # Everything else is backbone (embedding, attention, classifier, W mass heads)
        backbone_params = [p for p in nn.parameters() if id(p) not in head_ids]

        self._opt_backbone = optim.Adam(backbone_params, lr=1.2e-2)
        self._opt_onshell = optim.Adam(onshell_params, lr=1.2e-2)
        self._opt_offshell = optim.Adam(offshell_params, lr=1.2e-2)

        # LR decay: hold flat during batch ramp-up, then decay for fine-tuning
        lr_milestones = [38, 42, 45, 48]
        lr_gamma = 0.5
        self._lr_backbone = MultiStepLR(self._opt_backbone, milestones=lr_milestones, gamma=lr_gamma)
        self._lr_onshell = MultiStepLR(self._opt_onshell, milestones=lr_milestones, gamma=lr_gamma)
        self._lr_offshell = MultiStepLR(self._opt_offshell, milestones=lr_milestones, gamma=lr_gamma)

    def parameters(self):
        """Return a dummy parameter so the framework's optimizer is harmless.
        Real optimization is handled by three internal optimizers in train()."""
        if not hasattr(self, '_dummy_param'):
            self._dummy_param = torch.nn.Parameter(torch.zeros(1, device=self._device))
        return iter([self._dummy_param])

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

    def _unpack_forward(self, batch: BatchType):
        nu_pred_on, L_on, nu_pred_off, L_off, (logit_onshell, pz_hint_on) = self._nn(*_RegressorInput(batch, self._device))
        batch["pred_nu_on"] = nu_pred_on
        batch["cholesky_L_on"] = L_on
        batch["pred_nu_off"] = nu_pred_off
        batch["cholesky_L_off"] = L_off
        batch["logit_onshell"] = logit_onshell
        batch["pz_hint_on"] = pz_hint_on
        batch["ww_weights"] = self._nn._jet_weights

    def train(self, batch: BatchType) -> Tensor:
        self._ensure_optimizers()
        self._unpack_forward(batch)
        loss_backbone, loss_onshell, loss_offshell = self._loss(batch)

        # Train all three losses simultaneously
        self._opt_backbone.zero_grad()
        self._opt_onshell.zero_grad()
        self._opt_offshell.zero_grad()
        loss_backbone.backward(retain_graph=True)
        loss_onshell.backward(retain_graph=True)
        loss_offshell.backward()
        torch.nn.utils.clip_grad_norm_(self._nn.parameters(), max_norm=1.0)
        self._opt_backbone.step()
        self._opt_onshell.step()
        self._opt_offshell.step()
        total = loss_backbone.item() + loss_onshell.item() + loss_offshell.item()

        # Return a requires_grad tensor for the framework's loss.backward()/opt.step().
        dummy = torch.tensor(total, device=self._device, requires_grad=True)
        return dummy + 0  # enables .backward() but produces no real gradients

    def validate(self, batches: Iterable[BatchType]) -> dict[str]:
        weight = 0.0
        scalars = defaultdict(float)
        scalar_funcs = self._benchmarks.scalars

        for batch in batches:
            self._unpack_forward(batch)
            sumw = to_num(batch[Input.weight].sum())
            if scalar_funcs is None:
                l_bb, l_on, l_off = self._loss(batch)
                scalars["loss"] += (to_num(l_bb) + to_num(l_on) + to_num(l_off)) * sumw
                scalars["loss_backbone"] += to_num(l_bb) * sumw
                scalars["loss_onshell"] += to_num(l_on) * sumw
                scalars["loss_offshell"] += to_num(l_off) * sumw
            else:
                for func in scalar_funcs:
                    measured = func(batch)
                    for name, value in measured.items():
                        scalars[name] += to_num(value) * sumw
            weight += sumw
        for k in scalars:
            scalars[k] /= weight

        self._last_val_loss = scalars.get("loss", None)
        return {"scalars": scalars}

    def step(self, epoch: int = None):
        if self.ghost_batch is not None and self.ghost_batch.step(epoch):
            self._nn.setGhostBatches(self.ghost_batch.get_bs(), False)
        # Step LR schedulers (MultiStepLR — epoch-based)
        if self._opt_backbone is not None:
            self._lr_backbone.step()
            self._lr_onshell.step()
            self._lr_offshell.step()


class RegressorTraining(MultiStageTraining):
    def __init__(
        self,
        arch: RegressorArch,
        ghost_batch: GBNSchedule,
        cross_validation: Splitter,
        training_schedule: Schedule,
        finetuning_schedule: Schedule = None,
        benchmarks: RegressorBenchmarks = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._arch = arch
        self._ghost_batch = ghost_batch
        self._splitter = cross_validation
        self._training = training_schedule
        self._finetuning = finetuning_schedule
        self._benchmarks = benchmarks or RegressorBenchmarks()
        self._model: RegressorModel = None

    def stages(self):
        self._model = RegressorModel(
            device=self.device,
            arch=self._arch,
            benchmarks=self._benchmarks,
        )
        self._model.ghost_batch = self._ghost_batch
        self._model.to(self.device)
        self._splitter.setup(self.dataset)
        skim = _RegressorSkim(self._model._nn, self.device, self._splitter)
        yield TrainingStage(
            name="Initialization",
            model=skim,
            schedule=SkimStep(),
            training=self.dataset,
        )
        self._model.nn.initMeanStd()
        validation_sets = self._splitter.get()
        training_set = validation_sets[SplitterKeys.training]
        yield BenchmarkStage(
            name="Baseline",
            model=self._model,
            validation=validation_sets,
        )
        yield TrainingStage(
            name="Training",
            model=self._model,
            schedule=self._training,
            training=training_set,
            validation=validation_sets,
        )
        self._model.ghost_batch = None
        if self._finetuning is not None:
            layers = self._model._nn.layers
            layers.setLayerRequiresGrad(
                requires_grad=False, index=self._model._nn.embedding_layers()
            )
            yield TrainingStage(
                name="Finetuning",
                model=self._model,
                schedule=self._finetuning,
                training=training_set,
                validation=validation_sets,
            )
            self._model.ghost_batch = self._ghost_batch
            layers.setLayerRequiresGrad(requires_grad=True)


        output_stage = OutputStage(name="Final", path=f"{self.name}__{self.uuid}.pkl")
        output_path = output_stage.absolute_path
        if not output_path.is_null:
            logging.info(f"Saving model to {output_path}")
            with fsspec.open(output_path, "wb") as f:
                torch.save(
                    {
                        "model": self._model.nn.state_dict(),
                        "metadata": self.metadata,
                        "uuid": self.uuid,
                        "arch": self._arch.save(),
                        "input": {
                            k: getattr(InputBranch, k)
                            for k in (
                                "feature_ancillary",
                                "feature_bJetCand",
                                "feature_nonbJetCand",
                                "feature_leadingLep",
                                "feature_MET",
                            )
                        },
                    },
                    MemoryViewIO(f),
                )
            yield output_stage


class RegressorModelEval(Model):
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
        for k in saved["input"].keys():
            if getattr(InputBranch, k) != saved["input"][k]:
                raise ValueError(
                    f'Input features "{k}" mismatch: training={saved["input"][k]}, evaluation={getattr(InputBranch, k)}'
                )
        self._arch = RegressorArch.load(saved["arch"])
        self._nn = METRegressor(
            dijetFeatures=self._arch.n_features,
            ancillaryFeatures=InputBranch.feature_ancillary,
            device=device,
        )
        self._nn.load_state_dict(saved["model"], strict=False)

    @property
    def nn(self):
        return self._nn

    def evaluate(self, batch: BatchType) -> BatchType:
        selection = self._splitter.split(batch)[SplitterKeys.validation]
        selector = Selector(selection)

        nu_pred_on, L_on, nu_pred_off, L_off, (logit_onshell, pz_hint_on) = self._nn(*_RegressorInput(batch, self._device, selection))
        p_onshell = torch.sigmoid(logit_onshell)

        jet_weights = self._nn._jet_weights  # (n, 6): per-jet attention weights (2 heads × 3 jets)

        # Extract marginal sigmas from Cholesky: sigma_i = sqrt((L @ L^T)_{ii})
        cov_on = torch.bmm(L_on, L_on.transpose(-1, -2))
        sigma_on = cov_on.diagonal(dim1=-2, dim2=-1).sqrt()
        cov_off = torch.bmm(L_off, L_off.transpose(-1, -2))
        sigma_off = cov_off.diagonal(dim1=-2, dim2=-1).sqrt()

        # Select neutrino using classifier p_onshell directly
        use_on = (p_onshell > 0.55).unsqueeze(-1)  # (n, 1)
        nu_sel = torch.where(use_on, nu_pred_on[:, :3], nu_pred_off)
        sigma_sel = torch.where(use_on, sigma_on, sigma_off)

        output = {
            # Selected (best hypothesis) neutrino
            "nu_px": nu_sel[:, 0],
            "nu_py": nu_sel[:, 1],
            "nu_pz": nu_sel[:, 2],
            "nu_sigma_px": sigma_sel[:, 0],
            "nu_sigma_py": sigma_sel[:, 1],
            "nu_sigma_pz": sigma_sel[:, 2],
            # On-shell hypothesis
            "nu_px_on": nu_pred_on[:, 0],
            "nu_py_on": nu_pred_on[:, 1],
            "nu_pz_on": nu_pred_on[:, 2],
            # Off-shell hypothesis
            "nu_px_off": nu_pred_off[:, 0],
            "nu_py_off": nu_pred_off[:, 1],
            "nu_pz_off": nu_pred_off[:, 2],
            # Per-hypothesis sigmas (before selection)
            "nu_sigma_pz_on": sigma_on[:, 2],
            "nu_sigma_pz_off": sigma_off[:, 2],
            # Classifier
            "p_onshell": p_onshell,
            # Per-jet attention weights (2 heads × 4 jets)
            "jet_weight_0": jet_weights[:, 0],
            "jet_weight_1": jet_weights[:, 1],
            "jet_weight_2": jet_weights[:, 2],
            "jet_weight_3": jet_weights[:, 3],
            "jet_weight_4": jet_weights[:, 4],
            "jet_weight_5": jet_weights[:, 5],
            "jet_weight_6": jet_weights[:, 6],
            "jet_weight_7": jet_weights[:, 7],
        }
        return selector.pad(map_batch(self._mapping, output))


class RegressorEvaluation(Evaluation):
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
            saved = torch.load(f, **load_kw)
        self._regressor = RegressorModelEval(
            device=self.device,
            saved=saved,
            splitter=self._splitter,
            mapping=self._mapping,
        )
        self._regressor.to(self.device)
        yield EvaluationStage(
            name="Evaluation",
            model=self._regressor,
            dataset=self.dataset,
            dumper_kwargs={"name": self.name},
        )
