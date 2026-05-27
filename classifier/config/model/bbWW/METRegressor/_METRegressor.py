from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

from bbreww.classifier.config.setting.METRegressor import Input
from src.classifier.task import ArgParser, parse

from src.classifier.config.model._kfold import KFoldEval, KFoldTrain

_SCHEDULER = "classifier.config.scheduler"


if TYPE_CHECKING:
    from src.classifier.ml import BatchType
    from src.classifier.ml.skimmer import Splitter
    from torch import Tensor


class METRegressorTrain(KFoldTrain):
    model: str
    loss: Callable[[BatchType], Tensor]

    argparser = ArgParser()
    argparser.add_argument(
        "--architecture",
        type=parse.mapping,
        default="",
        help=f"METRegressor architecture {parse.EMBED}",
    )
    argparser.add_argument(
        "--ghost-batch",
        type=parse.mapping,
        default="",
        help=f"ghost batch normalization configuration {parse.EMBED}",
    )
    argparser.add_argument(
        "--training",
        nargs="+",
        default=["FixedStep"],
        metavar=("CLASS", "KWARGS"),
        help=f"training scheduler {parse.EMBED}",
    )
    argparser.add_argument(
        "--finetuning",
        nargs="+",
        default=[],
        metavar=("CLASS", "KWARGS"),
        help=f"fine-tuning scheduler {parse.EMBED}",
    )

    def initializer(self, splitter: Splitter, **kwargs):
        from bbreww.classifier.ml.models.METRegressor import (
            GBNSchedule,
            RegressorArch,
            RegressorBenchmarks,
            RegressorTraining,
        )

        arch = RegressorArch(**({"loss": self.loss} | self.opts.architecture))
        gbn = GBNSchedule(**self.opts.ghost_batch)
        training = parse.instance(self.opts.training, _SCHEDULER)
        finetuning = parse.instance(self.opts.finetuning, _SCHEDULER)

        return RegressorTraining(
            arch=arch,
            ghost_batch=gbn,
            cross_validation=splitter,
            training_schedule=training,
            finetuning_schedule=finetuning,
            benchmarks=RegressorBenchmarks(),
            model=self.model,
            **kwargs,
        )


class METRegressorEval(KFoldEval):
    model: str
    output_definition: Callable[[BatchType], BatchType]

    def initializer(self, model, splitter, **kwargs):
        from bbreww.classifier.ml.models.METRegressor import RegressorEvaluation

        return RegressorEvaluation(
            saved_model=model,
            cross_validation=splitter,
            output_definition=self.output_definition,
            model=self.model,
            **kwargs,
        )
