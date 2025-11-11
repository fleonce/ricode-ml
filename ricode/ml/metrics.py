from typing import (
    Any,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

import torch
import typing_extensions
from torcheval.metrics import Metric, MulticlassConfusionMatrix
from torcheval.metrics.classification.confusion_matrix import TMulticlassConfusionMatrix
from torcheval.metrics.toolkit import sync_and_compute_collection

from ricode.ml._metrics import ELSpan, Relation, RelationWithProbability, Span, TwoSpans

from ricode.ml._metrics.functional import (
    _f1_score_compute,
    BatchedOutputs,
    LabelGeneric,
    LabelType,
)
from ricode.ml._metrics.tasks.entity_linking import _el_score_update
from ricode.ml._metrics.tasks.fuzzy_ner import _fuzzy_ner_score_update
from ricode.ml._metrics.tasks.ner import (
    _ner_confusion_update,
    _ner_score_update,
    ComputeMode,
    TokenizedSpan,
    WordBoundarySpan,
)
from ricode.ml._metrics.tasks.relation_extraction import (
    _re_confusion_update,
    _re_score_update,
)

__all__ = [
    "MultiMetric",
    "NERF1Score",
    "NERPrecision",
    "NERRecall",
    "NERConfusionMatrix",
    "ContaminatedNERF1Score",
    "ContaminatedNERPrecision",
    "ContaminatedNERRecall",
    "FuzzyNERF1Score",
    "REF1Score",
    "REPrecision",
    "RERecall",
    "REConfusionMatrix",
    "ELF1Score",
    "ELPrecision",
    "ELRecall",
    "Span",
    "Relation",
    "RelationWithProbability",
    "ELSpan",
    "TwoSpans",
]


class MultiMetric:
    def __init__(self, prefix: Optional[str] = None, /, **kwargs: Metric[torch.Tensor]):
        self.prefix = prefix
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @property
    def metrics(self) -> Mapping[str, Metric[torch.Tensor]]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, Metric)
        }

    def compute(self) -> Mapping[str, torch.Tensor]:
        prefix = self.prefix or ""
        return {prefix + key: metric.compute() for key, metric in self.metrics.items()}

    def __getattr__(self, item: str) -> Metric[torch.Tensor]:
        return self.metrics[item]

    def __getitem__(self, item: str) -> Metric[torch.Tensor]:
        return self.metrics[item]

    def __add__(self, other):
        raise NotImplementedError

    def update(self, input: Any, target: Any):
        for metric in self.metrics.values():
            metric.update(input, target)

    def sync_and_compute(self):
        prefix = self.prefix or ""
        return sync_and_compute_collection(
            {prefix + key: metric for key, metric in self.metrics.items()}
        )


class _NERScore(Metric[torch.Tensor], Generic[LabelGeneric]):
    return_index: ClassVar[int] = 0

    labels: Optional[Sequence[LabelGeneric]]
    num_tp: torch.Tensor
    num_fp: torch.Tensor
    num_fn: torch.Tensor

    def __init__(
        self,
        labels: Optional[Sequence[LabelGeneric]] = None,
        strict: bool = True,
        average: Literal["micro", "macro", "none"] = "micro",
        position_aware: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.average = average
        self.position_aware = position_aware
        self.labels = labels
        self.strict = strict
        self.num_labels = len(labels) if labels is not None else 0
        if average == "micro":
            self._add_state("num_tp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fn", torch.tensor(0.0, device=self.device))
        elif average in {"macro", "none"}:
            self._add_state(
                "num_tp", torch.zeros((self.num_labels,), device=self.device)
            )
            self._add_state(
                "num_fp", torch.zeros((self.num_labels,), device=self.device)
            )
            self._add_state(
                "num_fn", torch.zeros((self.num_labels,), device=self.device)
            )
        else:
            raise NotImplementedError(average)

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _ner_score_update(
            output,
            target,
            self.average,
            self.strict,
            self.position_aware,
            self.labels,
            self.device,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn

    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[self.return_index]

    def compute_with_precision_recall(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _f1_score_compute(self.num_tp, self.num_fp, self.num_fn, self.average)

    def compute_per_label(
        self,
    ) -> Mapping[
        LabelGeneric, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if self.average not in {"none", "macro"}:
            raise ValueError(
                f"Average must be either 'none' or 'macro', not {self.average}"
            )
        if self.labels is None:
            raise ValueError(
                f"Average must be either 'none' or 'macro', not {self.average}"
            )
        precision, recall, f1 = _f1_score_compute(
            self.num_tp, self.num_fp, self.num_fn, "none"
        )
        return {
            label: (
                precision[pos],
                recall[pos],
                f1[pos],
                self.num_fn[pos] + self.num_tp[pos] + self.num_fp[pos],
            )
            for pos, label in enumerate(self.labels)
        }

    def merge_state(self, metrics: "Iterable[_NERScore]") -> typing_extensions.Self:
        for metric in metrics:
            self.num_tp += metric.num_tp
            self.num_fp += metric.num_fp
            self.num_fn += metric.num_fn
        return self


class _ContaminationNERScore(_NERScore):

    def __init__(
        self,
        labels: Optional[Sequence[LabelGeneric]] = None,
        strict: bool = True,
        average: Literal["micro", "macro", "none"] = "micro",
        position_aware: bool = False,
        device: Optional[torch.device] = None,
        contaminated_entities: Optional[set[TokenizedSpan | WordBoundarySpan]] = None,
        compute_mode: ComputeMode = ComputeMode.COMPUTE_CLEAN,
    ):
        super().__init__(labels, strict, average, position_aware, device)

        self.contaminated_entities = contaminated_entities or set()
        self.compute_mode = compute_mode

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _ner_score_update(
            output,
            target,
            self.average,
            self.strict,
            self.position_aware,
            self.labels,
            self.device,
            self.contaminated_entities,
            self.compute_mode,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn


class _FuzzyNERScore(_NERScore):

    def __init__(
        self,
        labels: Optional[Sequence[LabelGeneric]] = None,
        strict: bool = True,
        average: Literal["micro", "macro", "none"] = "micro",
        position_aware: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(labels, strict, average, position_aware, device)

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _fuzzy_ner_score_update(
            output,
            target,
            self.average,
            self.strict,
            self.position_aware,
            self.labels,
            self.device,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn


class NERF1Score(_NERScore, Generic[LabelGeneric]):
    return_index = 2


class NERPrecision(_NERScore):
    return_index = 0


class NERRecall(_NERScore):
    return_index = 1


class ContaminatedNERF1Score(_ContaminationNERScore, NERF1Score):
    pass


class ContaminatedNERPrecision(_ContaminationNERScore, NERPrecision):
    pass


class ContaminatedNERRecall(_ContaminationNERScore, NERRecall):
    pass


class FuzzyNERF1Score(_FuzzyNERScore, NERF1Score):
    pass


class _REScore(_NERScore):

    def __init__(
        self,
        labels: Optional[Sequence[LabelType]] = None,
        strict: bool = True,
        strict_relations: bool = True,
        average: Literal["micro", "macro", "none"] = "micro",
        position_aware: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(labels, strict, average, position_aware, device)
        self.strict_relations = strict_relations

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _re_score_update(
            output,
            target,
            self.average,
            self.position_aware,
            self.labels,
            self.strict,
            self.strict_relations,
            self.device,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn

    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def merge_state(self, metrics: "Iterable[_NERScore]") -> typing_extensions.Self:
        for metric in metrics:
            self.num_tp += metric.num_tp
            self.num_fp += metric.num_fp
            self.num_fn += metric.num_fn
        return self


class REF1Score(_REScore):
    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[2]


class REPrecision(_REScore):
    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[0]


class RERecall(_REScore):
    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[1]


class _ELScore(Metric[torch.Tensor], Generic[LabelGeneric]):
    return_index: ClassVar[int] = 0

    labels: Optional[Sequence[LabelGeneric]]
    num_tp: torch.Tensor
    num_fp: torch.Tensor
    num_fn: torch.Tensor

    def __init__(
        self,
        labels: Optional[Sequence[LabelGeneric]] = None,
        average: Literal["micro", "macro", "none"] = "micro",
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.average = average
        self.labels = labels
        self.num_labels = len(labels) if labels is not None else 0
        if average == "micro":
            self._add_state("num_tp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fn", torch.tensor(0.0, device=self.device))
        elif average in {"macro", "none"}:
            self._add_state(
                "num_tp", torch.zeros((self.num_labels,), device=self.device)
            )
            self._add_state(
                "num_fp", torch.zeros((self.num_labels,), device=self.device)
            )
            self._add_state(
                "num_fn", torch.zeros((self.num_labels,), device=self.device)
            )
        else:
            raise NotImplementedError(average)

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _el_score_update(
            output,
            target,
            self.average,
            self.labels,
            self.device,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn

    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[self.return_index]

    def compute_with_precision_recall(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _f1_score_compute(self.num_tp, self.num_fp, self.num_fn, self.average)

    def compute_per_label(
        self,
    ) -> Mapping[
        LabelGeneric, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if self.average not in {"none", "macro"}:
            raise ValueError(
                f"Average must be either 'none' or 'macro', not {self.average}"
            )
        if self.labels is None:
            raise ValueError(
                f"Average must be either 'none' or 'macro', not {self.average}"
            )
        precision, recall, f1 = _f1_score_compute(
            self.num_tp, self.num_fp, self.num_fn, "none"
        )
        return {
            label: (
                precision[pos],
                recall[pos],
                f1[pos],
                self.num_fn[pos] + self.num_tp[pos] + self.num_fp[pos],
            )
            for pos, label in enumerate(self.labels)
        }

    def merge_state(self, metrics: "Iterable[_ELScore]") -> typing_extensions.Self:
        for metric in metrics:
            self.num_tp += metric.num_tp
            self.num_fp += metric.num_fp
            self.num_fn += metric.num_fn
        return self


class ELF1Score(_ELScore, Generic[LabelGeneric]):
    return_index = 2


class ELPrecision(_ELScore):
    return_index = 0


class ELRecall(_ELScore):
    return_index = 1


class _MulticlassConfusionMatrix(MulticlassConfusionMatrix):

    def __init__(
        self,
        labels: Sequence[LabelType],
        position_aware: bool = False,
        *,
        normalize: Optional[Literal["all", "pred", "true", "none"]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(len(labels) + 1, normalize=normalize, device=device)
        self.position_aware = position_aware
        self.labels = labels

    @torch.inference_mode()
    def update(self, input: torch.Tensor, target: torch.Tensor):
        if input.numel() == 0 and target.numel() == 0:
            return self
        return super().update(input, target)

    @torch.inference_mode()
    def compute(self: TMulticlassConfusionMatrix) -> torch.Tensor:
        tensor = super().compute()
        if self.normalize is None or self.normalize == "none":
            return tensor.to(torch.long)
        return tensor


class NERConfusionMatrix(_MulticlassConfusionMatrix):
    @torch.inference_mode()
    def update(self, output: BatchedOutputs, target: BatchedOutputs):  # type: ignore[override]
        super().update(
            *_ner_confusion_update(
                output, target, self.position_aware, self.labels, self.device
            )
        )


class REConfusionMatrix(_MulticlassConfusionMatrix):

    def __init__(
        self,
        labels: list[str],
        position_aware: bool = False,
        strict: bool = True,
        *,
        normalize: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(labels, position_aware, normalize=normalize, device=device)
        self.strict = strict

    @torch.inference_mode()
    def update(self, output: BatchedOutputs, target: BatchedOutputs):  # type: ignore[override]
        super().update(
            *_re_confusion_update(
                output,
                target,
                self.position_aware,
                self.labels,
                self.strict,
                self.device,
            )
        )
