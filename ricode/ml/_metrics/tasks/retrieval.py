from typing import Iterable, Optional

import torch
import typing_extensions
from torcheval.metrics import Metric

from ricode.ml._metrics.tasks.functional.retrieval import (
    recall_at_k_compute,
    recall_at_k_update,
)


class RecallAtK(Metric[torch.Tensor]):
    num_tp: torch.Tensor
    num_target: torch.Tensor

    def __init__(
        self,
        k: int,
        ignore_index: int | None = None,
        largest: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)

        self.k = k
        self.ignore_index = ignore_index
        self.largest = largest
        self._add_state("num_tp", torch.tensor(0.0, device=self.device))
        self._add_state("num_target", torch.tensor(0.0, device=self.device))

    def update(self, output: torch.Tensor, target: torch.Tensor) -> "RecallAtK":
        num_tp, num_target = recall_at_k_update(
            output, target, self.k, self.ignore_index, self.largest
        )

        self.num_tp += num_tp
        self.num_target += num_target
        return self

    def compute(self) -> torch.Tensor:
        return recall_at_k_compute(self.num_tp, self.num_target)

    def merge_state(self, metrics: "Iterable[RecallAtK]") -> typing_extensions.Self:
        for metric in metrics:
            self.num_tp += metric.num_tp
            self.num_target += metric.num_target
        return self
