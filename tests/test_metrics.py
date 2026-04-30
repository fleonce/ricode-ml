import unittest

import torch

from ricode.ml._metrics.functional import _f1_score_support_compute, _tp_fp_fn_compute
from ricode.ml._metrics.tasks.ner import NamedEntity
from ricode.ml.metrics import NERMetrics
from tools.foreach import foreach
from tools.tensors import assert_tensor_equal
from torch import Tensor

_GENERATOR = torch.Generator().manual_seed(42)

TARGETS = (
    torch.arange(10, dtype=torch.long),
    torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
    torch.randint(0, 10, (100,), generator=_GENERATOR),
    torch.zeros(10, dtype=torch.long),
)
OUTPUTS = (
    torch.zeros(10, dtype=torch.long),
    torch.tensor([0, 1, 2, 2, 1, 0, 0, 1, 2]),
    torch.randint(0, 10, (100,), generator=_GENERATOR),
    torch.ones(10, dtype=torch.long),
)
NUM_LABELS = (
    10,
    3,
    10,
    2,
)
ENTITY_LABELS = ["PER", "ORG"]
ENTITY_TARGETS = (
    (
        NamedEntity(0, 1, "PER", "Dieter"),
        NamedEntity(2, 3, "ORG", "Dackel"),
    ),
)
ENTITY_OUTPUTS = (ENTITY_TARGETS[0],)


class MetricsTest(unittest.TestCase):

    @foreach(zipped=True, output=OUTPUTS, target=TARGETS, num_labels=NUM_LABELS)
    def test_support(self, output: Tensor, target: Tensor, num_labels: int):
        tp, fp, fn = _tp_fp_fn_compute(output, target, num_labels)

        labels = torch.arange(num_labels)
        compare_tp = (
            (output.view(-1, 1) == labels.view(1, -1))
            & (target.view(-1, 1) == labels.view(1, -1))
        ).sum(dim=0)
        compare_fn = (
            (
                (output.view(-1, 1) == labels.view(1, -1))
                != (target.view(-1, 1) == labels.view(1, -1))
            )
            * (target.view(-1, 1) == labels.view(1, -1))
        ).sum(dim=0)
        compare_fp = (
            (
                (output.view(-1, 1) == labels.view(1, -1))
                != (target.view(-1, 1) == labels.view(1, -1))
            )
            * (output.view(-1, 1) == labels.view(1, -1))
        ).sum(dim=0)
        compare_support = (target.view(-1, 1).eq(labels.view(1, -1))).sum(dim=0)
        self.assertEqual(tp.shape, (num_labels,))
        self.assertEqual(fp.shape, (num_labels,))
        self.assertEqual(fn.shape, (num_labels,))
        self.assertEqual(compare_tp.shape, (num_labels,))
        self.assertEqual(compare_fp.shape, (num_labels,))
        self.assertEqual(compare_fn.shape, (num_labels,))
        assert_tensor_equal(tp, compare_tp)
        assert_tensor_equal(fp, compare_fp)
        assert_tensor_equal(fn, compare_fn)

        support = _f1_score_support_compute(tp, fp, fn, "micro")[3]
        assert_tensor_equal(support, compare_support)

    @foreach(True, output=ENTITY_OUTPUTS, target=ENTITY_TARGETS)
    def test_ner_metrics(self, output, target):
        metrics = NERMetrics(ENTITY_LABELS, None)
        metrics.update([set(output)], [set(target)])
        self.assertIsNotNone(metrics.compute())
