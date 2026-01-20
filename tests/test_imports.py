import unittest


class ImportTest(unittest.TestCase):
    @staticmethod
    def test_import_basics():
        from ricode.ml.training_basics import BasicHparams, BasicMetrics  # noqa

    @staticmethod
    def test_import_dataset():
        from ricode.ml.training_datasets import BasicDataset  # noqa

    @staticmethod
    def test_import_dataset_mixins():
        from ricode.ml.training_dataset_mixins import (  # noqa
            ContaminatedMixin,
            TypesMixin,
        )

    @staticmethod
    def test_import_datasets():
        from ricode.ml.datasets.combined import CombinedDataset  # noqa
        from ricode.ml.datasets.concatenated import (  # noqa
            ConcatenatedDataset,
            ConcatenatedDatasetOld,
        )
        from ricode.ml.datasets.index import IndexDataset  # noqa
        from ricode.ml.datasets.proxy import ProxyDataset, ProxyTrainingArgs  # noqa
