import unittest

from more_itertools.more import first

from ricode.datasets.downloads.named_entity_recognition.ontonotes5 import OntoNotes5
from ricode.datasets.downloads.relation_extraction.ade import AdverseDrugEvents
from ricode.datasets.downloads.relation_extraction.conll2004 import CoNLL2004
from ricode.datasets.downloads.relation_extraction.scierc import SciERC
from ricode.utils.tempfiles import TemporaryDirectory


class DatasetDownloadTestCase(unittest.TestCase):
    def test_download_dataset(self):
        with TemporaryDirectory(delete_at="context-exit") as tempdir:
            for dataset_class in [OntoNotes5, CoNLL2004, SciERC, AdverseDrugEvents]:
                dataset = dataset_class(tempdir)

                for split in dataset.splits:
                    self.assertIsNotNone(first(dataset[split]))
