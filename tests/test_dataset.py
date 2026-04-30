import unittest

import torch

from ricode.ml.datasets.cumulative import FlattenedDataset
from ricode.utils.tempfiles import TemporaryDirectory


class DatasetTestCase(unittest.TestCase):
    def test_save_py_objects(self):
        ds = FlattenedDataset.new_empty()
        words = "Ich gehe einlaufen".split()
        for _ in range(1000):
            ds.append({"words": words})

        with TemporaryDirectory("context-exit") as tempdir:
            ds.save_to_disk(tempdir)
            lds = FlattenedDataset.from_preprocessed(tempdir)

        for item in range(len(ds)):
            out = ds[item]
            l_out = lds[item]
            for key in set(out.keys()) | set(l_out.keys()):
                self.assertIn(key, out.keys())
                self.assertIn(key, l_out.keys())

                value = out[key]
                l_value = l_out[key]

                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, l_value))
                else:
                    self.assertEqual(value, l_value)
