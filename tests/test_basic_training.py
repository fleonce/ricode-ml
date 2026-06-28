import functools
import unittest

import attrs
import torch

from ricode.ml.dataloaders import PaddingDataCollator1D, setup_dataloader
from ricode.ml.training import do_train
from ricode.ml.training_defaults import default_evaluate_function, setup_optimizers
from ricode.utils.decorators import attrs_to_json
from ricode.utils.tempfiles import TemporaryDirectory
from tools import foreach
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear
from transformers import PretrainedConfig, PreTrainedModel


class BasicTrainingTestCase(unittest.TestCase):

    @foreach(memory_training=(True, False, "full"))
    def test_basic_training(
        self,
        weight_in: int = 32,
        weight_out: int = 32,
        num_samples: int = 100,
        test_samples: int = 100,
        memory_training: bool = False,
    ):

        class BasicConfig(PretrainedConfig):
            def __init__(self, weight_in: int = 32, weight_out: int = 32, **kwargs):
                super().__init__(**kwargs)
                self.weight_in = weight_in
                self.weight_out = weight_out

        class BasicModel(PreTrainedModel):
            config_class = BasicConfig

            def __init__(self, config: BasicConfig):
                super().__init__(config)
                self.w1 = Linear(config.weight_in, config.weight_in, bias=False)
                self.w2 = Linear(config.weight_in, config.weight_out, bias=False)

                self.loss = CrossEntropyLoss(reduction="mean")

            def forward(self, inp: Tensor, target: Tensor):
                out = self.w2(self.w1(inp))
                return (self.loss(out.reshape(target.numel(), -1), target),)

        class DummyDataset:
            def __init__(self):
                self.inp = torch.randn((num_samples, weight_out))
                self.target = torch.zeros((num_samples,))

            def __getitem__(self, item):
                return {"inp": self.inp[item], "target": self.target[item]}

            def __len__(self):
                return num_samples

        class DummyDict(dict[str, DummyDataset]):
            pass

        @attrs_to_json
        @attrs.define
        class Hyperparameters:
            lr: float = 10
            weight_decay: float = 0
            lr_scheduler: str = "constant"
            warmup_steps: int | float = 0

            num_steps: int = 100
            eval_every_n_steps: int = 10
            optimize_for: str = "loss"
            patience: int = 0
            batch_size: int = 4
            gradient_accumulation: int = 1
            scale_lr: bool = False

        with TemporaryDirectory(delete_at="exit") as tempdir:
            do_train(
                DummyDict(
                    {split: DummyDataset() for split in ["train", "test", "eval"]}
                ),
                Hyperparameters(),
                BasicModel,
                BasicConfig,
                default_evaluate_function,
                setup_optimizers,
                functools.partial(
                    setup_dataloader,
                    collate_fn=PaddingDataCollator1D(
                        {
                            "inp": (torch.float, 0.0),
                            "target": (torch.long, 0),
                        }
                    ),
                ),
                None,
                memory_checkpoints=memory_training,
                model_path=tempdir,
                score_comparison=lambda f1, f2: f1 < f2,
            )
