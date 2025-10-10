import torch.cuda

from ricode.ml.model_setup.utils import identity
from ricode.ml.training_types import ModelUpdateProtocol, TModel

try:
    import torchao  # noqa: F401

    TORCHAO = True
except ImportError:
    TORCHAO = False

if TORCHAO:

    def setup_float8_model(*, disable: bool = False) -> ModelUpdateProtocol[TModel]:
        if disable:
            return identity

        def _model_setup(model: TModel) -> TModel:
            from torchao.float8 import convert_to_float8_training

            model = convert_to_float8_training(model)
            return model  # type: ignore

        return _model_setup

else:

    def setup_float8_model(*, disable: bool = False) -> ModelUpdateProtocol[TModel]:
        if disable:
            return identity

        raise ValueError(
            'torchao is required to use this feature, run "pip install torchao"'
        )


def _is_float8_supported():
    return torch.cuda.get_device_capability() >= (8, 9)
