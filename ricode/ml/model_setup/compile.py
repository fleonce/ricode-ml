import torch.nn

from ricode.ml.model_setup.job_config import CompileConfig
from ricode.ml.model_setup.utils import guess_model_block_types


def setup_compile(module: torch.nn.Module, config: CompileConfig) -> None:
    blocks = guess_model_block_types(module)

    for name, submodule in module.named_modules():
        if type(submodule) in blocks:
            pass

            compiled_submodule = torch.compile(
                submodule, fullgraph=True, dynamic=config.dynamic
            )
            module.register_module(name, compiled_submodule)
