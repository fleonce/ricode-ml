import torch.nn


def guess_model_block_type(module: torch.nn.Module) -> type:
    module_types = list(set(type(mod) for mod in module.modules()))
    module_type_names = list(map(lambda x: x.__name__, module_types))

    for pos, module_name in enumerate(module_type_names):
        if module_name.endswith("Layer") or module_name.endswith("Block"):
            return module_types[pos]

    raise ValueError(f"Cannot find a model block class out of {module_type_names}")
