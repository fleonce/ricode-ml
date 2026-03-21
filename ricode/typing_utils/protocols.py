from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    # @runtime_checkable
    class SupportsToList(torch.Tensor):
        pass

else:

    @runtime_checkable
    class SupportsToList(Protocol):
        pass
