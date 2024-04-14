import torch
from tango.common.registrable import Registrable

"""
"""


class Optimizer(torch.optim.Optimizer, Registrable):
    ...


class LRScheduler(torch.optim.lr_scheduler._LRScheduler, Registrable):
    ...


# Register all optimizers.
for name, cls in torch.optim.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, torch.optim.Optimizer)
        and not cls == torch.optim.Optimizer
    ):
        Optimizer.register("torch::" + name)(cls)

# Register all learning rate schedulers.
for name, cls in torch.optim.lr_scheduler.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, torch.optim.lr_scheduler._LRScheduler)
        and not cls == torch.optim.lr_scheduler._LRScheduler
    ):
        LRScheduler.register("torch::" + name)(cls)
