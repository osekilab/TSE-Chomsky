import torch
from tango.common import Registrable


class Activation(torch.nn.Module, Registrable):
    """
    Pytorch has a number of built-in activation functions.  We group those here under a common
    type, just to make it easier to configure and instantiate them `from_params` using
    `Registrable`.
    Note that we're only including element-wise activation functions in this list.  You really need
    to think about masking when you do a softmax or other similar activation function, so it
    requires a different API.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# There are no classes to decorate, so we hack these into Registrable._registry.
# If you want to instantiate it, you can do like this:
# Activation.by_name('relu')()
Registrable._registry[Activation] = {
    "relu": (torch.nn.ReLU, None),
    "relu6": (torch.nn.ReLU6, None),
    "elu": (torch.nn.ELU, None),
    "gelu": (torch.nn.GELU, None),
    "prelu": (torch.nn.PReLU, None),
    "leaky_relu": (torch.nn.LeakyReLU, None),
    "threshold": (torch.nn.Threshold, None),
    "hardtanh": (torch.nn.Hardtanh, None),
    "sigmoid": (torch.nn.Sigmoid, None),
    "tanh": (torch.nn.Tanh, None),
    "log_sigmoid": (torch.nn.LogSigmoid, None),
    "softplus": (torch.nn.Softplus, None),
    "softshrink": (torch.nn.Softshrink, None),
    "softsign": (torch.nn.Softsign, None),
    "tanhshrink": (torch.nn.Tanhshrink, None),
    "selu": (torch.nn.SELU, None),
}
