import torch
import torch.nn as nn
import math

from torch.nn.init import xavier_normal_, constant_, xavier_uniform_, kaiming_uniform_, kaiming_normal_
from torch import Tensor
from typing import Optional as _Optional


def _no_grad_normal_(tensor, mean, std, generator=None):
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)


def xavier_uniform_initialization(module):
    r"""using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def xavier_normal_initialization(module):
    r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def kaiming_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        kaiming_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        kaiming_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def kaiming_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        kaiming_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        kaiming_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    r"""Fill the input Tensor with values drawn from the normal distribution.

    :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            normal_, (tensor,), tensor=tensor, mean=mean, std=std, generator=generator
        )
    return _no_grad_normal_(tensor, mean, std, generator)