"""Base models/modules in PyTorch."""

import math
from torch import nn, save, Tensor

# to ignore imports for sphix-autoapidoc
__all__ = ['BaseModule', 'BaseModel', 'DiagLinear', 'CustomDataParallel']


class BaseModule(nn.Module):
    """Template for PyTorch modules."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __str__(self):
        """Pretty print module architecture."""
        raise NotImplementedError

    def build_model(self):
        """Build model from hparams."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Push data through module."""
        raise NotImplementedError

    def freeze(self):
        """Prevent updates to module parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Force updates to module parameters."""
        for param in self.parameters():
            param.requires_grad = True


class BaseModel(nn.Module):
    """Template for PyTorch models."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __str__(self):
        """Pretty print model architecture."""
        raise NotImplementedError

    def build_model(self):
        """Build model from hparams."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Push data through model."""
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        """Compute loss."""
        raise NotImplementedError

    def save(self, filepath):
        """Save model parameters."""
        save(self.state_dict(), filepath)

    def get_parameters(self):
        """Get all model parameters that have gradient updates turned on."""
        return filter(lambda p: p.requires_grad, self.parameters())


class DiagLinear(nn.Module):
    """Applies a diagonal linear transformation to the incoming data: :math:`y = xD^T + b`"""

    __constants__ = ['features']
    # features: int
    # weight: Tensor

    def __init__(self, features, bias=True):

        super(DiagLinear, self).__init__()

        self.features = features
        self.weight = nn.Parameter(Tensor(features))
        if bias:
            self.bias = nn.Parameter(Tensor(features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input.mul(self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return 'features={}, bias={}'.format(self.features, self.bias is not None)


class CustomDataParallel(nn.DataParallel):
    """Wrapper class for multi-gpu training.

    from https://github.com/pytorch/tutorials/issues/836
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
