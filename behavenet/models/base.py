"""Base models/modules in PyTorch."""

from torch import nn, save


class BaseModule(nn.Module):
    """Template for PyTorch modules."""

    def __init__(self, *args, **kwargs):
        super().__init__()

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


class CustomDataParallel(nn.DataParallel):
    """Wrapper class for multi-gpu training.

    from https://github.com/pytorch/tutorials/issues/836
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
