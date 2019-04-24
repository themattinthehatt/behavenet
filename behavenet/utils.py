import numpy as np
import torch
from torch.autograd import Variable


def estimate_model_footprint(model, input_size):
    """
    Adapted from http://jacobkimmel.github.io/pytorch_estimating_model_size/

    Args:
        model (pt model):
        input_size (tuple):

    Returns:
        int: bytes
    """

    allowed_modules = (
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.MaxPool2d,
        torch.nn.MaxUnpool2d,
        torch.nn.Linear
    )

    # assume everything is float32
    bytes = 4

    # estimate input size
    input_bytes = np.prod(input_size) * bytes

    # estimate model size
    mods = list(model.modules())
    for mod in mods:
        if isinstance(mod, allowed_modules):
            p = list(mod.parameters())
            sizes = []
            for p_ in p:
                sizes.append(np.array(p_.size()))

    model_bytes = 0
    for size in sizes:
        model_bytes += np.prod(np.array(size)) * bytes

    # estimate intermediate size
    # input_ = Variable(torch.FloatTensor(*input_size))
    # out_sizes = []
    # for mod in mods:
    #     if isinstance(mod, allowed_modules):
    #         out = mod(input_)
    #         if isinstance(out, tuple):
    #             out_sizes.append(np.array(out[0].size()))
    #         else:
    #             out_sizes.append(np.array(out.size()))
    #         input_ = out
    #     else:
    #         print(mod)
    x = Variable(torch.FloatTensor(*input_size))
    out_sizes = []
    for layer in model.encoding.encoder:
        if isinstance(layer, torch.nn.MaxPool2d):
            x, idx = layer(x)
        else:
            x = layer(x)
        out_sizes.append(x.size())

    int_bytes = 0
    for out_size in out_sizes:
        # multiply by 2 - assume decoder is symmetric
        int_bytes += np.prod(np.array(out_size)) * bytes * 2

    # multiply by 2 - we need to store values AND gradients
    int_bytes *= 2

    return (input_bytes + model_bytes + int_bytes) * 1.2  # safety blanket
