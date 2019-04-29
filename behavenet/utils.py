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


def get_best_model_version(model_path, measure='loss'):

    import pandas as pd
    import os

    # gather all versions
    def get_dirs(path):
        return next(os.walk(model_path))[1]

    versions = get_dirs(model_path)

    # load csv files with model metrics (saved out from test tube)
    metrics = []
    for i, version in enumerate(versions):
        # read metrics csv file
        try:
            metric = pd.read_csv(
                os.path.join(model_path, version, 'metrics.csv'))
        except:
            continue
        # get validation loss of best model # TODO: user-supplied measure
        val_loss = metric.val_loss.min()
        metrics.append(pd.DataFrame({
            'loss': val_loss,
            'version': version}, index=[i]))
    # put everything in pandas dataframe
    metrics_df = pd.concat(metrics, sort=False)
    # get version with smallest loss
    best_version = metrics_df['version'][metrics_df['loss'].idxmin()]

    return best_version
