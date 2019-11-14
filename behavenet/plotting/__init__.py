import os
import numpy as np


def make_dir_if_not_exists(save_file):
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def concat(ims_3channel, axis=1):
    """
    Concatenate two channels along x or y direction

    Args:
        ims_3channel (np array): (2, y_pix, x_pix)

    Returns:
        np array (2 * y_pix, x_pix) or (y_pix, 2 * x_pix)
    """
    return np.concatenate([ims_3channel[0, :, :], ims_3channel[1, :, :]], axis=axis)
