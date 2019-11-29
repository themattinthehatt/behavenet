"""Utility functions shared across multiple plotting modules."""

import os
import numpy as np


def make_dir_if_not_exists(save_file):
    """Utility function for creating necessary dictories for a specified filename.

    Parameters
    ----------
    save_file : :obj:`str`
        absolute path of save file

    """
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def concat(ims, axis=1):
    """Concatenate two channels along x or y direction (useful for data with multiple views).

    Parameters
    ----------
        ims : :obj:`np.ndarray`
            shape (2, y_pix, x_pix)

    Returns
    -------
    :obj:`np.ndarray`
        shape (2 * y_pix, x_pix) (if :obj:`axis=0`) or shape (y_pix, 2 * x_pix) (if :obj:`axis=1`)
    """
    return np.concatenate([ims[0, :, :], ims[1, :, :]], axis=axis)
