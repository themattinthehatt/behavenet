"""Utility functions shared across multiple plotting modules."""

import numpy as np
import os
import pandas as pd

from behavenet.fitting.utils import experiment_exists
from behavenet.fitting.utils import get_expt_dir
from behavenet.fitting.utils import get_session_dir
from behavenet.fitting.utils import get_best_model_version
from behavenet.fitting.utils import get_lab_example
from behavenet.fitting.utils import read_session_info_from_csv

# to ignore imports for sphix-autoapidoc
__all__ = ['load_metrics_csv_as_df']

# TODO: use load_metrics_csv_as_df in ae example notebook


def concat(ims, axis=1):
    """Concatenate two channels along x or y direction (useful for data with multiple views).

    Parameters
    ----------
    ims : :obj:`np.ndarray`
        shape (2, y_pix, x_pix)
    axis : :obj:`int`
        axis along which to concatenate; 0 = y dir, 1 = x dir

    Returns
    -------
    :obj:`np.ndarray`
        shape (2 * y_pix, x_pix) (if :obj:`axis=0`) or shape (y_pix, 2 * x_pix) (if :obj:`axis=1`)
    """
    return np.concatenate([ims[0, :, :], ims[1, :, :]], axis=axis)


def load_metrics_csv_as_df(hparams, lab, expt, metrics_list, version='best'):
    """Load metrics csv file and return as a pandas dataframe for easy plotting.

    Parameters
    ----------
    hparams : :obj:`dict`
        requires `sessions_csv`, `multisession`, `lab`, `expt`, `animal` and `session`
    lab : :obj:`str`
        for `get_lab_example`
    expt : :obj:`str`
        for `get_lab_example`
    metrics_list : :obj:`list`
        names of metrics to pull from csv; do not prepend with 'tr', 'val', or 'test
    version: :obj:`str`
        `best` to find best model in tt expt, None to find model with hyperparams defined in
        `hparams`, int to load specific model

    Returns
    -------
    :obj:`pandas.DataFrame` object

    """

    # programmatically fill out other hparams options
    get_lab_example(hparams, lab, expt)
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)

    # find metrics csv file
    if version is 'best':
        version = get_best_model_version(hparams['expt_dir'])[0]
    elif isinstance(version, int):
        version = version
    else:
        _, version = experiment_exists(hparams, which_version=True)
    version_dir = os.path.join(hparams['expt_dir'], 'version_%i' % version)
    metric_file = os.path.join(version_dir, 'metrics.csv')
    metrics = pd.read_csv(metric_file)

    # collect data from csv file
    sess_ids = read_session_info_from_csv(os.path.join(version_dir, 'session_info.csv'))
    sess_ids_strs = []
    for sess_id in sess_ids:
        sess_ids_strs.append(str('%s/%s' % (sess_id['animal'], sess_id['session'])))
    metrics_df = []
    for i, row in metrics.iterrows():
        dataset = 'all' if row['dataset'] == -1 else sess_ids_strs[row['dataset']]
        # make dict for val data
        val_dict = {
            'dataset': dataset,
            'epoch': row['epoch'],
            'dtype': 'val'}
        for metric in metrics_list:
            metrics_df.append(pd.DataFrame(
                {**val_dict, 'loss': metric, 'val': row['val_%s' % metric]}, index=[0]))
        # NOTE: grayed out lines are old version that returns a single dataframe row containing all
        # losses per epoch; new way creates one row per loss, making it easy to use with seaborn's
        # FacetGrid object for multi-axis plotting
        # for metric in metrics_list:
        #     val_dict[metric] = row['val_%s' % metric]
        # metrics_df.append(pd.DataFrame(val_dict, index=[0]))
        # make dict for train data
        tr_dict = {
            'dataset': dataset,
            'epoch': row['epoch'],
            'dtype': 'train'}
        for metric in metrics_list:
            metrics_df.append(pd.DataFrame(
                {**tr_dict, 'loss': metric, 'val': row['tr_%s' % metric]}, index=[0]))
        # for metric in metrics_list:
        #     tr_dict[metric] = row['tr_%s' % metric]
        # metrics_df.append(pd.DataFrame(tr_dict, index=[0]))
    return pd.concat(metrics_df, sort=True)
