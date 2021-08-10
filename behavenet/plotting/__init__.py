"""Utility functions shared across multiple plotting modules."""

from matplotlib.animation import FFMpegWriter
import numpy as np
import os
import pickle
import pandas as pd

from behavenet import make_dir_if_not_exists
from behavenet.fitting.utils import experiment_exists
from behavenet.fitting.utils import get_expt_dir
from behavenet.fitting.utils import get_session_dir
from behavenet.fitting.utils import get_best_model_version
from behavenet.fitting.utils import get_lab_example
from behavenet.fitting.utils import read_session_info_from_csv

# to ignore imports for sphix-autoapidoc
__all__ = ['concat', 'get_crop', 'load_latents', 'load_metrics_csv_as_df', 'save_movie']

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


def get_crop(im, y_0, y_ext, x_0, x_ext):
    """Get crop of image, filling in borders with zeros.

    Parameters
    ----------
    im : :obj:`np.ndarray`
        input image
    y_0 : :obj:`int`
        y-pixel center value
    y_ext : :obj:`int`
        y-pixel extent; crop in y-direction will be [y_0 - y_ext, y_0 + y_ext]
    x_0 : :obj:`int`
        y-pixel center value
    x_ext : :obj:`int`
        x-pixel extent; crop in x-direction will be [x_0 - x_ext, x_0 + x_ext]

    Returns
    -------
    :obj:`np.ndarray`
        cropped image

    """
    y_min = y_0 - y_ext
    y_max = y_0 + y_ext
    y_pix = y_max - y_min
    x_min = x_0 - x_ext
    x_max = x_0 + x_ext
    x_pix = x_max - x_min
    im_crop = np.copy(im[y_min:y_max, x_min:x_max])
    y_pix_, x_pix_ = im_crop.shape
    im_tmp = np.zeros((y_pix, x_pix))
    im_tmp[:y_pix_, :x_pix_] = im_crop
    return im_tmp


def load_latents(hparams, version, dtype='val'):
    """Load all latents as a single array.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify both a model and the associated data
    version : :obj:`int`
        version from test tube experiment defined in :obj:`hparams`
    dtype : :obj:`str`
        'train' | 'val' | 'test'

    Returns
    -------
    :obj:`np.ndarray`
        shape (time, n_latents)

    """
    sess_id = str('%s_%s_%s_%s_latents.pkl' % (
        hparams['lab'], hparams['expt'], hparams['animal'], hparams['session']))
    filename = os.path.join(
        hparams['expt_dir'], 'version_%i' % version, sess_id)
    if not os.path.exists(filename):
        raise FileNotFoundError('latents located at %s do not exist' % filename)
    latent_dict = pickle.load(open(filename, 'rb'))
    print('loaded latents from %s' % filename)
    # get all test latents
    latents = []
    for trial in latent_dict['trials'][dtype]:
        ls = latent_dict['latents'][trial]
        latents.append(ls)
    return np.concatenate(latents)


def load_metrics_csv_as_df(
        hparams, lab, expt, metrics_list, test=False, version='best', version_dir=None):
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
        names of metrics to pull from csv; do not prepend with 'tr', 'val', or 'test'
    test : :obj:`bool`
        True to only return test values (computed once at end of training)
    version: :obj:`str`
        `best` to find best model in tt expt, None to find model with hyperparams defined in
        `hparams`, int to load specific model

    Returns
    -------
    :obj:`pandas.DataFrame` object

    """

    # programmatically fill out other hparams options
    if version_dir is None:
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
        if test:
            test_dict = {
                'dataset': dataset,
                'epoch': row['epoch'],
                'dtype': 'test'}
            for metric in metrics_list:
                metrics_df.append(pd.DataFrame(
                    {**test_dict, 'loss': metric, 'val': row['test_%s' % metric]}, index=[0]))
        else:
            # make dict for val data
            val_dict = {
                'dataset': dataset,
                'epoch': row['epoch'],
                'dtype': 'val'}
            for metric in metrics_list:
                metrics_df.append(pd.DataFrame(
                    {**val_dict, 'loss': metric, 'val': row['val_%s' % metric]}, index=[0]))
            # NOTE: grayed out lines are old version that returns a single dataframe row containing
            # all losses per epoch; new way creates one row per loss, making it easy to use with
            # seaborn's FacetGrid object for multi-axis plotting for metric in metrics_list:
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


def save_movie(save_file, ani, frame_rate=15):
    """Save out matplotlib ArtistAnimation

    Parameters
    ----------
    save_file : :obj:`str`
        full save file (path and filename)
    ani : :obj:`matplotlib.animation.ArtistAnimation` object
        animation to save
    frame_rate : :obj:`int`, optional
        frame rate of saved movie

    """

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        if save_file[-3:] == 'gif':
            print('saving video to %s...' % save_file, end='')
            ani.save(save_file, writer='imagemagick', fps=frame_rate)
            print('done')
        else:
            if save_file[-3:] != 'mp4':
                save_file += '.mp4'
            writer = FFMpegWriter(fps=frame_rate, bitrate=-1)
            print('saving video to %s...' % save_file, end='')
            ani.save(save_file, writer=writer)
            print('done')
