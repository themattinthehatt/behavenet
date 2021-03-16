import os
import copy
import pickle
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from behavenet import get_user_dir
from behavenet import make_dir_if_not_exists
from behavenet.data.utils import build_data_generator
from behavenet.data.utils import load_labels_like_latents
from behavenet.fitting.eval import get_reconstruction
from behavenet.fitting.utils import experiment_exists
from behavenet.fitting.utils import get_best_model_and_data
from behavenet.fitting.utils import get_expt_dir
from behavenet.fitting.utils import get_lab_example
from behavenet.fitting.utils import get_session_dir
from behavenet.plotting import concat
from behavenet.plotting import get_crop
from behavenet.plotting import load_latents
from behavenet.plotting import load_metrics_csv_as_df
from behavenet.plotting import save_movie

# to ignore imports for sphix-autoapidoc
__all__ = [
    'get_input_range', 'compute_range', 'get_labels_2d_for_trial', 'get_model_input',
    'interpolate_2d', 'interpolate_1d', 'interpolate_point_path', 'plot_2d_frame_array',
    'plot_1d_frame_array', 'make_interpolated', 'make_interpolated_multipanel',
    'plot_psvae_training_curves', 'plot_hyperparameter_search_results',
    'plot_label_reconstructions', 'plot_latent_traversals', 'make_latent_traversal_movie',
    'plot_mspsvae_training_curves']


# ----------------------------------------
# low-level util functions
# ----------------------------------------

def get_input_range(
        input_type, hparams, sess_ids=None, sess_idx=0, model=None, data_gen=None, version=0,
        min_p=5, max_p=95, apply_label_masks=False):
    """Helper function to compute input range for a variety of data types.

    Parameters
    ----------
    input_type : :obj:`str`
        'latents' | 'labels' | 'labels_sc'
    hparams : :obj:`dict`
        needs to contain enough information to specify an autoencoder
    sess_ids : :obj:`list`, optional
        each entry is a session dict with keys 'lab', 'expt', 'animal', 'session'; for loading
        labels and labels_sc
    sess_idx : :obj:`int`, optional
        session index into data generator
    model : :obj:`AE` object, optional
        for generating latents if latent file does not exist
    data_gen : :obj:`ConcatSessionGenerator` object, optional
        for generating latents if latent file does not exist
    version : :obj:`int`, optional
        specify AE version for loading latents
    min_p : :obj:`int`, optional
        defines lower end of range; percentile in [0, 100]
    max_p : :obj:`int`, optional
        defines upper end of range; percentile in [0, 100]
    apply_label_masks : :obj:`bool`, optional
        `True` to set masked values to NaN in labels

    Returns
    -------
    :obj:`dict`
        keys are 'min' and 'max'

    """
    if input_type == 'latents':
        # load latents
        if sess_ids is not None and sess_idx is not None:
            latent_file = str('%s_%s_%s_%s_latents.pkl' % (
                sess_ids[sess_idx]['lab'], sess_ids[sess_idx]['expt'],
                sess_ids[sess_idx]['animal'], sess_ids[sess_idx]['session']))
        else:
            latent_file = str('%s_%s_%s_%s_latents.pkl' % (
                hparams['lab'], hparams['expt'], hparams['animal'], hparams['session']))
        filename = os.path.join(
            hparams['expt_dir'], 'version_%i' % version, latent_file)
        if not os.path.exists(filename):
            from behavenet.fitting.eval import export_latents
            print('latents file not found at %s' % filename)
            print('exporting latents...', end='')
            filenames = export_latents(data_gen, model)
            filename = filenames[0]
            print('done')
        latents = pickle.load(open(filename, 'rb'))
        inputs = latents['latents']
    elif input_type == 'labels':
        labels = load_labels_like_latents(hparams, sess_ids, sess_idx=sess_idx)
        inputs = labels['latents']
    elif input_type == 'labels_sc':
        hparams2 = copy.deepcopy(hparams)
        hparams2['conditional_encoder'] = True  # to actually return labels
        labels_sc = load_labels_like_latents(
            hparams2, sess_ids, sess_idx=sess_idx, data_key='labels_sc')
        inputs = labels_sc['latents']
    else:
        raise NotImplementedError

    if apply_label_masks:
        masks = load_labels_like_latents(
            hparams, sess_ids, sess_idx=sess_idx, data_key='labels_masks')
        for i, m in zip(inputs, masks):
            i[m == 0] = np.nan

    input_range = compute_range(inputs, min_p=min_p, max_p=max_p)
    return input_range


def compute_range(values_list, min_p=5, max_p=95):
    """Compute min and max of a list of numbers using percentiles.

    Parameters
    ----------
    values_list : :obj:`list`
        list of np.ndarrays; min/max calculated over axis 0 once all lists are vertically stacked
    min_p : :obj:`int`
        defines lower end of range; percentile in [0, 100]
    max_p : :obj:`int`
        defines upper end of range; percentile in [0, 100]

    Returns
    -------
    :obj:`dict`
        lower ['min'] and upper ['max'] range of input

    """
    if np.any([len(arr) == 0 for arr in values_list]):
        values_ = []
        for arr in values_list:
            if len(arr) != 0:
                values_.append(arr)
        values = np.vstack(values_)
    else:
        values = np.vstack(values_list)
    ranges = {
        'min': np.nanpercentile(values, min_p, axis=0),
        'max': np.nanpercentile(values, max_p, axis=0)}
    return ranges


def get_labels_2d_for_trial(
        hparams, sess_ids, trial=None, trial_idx=None, sess_idx=0, dtype='test', data_gen=None):
    """Return scaled labels (in pixel space) for a given trial.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to build a data generator
    sess_ids : :obj:`list` of :obj:`dict`
        each entry is a session dict with keys 'lab', 'expt', 'animal', 'session'
    trial : :obj:`int`, optional
        trial index into all possible trials (train, val, test); one of `trial` or `trial_idx`
        must be specified; `trial` takes precedence over `trial_idx`
    trial_idx : :obj:`int`, optional
        trial index into trial type defined by `dtype`; one of `trial` or `trial_idx` must be
        specified; `trial` takes precedence over `trial_idx`
    sess_idx : :obj:`int`, optional
        session index into data generator
    dtype : :obj:`str`, optional
        data type that is indexed by `trial_idx`; 'train' | 'val' | 'test'
    data_gen : :obj:`ConcatSessionGenerator` object, optional
        for generating labels

    Returns
    -------
    :obj:`tuple`
        - labels_2d_pt (:obj:`torch.Tensor`) of shape (batch, n_labels, y_pix, x_pix)
        - labels_2d_np (:obj:`np.ndarray`) of shape (batch, n_labels, y_pix, x_pix)

    """

    if (trial_idx is not None) and (trial is not None):
        raise ValueError('only one of "trial" or "trial_idx" can be specified')

    if data_gen is None:
        hparams_new = copy.deepcopy(hparams)
        hparams_new['conditional_encoder'] = True  # ensure scaled labels are returned
        hparams_new['device'] = 'cpu'
        hparams_new['as_numpy'] = False
        hparams_new['batch_load'] = True
        data_gen = build_data_generator(hparams_new, sess_ids, export_csv=False)

    # get trial
    if trial is None:
        trial = data_gen.datasets[sess_idx].batch_idxs[dtype][trial_idx]
    batch = data_gen.datasets[sess_idx][trial]
    labels_2d_pt = batch['labels_sc']
    labels_2d_np = labels_2d_pt.cpu().detach().numpy()

    return labels_2d_pt, labels_2d_np


def get_model_input(
        data_generator, hparams, model, trial=None, trial_idx=None, sess_idx=0, max_frames=200,
        compute_latents=False, compute_2d_labels=True, compute_scaled_labels=False, dtype='test'):
    """Return images, latents, and labels for a given trial.

    Parameters
    ----------
    data_generator: :obj:`ConcatSessionGenerator`
        for generating model input
    hparams : :obj:`dict`
        needs to contain enough information to specify both a model and the associated data
    model : :obj:`behavenet.models` object
        model type
    trial : :obj:`int`, optional
        trial index into all possible trials (train, val, test); one of `trial` or `trial_idx`
        must be specified; `trial` takes precedence over `trial_idx`
    trial_idx : :obj:`int`, optional
        trial index into trial type defined by `dtype`; one of `trial` or `trial_idx` must be
        specified; `trial` takes precedence over `trial_idx`
    sess_idx : :obj:`int`, optional
        session index into data generator
    max_frames : :obj:`int`, optional
        maximum size of batch to return
    compute_latents : :obj:`bool`, optional
        `True` to return latents
    compute_2d_labels : :obj:`bool`, optional
        `True` to return 2d label tensors of shape (batch, n_labels, y_pix, x_pix)
    compute_scaled_labels : :obj:`bool`, optional
        ignored if `compute_2d_labels` is `True`; if `compute_scaled_labels=True`, return scaled
        labels as shape (batch, n_labels) rather than 2d labels as shape
        (batch, n_labels, y_pix, x_pix).
    dtype : :obj:`str`, optional
        data type that is indexed by `trial_idx`; 'train' | 'val' | 'test'

    Returns
    -------
    :obj:`tuple`
        - ims_pt (:obj:`torch.Tensor`) of shape (max_frames, n_channels, y_pix, x_pix)
        - ims_np (:obj:`np.ndarray`) of shape (max_frames, n_channels, y_pix, x_pix)
        - latents_np (:obj:`np.ndarray`) of shape (max_frames, n_latents)
        - labels_pt (:obj:`torch.Tensor`) of shape (max_frames, n_labels)
        - labels_2d_pt (:obj:`torch.Tensor`) of shape (max_frames, n_labels, y_pix, x_pix)
        - labels_2d_np (:obj:`np.ndarray`) of shape (max_frames, n_labels, y_pix, x_pix)

    """

    if (trial_idx is not None) and (trial is not None):
        raise ValueError('only one of "trial" or "trial_idx" can be specified')
    if (trial_idx is None) and (trial is None):
        raise ValueError('one of "trial" or "trial_idx" must be specified')

    # get trial
    if trial is None:
        trial = data_generator.datasets[sess_idx].batch_idxs[dtype][trial_idx]
    batch = data_generator.datasets[sess_idx][trial]
    ims_pt = batch['images'][:max_frames]
    ims_np = ims_pt.cpu().detach().numpy()

    # continuous labels
    if hparams['model_class'] == 'ae' \
            or hparams['model_class'] == 'vae' \
            or hparams['model_class'] == 'beta-tcvae':
        labels_pt = None
        labels_np = None
    elif hparams['model_class'] == 'cond-ae' \
            or hparams['model_class'] == 'cond-vae' \
            or hparams['model_class'] == 'cond-ae-msp' \
            or hparams['model_class'] == 'ps-vae' \
            or hparams['model_class'] == 'msps-vae' \
            or hparams['model_class'] == 'labels-images':
        labels_pt = batch['labels'][:max_frames]
        labels_np = labels_pt.cpu().detach().numpy()
    else:
        raise NotImplementedError

    # one hot labels
    if hparams['conditional_encoder']:
        labels_2d_pt = batch['labels_sc'][:max_frames]
        labels_2d_np = labels_2d_pt.cpu().detach().numpy()
    else:
        if compute_2d_labels:
            hparams['session_dir'], sess_ids = get_session_dir(hparams)
            labels_2d_pt, labels_2d_np = get_labels_2d_for_trial(hparams, sess_ids, trial=trial)
        elif compute_scaled_labels:
            labels_2d_pt = None
            import h5py
            hdf5_file = data_generator.datasets[sess_idx].paths['labels']
            with h5py.File(hdf5_file, 'r', libver='latest', swmr=True) as f:
                labels_2d_np = f['labels_sc'][str('trial_%04i' % trial)][()].astype('float32')
        else:
            labels_2d_pt, labels_2d_np = None, None

    # latents
    if compute_latents:
        if hparams['model_class'] == 'cond-ae-msp' or hparams['model_class'] == 'ps-vae':
            latents_np = model.get_transformed_latents(ims_pt, dataset=sess_idx, as_numpy=True)
        else:
            _, latents_np = get_reconstruction(
                model, ims_pt, labels=labels_pt, labels_2d=labels_2d_pt, return_latents=True)
    else:
        latents_np = None

    return ims_pt, ims_np, latents_np, labels_pt, labels_np, labels_2d_pt, labels_2d_np


def interpolate_2d(
        interp_type, model, ims_0, latents_0, labels_0, labels_sc_0, mins, maxes, input_idxs,
        n_frames, crop_type=None, mins_sc=None, maxes_sc=None, crop_kwargs=None,
        marker_idxs=None, ch=0):
    """Return reconstructed images created by interpolating through latent/label space.

    Parameters
    ----------
    interp_type : :obj:`str`
        'latents' | 'labels'
    model : :obj:`behavenet.models` object
        autoencoder model
    ims_0 : :obj:`torch.Tensor`
        base images for interpolating labels, of shape (1, n_channels, y_pix, x_pix)
    latents_0 : :obj:`np.ndarray`
        base latents of shape (1, n_latents); only two of these dimensions will be changed if
        `interp_type='latents'`
    labels_0 : :obj:`np.ndarray`
        base labels of shape (1, n_labels)
    labels_sc_0 : :obj:`np.ndarray`
        base scaled labels in pixel space of shape (1, n_labels, y_pix, x_pix)
    mins : :obj:`array-like`
        minimum values of labels/latents, one for each dim
    maxes : :obj:`list`
        maximum values of labels/latents, one for each dim
    input_idxs : :obj:`list`
        indices of labels/latents that will be interpolated; for labels, must be y first, then x
        for proper marker recording
    n_frames : :obj:`int`
        number of interpolation points between mins and maxes (inclusive)
    crop_type : :obj:`str` or :obj:`NoneType`, optional
        currently only implements 'fixed'; if not None, cropped images are returned, and returned
        labels are also cropped so that they can be plotted on top of the cropped images; if None,
        returned cropped images are empty and labels are relative to original image size
    mins_sc : :obj:`list`, optional
        min values of scaled labels that correspond to min values of labels when using conditional
        encoders
    maxes_sc : :obj:`list`, optional
        max values of scaled labels that correspond to max values of labels when using conditional
        encoders
    crop_kwargs : :obj:`dict`, optional
        define center and extent of crop if `crop_type='fixed'`; keys are 'x_0', 'x_ext', 'y_0',
        'y_ext'
    marker_idxs : :obj:`list`, optional
        indices of `labels_sc_0` that will be interpolated; note that this is analogous but
        different from `input_idxs`, since the 2d tensor `labels_sc_0` has half as many label
        dimensions as `latents_0` and `labels_0`
    ch : :obj:`int`, optional
        specify which channel of input images to return (can only be a single value)

    Returns
    -------
    :obj:`tuple`
        - ims_list (:obj:`list` of :obj:`list` of :obj:`np.ndarray`) interpolated images
        - labels_list (:obj:`list` of :obj:`list` of :obj:`np.ndarray`) interpolated labels
        - ims_crop_list (:obj:`list` of :obj:`list` of :obj:`np.ndarray`) interpolated , cropped
          images

    """

    if interp_type == 'labels':
        from behavenet.data.transforms import MakeOneHot2D
        _, _, y_pix, x_pix = ims_0.shape
        one_hot_2d = MakeOneHot2D(y_pix, x_pix)

    # compute grid for relevant inputs
    n_interp_dims = len(input_idxs)
    assert n_interp_dims == 2

    # compute ranges for relevant inputs
    inputs = []
    inputs_sc = []
    for d in input_idxs:
        inputs.append(np.linspace(mins[d], maxes[d], n_frames))
        if mins_sc is not None and maxes_sc is not None:
            inputs_sc.append(np.linspace(mins_sc[d], maxes_sc[d], n_frames))
        else:
            if interp_type == 'labels':
                raise NotImplementedError

    ims_list = []
    ims_crop_list = []
    labels_list = []
    # latent_vals = []
    for i0 in range(n_frames):

        ims_tmp = []
        ims_crop_tmp = []
        labels_tmp = []
        # latents_tmp = []

        for i1 in range(n_frames):

            if interp_type == 'latents':

                # get (new) latents
                latents = np.copy(latents_0)
                latents[0, input_idxs[0]] = inputs[0][i0]
                latents[0, input_idxs[1]] = inputs[1][i1]

                # get scaled labels (for markers)
                labels_sc = _get_updated_scaled_labels(labels_sc_0)

                if model.hparams['model_class'] == 'cond-ae-msp':
                    # get reconstruction
                    im_tmp = get_reconstruction(
                        model,
                        torch.from_numpy(latents).float(),
                        apply_inverse_transform=True)
                else:
                    # get labels
                    if model.hparams['model_class'] == 'ae' \
                            or model.hparams['model_class'] == 'vae' \
                            or model.hparams['model_class'] == 'beta-tcvae' \
                            or model.hparams['model_class'] == 'ps-vae':
                        labels = None
                    elif model.hparams['model_class'] == 'cond-ae' \
                            or model.hparams['model_class'] == 'cond-vae':
                        labels = torch.from_numpy(labels_0).float()
                    else:
                        raise NotImplementedError
                    # get reconstruction
                    im_tmp = get_reconstruction(
                        model,
                        torch.from_numpy(latents).float(),
                        labels=labels)

            elif interp_type == 'labels':

                # get (new) scaled labels
                labels_sc = _get_updated_scaled_labels(
                    labels_sc_0, input_idxs, [inputs_sc[0][i0], inputs_sc[1][i1]])
                if len(labels_sc_0.shape) == 4:
                    # 2d scaled labels
                    labels_2d = torch.from_numpy(one_hot_2d(labels_sc)).float()
                else:
                    # 1d scaled labels
                    labels_2d = None

                if model.hparams['model_class'] == 'cond-ae-msp' \
                        or model.hparams['model_class'] == 'ps-vae':
                    # change latents that correspond to desired labels
                    latents = np.copy(latents_0)
                    latents[0, input_idxs[0]] = inputs[0][i0]
                    latents[0, input_idxs[1]] = inputs[1][i1]
                    # get reconstruction
                    im_tmp = get_reconstruction(model, latents, apply_inverse_transform=True)
                else:
                    # get (new) labels
                    labels = np.copy(labels_0)
                    labels[0, input_idxs[0]] = inputs[0][i0]
                    labels[0, input_idxs[1]] = inputs[1][i1]
                    # get reconstruction
                    im_tmp = get_reconstruction(
                        model,
                        ims_0,
                        labels=torch.from_numpy(labels).float(),
                        labels_2d=labels_2d)
            else:
                raise NotImplementedError

            ims_tmp.append(np.copy(im_tmp[0, ch]))

            if crop_type:
                x_min_tmp = crop_kwargs['x_0'] - crop_kwargs['x_ext']
                y_min_tmp = crop_kwargs['y_0'] - crop_kwargs['y_ext']
            else:
                x_min_tmp = 0
                y_min_tmp = 0

            if interp_type == 'labels':
                labels_tmp.append([
                    np.copy(labels_sc[0, input_idxs[0]]) - y_min_tmp,
                    np.copy(labels_sc[0, input_idxs[1]]) - x_min_tmp])
            elif interp_type == 'latents' and labels_sc_0 is not None:
                labels_tmp.append([
                    np.copy(labels_sc[0, marker_idxs[0]]) - y_min_tmp,
                    np.copy(labels_sc[0, marker_idxs[1]]) - x_min_tmp])
            else:
                labels_tmp.append([np.nan, np.nan])

            if crop_type:
                ims_crop_tmp.append(get_crop(
                    im_tmp[0, 0], crop_kwargs['y_0'], crop_kwargs['y_ext'], crop_kwargs['x_0'],
                    crop_kwargs['x_ext']))
            else:
                ims_crop_tmp.append([])

        ims_list.append(ims_tmp)
        ims_crop_list.append(ims_crop_tmp)
        labels_list.append(labels_tmp)

    return ims_list, labels_list, ims_crop_list


def interpolate_1d(
        interp_type, model, ims_0, latents_0, labels_0, labels_sc_0, mins, maxes, input_idxs,
        n_frames, crop_type=None, mins_sc=None, maxes_sc=None, crop_kwargs=None,
        marker_idxs=None, ch=0):
    """Return reconstructed images created by interpolating through latent/label space.

    Parameters
    ----------
    interp_type : :obj:`str`
        'latents' | 'labels'
    model : :obj:`behavenet.models` object
        autoencoder model
    ims_0 : :obj:`torch.Tensor`
        base images for interpolating labels, of shape (1, n_channels, y_pix, x_pix)
    latents_0 : :obj:`np.ndarray`
        base latents of shape (1, n_latents); only two of these dimensions will be changed if
        `interp_type='latents'`
    labels_0 : :obj:`np.ndarray`
        base labels of shape (1, n_labels)
    labels_sc_0 : :obj:`np.ndarray`
        base scaled labels in pixel space of shape (1, n_labels, y_pix, x_pix)
    mins : :obj:`array-like`
        minimum values of all labels/latents
    maxes : :obj:`array-like`
        maximum values of all labels/latents
    input_idxs : :obj:`array-like`
        indices of labels/latents that will be interpolated
    n_frames : :obj:`int`
        number of interpolation points between mins and maxes (inclusive)
    crop_type : :obj:`str` or :obj:`NoneType`, optional
        currently only implements 'fixed'; if not None, cropped images are returned, and returned
        labels are also cropped so that they can be plotted on top of the cropped images; if None,
        returned cropped images are empty and labels are relative to original image size
    mins_sc : :obj:`list`, optional
        min values of scaled labels that correspond to min values of labels when using conditional
        encoders
    maxes_sc : :obj:`list`, optional
        max values of scaled labels that correspond to max values of labels when using conditional
        encoders
    crop_kwargs : :obj:`dict`, optional
        define center and extent of crop if `crop_type='fixed'`; keys are 'x_0', 'x_ext', 'y_0',
        'y_ext'
    marker_idxs : :obj:`list`, optional
        indices of `labels_sc_0` that will be interpolated; note that this is analogous but
        different from `input_idxs`, since the 2d tensor `labels_sc_0` has half as many label
        dimensions as `latents_0` and `labels_0`
    ch : :obj:`int`, optional
        specify which channel of input images to return (can only be a single value)

    Returns
    -------
    :obj:`tuple`
        - ims_list (:obj:`list` of :obj:`list` of :obj:`np.ndarray`) interpolated images
        - labels_list (:obj:`list` of :obj:`list` of :obj:`np.ndarray`) interpolated labels
        - ims_crop_list (:obj:`list` of :obj:`list` of :obj:`np.ndarray`) interpolated , cropped
          images

    """

    if interp_type == 'labels':
        from behavenet.data.transforms import MakeOneHot2D
        _, _, y_pix, x_pix = ims_0.shape
        one_hot_2d = MakeOneHot2D(y_pix, x_pix)

    n_interp_dims = len(input_idxs)

    # compute ranges for relevant inputs
    inputs = []
    inputs_sc = []
    for d in input_idxs:
        inputs.append(np.linspace(mins[d], maxes[d], n_frames))
        if mins_sc is not None and maxes_sc is not None:
            inputs_sc.append(np.linspace(mins_sc[d], maxes_sc[d], n_frames))
        else:
            if interp_type == 'labels':
                raise NotImplementedError

    ims_list = []
    ims_crop_list = []
    labels_list = []
    # latent_vals = []
    for i0 in range(n_interp_dims):

        ims_tmp = []
        ims_crop_tmp = []
        labels_tmp = []

        for i1 in range(n_frames):

            if interp_type == 'latents':

                # get (new) latents
                latents = np.copy(latents_0)
                latents[0, input_idxs[i0]] = inputs[i0][i1]

                # get scaled labels (for markers)
                labels_sc = _get_updated_scaled_labels(labels_sc_0)

                if model.hparams['model_class'] == 'cond-ae-msp':
                    # get reconstruction
                    im_tmp = get_reconstruction(
                        model,
                        torch.from_numpy(latents).float(),
                        apply_inverse_transform=True)
                else:
                    # get labels
                    if model.hparams['model_class'] == 'ae' \
                            or model.hparams['model_class'] == 'vae' \
                            or model.hparams['model_class'] == 'beta-tcvae' \
                            or model.hparams['model_class'] == 'ps-vae':
                        labels = None
                    elif model.hparams['model_class'] == 'cond-ae' \
                            or model.hparams['model_class'] == 'cond-vae':
                        labels = torch.from_numpy(labels_0).float()
                    else:
                        raise NotImplementedError
                    # get reconstruction
                    im_tmp = get_reconstruction(
                        model,
                        torch.from_numpy(latents).float(),
                        labels=labels)

            elif interp_type == 'labels':

                # get (new) scaled labels
                labels_sc = _get_updated_scaled_labels(
                    labels_sc_0, input_idxs[i0], inputs_sc[i0][i1])
                if len(labels_sc_0.shape) == 4:
                    # 2d scaled labels
                    labels_2d = torch.from_numpy(one_hot_2d(labels_sc)).float()
                else:
                    # 1d scaled labels
                    labels_2d = None

                if model.hparams['model_class'] == 'cond-ae-msp' \
                        or model.hparams['model_class'] == 'ps-vae':
                    # change latents that correspond to desired labels
                    latents = np.copy(latents_0)
                    latents[0, input_idxs[i0]] = inputs[i0][i1]
                    # get reconstruction
                    im_tmp = get_reconstruction(model, latents, apply_inverse_transform=True)
                else:
                    # get (new) labels
                    labels = np.copy(labels_0)
                    labels[0, input_idxs[i0]] = inputs[i0][i1]
                    # get reconstruction
                    im_tmp = get_reconstruction(
                        model,
                        ims_0,
                        labels=torch.from_numpy(labels).float(),
                        labels_2d=labels_2d)
            else:
                raise NotImplementedError

            ims_tmp.append(np.copy(im_tmp[0, ch]))

            if crop_type:
                x_min_tmp = crop_kwargs['x_0'] - crop_kwargs['x_ext']
                y_min_tmp = crop_kwargs['y_0'] - crop_kwargs['y_ext']
            else:
                x_min_tmp = 0
                y_min_tmp = 0

            if interp_type == 'labels':
                labels_tmp.append([
                    np.copy(labels_sc[0, input_idxs[0]]) - y_min_tmp,
                    np.copy(labels_sc[0, input_idxs[1]]) - x_min_tmp])
            elif interp_type == 'latents' and labels_sc_0 is not None:
                labels_tmp.append([
                    np.copy(labels_sc[0, marker_idxs[0]]) - y_min_tmp,
                    np.copy(labels_sc[0, marker_idxs[1]]) - x_min_tmp])
            else:
                labels_tmp.append([np.nan, np.nan])

            if crop_type:
                ims_crop_tmp.append(get_crop(
                    im_tmp[0, 0], crop_kwargs['y_0'], crop_kwargs['y_ext'], crop_kwargs['x_0'],
                    crop_kwargs['x_ext']))
            else:
                ims_crop_tmp.append([])

        ims_list.append(ims_tmp)
        ims_crop_list.append(ims_crop_tmp)
        labels_list.append(labels_tmp)

    return ims_list, labels_list, ims_crop_list


def interpolate_point_path(
        interp_type, model, ims_0, labels_0, points, n_frames=10, ch=0, crop_kwargs=None,
        apply_inverse_transform=True):
    """Return reconstructed images created by interpolating through multiple points.

    This function is a simplified version of :func:`interpolate_1d()`; this function computes a
    traversal for a single dimension instead of all dimensions; also, this function does not
    support conditional encoders, nor does it attempt to compute the interpolated, scaled values
    of the labels as :func:`interpolate_1d()` does. This function should supercede
    :func:`interpolate_1d()` in a future refactor. Also note that this function is utilized by
    the code to make traversal movies, whereas :func:`interpolate_1d()` is utilized by the code to
    make traversal plots.

    Parameters
    ----------
    interp_type : :obj:`str`
        'latents' | 'labels'
    model : :obj:`behavenet.models` object
        autoencoder model
    ims_0 : :obj:`np.ndarray`
        base images for interpolating labels, of shape (1, n_channels, y_pix, x_pix)
    labels_0 : :obj:`np.ndarray`
        base labels of shape (1, n_labels); these values will be used if
        `interp_type='latents'`, and they will be ignored if `inter_type='labels'`
        (since `points` will be used)
    points : :obj:`list`
        one entry for each point in path; each entry is an np.ndarray of shape (n_latents,)
    n_frames : :obj:`int` or :obj:`array-like`
        number of interpolation points between each point; can be an integer that is used
        for all paths, or an array/list of length one less than number of points
    ch : :obj:`int`, optional
        specify which channel of input images to return; if not an int, all channels are
        concatenated in the horizontal dimension
    crop_kwargs : :obj:`dict`, optional
        if crop_type is not None, provides information about the crop (for a fixed crop window)
        keys : 'y_0', 'x_0', 'y_ext', 'x_ext'; window is
        (y_0 - y_ext, y_0 + y_ext) in vertical direction and
        (x_0 - x_ext, x_0 + x_ext) in horizontal direction
    apply_inverse_transform : :obj:`bool`
        if inputs are latents (and model class is 'cond-ae-msp' or 'ps-vae'), apply inverse
        transform to put in original latent space

    Returns
    -------
    :obj:`tuple`
        - ims_list (:obj:`list` of :obj:`np.ndarray`) interpolated images
        - inputs_list (:obj:`list` of :obj:`np.ndarray`) interpolated values

    """

    if model.hparams.get('conditional_encoder', False):
        raise NotImplementedError

    n_points = len(points)
    if isinstance(n_frames, int):
        n_frames = [n_frames] * (n_points - 1)
    assert len(n_frames) == (n_points - 1)

    ims_list = []
    inputs_list = []

    for p in range(n_points - 1):

        p0 = points[None, p]
        p1 = points[None, p + 1]
        p_vec = (p1 - p0) / n_frames[p]

        for pn in range(n_frames[p]):

            vec = p0 + pn * p_vec

            if interp_type == 'latents':

                if model.hparams['model_class'] == 'cond-ae' \
                        or model.hparams['model_class'] == 'cond-vae':
                    im_tmp = get_reconstruction(
                        model, vec, apply_inverse_transform=apply_inverse_transform,
                        labels=torch.from_numpy(labels_0).float().to(model.hparams['device']))
                else:
                    im_tmp = get_reconstruction(
                        model, vec, apply_inverse_transform=apply_inverse_transform)

            elif interp_type == 'labels':

                if model.hparams['model_class'] == 'cond-ae-msp' \
                        or model.hparams['model_class'] == 'ps-vae' \
                        or model.hparams['model_class'] == 'msps-vae':
                    im_tmp = get_reconstruction(
                        model, vec, apply_inverse_transform=True)
                else:  # cond-ae
                    im_tmp = get_reconstruction(
                        model, ims_0,
                        labels=torch.from_numpy(vec).float().to(model.hparams['device']))
            else:
                raise NotImplementedError

            if crop_kwargs is not None:
                if not isinstance(ch, int):
                    raise ValueError('"ch" must be an integer to use crop_kwargs')
                ims_list.append(get_crop(
                    im_tmp[0, ch],
                    crop_kwargs['y_0'], crop_kwargs['y_ext'],
                    crop_kwargs['x_0'], crop_kwargs['x_ext']))
            else:
                if isinstance(ch, int):
                    ims_list.append(np.copy(im_tmp[0, ch]))
                else:
                    ims_list.append(np.copy(concat(im_tmp[0])))

            inputs_list.append(vec)

    return ims_list, inputs_list


def _get_updated_scaled_labels(labels_og, idxs=None, vals=None):
    """Helper function for interpolate_xd functions."""

    if labels_og is not None:

        if len(labels_og.shape) == 4:
            # 2d scaled labels
            tmp = np.copy(labels_og)
            t, y, x = np.where(tmp[0] == 1)
            labels_sc = np.hstack([x, y])[None, :]
        else:
            # 1d scaled labels
            labels_sc = np.copy(labels_og)

        if idxs is not None:
            if isinstance(idxs, int):
                assert isinstance(vals, float)
                idxs = [idxs]
                vals = [vals]
            else:
                assert len(idxs) == len(vals)
            for idx, val in zip(idxs, vals):
                labels_sc[0, idx] = val

    else:
        labels_sc = None

    return labels_sc


# ----------------------------------------
# mid-level plotting functions
# ----------------------------------------

def plot_2d_frame_array(
        ims_list, markers=None, im_kwargs=None, marker_kwargs=None, figsize=None, save_file=None,
        format='pdf'):
    """Plot list of list of interpolated images output by :func:`interpolate_2d()` in a 2d grid.

    Parameters
    ----------
    ims_list : :obj:`list` of :obj:`list`
        each inner list element holds an np.ndarray of shape (y_pix, x_pix)
    markers : :obj:`list` of :obj:`list` or NoneType, optional
        each inner list element holds an array-like object with values (y_pix, x_pix); if None,
        markers are not plotted on top of frames
    im_kwargs : :obj:`dict` or NoneType, optional
        kwargs for `matplotlib.pyplot.imshow()` function (vmin, vmax, cmap, etc)
    marker_kwargs : :obj:`dict` or NoneType, optional
        kwargs for `matplotlib.pyplot.plot()` function (markersize, markeredgewidth, etc)
    figsize : :obj:`tuple`, optional
        (width, height) in inches
    save_file : :obj:`str` or NoneType, optional
        figure saved if not None
    format : :obj:`str`, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...

    """

    n_y = len(ims_list)
    n_x = len(ims_list[0])
    if figsize is None:
        y_pix, x_pix = ims_list[0][0].shape
        # how many inches per pixel?
        in_per_pix = 15 / (x_pix * n_x)
        figsize = (15, in_per_pix * y_pix * n_y)
    fig, axes = plt.subplots(n_y, n_x, figsize=figsize)

    if im_kwargs is None:
        im_kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}
    if marker_kwargs is None:
        marker_kwargs = {'markersize': 20, 'markeredgewidth': 3}

    for r, ims_list_y in enumerate(ims_list):
        for c, im in enumerate(ims_list_y):
            axes[r, c].imshow(im, **im_kwargs)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if markers is not None:
                axes[r, c].plot(
                    markers[r][c][1], markers[r][c][0], 'o', **marker_kwargs)
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0, top=1, right=1)
    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, bbox_inches='tight')
    plt.show()


def plot_1d_frame_array(
        ims_list, markers=None, im_kwargs=None, marker_kwargs=None, plot_ims=True, plot_diffs=True,
        figsize=None, save_file=None, format='pdf'):
    """Plot list of list of interpolated images output by :func:`interpolate_1d()` in a 2d grid.

    Parameters
    ----------
    ims_list : :obj:`list` of :obj:`list`
        each inner list element holds an np.ndarray of shape (y_pix, x_pix)
    markers : :obj:`list` of :obj:`list` or NoneType, optional
        each inner list element holds an array-like object with values (y_pix, x_pix); if None,
        markers are not plotted on top of frames
    im_kwargs : :obj:`dict` or NoneType, optional
        kwargs for `matplotlib.pyplot.imshow()` function (vmin, vmax, cmap, etc)
    marker_kwargs : :obj:`dict` or NoneType, optional
        kwargs for `matplotlib.pyplot.plot()` function (markersize, markeredgewidth, etc)
    plot_ims : :obj:`bool`, optional
        plot images
    plot_diffs : :obj:`bool`, optional
        plot differences
    figsize : :obj:`tuple`, optional
        (width, height) in inches
    save_file : :obj:`str` or NoneType, optional
        figure saved if not None
    format : :obj:`str`, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...

    """

    if not (plot_ims or plot_diffs):
        raise ValueError('Must plot at least one of ims or diffs')

    if plot_ims and plot_diffs:
        n_y = len(ims_list) * 2
        offset = 2
    else:
        n_y = len(ims_list)
        offset = 1
    n_x = len(ims_list[0])
    if figsize is None:
        y_pix, x_pix = ims_list[0][0].shape
        # how many inches per pixel?
        in_per_pix = 15 / (x_pix * n_x)
        figsize = (15, in_per_pix * y_pix * n_y)
    fig, axes = plt.subplots(n_y, n_x, figsize=figsize)

    if im_kwargs is None:
        im_kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}
    if marker_kwargs is None:
        marker_kwargs = {'markersize': 20, 'markeredgewidth': 3}

    for r, ims_list_y in enumerate(ims_list):
        base_im = ims_list_y[0]
        for c, im in enumerate(ims_list_y):
            # plot original images
            if plot_ims:
                axes[offset * r, c].imshow(im, **im_kwargs)
                axes[offset * r, c].set_xticks([])
                axes[offset * r, c].set_yticks([])
                if markers is not None:
                    axes[offset * r, c].plot(
                        markers[r][c][1], markers[r][c][0], 'o', **marker_kwargs)
            # plot differences
            if plot_diffs and plot_ims:
                axes[offset * r + 1, c].imshow(0.5 + (im - base_im), **im_kwargs)
                axes[offset * r + 1, c].set_xticks([])
                axes[offset * r + 1, c].set_yticks([])
            elif plot_diffs:
                axes[offset * r, c].imshow(0.5 + (im - base_im), **im_kwargs)
                axes[offset * r, c].set_xticks([])
                axes[offset * r, c].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0, top=1, right=1)
    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, bbox_inches='tight')
    plt.show()


def make_interpolated(
        ims, save_file, markers=None, text=None, text_title=None, text_color=[1, 1, 1],
        frame_rate=20, scale=3, markersize=10, markeredgecolor='w', markeredgewidth=1, ax=None):
    """Make a latent space interpolation movie.

    Parameters
    ----------
    ims : :obj:`list` of :obj:`np.ndarray`
        each list element is an array of shape (y_pix, x_pix)
    save_file : :obj:`str`
        absolute path of save file; does not need file extension, will automatically be saved as
        mp4. To save as a gif, include the '.gif' file extension in `save_file`. The movie will
        only be saved if `ax` is `NoneType`; else the list of animated frames is returned
    markers : :obj:`array-like`, optional
        array of size (n_frames, 2) which specifies the (x, y) coordinates of a marker on each
        frame
    text : :obj:`array-like`, optional
        array of size (n_frames) which specifies text printed in the lower left corner of each
        frame
    text_title : :obj:`array-like`, optional
        array of size (n_frames) which specifies text printed in the upper left corner of each
        frame
    text_color : :obj:`array-like`, optional
        rgb array specifying color of `text` and `text_title`, if applicable
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    scale : :obj:`float`, optional
        width of panel is (scale / 2) inches
    markersize : :obj:`float`, optional
        size of marker if `markers` is not `NoneType`
    markeredgecolor : :obj:`float`, optional
        color of marker edge if `markers` is not `NoneType`
    markeredgewidth : :obj:`float`, optional
        width of marker edge if `markers` is not `NoneType`
    ax : :obj:`matplotlib.axes.Axes` object
        optional axis in which to plot the frames; if this argument is not `NoneType` the list of
        animated frames is returned and the movie is not saved

    Returns
    -------
    :obj:`list`
        list of list of animated frames if `ax` is True; else save movie

    """

    y_pix, x_pix = ims[0].shape

    if ax is None:
        fig_width = scale / 2
        fig_height = y_pix / x_pix * scale / 2
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        ax = plt.gca()
        return_ims = False
    else:
        return_ims = True

    ax.set_xticks([])
    ax.set_yticks([])

    default_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    txt_kwargs = {
        'fontsize': 4, 'color': text_color, 'fontname': 'monospace',
        'horizontalalignment': 'left', 'verticalalignment': 'center',
        'transform': ax.transAxes}

    # ims is a list of lists, each row is a list of artists to draw in the current frame; here we
    # are just animating one artist, the image, in each frame
    ims_ani = []
    for i, im in enumerate(ims):
        im_tmp = []
        im_tmp.append(ax.imshow(im, **default_kwargs))
        # [s.set_visible(False) for s in ax.spines.values()]
        if markers is not None:
            im_tmp.append(ax.plot(
                markers[i, 0], markers[i, 1], '.r', markersize=markersize,
                markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)[0])
        if text is not None:
            im_tmp.append(ax.text(0.02, 0.06, text[i], **txt_kwargs))
        if text_title is not None:
            im_tmp.append(ax.text(0.02, 0.92, text_title[i], **txt_kwargs))
        ims_ani.append(im_tmp)

    if return_ims:
        return ims_ani
    else:
        plt.tight_layout(pad=0)
        ani = animation.ArtistAnimation(fig, ims_ani, blit=True, repeat_delay=1000)
        save_movie(save_file, ani, frame_rate=frame_rate)


def make_interpolated_multipanel(
        ims, save_file, markers=None, text=None, text_title=None, frame_rate=20, n_cols=3, scale=1,
        **kwargs):
    """Make a multi-panel latent space interpolation movie.

    Parameters
    ----------
    ims : :obj:`list` of :obj:`list` of :obj:`np.ndarray`
        each list element is used to for a single panel, and is another list that contains arrays
        of shape (y_pix, x_pix)
    save_file : :obj:`str`
        absolute path of save file; does not need file extension, will automatically be saved as
        mp4. To save as a gif, include the '.gif' file extension in `save_file`.
    markers : :obj:`list` of :obj:`array-like`, optional
        each list element is used for a single panel, and is an array of size (n_frames, 2)
        which specifies the (x, y) coordinates of a marker on each frame for that panel
    text : :obj:`list` of :obj:`array-like`, optional
        each list element is used for a single panel, and is an array of size (n_frames) which
        specifies text printed in the lower left corner of each frame for that panel
    text_title : :obj:`list` of :obj:`array-like`, optional
        each list element is used for a single panel, and is an array of size (n_frames) which
        specifies text printed in the upper left corner of each frame for that panel
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    n_cols : :obj:`int`, optional
        movie is `n_cols` panels wide
    scale : :obj:`float`, optional
        width of panel is (scale / 2) inches
    kwargs
        arguments are additional arguments to :func:`make_interpolated`, like 'markersize',
        'markeredgewidth', 'markeredgecolor', etc.

    """

    n_panels = len(ims)

    markers = [None] * n_panels if markers is None else markers
    text = [None] * n_panels if text is None else text

    y_pix, x_pix = ims[0][0].shape
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_width = scale / 2 * n_cols
    fig_height = y_pix / x_pix * scale / 2 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    # fill out empty panels with black frames
    while len(ims) < n_rows * n_cols:
        ims.append(np.zeros(ims[0].shape))
        markers.append(None)
        text.append(None)

    # ims is a list of lists, each row is a list of artists to draw in the current frame; here we
    # are just animating one artist, the image, in each frame
    ims_ani = []
    for i, (ims_curr, markers_curr, text_curr) in enumerate(zip(ims, markers, text)):
        col = i % n_cols
        row = int(np.floor(i / n_cols))
        if i == 0:
            text_title_str = text_title
        else:
            text_title_str = None
        if n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        ims_ani_curr = make_interpolated(
            ims=ims_curr, markers=markers_curr, text=text_curr, text_title=text_title_str, ax=ax,
            save_file=None, **kwargs)
        ims_ani.append(ims_ani_curr)

    # turn off other axes
    i += 1
    while i < n_rows * n_cols:
        col = i % n_cols
        row = int(np.floor(i / n_cols))
        axes[row, col].set_axis_off()
        i += 1

    # rearrange ims:
    # currently a list of length n_panels, each element of which is a list of length n_t
    # we need a list of length n_t, each element of which is a list of length n_panels
    n_frames = len(ims_ani[0])
    ims_final = [[] for _ in range(n_frames)]
    for i in range(n_frames):
        for j in range(n_panels):
            ims_final[i] += ims_ani[j][i]

    ani = animation.ArtistAnimation(fig, ims_final, blit=True, repeat_delay=1000)
    save_movie(save_file, ani, frame_rate=frame_rate)


# ----------------------------------------
# high-level plotting functions
# ----------------------------------------

def _get_psvae_hparams(**kwargs):
    hparams = {
        'data_dir': get_user_dir('data'),
        'save_dir': get_user_dir('save'),
        'model_class': 'ps-vae',
        'model_type': 'conv',
        'rng_seed_data': 0,
        'trial_splits': '8;1;1;0',
        'train_frac': 1.0,
        'rng_seed_model': 0,
        'fit_sess_io_layers': False,
        'learning_rate': 1e-4,
        'l2_reg': 0,
        'conditional_encoder': False,
        'vae.beta': 1}
    # update hparams
    for key, val in kwargs.items():
        if key == 'alpha' or key == 'beta':
            hparams['ps_vae.%s' % key] = val
        else:
            hparams[key] = val
    return hparams


def plot_psvae_training_curves(
        lab, expt, animal, session, alphas, betas, n_ae_latents, rng_seeds_model,
        experiment_name, n_labels, dtype='val', save_file=None, format='pdf', **kwargs):
    """Create training plots for each term in the ps-vae objective function.

    The `dtype` argument controls which type of trials are plotted ('train' or 'val').
    Additionally, multiple models can be plotted simultaneously by varying one (and only one) of
    the following parameters:

    - alpha
    - beta
    - number of unsupervised latents
    - random seed used to initialize model weights

    Each of these entries must be an array of length 1 except for one option, which can be an array
    of arbitrary length (corresponding to already trained models). This function generates a single
    plot with panels for each of the following terms:

    - total loss
    - pixel mse
    - label R^2 (note the objective function contains the label MSE, but R^2 is easier to parse)
    - KL divergence of supervised latents
    - index-code mutual information of unsupervised latents
    - total correlation of unsupervised latents
    - dimension-wise KL of unsupervised latents
    - subspace overlap

    Parameters
    ----------
    lab : :obj:`str`
        lab id
    expt : :obj:`str`
        expt id
    animal : :obj:`str`
        animal id
    session : :obj:`str`
        session id
    alphas : :obj:`array-like`
        alpha values to plot
    betas : :obj:`array-like`
        beta values to plot
    n_ae_latents : :obj:`array-like`
        unsupervised dimensionalities to plot
    rng_seeds_model : :obj:`array-like`
        model seeds to plot
    experiment_name : :obj:`str`
        test-tube experiment name
    n_labels : :obj:`int`
        dimensionality of supervised latent space
    dtype : :obj:`str`
        'train' | 'val'
    save_file : :obj:`str`, optional
        absolute path of save file; does not need file extension
    format : :obj:`str`, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...
    kwargs
        arguments are keys of `hparams`, for example to set `train_frac`, `rng_seed_model`, etc.

    """
    # check for arrays, turn ints into lists
    n_arrays = 0
    hue = None
    if len(alphas) > 1:
        n_arrays += 1
        hue = 'alpha'
    if len(betas) > 1:
        n_arrays += 1
        hue = 'beta'
    if len(n_ae_latents) > 1:
        n_arrays += 1
        hue = 'n latents'
    if len(rng_seeds_model) > 1:
        n_arrays += 1
        hue = 'rng seed'
    if n_arrays > 1:
        raise ValueError(
            'Can only set one of "alphas", "betas", "n_ae_latents", or "rng_seeds_model"' +
            'as an array')

    # set model info
    hparams = _get_psvae_hparams(experiment_name=experiment_name, **kwargs)

    metrics_list = [
        'loss', 'loss_data_mse', 'label_r2', 'loss_zs_kl', 'loss_zu_mi', 'loss_zu_tc',
        'loss_zu_dwkl']

    metrics_dfs = []
    i = 0
    for alpha in alphas:
        for beta in betas:
            for n_latents in n_ae_latents:
                for rng in rng_seeds_model:

                    # update hparams
                    hparams['ps_vae.alpha'] = alpha
                    hparams['ps_vae.beta'] = beta
                    hparams['n_ae_latents'] = n_latents + n_labels
                    hparams['rng_seed_model'] = rng

                    try:

                        get_lab_example(hparams, lab, expt)
                        hparams['animal'] = animal
                        hparams['session'] = session
                        hparams['session_dir'], sess_ids = get_session_dir(hparams)
                        hparams['expt_dir'] = get_expt_dir(hparams)
                        _, version = experiment_exists(hparams, which_version=True)

                        print(
                            'loading results with alpha=%i, beta=%i (version %i)' %
                            (alpha, beta, version))

                        metrics_dfs.append(load_metrics_csv_as_df(
                            hparams, lab, expt, metrics_list, version=None))

                        metrics_dfs[i]['alpha'] = alpha
                        metrics_dfs[i]['beta'] = beta
                        metrics_dfs[i]['n latents'] = hparams['n_ae_latents']
                        metrics_dfs[i]['rng seed'] = rng
                        i += 1

                    except TypeError:
                        print('could not find model for alpha=%i, beta=%i' % (alpha, beta))
                        continue

    metrics_df = pd.concat(metrics_dfs, sort=False)

    sns.set_style('white')
    sns.set_context('talk')
    data_queried = metrics_df[
        (metrics_df.epoch > 10) & ~pd.isna(metrics_df.val) & (metrics_df.dtype == dtype)]
    g = sns.FacetGrid(
        data_queried, col='loss', col_wrap=3, hue=hue, sharey=False, height=4)
    g = g.map(plt.plot, 'epoch', 'val').add_legend()  # , color=".3", fit_reg=False, x_jitter=.1);

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        g.savefig(save_file + '.' + format, dpi=300, format=format)


def plot_hyperparameter_search_results(
        lab, expt, animal, session, n_labels, label_names, alpha_weights, alpha_n_ae_latents,
        alpha_expt_name, beta_weights, beta_n_ae_latents, beta_expt_name, alpha, beta, save_file,
        batch_size=None, format='pdf', **kwargs):
    """Create a variety of diagnostic plots to assess the ps-vae hyperparameters.

    These diagnostic plots are based on the recommended way to perform a hyperparameter search in
    the ps-vae models; first, fix beta=1, and do a sweep over alpha values and number
    of latents (for example alpha=[50, 100, 500, 1000] and n_ae_latents=[2, 4, 8, 16]). The best
    alpha value is subjective because it involves a tradeoff between pixel mse and label mse. After
    choosing a suitable value, fix alpha and the number of latents and vary beta. This function
    will then plot the following panels:

    - pixel mse as a function of alpha/num latents (for fixed beta)
    - label mse as a function of alpha/num_latents (for fixed beta)
    - pixel mse as a function of beta (for fixed alpha/n_ae_latents)
    - label mse as a function of beta (for fixed alpha/n_ae_latents)
    - index-code mutual information (part of the KL decomposition) as a function of beta (for
      fixed alpha/n_ae_latents)
    - total correlation(part of the KL decomposition) as a function of beta (for fixed
      alpha/n_ae_latents)
    - dimension-wise KL (part of the KL decomposition) as a function of beta (for fixed
      alpha/n_ae_latents)
    - average correlation coefficient across all pairs of unsupervised latent dims as a function of
      beta (for fixed alpha/n_ae_latents)

    Parameters
    ----------
    lab : :obj:`str`
        lab id
    expt : :obj:`str`
        expt id
    animal : :obj:`str`
        animal id
    session : :obj:`str`
        session id
    n_labels : :obj:`str`
        number of label dims
    label_names : :obj:`array-like`
        names of label dims
    alpha_weights : :obj:`array-like`
        array of alpha weights for fixed values of beta
    alpha_n_ae_latents : :obj:`array-like`
        array of latent dimensionalities for fixed values of beta using alpha_weights
    alpha_expt_name : :obj:`str`
        test-tube experiment name of alpha-based hyperparam search
    beta_weights : :obj:`array-like`
        array of beta weights for a fixed value of alpha
    beta_n_ae_latents : :obj:`int`
        latent dimensionality used for beta hyperparam search
    beta_expt_name : :obj:`str`
        test-tube experiment name of beta hyperparam search
    alpha : :obj:`float`
        fixed value of alpha for beta search
    beta : :obj:`float`
        fixed value of beta for alpha search
    save_file : :obj:`str`
        absolute path of save file; does not need file extension
    batch_size : :obj:`int`, optional
        size of batches, used to compute correlation coefficient per batch; if NoneType, the
        correlation coefficient is computed across all time points
    format : :obj:`str`, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...
    kwargs
        arguments are keys of `hparams`, preceded by either `alpha_` or `beta_`. For example,
        to set the train frac of the alpha models, use `alpha_train_frac`; to set the rng_data_seed
        of the beta models, use `beta_rng_data_seed`.

    """

    def apply_masks(data, masks):
        return data[masks == 1]

    def get_label_r2(hparams, model, data_generator, version, dtype='val', overwrite=False):
        from sklearn.metrics import r2_score
        save_file = os.path.join(
            hparams['expt_dir'], 'version_%i' % version, 'r2_supervised.csv')
        if not os.path.exists(save_file) or overwrite:
            if not os.path.exists(save_file):
                print('R^2 metrics do not exist; computing from scratch')
            else:
                print('overwriting metrics at %s' % save_file)
            metrics_df = []
            data_generator.reset_iterators(dtype)
            for i_test in tqdm(range(data_generator.n_tot_batches[dtype])):
                # get next minibatch and put it on the device
                data, sess = data_generator.next_batch(dtype)
                x = data['images'][0]
                y = data['labels'][0].cpu().detach().numpy()
                if 'labels_masks' in data:
                    n = data['labels_masks'][0].cpu().detach().numpy()
                else:
                    n = np.ones_like(y)
                z = model.get_transformed_latents(x, dataset=sess)
                for i in range(n_labels):
                    y_true = apply_masks(y[:, i], n[:, i])
                    y_pred = apply_masks(z[:, i], n[:, i])
                    if len(y_true) > 10:
                        r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
                        mse = np.mean(np.square(y_true - y_pred))
                    else:
                        r2 = np.nan
                        mse = np.nan
                    metrics_df.append(pd.DataFrame({
                        'Trial': data['batch_idx'].item(),
                        'Label': label_names[i],
                        'R2': r2,
                        'MSE': mse,
                        'Model': 'PS-VAE'}, index=[0]))

            metrics_df = pd.concat(metrics_df)
            print('saving results to %s' % save_file)
            metrics_df.to_csv(save_file, index=False, header=True)
        else:
            print('loading results from %s' % save_file)
            metrics_df = pd.read_csv(save_file)
        return metrics_df

    # -----------------------------------------------------
    # load pixel/label MSE as a function of n_latents/alpha
    # -----------------------------------------------------

    # set model info
    hparams = _get_psvae_hparams(experiment_name=alpha_expt_name)
    # update hparams
    for key, val in kwargs.items():
        # hparam vals should be named 'alpha_[property]', for example 'alpha_train_frac'
        if key.split('_')[0] == 'alpha':
            prop = key[6:]
            hparams[prop] = val
        else:
            hparams[key] = val

    metrics_list = ['loss_data_mse']

    metrics_dfs_frame = []
    metrics_dfs_marker = []
    for n_latent in alpha_n_ae_latents:
        hparams['n_ae_latents'] = n_latent + n_labels
        for alpha_ in alpha_weights:
            hparams['ps_vae.alpha'] = alpha_
            hparams['ps_vae.beta'] = beta
            try:
                get_lab_example(hparams, lab, expt)
                hparams['animal'] = animal
                hparams['session'] = session
                hparams['session_dir'], sess_ids = get_session_dir(hparams)
                hparams['expt_dir'] = get_expt_dir(hparams)
                _, version = experiment_exists(hparams, which_version=True)
                print('loading results with alpha=%i, beta=%i (version %i)' % (
                    hparams['ps_vae.alpha'], hparams['ps_vae.beta'], version))
                # get frame mse
                metrics_dfs_frame.append(load_metrics_csv_as_df(
                    hparams, lab, expt, metrics_list, version=None, test=True))
                metrics_dfs_frame[-1]['alpha'] = alpha_
                metrics_dfs_frame[-1]['n_latents'] = hparams['n_ae_latents']
                # get marker mse
                model, data_gen = get_best_model_and_data(
                    hparams, Model=None, load_data=True, version=version)
                metrics_df_ = get_label_r2(hparams, model, data_gen, version, dtype='val')
                metrics_df_['alpha'] = alpha_
                metrics_df_['n_latents'] = hparams['n_ae_latents']
                metrics_dfs_marker.append(metrics_df_[metrics_df_.Model == 'PS-VAE'])
            except TypeError:
                print('could not find model for alpha=%i, beta=%i' % (
                    hparams['ps_vae.alpha'], hparams['ps_vae.beta']))
                continue
    metrics_df_frame = pd.concat(metrics_dfs_frame, sort=False)
    metrics_df_marker = pd.concat(metrics_dfs_marker, sort=False)
    print('done')

    # -----------------------------------------------------
    # load pixel/label MSE as a function of beta
    # -----------------------------------------------------
    # update hparams
    hparams['experiment_name'] = beta_expt_name
    for key, val in kwargs.items():
        # hparam vals should be named 'beta_[property]', for example 'beta_train_frac'
        if key.split('_')[0] == 'beta':
            prop = key[5:]
            hparams[prop] = val

    metrics_list = ['loss_data_mse', 'loss_zu_mi', 'loss_zu_tc', 'loss_zu_dwkl']

    metrics_dfs_frame_bg = []
    metrics_dfs_marker_bg = []
    metrics_dfs_corr_bg = []
    for beta in beta_weights:
        hparams['n_ae_latents'] = beta_n_ae_latents + n_labels
        hparams['ps_vae.alpha'] = alpha
        hparams['ps_vae.beta'] = beta
        try:
            get_lab_example(hparams, lab, expt)
            hparams['animal'] = animal
            hparams['session'] = session
            hparams['session_dir'], sess_ids = get_session_dir(hparams)
            hparams['expt_dir'] = get_expt_dir(hparams)
            _, version = experiment_exists(hparams, which_version=True)
            print('loading results with alpha=%i, beta=%i, (version %i)' % (
                hparams['ps_vae.alpha'], hparams['ps_vae.beta'], version))
            # get frame mse
            metrics_dfs_frame_bg.append(load_metrics_csv_as_df(
                hparams, lab, expt, metrics_list, version=None, test=True))
            metrics_dfs_frame_bg[-1]['beta'] = beta
            # get marker mse
            model, data_gen = get_best_model_and_data(
                hparams, Model=None, load_data=True, version=version)
            metrics_df_ = get_label_r2(hparams, model, data_gen, version, dtype='val')
            metrics_df_['beta'] = beta
            metrics_dfs_marker_bg.append(metrics_df_[metrics_df_.Model == 'PS-VAE'])
            # get corr
            latents = load_latents(hparams, version, dtype='test')
            if batch_size is None:
                corr = np.corrcoef(latents[:, n_labels + np.array([0, 1])].T)
                metrics_dfs_corr_bg.append(pd.DataFrame({
                    'loss': 'corr',
                    'dtype': 'test',
                    'val': np.abs(corr[0, 1]),
                    'beta': beta}, index=[0]))
            else:
                n_batches = int(np.ceil(latents.shape[0] / batch_size))
                for i in range(n_batches):
                    corr = np.corrcoef(
                        latents[i * batch_size:(i + 1) * batch_size,
                                n_labels + np.array([0, 1])].T)
                    metrics_dfs_corr_bg.append(pd.DataFrame({
                        'loss': 'corr',
                        'dtype': 'test',
                        'val': np.abs(corr[0, 1]),
                        'beta': beta}, index=[0]))
        except TypeError:
            print('could not find model for alpha=%i, beta=%i' % (
                hparams['ps_vae.alpha'], hparams['ps_vae.beta']))
            continue
        print()
    metrics_df_frame_bg = pd.concat(metrics_dfs_frame_bg, sort=False)
    metrics_df_marker_bg = pd.concat(metrics_dfs_marker_bg, sort=False)
    metrics_df_corr_bg = pd.concat(metrics_dfs_corr_bg, sort=False)
    print('done')

    # -----------------------------------------------------
    # ----------------- PLOT DATA -------------------------
    # -----------------------------------------------------
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.2)

    alpha_palette = sns.color_palette('Greens')
    beta_palette = sns.color_palette('Reds', len(metrics_df_corr_bg.beta.unique()))

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 7), dpi=300)

    n_rows = 2
    n_cols = 12
    gs = GridSpec(n_rows, n_cols, figure=fig)

    def despine(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    sns.set_palette(alpha_palette)

    # --------------------------------------------------
    # MSE per pixel
    # --------------------------------------------------
    ax_pixel_mse_alpha = fig.add_subplot(gs[0, 0:3])
    data_queried = metrics_df_frame[(metrics_df_frame.dtype == 'test')]
    sns.barplot(x='n_latents', y='val', hue='alpha', data=data_queried, ax=ax_pixel_mse_alpha)
    ax_pixel_mse_alpha.legend().set_visible(False)
    ax_pixel_mse_alpha.set_xlabel('Latent dimension')
    ax_pixel_mse_alpha.set_ylabel('MSE per pixel')
    ax_pixel_mse_alpha.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax_pixel_mse_alpha.set_title('Beta=1, Gamma=0')
    despine(ax_pixel_mse_alpha)

    # --------------------------------------------------
    # MSE per marker
    # --------------------------------------------------
    ax_marker_mse_alpha = fig.add_subplot(gs[0, 3:6])
    data_queried = metrics_df_marker
    sns.barplot(x='n_latents', y='MSE', hue='alpha', data=data_queried, ax=ax_marker_mse_alpha)
    ax_marker_mse_alpha.set_xlabel('Latent dimension')
    ax_marker_mse_alpha.set_ylabel('MSE per marker')
    ax_marker_mse_alpha.set_title('Beta=1, Gamma=0')
    ax_marker_mse_alpha.legend(frameon=True, title='Alpha')
    despine(ax_marker_mse_alpha)

    # --------------------------------------------------
    # MSE per pixel (beta)
    # --------------------------------------------------
    ax_pixel_mse_bg = fig.add_subplot(gs[0, 6:9])
    data_queried = metrics_df_frame_bg[
        (metrics_df_frame_bg.dtype == 'test') &
        (metrics_df_frame_bg.loss == 'loss_data_mse') &
        (metrics_df_frame_bg.epoch == 200)]
    sns.barplot(x='beta', y='val', data=data_queried, ax=ax_pixel_mse_bg)
    ax_pixel_mse_bg.legend().set_visible(False)
    ax_pixel_mse_bg.set_xlabel('Beta')
    ax_pixel_mse_bg.set_ylabel('MSE per pixel')
    ax_pixel_mse_bg.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax_pixel_mse_bg.set_title('Latents=%i, Alpha=%i' % (hparams['n_ae_latents'], alpha))
    despine(ax_pixel_mse_bg)

    # --------------------------------------------------
    # MSE per marker (beta)
    # --------------------------------------------------
    ax_marker_mse_bg = fig.add_subplot(gs[0, 9:12])
    data_queried = metrics_df_marker_bg
    sns.barplot(x='beta', y='MSE', data=data_queried, ax=ax_marker_mse_bg)
    ax_marker_mse_bg.set_xlabel('Beta')
    ax_marker_mse_bg.set_ylabel('MSE per marker')
    ax_marker_mse_bg.set_title('Latents=%i, Alpha=%i' % (hparams['n_ae_latents'], alpha))
    despine(ax_marker_mse_bg)

    # --------------------------------------------------
    # ICMI
    # --------------------------------------------------
    ax_icmi = fig.add_subplot(gs[1, 0:3])
    data_queried = metrics_df_frame_bg[
        (metrics_df_frame_bg.dtype == 'test') &
        (metrics_df_frame_bg.loss == 'loss_zu_mi') &
        (metrics_df_frame_bg.epoch == 200)]
    sns.lineplot(x='beta', y='val', data=data_queried, ax=ax_icmi, ci=None)
    ax_icmi.legend().set_visible(False)
    ax_icmi.set_xlabel('Beta')
    ax_icmi.set_ylabel('Index-code Mutual Information')
    ax_icmi.set_title('Latents=%i, Alpha=%i' % (hparams['n_ae_latents'], alpha))
    despine(ax_icmi)

    # --------------------------------------------------
    # TC
    # --------------------------------------------------
    ax_tc = fig.add_subplot(gs[1, 3:6])
    data_queried = metrics_df_frame_bg[
        (metrics_df_frame_bg.dtype == 'test') &
        (metrics_df_frame_bg.loss == 'loss_zu_tc') &
        (metrics_df_frame_bg.epoch == 200)]
    sns.lineplot(x='beta', y='val', data=data_queried, ax=ax_tc, ci=None)
    ax_tc.legend().set_visible(False)
    ax_tc.set_xlabel('Beta')
    ax_tc.set_ylabel('Total Correlation')
    ax_tc.set_title('Latents=%i, Alpha=%i' % (hparams['n_ae_latents'], alpha))
    despine(ax_tc)

    # --------------------------------------------------
    # DWKL
    # --------------------------------------------------
    ax_dwkl = fig.add_subplot(gs[1, 6:9])
    data_queried = metrics_df_frame_bg[
        (metrics_df_frame_bg.dtype == 'test') &
        (metrics_df_frame_bg.loss == 'loss_zu_dwkl') &
        (metrics_df_frame_bg.epoch == 200)]
    sns.lineplot(x='beta', y='val', data=data_queried, ax=ax_dwkl, ci=None)
    ax_dwkl.legend().set_visible(False)
    ax_dwkl.set_xlabel('Beta')
    ax_dwkl.set_ylabel('Dimension-wise KL')
    ax_dwkl.set_title('Latents=%i, Alpha=%i' % (hparams['n_ae_latents'], alpha))
    despine(ax_dwkl)

    # --------------------------------------------------
    # CC
    # --------------------------------------------------
    ax_cc = fig.add_subplot(gs[1, 9:12])
    data_queried = metrics_df_corr_bg
    sns.lineplot(x='beta', y='val', data=data_queried, ax=ax_cc, ci=None)
    ax_cc.legend().set_visible(False)
    ax_cc.set_xlabel('Beta')
    ax_cc.set_ylabel('Correlation Coefficient')
    ax_cc.set_title('Latents=%i, Alpha=%i' % (hparams['n_ae_latents'], alpha))
    despine(ax_cc)

    plt.tight_layout(h_pad=3)  # h_pad is fraction of font size

    # reset to default color palette
    # sns.set_palette(sns.color_palette(None, 10))
    sns.reset_orig()

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, format=format)


def plot_label_reconstructions(
        lab, expt, animal, session, n_ae_latents, experiment_name, n_labels, trials, version=None,
        plot_scale=0.5, sess_idx=0, save_file=None, format='pdf', xtick_locs=None, frame_rate=None,
        max_traces=8, add_r2=True, add_legend=True, colored_predictions=True, concat_trials=False,
        **kwargs):
    """Plot labels and their reconstructions from an ps-vae.

    Parameters
    ----------
    lab : :obj:`str`
        lab id
    expt : :obj:`str`
        expt id
    animal : :obj:`str`
        animal id
    session : :obj:`str`
        session id
    n_ae_latents : :obj:`str`
        dimensionality of unsupervised latent space; n_labels will be added to this
    experiment_name : :obj:`str`
        test-tube experiment name
    n_labels : :obj:`str`
        dimensionality of supervised latent space
    trials : :obj:`array-like`
        array of trials to reconstruct
    version : :obj:`str` or :obj:`int`, optional
        can be 'best' to load best model, and integer to load a specific model, or NoneType to use
        the values in hparams to load a specific model
    plot_scale : :obj:`float`
        scale the magnitude of reconstructions
    sess_idx : :obj:`int`, optional
        session index into data generator
    save_file : :obj:`str`, optional
        absolute path of save file; does not need file extension
    format : :obj:`str`, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...
    xtick_locs : :obj:`array-like`, optional
        tick locations in units of bins
    frame_rate : :obj:`float`, optional
        frame rate of behavorial video; to properly relabel xticks
    max_traces : :obj:`int`, optional
        maximum number of traces to plot, for easier visualization
    add_r2 : :obj:`bool`, optional
        print R2 value on plot
    add_legend : :obj:`bool`, optional
        print legend on plot
    colored_predictions : :obj:`bool`, optional
        color predictions using default seaborn colormap; else predictions are black
    concat_trials : :obj:`bool`, optional
        True to plot all trials together, separated by a small gap
    kwargs
        arguments are keys of `hparams`, for example to set `train_frac`, `rng_seed_model`, etc.

    """

    from behavenet.plotting.decoder_utils import plot_neural_reconstruction_traces

    if len(trials) == 1:
        concat_trials = False

    # set model info
    hparams = _get_psvae_hparams(
        experiment_name=experiment_name, n_ae_latents=n_ae_latents + n_labels, **kwargs)

    # programmatically fill out other hparams options
    get_lab_example(hparams, lab, expt)
    hparams['animal'] = animal
    hparams['session'] = session

    model, data_generator = get_best_model_and_data(
        hparams, Model=None, load_data=True, version=version, data_kwargs=None)
    print(data_generator)
    print('alpha: %i' % model.hparams['ps_vae.alpha'])
    print('beta: %i' % model.hparams['ps_vae.beta'])
    print('model seed: %i' % model.hparams['rng_seed_model'])

    n_blank = 5  # buffer time points between trials if concatenating
    labels_og_all = []
    labels_pred_all = []
    for trial in trials:
        # collect data
        batch = data_generator.datasets[sess_idx][trial]
        labels_og = batch['labels'].detach().cpu().numpy()
        labels_pred = model.get_predicted_labels(batch['images']).detach().cpu().numpy()
        if 'labels_masks' in batch:
            labels_masks = batch['labels_masks'].detach().cpu().numpy()
            labels_og[labels_masks == 0] = np.nan
        # store data
        labels_og_all.append(labels_og)
        labels_pred_all.append(labels_pred)
        if trial != trials[-1]:
            labels_og_all.append(np.nan * np.zeros((n_blank, labels_og.shape[1])))
            labels_pred_all.append(np.nan * np.zeros((n_blank, labels_pred.shape[1])))
        # plot data from single trial
        if not concat_trials:
            if save_file is not None:
                save_file_trial = save_file + '_trial-%i' % trial
            else:
                save_file_trial = None
            plot_neural_reconstruction_traces(
                labels_og, labels_pred, scale=plot_scale, save_file=save_file_trial, format=format,
                xtick_locs=xtick_locs, frame_rate=frame_rate, max_traces=max_traces, add_r2=add_r2,
                add_legend=add_legend, colored_predictions=colored_predictions)

    # plot data from all trials
    if concat_trials:
        if save_file is not None:
            save_file_trial = save_file + '_trial-{}'.format(trials)
        else:
            save_file_trial = None
        plot_neural_reconstruction_traces(
            np.vstack(labels_og_all), np.vstack(labels_pred_all), scale=plot_scale,
            save_file=save_file_trial, format=format,
            xtick_locs=xtick_locs, frame_rate=frame_rate, max_traces=max_traces, add_r2=add_r2,
            add_legend=add_legend, colored_predictions=colored_predictions)


def plot_latent_traversals(
        lab, expt, animal, session, model_class, alpha, beta, n_ae_latents, rng_seed_model,
        experiment_name, n_labels, label_idxs, label_min_p=5, label_max_p=95,
        channel=0, n_frames_zs=4, n_frames_zu=4, trial=None, trial_idx=1, batch_idx=1,
        crop_type=None, crop_kwargs=None, sess_idx=0, save_file=None, format='pdf', **kwargs):
    """Plot video frames representing the traversal of individual dimensions of the latent space.

    Parameters
    ----------
     lab : :obj:`str`
        lab id
    expt : :obj:`str`
        expt id
    animal : :obj:`str`
        animal id
    session : :obj:`str`
        session id
    model_class : :obj:`str`
        model class in which to perform traversal; currently supported models are:
        'ae' | 'vae' | 'cond-ae' | 'cond-vae' | 'beta-tcvae' | 'cond-ae-msp' | 'ps-vae'
        note that models with conditional encoders are not currently supported
    alpha : :obj:`float`
        ps-vae alpha value
    beta : :obj:`float`
        ps-vae beta value
    n_ae_latents : :obj:`int`
        dimensionality of unsupervised latents
    rng_seed_model : :obj:`int`
        model seed
    experiment_name : :obj:`str`
        test-tube experiment name
    n_labels : :obj:`str`
        dimensionality of supervised latent space (ignored when using fully unsupervised models)
    label_idxs : :obj:`array-like`, optional
        set of label indices (dimensions) to individually traverse
    label_min_p : :obj:`float`, optional
        lower percentile of training data used to compute range of traversal
    label_max_p : :obj:`float`, optional
        upper percentile of training data used to compute range of traversal
    channel : :obj:`int`, optional
        image channel to plot
    n_frames_zs : :obj:`int`, optional
        number of frames (points) to display for traversal through supervised dimensions
    n_frames_zu : :obj:`int`, optional
        number of frames (points) to display for traversal through unsupervised dimensions
    trial : :obj:`int`, optional
        trial index into all possible trials (train, val, test); one of `trial` or `trial_idx`
        must be specified; `trial` takes precedence over `trial_idx`
    trial_idx : :obj:`int`, optional
        trial index of base frame used for interpolation
    batch_idx : :obj:`int`, optional
        batch index of base frame used for interpolation
    crop_type : :obj:`str`, optional
        cropping method used on interpolated frames
        'fixed' | None
    crop_kwargs : :obj:`dict`, optional
        if crop_type is not None, provides information about the crop
        keys for 'fixed' type: 'y_0', 'x_0', 'y_ext', 'x_ext'; window is
        (y_0 - y_ext, y_0 + y_ext) in vertical direction and
        (x_0 - x_ext, x_0 + x_ext) in horizontal direction
    sess_idx : :obj:`int`, optional
        session index into data generator
    save_file : :obj:`str`, optional
        absolute path of save file; does not need file extension
    format : :obj:`str`, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...
    kwargs
        arguments are keys of `hparams`, for example to set `train_frac`, `rng_seed_model`, etc.

    """

    hparams = _get_psvae_hparams(
        model_class=model_class, alpha=alpha, beta=beta, n_ae_latents=n_ae_latents,
        experiment_name=experiment_name, rng_seed_model=rng_seed_model, **kwargs)

    if model_class == 'cond-ae-msp' or model_class == 'ps-vae':
        hparams['n_ae_latents'] += n_labels

    # programmatically fill out other hparams options
    get_lab_example(hparams, lab, expt)
    hparams['animal'] = animal
    hparams['session'] = session
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)
    _, version = experiment_exists(hparams, which_version=True)
    model_ae, data_generator = get_best_model_and_data(hparams, Model=None, version=version)

    # get latent/label info
    latent_range = get_input_range(
        'latents', hparams, model=model_ae, data_gen=data_generator, min_p=15, max_p=85,
        version=version)
    label_range = get_input_range(
        'labels', hparams, sess_ids=sess_ids, sess_idx=sess_idx,
        min_p=label_min_p, max_p=label_max_p)
    try:
        label_sc_range = get_input_range(
            'labels_sc', hparams, sess_ids=sess_ids, sess_idx=sess_idx,
            min_p=label_min_p, max_p=label_max_p)
    except KeyError:
        import copy
        label_sc_range = copy.deepcopy(label_range)

    # ----------------------------------------
    # label traversals
    # ----------------------------------------
    interp_func_label = interpolate_1d
    plot_func_label = plot_1d_frame_array
    save_file_new = save_file + '_label-traversals'

    if model_class == 'cond-ae' or model_class == 'cond-ae-msp' or model_class == 'ps-vae' or \
            model_class == 'cond-vae':

        # get model input for this trial
        ims_pt, ims_np, latents_np, labels_pt, labels_np, labels_2d_pt, labels_2d_np = \
            get_model_input(
                data_generator, hparams, model_ae, trial_idx=trial_idx, trial=trial,
                compute_latents=True, compute_scaled_labels=False, compute_2d_labels=False)

        if labels_2d_np is None:
            labels_2d_np = np.copy(labels_np)
        if crop_type == 'fixed':
            crop_kwargs_ = crop_kwargs
        else:
            crop_kwargs_ = None

        # perform interpolation
        ims_label, markers_loc_label, ims_crop_label = interp_func_label(
            'labels', model_ae, ims_pt[None, batch_idx, :], latents_np[None, batch_idx, :],
            labels_np[None, batch_idx, :], labels_2d_np[None, batch_idx, :],
            mins=label_range['min'], maxes=label_range['max'],
            n_frames=n_frames_zs, input_idxs=label_idxs, crop_type=crop_type,
            mins_sc=label_sc_range['min'], maxes_sc=label_sc_range['max'],
            crop_kwargs=crop_kwargs_, ch=channel)

        # plot interpolation
        if crop_type:
            marker_kwargs = {
                'markersize': 30, 'markeredgewidth': 8, 'markeredgecolor': [1, 1, 0],
                'fillstyle': 'none'}
            plot_func_label(
                ims_crop_label, markers=None, marker_kwargs=marker_kwargs, save_file=save_file_new,
                format=format)
        else:
            marker_kwargs = {
                'markersize': 20, 'markeredgewidth': 5, 'markeredgecolor': [1, 1, 0],
                'fillstyle': 'none'}
            plot_func_label(
                ims_label, markers=None, marker_kwargs=marker_kwargs, save_file=save_file_new,
                format=format)

    # ----------------------------------------
    # latent traversals
    # ----------------------------------------
    interp_func_latent = interpolate_1d
    plot_func_latent = plot_1d_frame_array
    save_file_new = save_file + '_latent-traversals'

    if hparams['model_class'] == 'cond-ae-msp' or hparams['model_class'] == 'ps-vae':
        latent_idxs = n_labels + np.arange(n_ae_latents)
    elif hparams['model_class'] == 'ae' \
            or hparams['model_class'] == 'vae' \
            or hparams['model_class'] == 'cond-vae' \
            or hparams['model_class'] == 'beta-tcvae':
        latent_idxs = np.arange(n_ae_latents)
    else:
        raise NotImplementedError

    # simplify options here
    scaled_labels = False
    twod_labels = False
    crop_type = None
    crop_kwargs = None
    labels_2d_np_sel = None

    # get model input for this trial
    ims_pt, ims_np, latents_np, labels_pt, labels_np, labels_2d_pt, labels_2d_np = \
        get_model_input(
            data_generator, hparams, model_ae, trial=trial, trial_idx=trial_idx,
            compute_latents=True, compute_scaled_labels=scaled_labels,
            compute_2d_labels=twod_labels)

    latents_np[:, n_labels:] = 0

    if hparams['model_class'] == 'ae' or hparams['model_class'] == 'beta-tcvae':
        labels_np_sel = labels_np
    else:
        labels_np_sel = labels_np[None, batch_idx, :]

    # perform interpolation
    ims_latent, markers_loc_latent_, ims_crop_latent = interp_func_latent(
        'latents', model_ae, ims_pt[None, batch_idx, :], latents_np[None, batch_idx, :],
        labels_np_sel, labels_2d_np_sel,
        mins=latent_range['min'], maxes=latent_range['max'],
        n_frames=n_frames_zu, input_idxs=latent_idxs, crop_type=crop_type,
        mins_sc=None, maxes_sc=None, crop_kwargs=crop_kwargs, ch=channel)

    # plot interpolation
    marker_kwargs = {
        'markersize': 20, 'markeredgewidth': 5, 'markeredgecolor': [1, 1, 0],
        'fillstyle': 'none'}
    plot_func_latent(
        ims_latent, markers=None, marker_kwargs=marker_kwargs, save_file=save_file_new,
        format=format)


def make_latent_traversal_movie(
        lab, expt, animal, session, model_class, alpha, beta, n_ae_latents,
        rng_seed_model, experiment_name, n_labels, trial_idxs, batch_idxs, trials,
        label_min_p=5, label_max_p=95, channel=0, sess_idx=0, n_frames=10, n_buffer_frames=5,
        crop_kwargs=None, n_cols=3, movie_kwargs={}, panel_titles=None, order_idxs=None,
        split_movies=False, save_file=None, **kwargs):
    """Create a multi-panel movie with each panel showing traversals of an individual latent dim.

    The traversals will start at a lower bound, increase to an upper bound, then return to a lower
    bound; the traversal of each dimension occurs simultaneously. It is also possible to specify
    multiple base frames for the traversals; the traversal of each base frame is separated by
    several blank frames. Note that support for plotting markers on top of the corresponding
    supervised dimensions is not supported by this function.

    Parameters
    ----------
    lab : :obj:`str`
        lab id
    expt : :obj:`str`
        expt id
    animal : :obj:`str`
        animal id
    session : :obj:`str`
        session id
    model_class : :obj:`str`
        model class in which to perform traversal; currently supported models are:
        'ae' | 'vae' | 'cond-ae' | 'cond-vae' | 'ps-vae'
        note that models with conditional encoders are not currently supported
    alpha : :obj:`float`
        ps-vae alpha value
    beta : :obj:`float`
        ps-vae beta value
    n_ae_latents : :obj:`int`
        dimensionality of unsupervised latents
    rng_seed_model : :obj:`int`
        model seed
    experiment_name : :obj:`str`
        test-tube experiment name
    n_labels : :obj:`str`
        dimensionality of supervised latent space (ignored when using fully unsupervised models)
    trial_idxs : :obj:`array-like` of :obj:`int`
        trial indices of base frames used for interpolation; if an entry is an integer, the
        corresponding entry in `trials` must be `None`. This value is a trial index into all
        *test* trials, and is not affected by how the test trials are shuffled. The `trials`
        argument (see below) takes precedence over `trial_idxs`.
    batch_idxs : :obj:`array-like` of :obj:`int`
        batch indices of base frames used for interpolation; correspond to entries in `trial_idxs`
        and `trials`
    trials : :obj:`array-like` of :obj:`int`
        trials of base frame used for interpolation; if an entry is an integer, the
        corresponding entry in `trial_idxs` must be `None`. This value is a trial index into all
        possible trials (train, val, test), whereas `trial_idxs` is an index only into test trials
    label_min_p : :obj:`float`, optional
        lower percentile of training data used to compute range of traversal
    label_max_p : :obj:`float`, optional
        upper percentile of training data used to compute range of traversal
    channel : :obj:`int`, optional
        image channel to plot
    sess_idx : :obj:`int`, optional
        session index into data generator
    n_frames : :obj:`int`, optional
        number of frames (points) to display for traversal across latent dimensions; the movie
        will display a traversal of `n_frames` across each dim, then another traversal of
        `n_frames` in the opposite direction
    n_buffer_frames : :obj:`int`, optional
        number of blank frames to insert between base frames
    crop_kwargs : :obj:`dict`, optional
        if crop_type is not None, provides information about the crop (for a fixed crop window)
        keys : 'y_0', 'x_0', 'y_ext', 'x_ext'; window is
        (y_0 - y_ext, y_0 + y_ext) in vertical direction and
        (x_0 - x_ext, x_0 + x_ext) in horizontal direction
    n_cols : :obj:`int`, optional
        movie is `n_cols` panels wide
    movie_kwargs : :obj:`dict`, optional
        additional kwargs for individual panels; possible keys are 'markersize', 'markeredgecolor',
        'markeredgewidth', and 'text_color'
    panel_titles : :obj:`list` of :obj:`str`, optional
        optional titles for each panel
    order_idxs : :obj:`array-like`, optional
        used to reorder panels (which are plotted in row-major order) if desired; can also be used
        to choose a subset of latent dimensions to include
    split_movies : :obj:`bool`, optional
        True to save a separate latent traversal movie for each latent dimension
    save_file : :obj:`str`, optional
        absolute path of save file; does not need file extension, will automatically be saved as
        mp4. To save as a gif, include the '.gif' file extension in `save_file`
    kwargs
        arguments are keys of `hparams`, for example to set `train_frac`, `rng_seed_model`, etc.

    """

    panel_titles = [''] * (n_labels + n_ae_latents) if panel_titles is None else panel_titles

    hparams = _get_psvae_hparams(
        model_class=model_class, alpha=alpha, beta=beta, n_ae_latents=n_ae_latents,
        experiment_name=experiment_name, rng_seed_model=rng_seed_model, **kwargs)

    if model_class == 'cond-ae-msp' or model_class == 'ps-vae':
        hparams['n_ae_latents'] += n_labels

    # programmatically fill out other hparams options
    get_lab_example(hparams, lab, expt)
    hparams['animal'] = animal
    hparams['session'] = session
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)
    _, version = experiment_exists(hparams, which_version=True)
    model_ae, data_generator = get_best_model_and_data(hparams, Model=None, version=version)

    # get latent/label info
    latent_range = get_input_range(
        'latents', hparams, model=model_ae, data_gen=data_generator, min_p=15, max_p=85,
        version=version)
    label_range = get_input_range(
        'labels', hparams, sess_ids=sess_ids, sess_idx=sess_idx,
        min_p=label_min_p, max_p=label_max_p)

    # ----------------------------------------
    # collect frames/latents/labels
    # ----------------------------------------
    if hparams['model_class'] == 'vae':
        csl = False
        c2dl = False
    else:
        csl = False
        c2dl = False

    ims_pt = []
    ims_np = []
    latents_np = []
    labels_pt = []
    labels_np = []
    # labels_2d_pt = []
    # labels_2d_np = []
    for trial, trial_idx in zip(trials, trial_idxs):
        ims_pt_, ims_np_, latents_np_, labels_pt_, labels_np_, labels_2d_pt_, labels_2d_np_ = \
            get_model_input(
                data_generator, hparams, model_ae, trial_idx=trial_idx, trial=trial,
                compute_latents=True, compute_scaled_labels=csl, compute_2d_labels=c2dl,
                max_frames=200)
        ims_pt.append(ims_pt_)
        ims_np.append(ims_np_)
        latents_np.append(latents_np_)
        labels_pt.append(labels_pt_)
        labels_np.append(labels_np_)
        # labels_2d_pt.append(labels_2d_pt_)
        # labels_2d_np.append(labels_2d_np_)

    if hparams['model_class'] == 'ps-vae':
        label_idxs = np.arange(n_labels)
        latent_idxs = n_labels + np.arange(n_ae_latents)
    elif hparams['model_class'] == 'vae':
        label_idxs = []
        latent_idxs = np.arange(hparams['n_ae_latents'])
    elif hparams['model_class'] == 'cond-vae':
        label_idxs = np.arange(n_labels)
        latent_idxs = np.arange(hparams['n_ae_latents'])
    else:
        raise Exception

    # ----------------------------------------
    # label traversals
    # ----------------------------------------
    ims_all = []
    txt_strs_all = []
    txt_strs_titles = []

    for label_idx in label_idxs:

        ims = []
        txt_strs = []

        for b, batch_idx in enumerate(batch_idxs):
            if hparams['model_class'] == 'ps-vae':
                points = np.array([latents_np[b][batch_idx, :]] * 3)
            elif hparams['model_class'] == 'cond-vae':
                points = np.array([labels_np[b][batch_idx, :]] * 3)
            else:
                raise Exception
            points[0, label_idx] = label_range['min'][label_idx]
            points[1, label_idx] = label_range['max'][label_idx]
            points[2, label_idx] = label_range['min'][label_idx]
            ims_curr, inputs = interpolate_point_path(
                'labels', model_ae, ims_pt[b][None, batch_idx, :],
                labels_np[b][None, batch_idx, :], points=points, n_frames=n_frames, ch=channel,
                crop_kwargs=crop_kwargs)
            ims.append(ims_curr)
            txt_strs += [panel_titles[label_idx] for _ in range(len(ims_curr))]

            if label_idx == 0:
                tmp = trial_idxs[b] if trial_idxs[b] is not None else trials[b]
                txt_strs_titles += [
                    'base frame %02i-%02i' % (tmp, batch_idx) for _ in range(len(ims_curr))]

            # add blank frames
            if len(batch_idxs) > 1:
                y_pix, x_pix = ims_curr[0].shape
                ims.append([np.zeros((y_pix, x_pix)) for _ in range(n_buffer_frames)])
                txt_strs += ['' for _ in range(n_buffer_frames)]
                if label_idx == 0:
                    txt_strs_titles += ['' for _ in range(n_buffer_frames)]

        ims_all.append(np.vstack(ims))
        txt_strs_all.append(txt_strs)

    # ----------------------------------------
    # latent traversals
    # ----------------------------------------
    crop_kwargs_ = None
    for latent_idx in latent_idxs:

        ims = []
        txt_strs = []

        for b, batch_idx in enumerate(batch_idxs):

            points = np.array([latents_np[b][batch_idx, :]] * 3)

            # points[:, latent_idxs] = 0
            points[0, latent_idx] = latent_range['min'][latent_idx]
            points[1, latent_idx] = latent_range['max'][latent_idx]
            points[2, latent_idx] = latent_range['min'][latent_idx]
            if hparams['model_class'] == 'vae':
                labels_curr = None
            else:
                labels_curr = labels_np[b][None, batch_idx, :]
            ims_curr, inputs = interpolate_point_path(
                'latents', model_ae, ims_pt[b][None, batch_idx, :],
                labels_curr, points=points, n_frames=n_frames, ch=channel,
                crop_kwargs=crop_kwargs_)
            ims.append(ims_curr)
            if hparams['model_class'] == 'cond-vae':
                txt_strs += [panel_titles[latent_idx + n_labels] for _ in range(len(ims_curr))]
            else:
                txt_strs += [panel_titles[latent_idx] for _ in range(len(ims_curr))]

            if latent_idx == 0 and len(label_idxs) == 0:
                # add frame ids here if skipping labels
                tmp = trial_idxs[b] if trial_idxs[b] is not None else trials[b]
                txt_strs_titles += [
                    'base frame %02i-%02i' % (tmp, batch_idx) for _ in range(len(ims_curr))]

            # add blank frames
            if len(batch_idxs) > 1:
                y_pix, x_pix = ims_curr[0].shape
                ims.append([np.zeros((y_pix, x_pix)) for _ in range(n_buffer_frames)])
                txt_strs += ['' for _ in range(n_buffer_frames)]
                if latent_idx == 0 and len(label_idxs) == 0:
                    txt_strs_titles += ['' for _ in range(n_buffer_frames)]

        ims_all.append(np.vstack(ims))
        txt_strs_all.append(txt_strs)

    # ----------------------------------------
    # make video
    # ----------------------------------------
    if order_idxs is None:
        # don't change order of latents
        order_idxs = np.arange(len(ims_all))

    if split_movies:
        for idx in order_idxs:
            if save_file.split('.')[-1] == 'gif':
                save_file_new = save_file[:-4] + '_latent-%i.gif' % idx
            elif save_file.split('.')[-1] == 'mp4':
                save_file_new = save_file[:-4] + '_latent-%i.mp4' % idx
            else:
                save_file_new = save_file + '_latent-%i' % 0
            make_interpolated(
                ims=ims_all[idx],
                text=txt_strs_all[idx],
                text_title=txt_strs_titles,
                save_file=save_file_new, scale=3, **movie_kwargs)
    else:
        make_interpolated_multipanel(
            ims=[ims_all[i] for i in order_idxs],
            text=[txt_strs_all[i] for i in order_idxs],
            text_title=txt_strs_titles,
            save_file=save_file, scale=2, n_cols=n_cols, **movie_kwargs)


def plot_mspsvae_training_curves(
        hparams, alpha, beta, delta, rng_seed_model, n_latents, n_background, n_labels, lab=None,
        expt=None, dtype='val', version_dir=None, save_file=None, format='pdf', **kwargs):
    """Create training plots for each term in the ps-vae objective function.

    The `dtype` argument controls which type of trials are plotted ('train' or 'val').
    Additionally, multiple models can be plotted simultaneously by varying one (and only one) of
    the following parameters:

    - alpha
    - beta
    - gamma
    - number of unsupervised latents
    - random seed used to initialize model weights

    Each of these entries must be an array of length 1 except for one option, which can be an array
    of arbitrary length (corresponding to already trained models). This function generates a single
    plot with panels for each of the following terms:

    - total loss
    - pixel mse
    - label R^2 (note the objective function contains the label MSE, but R^2 is easier to parse)
    - KL divergence of supervised latents
    - index-code mutual information of unsupervised latents
    - total correlation of unsupervised latents
    - dimension-wise KL of unsupervised latents
    - subspace overlap

    Parameters
    ----------
    hparams
    alpha : :obj:`array-like`
        alpha values to plot
    beta : :obj:`array-like`
        beta values to plot
    delta : :obj:`array-like`
        delta values to plot
    n_ae_latents : :obj:`array-like`
        unsupervised dimensionalities to plot
    rng_seeds_model : :obj:`array-like`
        model seeds to plot
    n_labels : :obj:`int`
        dimensionality of supervised latent space
    dtype : :obj:`str`
        'train' | 'val'
    save_file : :obj:`str`, optional
        absolute path of save file; does not need file extension
    format : :obj:`str`, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...
    kwargs
        arguments are keys of `hparams`, for example to set `train_frac`, `rng_seed_model`, etc.

    """
    if dtype == 'val':
        hue = 'dataset'
    else:
        hue = None

    metrics_list = [
        'loss', 'loss_data_mse', 'label_r2',
        'loss_zs_kl', 'loss_zu_mi', 'loss_zu_tc', 'loss_zu_dwkl', 'loss_triplet']

    # update hparams
    hparams['ps_vae.alpha'] = alpha
    hparams['ps_vae.beta'] = beta
    hparams['ps_vae.delta'] = delta
    hparams['n_ae_latents'] = n_latents + n_background + n_labels
    hparams['rng_seed_model'] = rng_seed_model

    if version_dir is None:
        try:
            _, version = experiment_exists(hparams, which_version=True)
            print(
                'loading results with alpha=%i, beta=%i, delta=%i (version %i)' %
                (alpha, beta, delta, version))
            metrics_df = load_metrics_csv_as_df(hparams, lab, expt, metrics_list, version=None)
        except TypeError:
            print('could not find model for alpha=%i, beta=%i, delta=%i' % (alpha, beta, delta))
            return None
    else:
        metrics_df = load_metrics_csv_as_df(
            hparams, lab=None, expt=None, metrics_list=metrics_list, version_dir=version_dir)

    sns.set_style('white')
    sns.set_context('talk')
    data_queried = metrics_df[
        (metrics_df.epoch > 10) & ~pd.isna(metrics_df.val) & (metrics_df.dtype == dtype)]
    g = sns.FacetGrid(
        data_queried, col='loss', col_wrap=3, hue=hue, sharey=False, height=4)
    g = g.map(plt.plot, 'epoch', 'val').add_legend()  # , color=".3", fit_reg=False, x_jitter=.1);

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('alpha=%i, beta=%i, delta=%i, rng=%i' % (alpha, beta, delta, rng_seed_model))

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        g.savefig(save_file + '.' + format, dpi=300, format=format)
