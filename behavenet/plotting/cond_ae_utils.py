import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from behavenet import make_dir_if_not_exists
from behavenet.data.utils import load_labels_like_latents
from behavenet.fitting.eval import get_reconstruction
from behavenet.fitting.utils import get_session_dir

# to ignore imports for sphix-autoapidoc
__all__ = [
    'get_crop', 'get_input_range', 'compute_range', 'get_labels_2d_for_trial', 'get_model_input',
    'interpolate_2d', 'interpolate_1d', 'plot_2d_frame_array', 'plot_1d_frame_array']


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

    Returns
    -------
    :obj:`dict`
        keys are 'min' and 'max'

    """
    if input_type == 'latents':
        # load latents
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
        from behavenet.fitting.utils import build_data_generator
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
        data_generator, hparams, model, trial=None, trial_idx=None, sess_idx=0, max_frames=100,
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
            or hparams['model_class'] == 'sss-vae' \
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
        if hparams['model_class'] == 'cond-ae-msp' or hparams['model_class'] == 'sss-vae':
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
                            or model.hparams['model_class'] == 'sss-vae':
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
                        or model.hparams['model_class'] == 'sss-vae':
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
                            or model.hparams['model_class'] == 'sss-vae':
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
                        or model.hparams['model_class'] == 'sss-vae':
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


def plot_2d_frame_array(
        ims_list, markers=None, im_kwargs=None, marker_kwargs=None, figsize=None, save_file=None):
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
    figsize : :obj:`tuple`
        (width, height) in inches
    save_file : :obj:`str` or NoneType, optional
        figure saved if not None

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
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()


def plot_1d_frame_array(
        ims_list, markers=None, im_kwargs=None, marker_kwargs=None, figsize=None, save_file=None,
        plot_ims=True, plot_diffs=True):
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
    figsize : :obj:`tuple`
        (width, height) in inches
    save_file : :obj:`str` or NoneType, optional
        figure saved if not None
    plot_ims : :obj:`bool`
        plot images
    plot_diffs : :obj:`bool`
        plot differences

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
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
