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
        min_p=5, max_p=95):
    """Helper function to compute input range for a variety of data types.

    Parameters
    ----------
    input_type : :obj:`str`
        'latents' | 'labels' | 'labels_sc'
    hparams : :obj:`dict`
    sess_ids : :obj:`list`
    sess_idx : :obj:`int`
    model : :obj:`AE` object
    data_gen : :obj:`ConcatSessionGenerator` object
    version
    min_p : :obj:`int`
    max_p : :obj:`int`

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
    input_range = compute_range(inputs, min_p=min_p, max_p=max_p)
    return input_range


def compute_range(values_list, min_p=5, max_p=95):
    """Compute min and max of a list of numbers using percentiles.

    Parameters
    ----------
    values_list : :obj:`list`
        list of np.ndarrays; min/max calculated over axis 0
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
        'min': np.percentile(values, min_p, axis=0),
        'max': np.percentile(values, max_p, axis=0)}
    return ranges


def get_labels_2d_for_trial(
        hparams, sess_ids, trial=None, trial_idx=None, sess_idx=0, dtype='test', data_gen=None):
    """Return framespace [scaled] labels for a given trial.

    Parameters
    ----------
    hparams
    sess_ids
    trial
    trial_idx
    sess_idx
    dtype
    data_gen

    Returns
    -------

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
    data_generator : :obj:`ConcatSessionsGenerator`
    hparams : :obj:`dict`
    model : pytorch model
    trial : :obj:`int`
        actual trial number
    trial_idx : :obj:`int`
        index into trials of type `dtype`
    sess_idx : :obj:`int`
    max_frames : :obj:`int`
    compute_latents : :obj:`bool`
    compute_2d_labels : :obj:`bool`
    compute_scaled_labels : :obj:`bool`
        ignored if `compute_2d_labels` is `True`; if `compute_scaled_labels=True`, return scaled
        labels as shape (batch, n_labels) rather than 2d labels as shape
        (batch, n_labels, y_pix, x_pix).
    dtype : :obj:`str`

    Returns
    -------
    :obj:`tuple`
        - ims_pt
        - ims_np
        - latents_np
        - labels_pt
        - labels_2d_pt
        - labels_2d_np

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
    if hparams['model_class'] == 'ae':
        labels_pt = None
        labels_np = None
    elif hparams['model_class'] == 'cond-ae' \
            or hparams['model_class'] == 'cond-ae-msp' \
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
        if hparams['model_class'] == 'cond-ae-msp':
            latents_np = model.get_transformed_latents(
                ims_pt, dataset=sess_idx, labels_2d=labels_2d_pt, as_numpy=True)
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
    """

    Parameters
    ----------
    interp_type
    model
    ims_0
    latents_0
    labels_0
    labels_sc_0
    mins
    maxes
    input_idxs
        must be y first, then x for proper marker recording
    n_frames
    crop_type
        currently only implements 'fixed'
    mins_sc
    maxes_sc
    crop_kwargs
    marker_idxs
        indicate which indices of ``labels_sc_0'' should be used for the marker when
        interp_type='latent' (otherwise the chosen marker defined by ``input_idxs''
        is used)
    ch : :obj:`int`, optional
        specify which channel of input images to return (can only be one)

    Returns
    -------

    """

    if interp_type == 'labels':
        from behavenet.data.transforms import MakeOneHot2D
        _, _, y_pix, x_pix = ims_0.shape
        one_hot_2d = MakeOneHot2D(y_pix, x_pix)

    # compute grid for relevant inputs
    inputs_0 = np.linspace(mins[0], maxes[0], n_frames)
    inputs_1 = np.linspace(mins[1], maxes[1], n_frames)
    if mins_sc is not None and maxes_sc is not None:
        inputs_0_sc = np.linspace(mins_sc[0], maxes_sc[0], n_frames)
        inputs_1_sc = np.linspace(mins_sc[1], maxes_sc[1], n_frames)
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
                latents[0, input_idxs[0]] = inputs_0[i0]
                latents[0, input_idxs[1]] = inputs_1[i1]

                # get scaled labels (for markers)
                if labels_sc_0 is not None:
                    if len(labels_sc_0.shape) == 3:
                        # 2d scaled labels
                        tmp = np.copy(labels_sc_0)
                        t, y, x = np.where(tmp[0] == 1)
                        labels_sc = np.hstack([x, y])[None, :]
                    else:
                        # 1d scaled labels
                        labels_sc = np.copy(labels_sc_0)

                if model.hparams['model_class'] == 'cond-ae-msp':
                    # get reconstruction
                    im_tmp = get_reconstruction(
                        model,
                        torch.from_numpy(latents).float(),
                        apply_inverse_transform=True)
                else:
                    # get labels
                    if model.hparams['model_class'] == 'ae':
                        labels = None
                    elif model.hparams['model_class'] == 'cond-ae':
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
                if len(labels_sc_0.shape) == 3:
                    # 2d scaled labels
                    tmp = np.copy(labels_sc_0)
                    t, y, x = np.where(tmp[0] == 1)
                    labels_sc = np.hstack([x, y])[None, :]
                    labels_sc[0, input_idxs[0]] = inputs_0_sc[i0]
                    labels_sc[0, input_idxs[1]] = inputs_1_sc[i1]
                    labels_2d = torch.from_numpy(one_hot_2d(labels_sc)).float()
                else:
                    # 1d scaled labels
                    labels_sc = np.copy(labels_sc_0)
                    labels_sc[0, input_idxs[0]] = inputs_0_sc[i0]
                    labels_sc[0, input_idxs[1]] = inputs_1_sc[i1]
                    labels_2d = None

                if model.hparams['model_class'] == 'cond-ae-msp':
                    # change latents that correspond to desired labels
                    latents = np.copy(latents_0)
                    latents[0, input_idxs[0]] = inputs_0[i0]
                    latents[0, input_idxs[1]] = inputs_1[i1]
                    # get reconstruction
                    im_tmp = get_reconstruction(model, latents, apply_inverse_transform=True)
                else:
                    # get (new) labels
                    labels = np.copy(labels_0)
                    labels[0, input_idxs[0]] = inputs_0[i0]
                    labels[0, input_idxs[1]] = inputs_1[i1]
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


def plot_2d_frame_array(
        ims_list, markers=None, im_kwargs=None, marker_kwargs=None, figsize=(15, 15),
        save_file=None):
    """

    Parameters
    ----------
    ims_list : :obj:`list` of :obj:`list
        each inner list element holds an np.array of shape (y_pix, x_pix)
    markers : :obj:`list` of :obj:`list` or NoneType, optional
        each inner list element holds an array-like object with values (y_pix, x_pix);
        if None, markers are not plotted on top of frames
    im_kwargs : :obj:`dict` or NoneType, optional
    marker_kwargs : :obj:`dict` or NoneType, optional
    figsize : :obj:`tuple`
    save_file : :obj:`str` or NoneType, optional
        figure saved if not None

    """
    n_x = len(ims_list)
    n_y = len(ims_list[0])
    fig, axes = plt.subplots(n_x, n_y, figsize=figsize)

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
