"""Utility functions for automatically constructing hdf5 files."""

import cv2
import h5py
from numpy import genfromtxt
import numpy as np
import os

# to ignore imports for sphinx-autoapidoc
__all__ = ['build_hdf5', 'load_raw_labels', 'resize_labels', 'get_frames_from_idxs']


def build_hdf5(
        save_file, video_file, label_file=None, pose_algo=None, batch_size=128, xpix=None,
        ypix=None, label_likelihood_thresh=0.9, zscore=True):
    """Build Behavenet-style HDF5 file from video file and optional label file.

    This function provides a basic example for how to convert raw video and label files into the
    processed version required by Behavenet. In doing so no additional assumptions are made about
    a possible trial structure; equally-sized batches are created. For more complex data, users
    will need to adapt this function to suit their own needs.

    Parameters
    ----------
    save_file : :obj:`str`
        absolute file path of new HDF5 file; the directory does not need to be created beforehand
    video_file : :obj:`str`
        absolute file path of the video (.mp4, .avi)
    label_file : :obj:`str`, optional
        absolute file path of the labels; current formats include DLC/DGP csv or h5 files
    pose_algo : :obj:`str`, optional
        'dlc' | 'dgp'
    batch_size : :obj:`int`, optional
        uniform batch size of data
    xpix : :obj:`int`, optional
        if not None, video frames will be reshaped before storing in the HDF5
    ypix : :obj:`int`, optional
        if not None, video frames will be reshaped before storing in the HDF5
    label_likelihood_thresh : :obj:`float`, optional
        likelihood threshold used to define masks; any labels/timepoints with a likelihood below
        this value will be set to NaN
    zscore : :obj:`bool`, optional
        individually z-score each label before saving in the HDF5

    """

    # load video capture
    video_cap = cv2.VideoCapture(video_file)
    n_total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    xpix_og = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ypix_og = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # load labels
    if label_file is not None:
        labels, masks = load_raw_labels(
            label_file, pose_algo=pose_algo, likelihood_thresh=label_likelihood_thresh)
        # error check
        n_total_labels = labels.shape[0]
        assert n_total_frames == n_total_labels, 'Number of frames does not match number of labels'
    else:
        labels = None

    n_trials = int(np.ceil(n_total_frames / batch_size))
    trials = np.arange(n_trials)

    timestamps = np.arange(n_total_frames)

    # compute z-score params
    if label_file is not None and zscore:
        means = np.nanmean(labels, axis=0)
        stds = np.nanstd(labels, axis=0)
    else:
        means = None
        stds = None

    # create directory for hdf5 if it doesn't already exist
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    with h5py.File(save_file, 'w', libver='latest', swmr=True) as f:

        # single write multi-read
        f.swmr_mode = True

        # create image group
        group_i = f.create_group('images')

        if label_file is not None:
            # create labels group (z-scored)
            group_l = f.create_group('labels')

            # create label mask group
            group_m = f.create_group('labels_masks')

            # create labels group (not z-scored, but downsampled if necessary)
            group_ls = f.create_group('labels_sc')

        # create a dataset for each trial within groups
        for tr_idx, trial in enumerate(trials):

            # find video timestamps during this trial
            trial_beg = trial * batch_size
            trial_end = (trial + 1) * batch_size

            ts_idxs = np.where((timestamps >= trial_beg) & (timestamps < trial_end))[0]

            # ----------------------------------------------------------------------------
            # image data
            # ----------------------------------------------------------------------------
            # collect from video capture, downsample
            frames_tmp = get_frames_from_idxs(video_cap, ts_idxs)
            if xpix is not None and ypix is not None:
                # Nones to add batch/channel dims
                frames_tmp = [cv2.resize(f[0], (xpix, ypix))[None, None, ...] for f in frames_tmp]
            else:
                frames_tmp = [f[None, ...] for f in frames_tmp]
            group_i.create_dataset(
                'trial_%04i' % tr_idx, data=np.vstack(frames_tmp), dtype='uint8')

            # ----------------------------------------------------------------------------
            # label data
            # ----------------------------------------------------------------------------
            if label_file is not None:
                # label masks
                group_m.create_dataset('trial_%04i' % tr_idx, data=masks[ts_idxs], dtype='float32')

                # label data (zscored, masked)
                labels_tmp = (labels[ts_idxs] - means) / stds
                labels_tmp[masks[ts_idxs] == 0] = 0  # pytorch doesn't play well with nans
                assert ~np.any(np.isnan(labels_tmp))
                group_l.create_dataset('trial_%04i' % tr_idx, data=labels_tmp, dtype='float32')

                # label data (non-zscored, masked)
                labels_tmp = labels[ts_idxs]
                labels_tmp = resize_labels(labels_tmp, xpix, ypix, xpix_og, ypix_og)
                labels_tmp[masks[ts_idxs] == 0] = 0
                group_ls.create_dataset('trial_%04i' % tr_idx, data=labels_tmp, dtype='float32')


def load_raw_labels(file_path, pose_algo, likelihood_thresh=0.9):
    """Load labels and build masks from a variety of standardized source files.

    This function currently supports the loading of csv and h5 files output by DeepLabCut (DLC) and
    Deep Graph Pose (DGP).

    Parameters
    ----------
    file_path : :obj:`str`
        absolute file path of label file
    pose_algo : :obj:`str`
        'dlc' | 'dgp'
    likelihood_thresh : :obj:`float`
        likelihood threshold used to define masks; any labels/timepoints with a likelihood below
        this value will be set to NaN and the corresponding masks file with have a 0

    Returns
    -------
    :obj:`tuple`
        - (array-like): labels, all x-values first, then all y-values
        - (array-like): masks; 1s correspond to good values, 0s correspond to bad values

    """
    if pose_algo == 'dlc' or pose_algo == 'dgp':
        file_ext = file_path.split('.')[-1]
        if file_ext == 'csv':
            labels_tmp = genfromtxt(file_path, delimiter=',', dtype=None, encoding=None)
            labels_tmp = labels_tmp[3:, 1:].astype('float')  # get rid of headers, etc.
        elif file_ext == 'h5':
            with h5py.File(file_path, 'r') as f:
                t = f['df_with_missing']['table'][()]
            labels_tmp = np.concatenate([t[i][1][None, :] for i in range(len(t))])
        else:
            raise NotImplementedError(
                '"%s" is an unsupported file extentsion for %s' % (file_ext, pose_algo))
        xvals = labels_tmp[:, 0::3]
        yvals = labels_tmp[:, 1::3]
        likes = labels_tmp[:, 2::3]
        labels = np.hstack([xvals, yvals])
        likes = np.hstack([likes, likes])
        masks = 1.0 * (likes >= likelihood_thresh)
        labels[masks != 1] = np.nan
    elif pose_algo == 'dpk':
        raise NotImplementedError
    elif pose_algo == 'leap':
        raise NotImplementedError
    else:
        raise NotImplementedError('the pose algorithm "%s" is currently unsupported' % pose_algo)

    return labels, masks


def resize_labels(labels, xpix_new, ypix_new, xpix_old, ypix_old):
    """Update label values to reflect scale of corresponding images.

    Parameters
    ----------
    labels : :obj:`array-like`
        np.ndarray of shape (n_time, 2 * n_labels); for a given row, all x-values come first,
        followed by all y-values
    xpix_new : :obj:`int`
        xpix of new images
    ypix_new : :obj:`int`
        ypix of new images
    xpix_old : :obj:`int`
        xpix of original images
    ypix_old : :obj:`int`
        ypix of original images


    Returns
    -------
    array-like
        resized label values

    """
    if xpix_new is None or ypix_new is None:
        return labels
    else:
        n_labels = labels.shape[1] // 2
        old = np.array([xpix_old] * n_labels + [ypix_old] * n_labels)
        new = np.array([xpix_new] * n_labels + [ypix_new] * n_labels)
        labels_scale = (labels / old) * new
        return labels_scale


def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : :obj:`cv2.VideoCapture` object
    idxs : :obj:`array-like`
        frame indices into video

    Returns
    -------
    obj:`array-like`
        returned frames of shape shape (n_frames, y_pix, x_pix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames
