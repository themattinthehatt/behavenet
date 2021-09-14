"""Helper functions to preprocess IBL data from raw data to HDF5 files for Behavenet."""

import numpy as np
import os


# -------------------------------------------------------------------------------------------------
# Markers
# -------------------------------------------------------------------------------------------------

def get_markers(marker_path, view, likelihood_thresh=0.9):
    """Load DLC markers (in original frame resolution) and likelihood masks from alf directory.

    Parameters
    ----------
    marker_path : str
        path to directory that contains markers
    view : str
        'left' | 'right'
    likelihood_thresh : float
        dlc likelihoods below this value returned as NaNs

    Returns
    -------
    tuple
        - XYs (dict): keys are body parts, values are np.ndarrays of shape (n_t, 2)
        - masks (dict): keys are body parts, values are np.ndarrays of shape (n_t, 2)

    """

    import pandas as pd

    dlc_path = os.path.join(marker_path, '_ibl_%sCamera.dlc.pqt' % view)
    cam = pd.read_parquet(dlc_path)
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    d = list(points)
    points = np.array(d)

    # Set values to nan if likelihood is too low
    XYs = {}
    masks = {}
    likelihoods = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'].to_numpy() < likelihood_thresh,
            cam[point + '_x'].to_numpy())
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'].to_numpy() < likelihood_thresh,
            cam[point + '_y'].to_numpy())
        y = y.filled(np.nan)
        XYs[point] = np.hstack([x[:, None], y[:, None]])
        masks[point] = np.ones_like(XYs[point])
        masks[point][np.isnan(XYs[point])] = 0
        likelihoods[point] = cam[point + '_likelihood']

    return XYs, masks, likelihoods


def get_pupil_position(markers):
    """Find median x/y position of pupil in left/right videos.

    Parameters
    ----------
    markers : dict
        keys are body parts, values are np.ndarrays of shape (n_t, 2); must contain
        `'pupil_bottom_r'`, `'pupil_left_r'`, `'pupil_right_r'`, `'pupil_top_r'` or equivalent for
        left side

    Returns
    -------
    tuple
        - x-value (float)
        - y-value (float)

    """
    if 'pupil_bottom_r' in list(markers.keys()):
        pupil_markers = ['pupil_bottom_r', 'pupil_left_r', 'pupil_right_r', 'pupil_top_r']
    else:
        pupil_markers = ['pupil_bottom_l', 'pupil_left_l', 'pupil_right_l', 'pupil_top_l']
    pupil_x = []
    pupil_y = []
    for pm in pupil_markers:
        pupil_x.append(markers[pm][:, 0, None])
        pupil_y.append(markers[pm][:, 1, None])
    pupil_x = np.hstack(pupil_x)
    pupil_y = np.hstack(pupil_y)
    median_x = np.nanmedian(pupil_x)
    median_y = np.nanmedian(pupil_y)
    return median_x, median_y


def get_nose_position(markers):
    """Find median x/y position of nose tip in left/right videos.

    Parameters
    ----------
    markers : dict
        keys are body parts, values are np.ndarrays of shape (n_t, 2); must contain `'nose_tip'`

    Returns
    -------
    tuple
        - x-value (float)
        - y-value (float)

    """
    return np.nanmedian(markers['nose_tip'], axis=0)


def crop_markers(markers, xmin, xmax, ymin, ymax):
    """Update marker values to reflect crop of corresponding images.

    Parameters
    ----------
    markers : dict or array-like
        if dict, keys are body parts, values are np.ndarrays of shape (n_time, 2)
        if array-like, np.ndarray of shape (n_time, 2)
    xmin : float
        min x value from image crop
    xmax : float
        max x value from image crop
    ymin : float
        min y value from image crop
    ymax : float
        max y value from image crop

    Returns
    -------
    variable
        same type as input, with updated marker values

    """
    if isinstance(markers, dict):
        marker_names = list(markers.keys())
        markers_crop = {}
        for m in marker_names:
            markers_crop[m] = markers[m] - np.array([xmin, ymin])
    else:
        markers_crop = markers - np.array([xmin, ymin])
    return markers_crop


def scale_markers(markers, xpix_old, xpix_new, ypix_old, ypix_new):
    """Update marker values to reflect scale of corresponding images.

    Parameters
    ----------
    markers : dict or array-like
        if dict, keys are body parts, values are np.ndarrays of shape (n_time, 2)
        if array-like, np.ndarray of shape (n_time, 2)
    xpix_old : int
        xpix of original images
    xpix_new
        xpix of new images
    ypix_old
        ypix of old images
    ypix_new
        ypix of new images

    Returns
    -------
    variable
        same type as input, with updated marker values

    """
    old = np.array([xpix_old, ypix_old])
    new = np.array([xpix_new, ypix_new])
    if isinstance(markers, dict):
        marker_names = list(markers.keys())
        markers_scale = {}
        for m in marker_names:
            markers_scale[m] = (markers[m] / old) * new
    else:
        markers_scale = (markers / old) * new
    return markers_scale


# -------------------------------------------------------------------------------------------------
# Frames
# -------------------------------------------------------------------------------------------------

def crop_frame(fr, xmin, xmax, ymin, ymax):
    """Crop frame, inserting zeros (black pixels) if bounds go beyond frame dimensions.

    Parameters
    ----------
    fr : array-like
        frame to crop, of shape (ypix, xpix)
    xmin : int
    xmax : int
    ymin : int
    ymax : int

    Returns
    -------
    array-like
        cropped frame

    """
    ypix, xpix = fr.shape

    if xmin < 0 or ymin < 0 or xmax > xpix or ymax > ypix:
        frame = np.zeros((ymax-ymin, xmax-xmin))
        # indices into original frame - don't go outside bounds!
        xmn = np.max([0, xmin])
        ymn = np.max([0, ymin])
        xmx = np.min([xmax, xpix])
        ymx = np.min([ymax, ypix])
        # indices into new frame - might contain black edges
        x_l = np.min([0, xmin])  # xmin>0, start at edge (0), else offset (-xmin)
        y_l = np.min([0, ymin])  # ymin>0, start at edge (0), else offset (-ymin)
        frame[-y_l:(-y_l + ymx - ymn), -x_l:(-x_l + xmx - xmn)] = fr[ymn:ymx, xmn:xmx]
    else:
        frame = fr[ymin:ymax, xmin:xmax]
    return frame


def get_frame_lims(x_eye, y_eye, x_nose, y_nose, view, vertical_align='eye'):
    """Automatically compute the crop parameters of a view using the eye and nose and reference.

    Note that horizontal/vertical proportions are currently hard-coded.

    Parameters
    ----------
    x_eye : float
        x position of the eye
    y_eye : float
        y position of the eye
    x_nose : float
        x position of the nose
    y_nose : float
        y position of the nose
    view : str
        'left' | 'right'
    vertical_align : str
        defines which feature controls the vertical alignment
        'eye' | 'nose'

    Returns
    -------
    tuple
        - xmin (float)
        - xmax (float)
        - ymin (float)
        - ymax (float)

    """
    # horizontal proportions
    edge2nose = 0.02
    nose2eye = 0.33
    eye2edge = 0.65
    # vertical proportions
    eye2top = 0.10
    eye2bot = 0.90
    nose2top = 0.25
    nose2bot = 0.75
    # horizontal calc
    nose2eye_pix = np.abs(x_eye - x_nose)
    edge2nose_pix = edge2nose / nose2eye * nose2eye_pix
    eye2edge_pix = eye2edge / nose2eye * nose2eye_pix
    total_x_pix = np.round(nose2eye_pix + edge2nose_pix + eye2edge_pix)
    if view == 'left':
        xmin = int(x_nose - edge2nose_pix)
        xmax = int(x_eye + eye2edge_pix)
    elif view == 'right':
        xmin = int(x_eye - eye2edge_pix)
        xmax = int(x_nose + edge2nose_pix)
    else:
        raise Exception
    # vertical calc (assume we want a square image out)
    if vertical_align == 'eye':
        # based on eye
        eye2top_pix = eye2top * total_x_pix
        eye2bot_pix = eye2bot * total_x_pix
        ymin = int(y_eye - eye2top_pix)
        ymax = int(y_eye + eye2bot_pix)
    else:
        # based on nose
        nose2top_pix = nose2top * total_x_pix
        nose2bot_pix = nose2bot * total_x_pix
        ymin = int(y_nose - nose2top_pix)
        ymax = int(y_nose + nose2bot_pix)
    return xmin, xmax, ymin, ymax


def make_labeled_movie(save_file, frames, points, framerate=20, height=4):
    """Behavioral video overlaid with markers.

    Parameters
    ----------
    save_file : str
        absolute path of video (including file extension)
    frames : np.ndarray
        frame array of shape (n_frames, n_channels, ypix, xpix)
    points : dict
        keys of marker names and vals of marker values,
        i.e. `points['paw_l'].shape = (n_t, 2)`
    framerate : float, optional
        framerate of video
    height : float, optional
        height of movie in inches

    """
    import matplotlib.pyplot as plt

    tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    n_frames, _, img_height, img_width = frames.shape

    h = height
    w = h * (img_width / img_height)
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    ax.set_yticks([])
    ax.set_xticks([])
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    for n in range(n_frames):

        ax.clear()  # important!! otherwise each frame will plot on top of the last

        if n % 100 == 0:
            print('processing frame %03i/%03i' % (n, n_frames))

        # plot original frame
        ax.imshow(frames[n, 0], vmin=0, vmax=255, cmap='gray')
        # plot markers
        for m, (marker_name, marker_vals) in enumerate(points.items()):
            ax.plot(
                marker_vals[n, 0], marker_vals[n, 1], 'o', markersize=8)

        ax.set_xlim([0, img_width])
        ax.set_ylim([img_height, 0])

        plt.savefig(os.path.join(tmp_dir, 'frame_%06i.jpeg' % n))

    save_video(save_file, tmp_dir, framerate)


def save_video(save_file, tmp_dir, framerate=20):
    """Create video with ffmepg from a directory of images.

    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    tmp_dir : str
        temporary directory that stores frames of video; this directory will be deleted
    framerate : float, optional
        framerate of final video

    """
    import subprocess
    import shutil

    if os.path.exists(save_file):
        os.remove(save_file)

    # make mp4 from images using ffmpeg
    call_str = \
        'ffmpeg -r %f -i %s -c:v libx264 %s' % (
            framerate, os.path.join(tmp_dir, 'frame_%06d.jpeg'), save_file)
    print(call_str)
    subprocess.run(['/bin/bash', '-c', call_str], check=True)

    # delete tmp directory
    shutil.rmtree(tmp_dir)


# -------------------------------------------------------------------------------------------------
# Misc
# -------------------------------------------------------------------------------------------------

def get_highest_me_trials(markers_2d, batch_size, n_batches):
    """Find trials with highest motion energy to help with batch selection.

    Parameters
    ----------
    markers_2d : dict
        keys are camera names; vals are themselves dicts with marker names; those vals are arrays
        of shape (n_timepoints, 2), i.e.
        >> points_2d['left']['paw_l'].shape
        >> (100, 2)
    batch_size : int
        number of contiguous time points per batch
    n_batches : int
        total number of batches to add to hdf5

    Returns
    -------
    array-like
        trial indices of the `n_batches` trials with highest motion energy (sorted low to high)

    """

    # just use paws to compute motion energy
    if isinstance(markers_2d, dict):
        vll = np.vstack([np.zeros((1, 2)), np.diff(markers_2d['left']['paw_l'], axis=0)])
        vlr = np.vstack([np.zeros((1, 2)), np.diff(markers_2d['left']['paw_r'], axis=0)])
        vrr = np.vstack([np.zeros((1, 2)), np.diff(markers_2d['right']['paw_r'], axis=0)])
        vrl = np.vstack([np.zeros((1, 2)), np.diff(markers_2d['right']['paw_l'], axis=0)])
        me_all = np.abs(np.hstack([vll, vlr, vrr, vrl]))
    else:
        me_all = np.abs(
            np.vstack([np.zeros((1, markers_2d.shape[1])), np.diff(markers_2d, axis=0)]))

    n_total_frames = me_all.shape[0]
    n_trials = int(np.ceil(n_total_frames / batch_size))
    assert n_trials >= batch_size

    total_me = np.zeros(n_trials)
    for trial in range(n_trials):
        trial_beg = trial * batch_size
        trial_end = (trial + 1) * batch_size
        total_me[trial] = np.nanmean(me_all[trial_beg:trial_end])

    total_me[np.isnan(total_me)] = -100  # nans get pushed to end of sorted array
    sorted_me_idxs = np.argsort(total_me)
    best_trials = sorted_me_idxs[-n_batches:]

    return best_trials


def nanargmax(array):
    """Sorts non-nan values in a non-negative array, highest to lowest.

    Parameters
    ----------
    array : array-like

    Returns
    -------
    np.ndarray

    """
    array_c = np.copy(array)
    nan_idxs = np.where(np.isnan(array_c))[0]
    array_c[nan_idxs] = 0
    array_sorted = np.argsort(array_c)
    return array_sorted[::-1]
