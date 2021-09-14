"""Paw processing pipeline classes."""

import cv2
import h5py
import numpy as np

from .utils import *

# camera constants (full size of right view, half size of left view)
IMG_WIDTH = 640
IMG_HEIGHT = 512

# offsets to fix timestamp alignment issues
# this has no effect on the creation of the hdf5 or the running of behavenet, but is necessary if
# users want to perform downstream analyses that require precise timing info (relating behavior to
# the trial structure, neural activity, etc.)
OFFSETS = {
    # churchlandlab/CSHL047/2020-01-20-001
    '89f0d6ff-69f4-45bc-b89e-72868abb042a': {'right': -1, 'left': -183},

    # churchlandlab/CSHL049/2020-01-08-001
    '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b': {'right': -9, 'left': -4},

    # cortexlab/KS023/2019-12-10-001
    'aad23144-0e52-4eac-80c5-c4ee2decb198': {'right': -5, 'left': -3},

    # hoferlab/SWC_043/2020-09-21-001
    '4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a': {'right': 0, 'left': 0},
}


class PawProcessor(object):
    """Video processing pipeline class for single-view IBL paw videos.

    For a given session, this class provides methods to:
    - download raw data from public IBL repository
    - load video (and associated timestamps) with opencv
    - load DLC markers
    - build hdf5 file for Behavenet

    Examples
    --------
    For example, to load markers:

    ```
    # initialize model
    vp = PawProcessor(one, view, eid, lab, animal, date, number)
    # compute paths
    vp.compute_paths(data_path_raw='/path/to/raw/data')
    # download data from public server
    vp.download_data()
    # load markers
    vp.load_2d_markers(likelihood_thresh=0.9)
    ```

    """

    # assume we're analyzing both paws
    marker_names = ['paw_r', 'paw_l']

    def __init__(self, one, view, eid, lab, animal, date, number):
        """Initialize video processing pipeline.

        Parameters
        ----------
        one : ONE object
            for searching/loading data on server
        view : str
            'left' | 'right'
        eid : str
            session eid
        lab : str
            lab name for this eid
        animal : str
            animal name for this eid
        date :  str
            date for this eid
        number : str
            expt number for this eid

        """

        # --------------------------------
        # metadata
        # --------------------------------
        self.eid = eid
        self.one = one
        self.view = view

        self.lab = lab          # e.g. 'cortexlab'
        self.animal = animal    # e.g. 'KS023'
        self.date = date        # e.g. '2019-12-10'
        self.number = number    # e.g. '001'
        self.session = '%s-%s' % (date, number)
        self.sess_str = '%s_%s_%s' % (lab, animal, self.session)

        # --------------------------------
        # pipeline checks
        # --------------------------------
        # timestamps loaded from numpy array
        self.is_load_timestamps = False

        # open cv video capture objects initialized
        self.is_load_video_cap = False

        # 2d markers loaded from local machine
        self.is_load_2d_markers = False

        # frame crop params computed for video plotting
        self.is_find_crop_params = False

        # --------------------------------
        # data objects
        # --------------------------------
        # object to handle paths for io
        self.paths = Paths()

        # object to handle video data
        self.video = Video()

        # object to handle associated 2D marker data
        self.markers = PawMarkers(self.marker_names)

        # crop lims
        # type : dict with keys 'xmin', 'xmax', 'ymin', 'ymax'
        self.crop_lims = None

    def __str__(self):
        """Pretty print session details."""
        print_str = '-------------------------------------\n'
        print_str += 'eid: %s\n' % self.eid
        print_str += 'lab: %s\n' % self.lab
        print_str += 'animal: %s\n' % self.animal
        print_str += 'date: %s\n' % self.date
        print_str += 'number: %s\n' % self.number
        print_str += '-------------------------------------'
        return print_str

    def compute_paths(self, data_path_raw):
        """Set various paths, stored in Paths object (accessed through self.paths)

        IBL paths
            - data_path_raw: base location of raw data (mp4, alf directory, etc)
            - session_path: location of raw data for this session
                session_path = data_path_raw/lab/Subjects/animal/date/number
            - alf_path: alf path (which contains timestamp and marker files)
                alf_path = session_path/alf
            - video_path: raw video data path (which contains mp4 files)
                video_path = session_path/raw_video_data

        Parameters
        ----------
        data_path_raw : str
            base path for raw data; for example, raw data for a specific session will be stored
            as `data_path_raw/lab/Subjects/animal/date/number`

        """

        # data from flatiron
        self.paths.data_path_raw = data_path_raw
        self.paths.session_path = os.path.join(
            data_path_raw, self.lab, 'Subjects', self.animal, self.date, self.number)
        self.paths.alf_path = os.path.join(self.paths.session_path, 'alf')
        self.paths.video_path = os.path.join(self.paths.session_path, 'raw_video_data')

    def download_data(self):
        """Download dlc markers and raw videos from server."""

        dataset_types = [
            '_iblrig_%sCamera.raw.mp4' % self.view,  # raw videos
            '_ibl_%sCamera.times.npy' % self.view,   # alf camera times
            '_ibl_%sCamera.dlc.pqt' % self.view,     # alf dlc traces
        ]
        for dataset_type in dataset_types:
            print('downloading %s...' % dataset_type, end='')
            self.one.load_dataset(self.eid, dataset_type, download_only=True)
            print('done')

    def load_timestamps(self, data_path_raw=None):
        """Load camera timestamps from alf directory.

        Parameters
        ----------
        data_path_raw : str, optional
            base path for raw data; required if paths have not already been computed

        """

        # make sure paths exist
        if self.paths.alf_path is None:
            self.compute_paths(data_path_raw=data_path_raw)

        # load data
        self.video.load_timestamps(
            os.path.join(self.paths.alf_path, '_ibl_%sCamera.times.npy' % self.view))

        # print message
        print('average %s camera framerate: %f Hz' % (
            self.view, 1.0 / np.mean(np.diff(self.video.timestamps))))

        # update pipeline check
        self.is_load_timestamps = True

    def load_video_cap(self, data_path_raw=None):
        """Initialize opencv video capture objects from videos in raw video directory.

        Parameters
        ----------
        data_path_raw : str, optional
            base path for raw data; required if paths have not already been computed

        """

        # make sure paths exist
        if self.paths.video_path is None:
            self.compute_paths(data_path_raw=data_path_raw)

        # load data
        filepath = os.path.join(self.paths.video_path, '_iblrig_%sCamera.raw.mp4' % self.view)
        if not os.path.exists(filepath):
            raise FileNotFoundError('Raw video has not been downloaded')
        self.video.load_video_cap(filepath)

        # print message
        print('%s camera:' % self.view)
        print('location: %s' % filepath)
        print('\tframes: %i' % self.video.total_frames)
        print('\tx_pix: %i' % self.video.frame_width)
        print('\ty_pix: %i' % self.video.frame_height)

        # update pipeline check
        self.is_load_video_cap = True

    def load_video_caps(self, **kwargs):
        """Alias for `load_video_cap()`"""
        self.load_video_cap(**kwargs)

    def load_2d_markers(self, likelihood_thresh=0.9, data_path_raw=None):
        """Load 2d markers from alf directory.

        Requires (will be automatically run if not already):
            - load_timestamps

        Parameters
        ----------
        likelihood_thresh : float, optional
            marker values are set to NaN when the corresponding likelihood is below this threshold
        data_path_raw : str, optional
            base path for raw data; required if paths have not already been computed

        """

        # make sure paths exist
        if self.paths.alf_path is None:
            self.compute_paths(data_path_raw=data_path_raw)

        # make sure necessary pipeline functions have run
        if not self.is_load_timestamps:
            self.load_timestamps(data_path_raw=data_path_raw)

        # collect markers
        self.markers.load_markers(
            self.paths.alf_path, view=self.view, likelihood_thresh=likelihood_thresh)

        # align timestamps
        t_timestamps = self.video.timestamps.shape[0]
        t_markers = self.markers.vals[self.markers.marker_names[0]].shape[0]
        if t_timestamps != t_markers:
            print('warning! timestamp mismatch')
            print('timestamps: %i' % t_timestamps)
            print('markers: %i' % t_markers)

            offset = OFFSETS.get(self.eid, {'right': None, 'left': None}).get(self.view, None)

            if offset <= 0:
                ab_off = abs(offset)
                self.video.timestamps = self.video.timestamps[ab_off:(t_markers + ab_off)]
                if t_markers > self.video.timestamps.shape[0]:
                    self.video.timestamps = np.concatenate([
                        self.video.timestamps,
                        np.nan * np.zeros((t_markers - self.video.timestamps.shape[0]))])
            else:
                self.video.timestamps = np.concatenate([
                    np.nan * np.zeros((offset,)),
                    self.video.timestamps[:(t_markers - offset)]])

            assert t_markers == self.video.timestamps.shape[0]

        # update pipeline check
        self.is_load_2d_markers = True

    def find_crop_params(self, load_kwargs={}):
        """Compute crop params that align each session.

        Requires (will be automatically run if not already):
            - load_2d_markers

        Parameters
        ----------
        load_kwargs : dict, optional
            arguments for `load_2d_markers` if this step has not already run

        """

        # make sure necessary pipeline functions have run
        if not self.is_load_2d_markers:
            self.load_2d_markers(**load_kwargs)

        # compute location of pupil/nose
        med_eye = {}
        med_nose = {}
        if self.view == 'left':
            mx, my = get_pupil_position(self.markers.vals)
            med_eye['x'], med_eye['y'] = mx / 2, my / 2
            mx, my = get_nose_position(self.markers.vals)
            med_nose['x'], med_nose['y'] = mx / 2, my / 2
        elif self.view == 'right':
            med_eye['x'], med_eye['y'] = get_pupil_position(self.markers.vals)
            med_nose['x'], med_nose['y'] = get_nose_position(self.markers.vals)
        # compute window
        xmin, xmax, ymin, ymax = get_frame_lims(
            med_eye['x'], med_eye['y'], med_nose['x'], med_nose['y'], self.view,
            vertical_align='nose')
        self.crop_lims = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}

        # update pipeline check
        self.is_find_crop_params = True

    def build_hdf5(
            self, hdf5_file, batch_size, xpix, ypix, n_batches, batch_selection=None):
        """Build hdf5 file with video and markers from a single view for Behavenet modeling.

        Requires:
            - load_timestamps
            - load_2d_markers
            - find_crop_params

        Parameters
        ----------
        hdf5_file : str
            filepath for final hdf5 file
        batch_size : int
            number of contiguous time points per batch
        xpix : int
            xpix of downsampled frames
        ypix : int
            ypix of downsampled frames
        n_batches : int
            total number of batches to add to hdf5
        batch_selection : str, optional
            'me': batches with highest motion energy
            'random': random batches
            None: use every time point in a batch

        """

        if not os.path.exists(os.path.dirname(hdf5_file)):
            os.makedirs(os.path.dirname(hdf5_file))

        n_total_frames = self.video.total_frames
        timestamps = np.arange(n_total_frames)

        # rearrange points/likelihoods data
        xs = np.hstack([self.markers.vals[m][:, 0, None] for m in self.marker_names])
        ys = np.hstack([self.markers.vals[m][:, 1, None] for m in self.marker_names])
        points_2d = np.hstack([xs, ys])

        ls = np.hstack([self.markers.masks[m][:, 0, None] for m in self.marker_names])
        likelihoods_2d = np.hstack([ls, ls])

        # select batches to include in hdf5
        if batch_selection is None:
            # use all frames
            n_trials = int(np.ceil(n_total_frames / batch_size))
            trials = np.arange(n_trials)
        elif batch_selection == 'random':
            # use a random subset of frames
            n_trials = n_batches
            trials = np.random.choice(
                int(np.ceil(n_total_frames / batch_size)), n_batches, replace=False)
        elif batch_selection == 'me':
            # use trials with highest motion energy
            n_trials = n_batches
            trials = get_highest_me_trials(points_2d, batch_size, n_batches)
            np.random.seed(0)
            np.random.shuffle(trials)
            print('using trials {}'.format(trials))
        else:
            raise ValueError('{} is an invalid batch selection method'.format(batch_selection))

        means = np.nanmean(points_2d, axis=0)
        stds = np.nanstd(points_2d, axis=0)

        # copy to save space
        lims = self.crop_lims

        with h5py.File(hdf5_file, 'w', libver='latest') as f:

            # single write multi-read
            f.swmr_mode = True

            # create image group
            group_i = f.create_group('images')

            # create labels group (z-scored)
            group_l = f.create_group('labels')

            # create original labels group (scaled to new image dims)
            group_l_sc = f.create_group('labels_sc')

            # create label mask group
            group_m = f.create_group('labels_masks')

            # create a dataset for each trial within groups
            for tr_idx, trial in enumerate(trials):

                if tr_idx % 25 == 0:
                    print('processing trial %04i/%04i' % (tr_idx, n_trials))

                # find video timestamps during this trial
                trial_beg = trial * batch_size
                trial_end = (trial + 1) * batch_size

                ts_idxs = np.where((timestamps >= trial_beg) & (timestamps < trial_end))[0]

                # ----------------------------------------------------------------------------
                # image data
                # ----------------------------------------------------------------------------
                # collect from video capture, peform initial downsampling
                frames_tmp = self.video.get_frames_from_idxs(ts_idxs)
                frames = [cv2.resize(f[0], (IMG_WIDTH, IMG_HEIGHT)) for f in frames_tmp]
                # crop and scale
                bs = len(frames)
                frames_proc = np.zeros((bs, 1, ypix, xpix), dtype='uint8')
                for b in range(bs):
                    frames_proc[b, 0, :, :] = cv2.resize(crop_frame(
                        frames[b], lims['xmin'], lims['xmax'], lims['ymin'], lims['ymax']),
                        (xpix, ypix))

                group_i.create_dataset('trial_%04i' % tr_idx, data=frames_proc, dtype='uint8')

                # ----------------------------------------------------------------------------
                # marker masks
                # ----------------------------------------------------------------------------
                group_m.create_dataset(
                    'trial_%04i' % tr_idx, data=likelihoods_2d[ts_idxs], dtype='float32')

                # ----------------------------------------------------------------------------
                # marker data (z-scored, masked)
                # ----------------------------------------------------------------------------
                points_tmp = (points_2d[ts_idxs] - means) / stds
                # because pytorch doesn't play well with nans
                points_tmp[likelihoods_2d[ts_idxs] == 0] = 0
                assert ~np.any(np.isnan(points_tmp))
                group_l.create_dataset('trial_%04i' % tr_idx, data=points_tmp, dtype='float32')

                # ----------------------------------------------------------------------------
                # marker data (scaled, masked)
                # ----------------------------------------------------------------------------
                points_tmp = {m: self.markers.vals[m][ts_idxs] for m in self.marker_names}
                points_tmp = crop_markers(
                    points_tmp, lims['xmin'], lims['xmax'], lims['ymin'], lims['ymax'])
                points_tmp = scale_markers(
                    points_tmp, lims['xmax'] - lims['xmin'], xpix, lims['ymax'] - lims['ymin'],
                    ypix)
                points_x = np.hstack([points_tmp[m][:, 0, None] for m in self.marker_names])
                points_y = np.hstack([points_tmp[m][:, 1, None] for m in self.marker_names])
                points_2d_tmp = np.hstack([points_x, points_y])
                group_l_sc.create_dataset(
                    'trial_%04i' % tr_idx, data=points_2d_tmp, dtype='float32')

    @staticmethod
    def test_hdf5_build(hdf5_file, save_file=None, idxs=[0], framerate=20):
        """Test hdf5 build by making a video from a single batch that includes frames and markers.

        Parameters
        ----------
        hdf5_file : str
            filepath of hdf5 file
        save_file : str, optional
            filepath of saved movie
        idxs : array-like, optional
            trial index to plot
        framerate : float, optional
            framerate of video

        Returns
        -------
        dict
            - 'images' (array-like): frames
            - 'labels' (array-like): 2d markers (z-scored)
            - 'labels_masks' (array-like): masks on 2d markers
            - 'labels_sc' (array-like): 2d markers (scaled to image)

        """

        if isinstance(idxs, int):
            idxs = [idxs]

        n_buffer = 5
        labels_2d = {'paw_l': [], 'paw_r': []}
        frames = []
        labels_masks = []

        with h5py.File(hdf5_file, 'r', libver='latest', swmr=True) as file:

            for idx in idxs:
                # load data
                frames_tmp = file['images'][str('trial_%04i' % idx)][()]
                labels_tmp = file['labels_sc'][str('trial_%04i' % idx)][()]
                labels_masks_tmp = file['labels_masks'][str('trial_%04i' % idx)][()]
                labels_tmp[labels_masks_tmp == 0] = np.nan
                # concatenate to previous data
                frames.append(frames_tmp)
                labels_masks.append(labels_masks_tmp)
                labels_2d['paw_l'].append(labels_tmp[:, np.array([0, 2])])
                labels_2d['paw_r'].append(labels_tmp[:, np.array([1, 3])])
                # concatenate blanks
                if idx != idxs[-1]:
                    n_t, n_ch, ypix, xpix = frames[-1].shape
                    frames.append(np.zeros((n_buffer, n_ch, ypix, xpix)))
                    labels_masks.append(np.zeros((n_buffer, labels_masks[-1].shape[1])))
                    labels_2d['paw_l'].append(np.nan * np.zeros((n_buffer, 2)))
                    labels_2d['paw_r'].append(np.nan * np.zeros((n_buffer, 2)))

            frames = np.vstack(frames)
            labels_masks = np.vstack(labels_masks)
            labels_2d['paw_l'] = np.vstack(labels_2d['paw_l'])
            labels_2d['paw_r'] = np.vstack(labels_2d['paw_r'])

        if save_file is not None:
            make_labeled_movie(
                save_file,
                frames=frames,
                points=labels_2d,
                framerate=framerate)

        data_dict = {
            'images': frames,
            'labels': labels_2d,
            'labels_masks': labels_masks,
        }

        return data_dict


class Video(object):
    """Simple class for loading videos and timestamps."""

    def __init__(self):

        # video timestamps
        # type : array of shape (n_timepoints,)
        self.timestamps = None

        # location of video timestamps
        # type : str
        self.timestamps_path = None

        # opencv video capture
        # type : cv2.VideoCapture object
        self.video_cap = None

        # location of opencv video capture
        # type : str
        self.video_cap_path = None

        # total frames
        # type : int
        self.total_frames = np.nan

        # frame width (pixels)
        # type : int
        self.frame_width = None

        # frame height (pixels)
        # type : int
        self.frame_height = None

    def load_timestamps(self, filepath):
        """Load camera timestamps from alf directory.

        Parameters
        ----------
        filepath : str
            absolute location of timestamps (.npy)

        """

        # save filepath
        self.timestamps_path = filepath

        # load data
        self.timestamps = np.load(filepath)

    def load_video_cap(self, filepath):
        """Initialize opencv video capture objects from video file.

        Parameters
        ----------
        filepath : str
            absolute location of video (.mp4, .avi)

        """

        # save filepath
        self.video_cap_path = filepath

        # load video cap
        self.video_cap = cv2.VideoCapture(filepath)
        if not self.video_cap.isOpened():
            raise IOError('error opening video file at %s' % filepath)

        # save frame info
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frames_from_idxs(self, idxs):
        """Helper function to load video segments.

        Parameters
        ----------
        idxs : array-like
            frame indices into video

        Returns
        -------
        np.ndarray
            returned frames of shape (n_frames, n_channels, y_pix, x_pix)

        """
        is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
        n_frames = len(idxs)
        for fr, i in enumerate(idxs):
            if fr == 0 or not is_contiguous:
                self.video_cap.set(1, i)
            ret, frame = self.video_cap.read()
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


class PawMarkers(object):
    """Class for loading paw markers."""

    def __init__(self, marker_names=[]):

        # marker names
        # type : list of strs
        self.marker_names = marker_names

        # 2D markers
        # type : dict with keys `self.marker_names`, vals are arrays of shape (n_t, 2)
        self.vals = {}

        # 2D marker likelihoods
        # type : dict with keys `self.marker_names`, vals are arrays of shape (n_t,)
        self.likelihoods = {}

        # 2D marker masks
        # type : dict with keys `self.marker_names`, vals are bool arrays of shape (n_t,)
        self.masks = {}

    def load_markers(self, markers_path, view, likelihood_thresh=0.9):
        """Load markers corresponding to videos.

        Parameters
        ----------
        markers_path : str
            absolute path of marker directory
        view : str
            'left' | 'right'
        likelihood_thresh : float, optional
            marker values are set to NaN when the corresponding likelihood is below this threshold

        """
        # load marker data
        self.vals, self.masks, self.likelihoods = get_markers(
            marker_path=markers_path, view=view, likelihood_thresh=likelihood_thresh)
        # print marker info
        print('%s camera: fraction of good markers' % view)
        for m in self.marker_names:
            frac = np.sum(self.masks[m]) / np.prod(self.masks[m].shape)
            print('%s: %1.2f' % (m, frac))


class Paths(object):
    """Class to store paths and allow for easy access."""

    def __init__(self):

        # data from flatiron
        self.data_path_raw = None
        self.session_path = None
        self.alf_path = None
        self.video_path = None

        # processed data in behavenet format
        self.data_path_proc = None
        self.hdf5_file = None

    def __str__(self):
        """Pretty print paths."""
        print_str = 'data_path_raw: {}\n'.format(self.data_path_raw)
        print_str += 'session_path: {}\n'.format(self.session_path)
        print_str += 'alf_path: {}\n'.format(self.alf_path)
        print_str += 'video_path: {}\n'.format(self.video_path)
        print_str += 'data_path_proc: {}\n'.format(self.data_path_proc)
        print_str += 'hdf5_file: {}\n'.format(self.hdf5_file)
        return print_str
