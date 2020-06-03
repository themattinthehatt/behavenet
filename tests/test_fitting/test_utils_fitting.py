import os
import pytest
from behavenet.fitting import utils


def dict2str(dictionary):
    """Helper function to easily compare lists of dicts whose values are strings."""
    s = ''
    for k, v in dictionary.items():
        s += str(v)
    return s


class TestClass:

    @pytest.fixture(autouse=True)
    def test_init(self, tmpdir):
        """Build temp directory structure for testing io functions.

        tmpdir
        ├── multisession-06
        |   └── session_info.csv
        ├── lab0
        |   ├── multisession-00
        |   |   └── session_info.csv
        |   └── expt0
        |   |   ├── multisession-00
        |   |   |   └── session_info.csv
        |   |   ├── multisession-01
        |   |   |   └── session_info.csv
        |   |   ├── animal0
        |   |   |   ├── multisession-00
        |   |   |   |   └── session_info.csv
        |   |   |   ├── multisession-01
        |   |   |   |   └── session_info.csv
        |   |   |   ├── session-00
        |   |   |   ├── session-01
        |   |   |   └── session-02
        |   |   └── animal1
        |   |       ├── multisession-03
        |   |       |   └── session_info.csv
        |   |       ├── multisession-04
        |   |       |   └── session_info.csv
        |   |       └── session-00
        |   └── expt1
        |       ├── animal0
        |       |   └── session-00
        |       └── animal1
        |           └── session-00
        └── lab1
            └── expt0
                └── animal0
                    ├── session-00
                    └── session-01

        """

        self.tmpdir = tmpdir

        # build tmp directory structure
        self.l0_dir = self.tmpdir.mkdir('lab0')
        self.l1_dir = self.tmpdir.mkdir('lab1')
        self.tmpdir.mkdir('multisession-06')

        # lab0 subdirs
        self.l0e0_dir = self.l0_dir.mkdir('expt0')
        self.l0e1_dir = self.l0_dir.mkdir('expt1')

        # lab0/expt0 subdirs
        self.l0e0a0_dir = self.l0e0_dir.mkdir('animal0')
        self.l0e0a1_dir = self.l0e0_dir.mkdir('animal1')
        self.l0e0_dir.mkdir('multisession-00')
        self.l0e0_dir.mkdir('multisession-01')

        # lab0/expt0/animal0 subdirs
        self.l0e0a0_dir.mkdir('session-00')
        self.l0e0a0_dir.mkdir('session-01')
        self.l0e0a0_dir.mkdir('session-02')
        self.l0e0a0_dir.mkdir('multisession-00')
        self.l0e0a0_dir.mkdir('multisession-01')

        # lab0/expt0/animal1 subdirs
        self.l0e0a1_dir.mkdir('session-00')
        self.l0e0a1_dir.mkdir('multisession-03')
        self.l0e0a1_dir.mkdir('multisession-04')

        # lab0/expt1 subdirs
        self.l0e1a0_dir = self.l0e1_dir.mkdir('animal0')
        self.l0e1a1_dir = self.l0e1_dir.mkdir('animal1')

        # lab0/expt1/animal0 subdirs
        self.l0e1a0_dir.mkdir('session-00')

        # lab0/expt1/animal1 subdirs
        self.l0e1a1_dir.mkdir('session-00')

        # lab1 subdirs
        self.l1e0_dir = self.l1_dir.mkdir('expt0')

        # lab1/expt0 subdirs
        self.l1e0a0_dir = self.l1e0_dir.mkdir('animal0')

        # lab1/expt0/animal0 subdirs
        self.l1e0a0_dir.mkdir('session-00')
        self.l1e0a0_dir.mkdir('session-01')

        # collect session ids in a list
        self.sess_ids = [
            {'lab': 'lab0', 'expt': 'expt0', 'animal': 'animal0', 'session': 'session-00'},
            {'lab': 'lab0', 'expt': 'expt0', 'animal': 'animal0', 'session': 'session-01'},
            {'lab': 'lab0', 'expt': 'expt0', 'animal': 'animal0', 'session': 'session-02'},
            {'lab': 'lab0', 'expt': 'expt0', 'animal': 'animal1', 'session': 'session-00'},
            {'lab': 'lab0', 'expt': 'expt1', 'animal': 'animal0', 'session': 'session-00'},
            {'lab': 'lab0', 'expt': 'expt1', 'animal': 'animal1', 'session': 'session-00'},
            {'lab': 'lab1', 'expt': 'expt0', 'animal': 'animal0', 'session': 'session-00'},
            {'lab': 'lab1', 'expt': 'expt0', 'animal': 'animal0', 'session': 'session-01'},
        ]

        # create csv file for expt0/animal0/multisession-00
        self.l0e0a0_idxs = [0, 1, 2]
        self.l0e0a0_id = 0
        csv_path = os.path.join(self.l0e0a0_dir, 'multisession-%02i' % self.l0e0a0_id)
        self.l0e0a0_csv = os.path.join(csv_path, 'session_info.csv')
        utils.export_session_info_to_csv(csv_path, [self.sess_ids[i] for i in self.l0e0a0_idxs])

        # create csv file for expt0/animal0/multisession-01
        self.l0e0a0m1_idxs = [1, 2]
        self.l0e0a0m1_id = 1
        csv_path = os.path.join(self.l0e0a0_dir, 'multisession-%02i' % self.l0e0a0m1_id)
        self.l0e0a0m1_csv = os.path.join(csv_path, 'session_info.csv')
        utils.export_session_info_to_csv(csv_path, [self.sess_ids[i] for i in self.l0e0a0m1_idxs])

        # create csv file for expt0/animal1/multisession-03
        self.l0e0a1_idxs = [3]
        self.l0e0a1_id = 3
        csv_path = os.path.join(self.l0e0a1_dir, 'multisession-%02i' % self.l0e0a1_id)
        self.l0e0a1_csv = os.path.join(csv_path, 'session_info.csv')
        utils.export_session_info_to_csv(csv_path, [self.sess_ids[i] for i in self.l0e0a1_idxs])

        # create csv file for expt0/animal1/multisession-04
        self.l0e0a1m4_idxs = [3]
        self.l0e0a1m4_id = 4
        csv_path = os.path.join(self.l0e0a1_dir, 'multisession-%02i' % self.l0e0a1m4_id)
        self.l0e0a1m4_csv = os.path.join(csv_path, 'session_info.csv')
        utils.export_session_info_to_csv(csv_path, [self.sess_ids[i] for i in self.l0e0a1m4_idxs])

        # create csv file for expt0/multisession-00
        self.l0e0_idxs = [0, 1, 2, 3]
        self.l0e0_id = 0
        csv_path = os.path.join(self.l0e0_dir, 'multisession-%02i' % self.l0e0_id)
        self.l0e0_csv = os.path.join(csv_path, 'session_info.csv')
        utils.export_session_info_to_csv(csv_path, [self.sess_ids[i] for i in self.l0e0_idxs])

        # create csv file for expt0/multisession-01
        self.l0e0m1_idxs = [0, 3]
        self.l0e0m1_id = 1
        csv_path = os.path.join(self.l0e0_dir, 'multisession-%02i' % self.l0e0m1_id)
        self.l0e0m1_csv = os.path.join(csv_path, 'session_info.csv')
        utils.export_session_info_to_csv(csv_path, [self.sess_ids[i] for i in self.l0e0m1_idxs])

        # create csv file for lab0/multisession-00
        self.l0_idxs = [0, 1, 2, 3, 4, 5]
        self.l0_id = 0
        csv_path = os.path.join(self.l0_dir, 'multisession-%02i' % self.l0_id)
        self.l0_csv = os.path.join(csv_path, 'session_info.csv')
        utils.export_session_info_to_csv(csv_path, [self.sess_ids[i] for i in self.l0_idxs])

        # create csv file for multisession-06
        self.l_idxs = [0, 1, 2, 3, 4, 5, 6]
        self.l_id = 6
        csv_path = os.path.join(self.tmpdir, 'multisession-%02i' % self.l_id)
        self.l_csv = os.path.join(csv_path, 'session_info.csv')
        utils.export_session_info_to_csv(csv_path, [self.sess_ids[i] for i in self.l_idxs])

        self.l0e0a0s0_idxs = [0]
        self.l0e0a0s0_id = 0

        self.l1e0a0_idxs = [6, 7]
        self.l1e0a0_id = 0

    def test_get_subdirs(self):

        # raise exception when no subdirs
        with pytest.raises(StopIteration):
            utils.get_subdirs(os.path.join(self.tmpdir, self.l0_dir, 'multisession-00'))

        # find subdirs
        subdirs = utils.get_subdirs(os.path.join(self.tmpdir, self.l0_dir))
        assert sorted(subdirs) == ['expt0', 'expt1', 'multisession-00']

        # raise exception when not a path
        with pytest.raises(ValueError):
            utils.get_subdirs('/ZzZtestingZzZ')

    def test_get_multisession_paths(self):

        # multisessions at base dir level
        paths = utils._get_multisession_paths(self.l1_dir)
        assert paths == []

        # multisessions at lab level
        lab_subdir = os.path.join(self.tmpdir, 'lab0')
        paths = utils._get_multisession_paths(self.tmpdir, lab='lab0')
        assert paths == [os.path.join(lab_subdir, 'multisession-00')]

        # multisessions at lab level
        expt_subdir = os.path.join(lab_subdir, 'expt0')
        paths = utils._get_multisession_paths(self.tmpdir, lab='lab0', expt='expt0')
        assert sorted(paths) == [
            os.path.join(expt_subdir, 'multisession-00'),
            os.path.join(expt_subdir, 'multisession-01')]

        # multisessions at animal0 level
        animal0_subdir = os.path.join(expt_subdir, 'animal0')
        paths = utils._get_multisession_paths(
            self.tmpdir, lab='lab0', expt='expt0', animal='animal0')
        assert sorted(paths) == [
            os.path.join(animal0_subdir, 'multisession-00'),
            os.path.join(animal0_subdir, 'multisession-01')]

        # multisessions at animal1 level
        animal1_subdir = os.path.join(expt_subdir, 'animal1')
        paths = utils._get_multisession_paths(
            self.tmpdir, lab='lab0', expt='expt0', animal='animal1')
        assert sorted(paths) == [
            os.path.join(animal1_subdir, 'multisession-03'),
            os.path.join(animal1_subdir, 'multisession-04')]

        # no multisessions
        paths = utils._get_multisession_paths(
            self.tmpdir, lab='lab1', expt='expt0', animal='animal0')
        assert len(paths) == 0

    def test_get_single_sessions(self):

        sess_ret = utils._get_single_sessions(self.tmpdir, depth=4, curr_depth=0)

        for sess in self.sess_ids:
            if sess not in sess_ret:
                raise ValueError

    def test_get_session_dir(self):

        hparams = {'data_dir': self.tmpdir, 'save_dir': self.tmpdir}

        # ------------------------------------------------------------
        # csv contained in multisession directory
        # ------------------------------------------------------------
        # single session from one animal
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'animal1'
        hparams['session'] = 'session-00'
        hparams['sessions_csv'] = self.l0e0a1_csv
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            hparams['session'])
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        assert sess_dir == sess_dir_
        assert sess_single == [self.sess_ids[i] for i in self.l0e0a1_idxs]

        # multiple sessions from one animal
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'animal0'
        hparams['sessions_csv'] = self.l0e0a0_csv
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            'multisession-%02i' % self.l0e0a0_id)
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        assert sess_dir == sess_dir_
        assert sess_single == [self.sess_ids[i] for i in self.l0e0a0_idxs]

        # multiple sessions from one experiment
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt0'
        hparams['sessions_csv'] = self.l0e0_csv
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'],
            'multisession-%02i' % self.l0e0_id)
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        assert sess_dir == sess_dir_
        assert sess_single == [self.sess_ids[i] for i in self.l0e0_idxs]

        # multiple sessions from one lab
        hparams['lab'] = 'lab0'
        hparams['sessions_csv'] = self.l0_csv
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], 'multisession-%02i' % self.l0_id)
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        assert sess_dir == sess_dir_
        assert sess_single == [self.sess_ids[i] for i in self.l0_idxs]

        # multiple sessions from multiple labs
        hparams['sessions_csv'] = self.l_csv
        with pytest.raises(NotImplementedError):
            utils.get_session_dir(hparams, session_source='save')

        # ------------------------------------------------------------
        # use 'all' in hparams instead of csv file
        # ------------------------------------------------------------
        hparams['sessions_csv'] = ''

        # all labs
        hparams['lab'] = 'all'
        with pytest.raises(NotImplementedError):
            utils.get_session_dir(hparams, session_source='save')

        # all experiments
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'all'
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], 'multisession-%02i' % self.l0_id)
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        sess_single = [dict2str(d) for d in sess_single]
        sess_single_ = [dict2str(self.sess_ids[i]) for i in self.l0_idxs]
        assert sess_dir == sess_dir_
        assert sorted(sess_single) == sorted(sess_single_)

        # all animals
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'all'
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'],
            'multisession-%02i' % self.l0e0_id)
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        sess_single = [dict2str(d) for d in sess_single]
        sess_single_ = [dict2str(self.sess_ids[i]) for i in self.l0e0_idxs]
        assert sess_dir == sess_dir_
        assert sorted(sess_single) == sorted(sess_single_)

        # all sessions
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'animal0'
        hparams['session'] = 'all'
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            'multisession-%02i' % self.l0e0a0_id)
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        sess_single = [dict2str(d) for d in sess_single]
        sess_single_ = [dict2str(self.sess_ids[i]) for i in self.l0e0a0_idxs]
        assert sess_dir == sess_dir_
        assert sorted(sess_single) == sorted(sess_single_)

        # single session
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'animal0'
        hparams['session'] = 'session-%02i' % self.l0e0a0s0_id
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            hparams['session'])
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        sess_single = [dict2str(d) for d in sess_single]
        sess_single_ = [dict2str(self.sess_ids[i]) for i in self.l0e0a0s0_idxs]
        assert sess_dir == sess_dir_
        assert sorted(sess_single) == sorted(sess_single_)

        # ------------------------------------------------------------
        # use 'all' to define level, then define existing multisession
        # ------------------------------------------------------------
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'animal0'
        hparams['session'] = 'all'
        hparams['multisession'] = 1
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            'multisession-%02i' % self.l0e0a0m1_id)
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        sess_single = [dict2str(d) for d in sess_single]
        sess_single_ = [dict2str(self.sess_ids[i]) for i in self.l0e0a0m1_idxs]
        assert sess_dir == sess_dir_
        assert sorted(sess_single) == sorted(sess_single_)

        # TODO: return correct single session if multisession returns single

        # ------------------------------------------------------------
        #  use 'all' to define level, no existing multisession
        # ------------------------------------------------------------
        hparams['lab'] = 'lab1'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'animal0'
        hparams['session'] = 'all'
        hparams['multisession'] = None
        sess_dir_ = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            'multisession-%02i' % 0)
        sess_dir, sess_single = utils.get_session_dir(hparams, session_source='save')
        sess_single = [dict2str(d) for d in sess_single]
        sess_single_ = [dict2str(self.sess_ids[i]) for i in self.l1e0a0_idxs]
        assert sess_dir == sess_dir_
        assert sorted(sess_single) == sorted(sess_single_)

        # ------------------------------------------------------------
        # other
        # ------------------------------------------------------------
        # bad 'session_source'
        with pytest.raises(ValueError):
            utils.get_session_dir(hparams, session_source='test')

    def test_get_expt_dir(self):

        hparams = {
            'data_dir': 'ddir', 'save_dir': 'sdir', 'lab': 'lab0', 'expt': 'expt0',
            'animal': 'animal0', 'session': 'session-00'}
        session_dir = os.path.join(
            hparams['data_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            hparams['session'])
        hparams['session_dir'] = session_dir

        # -------------------------
        # ae
        # -------------------------
        hparams['model_class'] = 'ae'
        hparams['model_type'] = 'conv'
        hparams['n_ae_latents'] = 8
        hparams['experiment_name'] = 'tt_expt'
        model_path = os.path.join(
            session_dir, hparams['model_class'], hparams['model_type'],
            '%02i_latents' % hparams['n_ae_latents'], hparams['experiment_name'])

        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        expt_dir = utils.get_expt_dir(hparams, model_class=None, model_type=None, expt_name=None)
        assert expt_dir == model_path

        # multisession
        hparams['save_dir'] = self.tmpdir
        hparams['ae_multisession'] = 0
        model_path = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            'multisession-%02i' % hparams['ae_multisession'], hparams['model_class'],
            hparams['model_type'], '%02i_latents' % hparams['n_ae_latents'],
            hparams['experiment_name'])
        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path
        hparams['ae_multisession'] = None
        hparams['save_dir'] = 'sdir'

        # -------------------------
        # cond-ae [-msp]
        # -------------------------
        hparams['model_class'] = 'cond-ae'
        hparams['model_type'] = 'conv'
        hparams['n_ae_latents'] = 8
        hparams['experiment_name'] = 'tt_expt'
        model_path = os.path.join(
            session_dir, hparams['model_class'], hparams['model_type'],
            '%02i_latents' % hparams['n_ae_latents'], hparams['experiment_name'])

        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        hparams['model_class'] = 'cond-ae-msp'
        model_path = os.path.join(
            session_dir, hparams['model_class'], hparams['model_type'],
            '%02i_latents' % hparams['n_ae_latents'], hparams['experiment_name'])
        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        # -------------------------
        # neural-ae/ae-neural
        # -------------------------
        hparams['model_class'] = 'neural-ae'
        hparams['model_type'] = 'ff'
        hparams['n_ae_latents'] = 8
        hparams['experiment_name'] = 'tt_expt'
        model_path = os.path.join(
            session_dir, hparams['model_class'], '%02i_latents' % hparams['n_ae_latents'],
            hparams['model_type'], 'all', hparams['experiment_name'])

        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        hparams['model_class'] = 'ae-neural'
        model_path = os.path.join(
            session_dir, hparams['model_class'], '%02i_latents' % hparams['n_ae_latents'],
            hparams['model_type'], 'all', hparams['experiment_name'])
        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        # -------------------------
        # neural-arhmm/arhmm-neural
        # -------------------------
        hparams['model_class'] = 'neural-arhmm'
        hparams['model_type'] = 'ff'
        hparams['n_ae_latents'] = 8
        hparams['n_arhmm_states'] = 10
        hparams['kappa'] = 0
        hparams['experiment_name'] = 'tt_expt'
        model_path = os.path.join(
            session_dir, hparams['model_class'], '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'], '%.0e_kappa' % hparams['kappa'],
            hparams['model_type'], 'all', hparams['experiment_name'])

        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        hparams['model_class'] = 'arhmm-neural'
        model_path = os.path.join(
            session_dir, hparams['model_class'], '%02i_latents' % hparams['n_ae_latents'],
             '%02i_states' % hparams['n_arhmm_states'], '%.0e_kappa' % hparams['kappa'],
            hparams['model_type'], 'all', hparams['experiment_name'])
        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        # -------------------------
        # arhmm/hmm
        # -------------------------
        hparams['model_class'] = 'arhmm'
        hparams['n_ae_latents'] = 8
        hparams['n_arhmm_states'] = 10
        hparams['kappa'] = 0
        hparams['noise_type'] = 'gaussian'
        hparams['experiment_name'] = 'tt_expt'
        model_path = os.path.join(
            session_dir, hparams['model_class'], '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'], '%.0e_kappa' % hparams['kappa'],
            hparams['noise_type'], hparams['experiment_name'])

        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        # multisession
        hparams['save_dir'] = self.tmpdir
        hparams['arhmm_multisession'] = 0
        model_path = os.path.join(
            hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
            'multisession-%02i' % hparams['arhmm_multisession'], hparams['model_class'],
            '%02i_latents' % hparams['n_ae_latents'], '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'], hparams['noise_type'], hparams['experiment_name'])
        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path
        hparams['arhmm_multisession'] = None
        hparams['save_dir'] = 'sdir'

        # -------------------------
        # arhmm-labels
        # -------------------------
        hparams['model_class'] = 'arhmm-labels'
        hparams['n_arhmm_states'] = 10
        hparams['kappa'] = 0
        hparams['noise_type'] = 'studentst'
        hparams['experiment_name'] = 'tt_expt'
        model_path = os.path.join(
            session_dir, hparams['model_class'], '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'], hparams['noise_type'], hparams['experiment_name'])

        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        # -------------------------
        # bayesian decoding
        # -------------------------
        hparams['model_class'] = 'bayesian-decoding'
        hparams['n_ae_latents'] = 8
        hparams['n_arhmm_states'] = 10
        hparams['kappa'] = 0
        hparams['noise_type'] = 'studentst'
        hparams['experiment_name'] = 'tt_expt'
        model_path = os.path.join(
            session_dir, hparams['model_class'], '%02i_latents' % hparams['n_ae_latents'],
             '%02i_states' % hparams['n_arhmm_states'], '%.0e_kappa' % hparams['kappa'],
            hparams['noise_type'], 'all', hparams['experiment_name'])

        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        # -------------------------
        # labels-images
        # -------------------------
        hparams['model_class'] = 'labels-images'
        hparams['model_type'] = 'conv'
        hparams['experiment_name'] = 'tt_expt'
        model_path = os.path.join(
            session_dir, hparams['model_class'], hparams['model_type'], hparams['experiment_name'])

        expt_dir = utils.get_expt_dir(
            hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
            expt_name=hparams['experiment_name'])
        assert expt_dir == model_path

        # -------------------------
        # other
        # -------------------------
        hparams['model_class'] = 'testing'
        hparams['model_type'] = 'conv'
        hparams['experiment_name'] = 'tt_expt'
        with pytest.raises(ValueError):
            utils.get_expt_dir(
                hparams, model_class=hparams['model_class'], model_type=hparams['model_type'],
                expt_name=hparams['experiment_name'])

    def test_read_session_info_from_csv(self):

        sessions = utils.read_session_info_from_csv(self.l0e0a0_csv)
        assert sessions == [self.sess_ids[i] for i in self.l0e0a0_idxs]

    def test_export_session_info_to_csv(self):

        # create csv file
        sess_idxs = [0, 1, 2, 4, 5]
        csv_file = os.path.join(self.tmpdir, 'session_info.csv')
        utils.export_session_info_to_csv(self.tmpdir, [self.sess_ids[i] for i in sess_idxs])
        # load csv file
        sessions = utils.read_session_info_from_csv(csv_file)
        assert sessions == [self.sess_ids[i] for i in sess_idxs]

    def test_contains_session(self):

        # self.l0e0a0_idxs = [0, 1, 2]
        csv_path = os.path.join(self.l0e0a0_dir, 'multisession-%02i' % self.l0e0a0_id)

        # true
        assert utils.contains_session(csv_path, self.sess_ids[self.l0e0a0_idxs[0]])

        # false
        assert not utils.contains_session(csv_path, self.sess_ids[3])

    def test_find_session_dirs(self):

        hparams = {'save_dir': self.tmpdir}

        # no multisessions
        hparams['lab'] = 'lab1'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'animal0'
        hparams['session'] = 'session-01'
        sess_dirs = [
            os.path.join(hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
                         hparams['session'])]
        sess_ids = [
            {'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': hparams['animal'],
             'session': hparams['session'], 'multisession': None}]
        sess_dirs_, sess_ids_ = utils.find_session_dirs(hparams)
        assert sorted(sess_dirs) == sorted(sess_dirs_)
        assert sorted([dict2str(s) for s in sess_ids]) == sorted([dict2str(s) for s in sess_ids_])

        # single multisession
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt1'
        hparams['animal'] = 'animal1'
        hparams['session'] = 'session-00'
        sess_dirs = [
            os.path.join(hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
                         hparams['session']),
            os.path.join(hparams['save_dir'], hparams['lab'], 'multisession-00')]
        sess_ids = [
            {'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': hparams['animal'],
             'session': hparams['session'], 'multisession': None},
            {'lab': hparams['lab'], 'expt': 'all', 'animal': '', 'session': '',
             'multisession': 0}]
        sess_dirs_, sess_ids_ = utils.find_session_dirs(hparams)
        assert sorted(sess_dirs) == sorted(sess_dirs_)
        assert sorted([dict2str(s) for s in sess_ids]) == sorted([dict2str(s) for s in sess_ids_])

        # several multisessions
        hparams['lab'] = 'lab0'
        hparams['expt'] = 'expt0'
        hparams['animal'] = 'animal0'
        hparams['session'] = 'session-00'
        sess_dirs = [
            os.path.join(hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
                         hparams['session']),
            os.path.join(hparams['save_dir'], hparams['lab'], 'multisession-00'),
            os.path.join(hparams['save_dir'], hparams['lab'], hparams['expt'], 'multisession-00'),
            os.path.join(hparams['save_dir'], hparams['lab'], hparams['expt'], 'multisession-01'),
            os.path.join(hparams['save_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
                         'multisession-00')]
        sess_ids = [
            {'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': hparams['animal'],
             'session': hparams['session'], 'multisession': None},
            {'lab': hparams['lab'], 'expt': 'all', 'animal': '', 'session': '',
             'multisession': 0},
            {'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': 'all', 'session': '',
             'multisession': 0},
            {'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': 'all', 'session': '',
             'multisession': 1},
            {'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': hparams['animal'],
             'session': 'all', 'multisession': 0}]
        sess_dirs_, sess_ids_ = utils.find_session_dirs(hparams)
        assert sorted(sess_dirs) == sorted(sess_dirs_)
        assert sorted([dict2str(s) for s in sess_ids]) == sorted([dict2str(s) for s in sess_ids_])

    def test_get_model_params(self):

        misc_hparams = {
            'data_dir': '/tmp/path',
            'save_dir': '/tmp/path2',
            'export_train_plots': True}
        base_hparams = {
            'rng_seed_data': 4,
            'trial_splits': '4;1;1;0',
            'train_frac': 0.9,
            'rng_seed_model': 11}

        # -----------------
        # ae/vae/cond-ae
        # -----------------
        # ae/vae
        model_hparams = {
            'model_class': 'ae',
            'model_type': 'conv',
            'n_ae_latents': 5,
            'fit_sess_io_layers': False,
            'learning_rate': 1e-4,
            'l2_reg': 1e-2}
        ret_hparams = utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})
        assert ret_hparams == {**base_hparams, **model_hparams}

        # cond-ae
        model_hparams = {
            'model_class': 'cond-ae',
            'model_type': 'conv',
            'n_ae_latents': 5,
            'fit_sess_io_layers': False,
            'learning_rate': 1e-4,
            'l2_reg': 1e-2,
            'conditional_encoder': False}
        ret_hparams = utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})
        assert ret_hparams == {**base_hparams, **model_hparams}

        # cond-ae-msp
        model_hparams = {
            'model_class': 'cond-ae-msp',
            'model_type': 'conv',
            'n_ae_latents': 5,
            'fit_sess_io_layers': False,
            'learning_rate': 1e-4,
            'l2_reg': 1e-2,
            'msp_weight': 1e-5,
            'conditional_encoder': False}
        ret_hparams = utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})
        assert ret_hparams == {**base_hparams, **model_hparams}

        # -----------------
        # arhmm/hmm
        # -----------------
        model_hparams = {
            'model_class': 'arhmm',
            'model_type': '',
            'n_arhmm_lags': 2,
            'noise_type': 'gaussian',
            'kappa': 0,
            'ae_experiment_name': 'ae_expt',
            'ae_version': 4,
            'ae_model_type': 'conv',
            'n_ae_latents': 5}
        ret_hparams = utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})
        assert ret_hparams == {**base_hparams, **model_hparams}

        # -----------------
        # arhmm-/hmm-labels
        # -----------------
        model_hparams = {
            'model_class': 'arhmm-labels',
            'model_type': '',
            'n_arhmm_lags': 2,
            'noise_type': 'gaussian',
            'kappa': 0}
        ret_hparams = utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})
        assert ret_hparams == {**base_hparams, **model_hparams}

        # -----------------
        # neural-ae/ae-neural
        # -----------------
        model_hparams = {
            'model_class': 'neural-ae',
            'model_type': 'ff',
            'ae_experiment_name': 'ae_expt',
            'ae_version': 4,
            'ae_model_type': 'conv',
            'n_ae_latents': 5,
            'n_lags': 3,
            'l2_reg': 1,
            'n_hid_layers': 0,
            'activation': 'relu',
            'subsample_method': 'none'}
        ret_hparams = utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})
        assert ret_hparams == {**base_hparams, **model_hparams}

        # -----------------
        # neural-arhmm/arhmm-neural
        # -----------------
        model_hparams = {
            'model_class': 'neural-arhmm',
            'model_type': 'ff',
            'arhmm_experiment_name': 'arhmm_expt',
            'arhmm_version': 12,
            'n_arhmm_states': 4,
            'n_arhmm_lags': 1,
            'noise_type': 'gaussian',
            'kappa': 10,
            'ae_model_type': 'conv',
            'n_ae_latents': 5,
            'n_lags': 3,
            'l2_reg': 1,
            'n_hid_layers': 2,
            'n_hid_units': 10,
            'activation': 'relu',
            'subsample_method': 'single',
            'subsample_idxs_name': 'a',
            'subsample_idxs_group_0': 'b',
            'subsample_idxs_group_1': 'c'}
        ret_hparams = utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})
        assert ret_hparams == {**base_hparams, **model_hparams}

        # -----------------
        # bayesian decoding
        # -----------------
        model_hparams = {'model_class': 'bayesian-decoding', 'model_type': ''}
        with pytest.raises(NotImplementedError):
            utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})

        # -----------------
        # labels-images
        # -----------------
        model_hparams = {
            'model_class': 'labels-images',
            'model_type': '',
            'fit_sess_io_layers': True,
            'learning_rate': 1e-1,
            'l2_reg': 10}
        ret_hparams = utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})
        assert ret_hparams == {**base_hparams, **model_hparams}

        # -----------------
        # other
        # -----------------
        model_hparams = {'model_class': 'test', 'model_type': ''}
        with pytest.raises(NotImplementedError):
            utils.get_model_params({**misc_hparams, **base_hparams, **model_hparams})

    def test_get_region_dir(self):

        # no subsample method specified
        hparams = {}
        region_dir = utils.get_region_dir(hparams)
        assert region_dir == 'all'

        # subsample method = 'none'
        hparams = {'subsample_method': 'none'}
        region_dir = utils.get_region_dir(hparams)
        assert region_dir == 'all'

        # subsample method = 'single'
        hparams = {'subsample_method': 'single', 'subsample_idxs_name': 'mctx'}
        region_dir = utils.get_region_dir(hparams)
        assert region_dir == 'mctx-single'

        # subsample method = 'loo'
        hparams = {'subsample_method': 'loo', 'subsample_idxs_name': 'mctx'}
        region_dir = utils.get_region_dir(hparams)
        assert region_dir == 'mctx-loo'

        # invalid subsample method
        hparams = {'subsample_method': 'test'}
        with pytest.raises(ValueError):
            utils.get_region_dir(hparams)
