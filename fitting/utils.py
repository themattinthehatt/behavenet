import os
import pickle
import numpy as np
import torch
from torch.autograd import Variable
from behavenet.utils import export_latents, export_predictions


def get_subdirs(path):
    try:
        return next(os.walk(path))[1]
    except StopIteration:
        raise Exception('%s does not contain any subdirectories' % path)


def get_output_dirs(hparams, model_class=None, model_type=None, expt_name=None):

    if model_class is None:
        model_class = hparams['model_class']

    if model_type is None:
        model_type = hparams['model_type']

    if expt_name is None:
        expt_name = hparams['experiment_name']

    sess_dir = os.path.join(
            hparams['tt_save_path'], hparams['lab'], hparams['expt'],
            hparams['animal'], hparams['session'])

    if model_class == 'ae':
        results_dir = os.path.join(
            sess_dir, 'ae', model_type,
            '%02i_latents' % hparams['n_ae_latents'])
    elif model_class == 'neural-ae':
        # TODO: include brain region, ae version
        results_dir = os.path.join(
            sess_dir, 'neural-ae',
            '%02i_latents' % hparams['n_ae_latents'],
            model_type)
    elif model_class == 'neural-arhmm':
        results_dir = os.path.join(
            sess_dir, 'neural-arhmm',
            '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'],
            model_type)
    elif model_class == 'arhmm':
        results_dir = os.path.join(
            sess_dir, 'arhmm',
            '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'],
            hparams['noise_type'])
    elif model_class == 'arhmm-decoding':
        results_dir = os.path.join(
            sess_dir, 'arhmm',
            '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'],
            hparams['noise_type'])
    else:
        raise ValueError('"%s" is an invalid model class' % model_class)

    expt_dir = os.path.join(results_dir, 'test_tube_data', expt_name)

    return sess_dir, results_dir, expt_dir


def estimate_model_footprint(model, input_size, cutoff_size=20):
    """
    Adapted from http://jacobkimmel.github.io/pytorch_estimating_model_size/

    Args:
        model (pt model):
        input_size (tuple):
        cutoff_size (float): GB

    Returns:
        int: bytes
    """

    allowed_modules = (
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.MaxPool2d,
        torch.nn.MaxUnpool2d,
        torch.nn.Linear
    )

    curr_bytes = 0

    # assume everything is float32
    bytes = 4

    # estimate input size
    curr_bytes += np.prod(input_size) * bytes

    # estimate model size
    mods = list(model.modules())
    for mod in mods:
        if isinstance(mod, allowed_modules):
            p = list(mod.parameters())
            for p_ in p:
                curr_bytes += np.prod(np.array(p_.size())) * bytes

    # estimate intermediate size
    x = Variable(torch.FloatTensor(*input_size))
    for layer in model.encoding.encoder:
        if isinstance(layer, torch.nn.MaxPool2d):
            x, idx = layer(x)
        else:
            x = layer(x)
        # multiply by 2 - assume decoder is symmetric
        # multiply by 2 - we need to store values AND gradients
        curr_bytes += np.prod(x.size()) * bytes * 2 * 2
        if curr_bytes / 1e9 > cutoff_size:
            break

    return curr_bytes * 1.2  # safety blanket


def get_best_model_version(model_path, measure='val_loss', n_best=1, best_def='min'):
    """

    Args:
        model_path (str): test tube experiment directory containing version_%i
            subdirectories
        measure (str):

    Returns:
        str

    """

    import pandas as pd

    # gather all versions
    versions = get_subdirs(model_path)

    # load csv files with model metrics (saved out from test tube)
    metrics = []
    for i, version in enumerate(versions):
        # read metrics csv file
        try:
            metric = pd.read_csv(
                os.path.join(model_path, version, 'metrics.csv'))
        except:
            continue
        # get validation loss of best model
        if best_def == 'min':
            val_loss = metric[measure].min()
        elif best_def == 'max':
            val_loss = metric[measure].max()
        metrics.append(pd.DataFrame({
            'loss': val_loss,
            'version': version}, index=[i]))
    # put everything in pandas dataframe
    metrics_df = pd.concat(metrics, sort=False)
    # get version with smallest loss
    
    if n_best == 1:
        if best_def == 'min':
            best_versions = [metrics_df['version'][metrics_df['loss'].idxmin()]]
        elif best_def == 'max':
            best_versions = [metrics_df['version'][metrics_df['loss'].idxmax()]]
    else:
        if best_dir == 'min':
            best_versions = np.asarray(metrics_df['version'][metrics_df['loss'].nsmallest(n_best,'all').index])
        elif best_def == 'max':
            raise NotImplementedError
        if best_versions.shape[0] != n_best:
            print('More versions than specified due to same validation loss')
        
    return best_versions


def get_best_model_and_data(hparams, Model, load_data=True, version='best'):

    from data.data_generator import ConcatSessionsGenerator

    # get best model version
    sess_dir, results_dir, expt_dir = get_output_dirs(hparams)
    if version == 'best':
        best_version = get_best_model_version(expt_dir)[0]
    else:
        if isinstance(version, str) and version[0] == 'v':
            # assume we got a string of the form 'version_XX'
            best_version = version
        else:
            best_version = str('version_{}'.format(version))
    version_dir = os.path.join(expt_dir, best_version)
    arch_file = os.path.join(version_dir, 'meta_tags.pkl')
    model_file = os.path.join(version_dir, 'best_val_model.pt')
    if not os.path.exists(model_file) and not os.path.exists(model_file + '.meta'):
        model_file = os.path.join(version_dir, 'best_val_model.ckpt')
    print('Loading model defined in %s' % arch_file)

    with open(arch_file, 'rb') as f:
        hparams_new = pickle.load(f)

    # update paths if performing analysis on a different machine
    hparams_new['data_dir'] = hparams['data_dir']
    hparams_new['session_dir'] = sess_dir
    hparams_new['results_dir'] = results_dir
    hparams_new['expt_dir'] = expt_dir
    hparams_new['use_output_mask'] = hparams['use_output_mask'] # TODO: get rid of eventually
    hparams_new['device']='cpu'
    
    # build data generator
    hparams_new, signals, transforms, load_kwargs = get_data_generator_inputs(
        hparams_new)
    ids = {
        'lab': hparams_new['lab'],
        'expt': hparams_new['expt'],
        'animal': hparams_new['animal'],
        'session': hparams_new['session']}
    if load_data:
        # sometimes we want a single data_generator for multiple models
        data_generator = ConcatSessionsGenerator(
            hparams_new['data_dir'], ids,
            signals=signals, transforms=transforms, load_kwargs=load_kwargs,
            device=hparams_new['device'], as_numpy=hparams_new['as_numpy'],
            batch_load=hparams_new['batch_load'], rng_seed=hparams_new['rng_seed'])
    else:
        data_generator = None

    # build models
    if 'lib' not in hparams_new:
        hparams_new['lib'] = 'pt'

    model = Model(hparams_new)
    model.version = best_version
    if hparams_new['lib'] == 'pt' or hparams_new['lib'] == 'pytorch':
        model.load_state_dict(torch.load(model_file,map_location=lambda storage, loc: storage))
        model.to(hparams_new['device'])
        model.eval()
    elif hparams_new['lib'] == 'tf':
        import tensorflow as tf
        # load trained weights into model
        if not hasattr(model, 'encoder_input'):
            next_batch = tf.placeholder(
                dtype=tf.float32,
                shape=(
                    None,
                    hparams_new['y_pixels'],
                    hparams_new['x_pixels'],
                    hparams_new['n_input_channels']))
            model.encoder_input = next_batch
            model.forward(next_batch)

        sess_config = tf.ConfigProto(device_count={'GPU': 0})
        saver = tf.train.Saver()
        sess = tf.Session(config=sess_config)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)
        model.sess = sess
    else:
        raise ValueError('"%s" is not a valid lib' % hparams_new['lib'])

    return model, data_generator


def experiment_exists(hparams):

    import pickle
    import copy

    try:
        tt_versions = get_subdirs(hparams['expt_dir'])
    except StopIteration:
        # no versions yet
        return False

    # get rid of extra dict
    # TODO: this is ugly and not easy to maintain
    hparams_less = copy.copy(hparams)
    hparams_less.pop('architecture_params', None)
    hparams_less.pop('list_index', None)
    hparams_less.pop('lab_example', None)
    hparams_less.pop('tt_nb_gpu_trials', None)
    hparams_less.pop('tt_nb_cpu_trials', None)
    hparams_less.pop('tt_nb_cpu_workers', None)
    hparams_less.pop('lib', None)
    hparams_less.pop('use_output_mask', None)
    hparams_less.pop('ae_model_type', None)

    found_match = False
    for version in tt_versions:
        # load hparams
        version_file = os.path.join(hparams['expt_dir'], version, 'meta_tags.pkl')
        try:
            with open(version_file, 'rb') as f:
                hparams_ = pickle.load(f)
            if all([hparams_[key] == hparams_less[key] for key in hparams_less.keys()]):
                # found match - did it finish training?
                if hparams_['training_completed']:
                    found_match = True
                    print('model found with complete training; aborting')
                    break
            # else:
            #     print()
            #     print()
            #     for key in hparams_less.keys():
            #         val1 = hparams_[key]
            #         val2 = hparams_less[key]
            #         if val1 != val2:
            #             print('Key: {}; val1: {}; val2 {}'.format(key, val1, val2))
        except IOError:
            continue

    return found_match


def export_hparams(hparams, exp):
    """
    Export hyperparameter dictionary as csv file (for easy human reading) and
    as a pickled dict (for easy loading)

    Args:
        hparams (dict):
        exp (test_tube.Experiment object):
    """

    import pickle

    # save out as pickle
    meta_file = os.path.join(
        hparams['expt_dir'], 'version_%i' % exp.version, 'meta_tags.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(hparams, f)

    # save out as csv
    exp.tag(hparams)
    exp.save()


def get_data_generator_inputs(hparams):
    """
    Helper function for generating signals, transforms and load_kwargs for
    common models
    """

    from data.transforms import Threshold, ZScore, BlockShuffle

    # get neural signals/transforms/load_kwargs
    if hparams['model_class'].find('neural') > -1:
        neural_transforms = None  # neural_region
        neural_kwargs = None
        if hparams['neural_type'] == 'spikes':
            if hparams['neural_thresh'] > 0:
                neural_transforms = Threshold(
                    threshold=hparams['neural_thresh'],
                    bin_size=hparams['neural_bin_size'])
        elif hparams['neural_type'] == 'ca':
            neural_transforms = ZScore()
        else:
            raise ValueError(
                '"%s" is an invalid neural type' % hparams['neural_type'])
    else:
        neural_transforms = None
        neural_kwargs = None

    # get model-specific signals/transforms/load_kwargs
    if hparams['model_class'] == 'ae':

        if hparams['use_output_mask']:
            signals = [hparams['signals'], 'masks']
            transforms = [hparams['transforms'], None]
            load_kwargs = [None, None]
        else:
            signals = [hparams['signals']]
            transforms = [hparams['transforms']]
            load_kwargs = [None]

    elif hparams['model_class'] == 'neural-ae':

        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'ae'
        hparams['output_size'] = hparams['n_ae_latents']
        if hparams['model_type'][-2:] == 'mv':
            hparams['noise_dist'] = 'gaussian-full'
        else:
            hparams['noise_dist'] = 'gaussian'

        _, _, ae_dir = get_output_dirs(
            hparams, model_class='ae',
            expt_name=hparams['ae_experiment_name'],
            model_type=hparams['ae_model_type'])

        ae_transforms = None
        ae_kwargs = {
            'model_dir': ae_dir,
            'model_version': hparams['ae_version']}

        signals = ['neural', 'ae']
        transforms = [neural_transforms, ae_transforms]
        load_kwargs = [neural_kwargs, ae_kwargs]

    elif hparams['model_class'] == 'neural-arhmm':

        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'arhmm'
        hparams['output_size'] = hparams['n_arhmm_states']
        hparams['noise_dist'] = 'categorical'

        _, _, arhmm_dir = get_output_dirs(
            hparams, model_class='arhmm',
            expt_name=hparams['arhmm_experiment_name'])

        if 'shuffle_rng_seed' in hparams:
            arhmm_transforms = BlockShuffle(hparams['shuffle_rng_seed'])
        else:
            arhmm_transforms = None
        arhmm_kwargs = {
            'model_dir': arhmm_dir,
            'model_version': hparams['arhmm_version']}

        signals = ['neural', 'arhmm']
        transforms = [neural_transforms, arhmm_transforms]
        load_kwargs = [neural_kwargs, arhmm_kwargs]

    elif hparams['model_class'] == 'arhmm':

        _, _, ae_dir = get_output_dirs(
            hparams, model_class='ae',
            expt_name=hparams['ae_experiment_name'],
            model_type=hparams['ae_model_type'])

        ae_transforms = None
        ae_kwargs = {
            'model_dir': ae_dir,
            'model_version': hparams['ae_version']}

        if hparams['use_output_mask']:
            signals = ['ae', 'images', 'masks']
            transforms = [ae_transforms, None, None]
            load_kwargs = [ae_kwargs, None, None]
        else:
            signals = ['ae', 'images']
            transforms = [ae_transforms, None]
            load_kwargs = [ae_kwargs, None]
    elif hparams['model_class'] == 'arhmm-decoding':

        _, _, ae_dir = get_output_dirs(
            hparams, model_class='ae',
            expt_name=hparams['ae_experiment_name'],
            model_type=hparams['ae_model_type'])

        ae_transforms = None
        ae_kwargs = {
            'model_dir': ae_dir,
            'model_version': hparams['ae_version']} 

        if hparams['use_output_mask']:
            signals = ['ae', 'images', 'masks','ae_predictions','arhmm_predictions','arhmm']
            transforms = [ae_transforms, None, None, None, None, None]
            load_kwargs = [ae_kwargs, None, None, None, None, None]
        else:
            signals = ['ae', 'images', 'ae_predictions','arhmm_predictions','arhmm']
            transforms = [ae_transforms, None, None, None, None]
            load_kwargs = [ae_kwargs, None, None, None, None]

    else:
        raise ValueError('"%s" is an invalid model_class' % hparams['model_class'])

    return hparams, signals, transforms, load_kwargs


def add_lab_defaults_to_parser(parser, lab=None):

    if lab == 'musall':
        parser.add_argument('--n_input_channels', '-i', default=2, help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', '-x', default=128, help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', '-y', default=128, help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=False, action='store_true')
        parser.add_argument('--approx_batch_size', '-b', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', '-l', default='musall', type=str)
        parser.add_argument('--expt', '-e', default='vistrained', type=str)
        parser.add_argument('--animal', '-a', default='mSM30', type=str)
        parser.add_argument('--session', '-s', default='10-Oct-2017', type=str)
        parser.add_argument('--neural_bin_size', default=None, help='ms')
        parser.add_argument('--neural_type', default='ca', choices=['spikes', 'ca'])
    elif lab == 'steinmetz':
        parser.add_argument('--n_input_channels', '-i', default=1, help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', '-x', default=192, help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', '-y', default=112, help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=False, action='store_true')
        parser.add_argument('--approx_batch_size', '-b', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', '-l', default='steinmetz', type=str)
        parser.add_argument('--expt', '-e', default='2-probe', type=str)
        parser.add_argument('--animal', '-a', default='mouse-01', type=str)
        parser.add_argument('--session', '-s', default='session-01', type=str)
        parser.add_argument('--neural_bin_size', default=39.61, help='ms')
        parser.add_argument('--neural_type', default='spikes', choices=['spikes', 'ca'])
    elif lab == 'steinmetz-face':
        parser.add_argument('--n_input_channels', '-i', default=1, help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', '-x', default=128, help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', '-y', default=128, help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=False, action='store_true')
        parser.add_argument('--approx_batch_size', '-b', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', '-l', default='steinmetz', type=str)
        parser.add_argument('--expt', '-e', default='2-probe-face', type=str)
        parser.add_argument('--animal', '-a', default='mouse-01', type=str)
        parser.add_argument('--session', '-s', default='session-01', type=str)
        parser.add_argument('--neural_bin_size', default=39.61, help='ms')
        parser.add_argument('--neural_type', default='spikes', choices=['spikes', 'ca'])
    elif lab == 'datta':
        parser.add_argument('--n_input_channels', '-i', default=1, help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', '-x', default=80, help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', '-y', default=80, help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=True, action='store_true')
        parser.add_argument('--approx_batch_size', '-b', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', '-l', default='datta', type=str)
        parser.add_argument('--expt', '-e', default='inscopix', type=str)
        parser.add_argument('--animal', '-a', default='15566', type=str)
        parser.add_argument('--session', '-s', default='2018-11-27', type=str)
        parser.add_argument('--neural_bin_size', default=None, help='ms')
        parser.add_argument('--neural_type', default='ca', choices=['spikes', 'ca'])
    else:
        parser.add_argument('--n_input_channels', '-i', help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', '-x', help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', '-y', help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=False, action='store_true')
        parser.add_argument('--approx_batch_size', '-b', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', '-l', type=str)
        parser.add_argument('--expt', '-e', type=str)
        parser.add_argument('--animal', '-a', type=str)
        parser.add_argument('--session', '-s', type=str)
        parser.add_argument('--neural_bin_size', default=None, help='ms')
        parser.add_argument('--neural_type', default='spikes', choices=['spikes', 'ca'])


def get_lab_example(hparams, lab):
    if lab == 'steinmetz':
        hparams['lab'] = 'steinmetz'
        hparams['expt'] = '2-probe'
        hparams['animal'] = 'mouse-01'
        hparams['session'] = 'session-01'
        hparams['n_ae_latents'] = 12
        hparams['use_output_mask'] = False
        hparams['frame_rate']=25
    if lab == 'steinmetz-face':
        hparams['lab'] = 'steinmetz'
        hparams['expt'] = '2-probe-face'
        hparams['animal'] = 'mouse-01'
        hparams['session'] = 'session-01'
        hparams['n_ae_latents'] = 12
        hparams['use_output_mask'] = False
        hparams['frame_rate']=25
    elif lab == 'musall':
        hparams['lab'] = 'musall'
        hparams['expt'] = 'vistrained'
        hparams['animal'] = 'mSM30'
        hparams['session'] = '10-Oct-2017'
        hparams['n_ae_latents'] = 16
        hparams['use_output_mask'] = False
        hparams['frame_rate']=30 # is this correct?
    elif lab == 'datta':
        hparams['lab'] = 'datta'
        hparams['expt'] = 'inscopix'
        hparams['animal'] = '15566'
        hparams['session'] = '2018-11-27'
        hparams['n_ae_latents'] = 8
        hparams['use_output_mask'] = True
        hparams['frame_rate']=30


def get_reconstruction(model, inputs):
    """
    Reconstruct an image from either image or latent inputs

    Args:
        model: pt or tf Model
        inputs (torch.Tensor object):
            images (batch x channels x y_pix x x_pix)
            latents (batch x n_ae_latents)

    Returns:
        np array (batch x channels x y_pix x x_pix)
    """

    # check to see if inputs are images or latents
    if len(inputs.shape) == 2:
        input_type = 'latents'
    else:
        input_type = 'images'

    if isinstance(model, torch.nn.Module):
        if input_type == 'images':
            ims_recon, _ = model(inputs)
        else:
            # TODO: how to incorporate maxpool layers for decoding only?
            ims_recon = model.decoding(inputs, None, None)
        ims_recon = ims_recon.cpu().detach().numpy()
    else:
        if input_type == 'images':
            ims_ = np.transpose(inputs.cpu().detach().numpy(), (0, 2, 3, 1))
            feed_dict = {model.encoder_input: ims_}
        else:
            feed_dict = {model.decoder_input: inputs}

        ims_recon = model.sess.run(model.y, feed_dict=feed_dict)
        ims_recon = np.transpose(ims_recon, (0, 3, 1, 2))

    return ims_recon


def export_latents_best(hparams):
    """
    Export predictions for the best decoding model in a test tube experiment.
    Predictions are saved in the corresponding model directory.

    Args:
        hparams (dict):
    """

    if hparams['lib'] == 'pt' or hparams['lib'] == 'pytorch':
        from behavenet.models import AE
        model, data_generator = get_best_model_and_data(hparams, AE)
        export_latents(data_generator, model)
    elif hparams['lib'] == 'tf':
        from behavenet.models_tf import AE
        model, data_generator = get_best_model_and_data(hparams, AE)
        export_latents_tf(data_generator, model)
    else:
        raise ValueError('"%s" is an invalid model library')


def export_predictions_best(hparams):
    """
    Export predictions for the best decoding model in a test tube experiment.
    Predictions are saved in the corresponding model directory.

    Args:
        hparams (dict):
    """

    from behavenet.models import Decoder

    if hparams['lib'] == 'tf':
        raise NotImplementedError

    model, data_generator = get_best_model_and_data(hparams, Decoder)
    export_predictions(data_generator, model)


def export_latents_tf(data_generator, model, filename=None):
    """Port of behavenet.fitting.utils.export_latents for tf models"""

    # initialize container for latents
    latents = [[] for _ in range(data_generator.num_datasets)]
    for i, dataset in enumerate(data_generator.datasets):
        trial_len = dataset.trial_len
        num_trials = dataset.num_trials
        latents[i] = np.full(
            shape=(num_trials, trial_len, model.hparams['n_ae_latents']),
            fill_value=np.nan)

        # partially fill container (gap trials will be included as nans)
        dtypes = ['train', 'val', 'test']
        for dtype in dtypes:
            data_generator.reset_iterators(dtype)
            for i in range(data_generator.num_tot_batches[dtype]):
                data, dataset = data_generator.next_batch(dtype)

                # process batch, perhaps in chunks if full batch is too large
                # to fit on gpu
                chunk_size = 200
                y = np.transpose(
                    data[model.hparams['signals']][0].cpu().detach().numpy(),
                    (0, 2, 3, 1))
                batch_size = y.shape[0]
                if batch_size > chunk_size:
                    # split into chunks
                    num_chunks = int(np.ceil(batch_size / chunk_size))
                    for chunk in range(num_chunks):
                        # take chunks of size chunk_size, plus overlap due to
                        # max_lags
                        indx_beg = chunk * chunk_size
                        indx_end = np.min([(chunk + 1) * chunk_size, batch_size])

                        curr_latents = model.sess.run(
                            model.x,
                            feed_dict={model.encoder_input: y[indx_beg:indx_end]})

                        latents[dataset][data['batch_indx'].item(),
                        indx_beg:indx_end, :] = curr_latents
                else:
                    curr_latents = model.sess.run(
                        model.x, feed_dict={model.encoder_input: y})
                    latents[dataset][data['batch_indx'].item(), :, :] = \
                        curr_latents

    # save latents separately for each dataset
    for i, dataset in enumerate(data_generator.datasets):
        # get save name which includes lab/expt/animal/session
        # sess_id = str(
        #     '%s_%s_%s_%s_latents.pkl' % (
        #         dataset.lab, dataset.expt, dataset.animal,
        #         dataset.session))
        if filename is None:
            sess_id = 'latents.pkl'
            filename = os.path.join(
                model.hparams['results_dir'], 'test_tube_data',
                model.hparams['experiment_name'], model.version,
                sess_id)
        # save out array in pickle file
        pickle.dump({
            'latents': latents[i],
            'trials': data_generator.batch_indxs[i]},
            open(filename, 'wb'))
