import os
import numpy as np
import torch
from torch.autograd import Variable
from behavenet.utils import export_latents, export_predictions


def get_subdirs(path):
    return next(os.walk(path))[1]


def get_output_dirs(hparams, model_class=None, expt_name=None):

    if model_class is None:
        model_class = hparams['model_class']

    if expt_name is None:
        expt_name = hparams['experiment_name']

    sess_dir = os.path.join(
            hparams['tt_save_path'], hparams['lab'], hparams['expt'],
            hparams['animal'], hparams['session'])

    if model_class == 'ae':
        results_dir = os.path.join(
            sess_dir, 'ae', hparams['model_type'],
            '%02i_latents' % hparams['n_ae_latents'])
    elif model_class == 'neural-ae':
        # TODO: include brain region, ae version
        results_dir = os.path.join(
            sess_dir, 'neural-ae',
            '%02i_latents' % hparams['n_ae_latents'],
            hparams['model_type'])
    elif model_class == 'neural-arhmm':
        results_dir = os.path.join(
            sess_dir, 'neural-arhmm',
            '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            hparams['model_type'])
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


def get_best_model_version(model_path, measure='loss',n_best=1):
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
        # get validation loss of best model # TODO: user-supplied measure
        val_loss = metric.val_loss.min()
        metrics.append(pd.DataFrame({
            'loss': val_loss,
            'version': version}, index=[i]))
    # put everything in pandas dataframe
    metrics_df = pd.concat(metrics, sort=False)
    # get version with smallest loss
    
    if n_best==1:
        best_versions = [metrics_df['version'][metrics_df['loss'].idxmin()]]
    else:
        best_versions = np.asarray(metrics_df['version'][metrics_df['loss'].nsmallest(n_best,'all').index])
        if best_versions.shape[0]!=n_best:
            print('More versions than specified due to same validation loss')
        
    return best_versions


def experiment_exists(hparams):

    import pickle

    try:
        tt_versions = get_subdirs(hparams['expt_dir'])
    except StopIteration:
        # no versions yet
        return False

    print(tt_versions)

    # get rid of extra dict
    input_arch_params = hparams['architecture_params']

    found_match = False
    for version in tt_versions:
        # load hparams
        version_file = os.path.join(hparams['expt_dir'], version, 'meta_tags.pkl')
        try:
            with open(version_file, 'rb') as f:
                hparams_ = pickle.load(f)

            curr_arch_params = hparams_['architecture_params']
            if all([
                    curr_arch_params[key] == input_arch_params[key] for key in \
                    input_arch_params.keys()]):
                # found match - did it finish training?
                if hparams_['training_completed']:
                    found_match = True
                    print('model found with complete training; aborting')
                    break
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

    from data.transforms import Threshold

    # get neural signals/transforms/load_kwargs
    if hparams['model_class'].find('neural') > -1:
        if hparams['neural_thresh'] > 0 and hparams['neural_type'] == 'spikes':
            neural_transforms = Threshold(
                threshold=hparams['neural_thresh'],
                bin_size=hparams['neural_bin_size'])
        else:
            neural_transforms = None  # neural_region
        neural_kwargs = None
    else:
        neural_transforms = None
        neural_kwargs = None

    # get model-specific signals/transforms/load_kwargs
    if hparams['model_class'] == 'ae':

        signals = [hparams['signals']]
        transforms = [hparams['transforms']]
        load_kwargs = [None]

    elif hparams['model_class'] == 'neural-ae':

        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'ae'
        hparams['output_size'] = hparams['n_ae_latents']
        hparams['noise_dist'] = 'gaussian'

        _, _, ae_dir = get_output_dirs(
            hparams, model_class='ae',
            expt_name=hparams['ae_experiment_name'])

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

        arhmm_transforms = None
        arhmm_kwargs = {
            'model_dir': arhmm_dir,
            'model_version': hparams['arhmm_version']}

        signals = ['neural', 'arhmm']
        transforms = [neural_transforms, arhmm_transforms]
        load_kwargs = [neural_kwargs, arhmm_kwargs]

    else:
        raise ValueError('"%s" is an invalid model_class' % hparams['model_class'])

    return hparams, signals, transforms, load_kwargs


def export_latents_best(hparams):
    """
    Export predictions for the best decoding model in a test tube experiment.
    Predictions are saved in the corresponding model directory.

    Args:
        hparams (dict):
    """

    import os
    import pickle
    from data.data_generator import ConcatSessionsGenerator
    from behavenet.models import AE

    # ###########################
    # ### Get Best Experiment ###
    # ###########################

    # get session_dir, results_dir (session_dir + ae details), expt_dir (
    # results_dir + experiment details)
    # expt_dir contains version_%i directories
    hparams['session_dir'], hparams['results_dir'], hparams['expt_dir'] = \
        get_output_dirs(hparams)

    best_version = get_best_model_version(hparams['expt_dir'])[0]
    best_model_file = os.path.join(
        hparams['expt_dir'], best_version, 'best_val_model.pt')

    # copy over hparams from best model
    hparams_file = os.path.join(
        hparams['expt_dir'], best_version, 'meta_tags.pkl')
    with open(hparams_file, 'rb') as f:
        hparams = pickle.load(f)

    # ###########################
    # ### LOAD DATA GENERATOR ###
    # ###########################

    hparams, signals, transforms, load_kwargs = get_data_generator_inputs(hparams)
    ids = {
        'lab': hparams['lab'],
        'expt': hparams['expt'],
        'animal': hparams['animal'],
        'session': hparams['session']}
    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], ids,
        signals=signals, transforms=transforms, load_kwargs=load_kwargs,
        device=hparams['device'], as_numpy=hparams['as_numpy'],
        batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed'])

    # ####################
    # ### CREATE MODEL ###
    # ####################

    model = AE(hparams)

    # load best model params
    model.version = int(best_version[8:])  # omg this is awful
    model.load_state_dict(torch.load(best_model_file))
    model.to(hparams['device'])
    model.eval()

    # push data through model
    export_latents(data_generator, model)


def export_predictions_best(hparams):
    """
    Export predictions for the best decoding model in a test tube experiment.
    Predictions are saved in the corresponding model directory.

    Args:
        hparams (dict):
    """

    import os
    import pickle
    from data.data_generator import ConcatSessionsGenerator
    from behavenet.models import Decoder

    # ###########################
    # ### Get Best Experiment ###
    # ###########################

    # get session_dir, results_dir (session_dir + decoding details),
    # expt_dir (results_dir + experiment details)
    # expt_dir contains version_%i directories
    hparams['session_dir'], hparams['results_dir'], hparams['expt_dir'] = \
        get_output_dirs(hparams)
    best_version = get_best_model_version(hparams['expt_dir'])[0]
    best_model_file = os.path.join(
        hparams['expt_dir'], best_version, 'best_val_model.pt')

    # copy over hparams from best model
    hparams_file = os.path.join(
        hparams['expt_dir'], best_version, 'meta_tags.pkl')
    with open(hparams_file, 'rb') as f:
        hparams = pickle.load(f)

    # ###########################
    # ### LOAD DATA GENERATOR ###
    # ###########################

    hparams, signals, transforms, load_kwargs = get_data_generator_inputs(hparams)
    ids = {
        'lab': hparams['lab'],
        'expt': hparams['expt'],
        'animal': hparams['animal'],
        'session': hparams['session']}
    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], ids,
        signals=signals, transforms=transforms, load_kwargs=load_kwargs,
        device=hparams['device'], as_numpy=hparams['as_numpy'],
        batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed'])
    hparams['input_size'] = data_generator.datasets[0].dims[hparams['input_signal']][2]

    # ####################
    # ### CREATE MODEL ###
    # ####################

    model = Decoder(hparams)

    # load best model params
    model.version = int(best_version[8:])  # omg this is awful
    model.load_state_dict(torch.load(best_model_file))
    model.to(hparams['device'])
    model.eval()

    # push data through model
    export_predictions(data_generator, model)
