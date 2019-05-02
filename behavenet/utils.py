import numpy as np
import torch
from torch.autograd import Variable


def estimate_model_footprint(model, input_size):
    """
    Adapted from http://jacobkimmel.github.io/pytorch_estimating_model_size/

    Args:
        model (pt model):
        input_size (tuple):

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

    # assume everything is float32
    bytes = 4

    # estimate input size
    input_bytes = np.prod(input_size) * bytes

    # estimate model size
    mods = list(model.modules())
    for mod in mods:
        if isinstance(mod, allowed_modules):
            p = list(mod.parameters())
            sizes = []
            for p_ in p:
                sizes.append(np.array(p_.size()))

    model_bytes = 0
    for size in sizes:
        model_bytes += np.prod(np.array(size)) * bytes

    # estimate intermediate size
    # input_ = Variable(torch.FloatTensor(*input_size))
    # out_sizes = []
    # for mod in mods:
    #     if isinstance(mod, allowed_modules):
    #         out = mod(input_)
    #         if isinstance(out, tuple):
    #             out_sizes.append(np.array(out[0].size()))
    #         else:
    #             out_sizes.append(np.array(out.size()))
    #         input_ = out
    #     else:
    #         print(mod)
    x = Variable(torch.FloatTensor(*input_size))
    out_sizes = []
    for layer in model.encoding.encoder:
        if isinstance(layer, torch.nn.MaxPool2d):
            x, idx = layer(x)
        else:
            x = layer(x)
        out_sizes.append(x.size())

    int_bytes = 0
    for out_size in out_sizes:
        # multiply by 2 - assume decoder is symmetric
        int_bytes += np.prod(np.array(out_size)) * bytes * 2

    # multiply by 2 - we need to store values AND gradients
    int_bytes *= 2

    return (input_bytes + model_bytes + int_bytes) * 1.2  # safety blanket


def get_best_model_version(model_path, measure='loss'):
    """

    Args:
        model_path (str): test tube experiment directory containing version_%i
            subdirectories
        measure (str):

    Returns:
        str

    """

    import pandas as pd
    import os

    # gather all versions
    def get_dirs(path):
        return next(os.walk(model_path))[1]

    versions = get_dirs(model_path)

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
    best_version = metrics_df['version'][metrics_df['loss'].idxmin()]

    return best_version


def export_latents(data_generator, model, filename=None):
    """
    Export predicted latents using an already initialized data_generator and
    model; latents are saved based on the model's hparams dict unless another
    file is provided.

    Args:
        data_generator (ConcatSessionGenerator):
        model (AE):
        filename (str): absolute path
    """

    import pickle
    import os

    model.eval()

    # initialize container for latents
    latents = [[] for _ in range(data_generator.num_datasets)]
    for i, dataset in enumerate(data_generator.datasets):
        trial_len = dataset.trial_len
        num_trials = dataset.num_trials
        latents[i] = np.full(
            shape=(num_trials, trial_len, model.hparams['n_latents']),
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
            y = data[model.hparams['signals']][0]
            batch_size = y.shape[0]
            if batch_size > chunk_size:
                # split into chunks
                num_chunks = int(np.ceil(batch_size / chunk_size))
                for chunk in range(num_chunks):
                    # take chunks of size chunk_size, plus overlap due to
                    # max_lags
                    indx_beg = chunk * chunk_size
                    indx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                    curr_latents, _, _ = model.encoding(
                        y[indx_beg:indx_end])
                    latents[dataset][data['batch_indx'].item(),
                    indx_beg:indx_end, :] = \
                        curr_latents.cpu().detach().numpy()
            else:
                curr_latents, _, _ = model.encoding(y)
                latents[dataset][data['batch_indx'].item(), :, :] = \
                    curr_latents.cpu().detach().numpy()

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
                model.hparams['experiment_name'], 'version_%i' % model.version,
                sess_id)
        # save out array in pickle file
        pickle.dump({
            'latents': latents[i],
            'trials': data_generator.batch_indxs[i]},
            open(filename, 'wb'))


def export_predictions(data_generator, model, filename=None):
    """
    Export predictions using an already initialized data_generator and model;
    predictions are saved based on the model's hparams dict unless another file
    is provided.

    Args:
        data_generator (ConcatSessionGenerator):
        model (NN):
        filename (str): absolute path
    """

    import pickle
    import os

    model.eval()

    # initialize container for latents
    predictions = [[] for _ in range(data_generator.num_datasets)]
    for i, dataset in enumerate(data_generator.datasets):
        trial_len = dataset.trial_len
        num_trials = dataset.num_trials
        predictions[i] = np.full(
            shape=(num_trials, trial_len, model.hparams['output_size']),
            fill_value=np.nan)

    # partially fill container (gap trials will be included as nans)
    max_lags = model.hparams['n_max_lags']
    dtypes = ['train', 'val', 'test']
    for dtype in dtypes:
        data_generator.reset_iterators(dtype)
        for i in range(data_generator.num_tot_batches[dtype]):
            data, dataset = data_generator.next_batch(dtype)

            predictors = data[model.hparams['input_signal']][0]
            targets = data[model.hparams['output_signal']][0]

            # process batch, perhaps in chunks if full batch is too large
            # to fit on gpu
            chunk_size = 200
            batch_size = targets.shape[0]
            if batch_size > chunk_size:
                # split into chunks
                num_chunks = int(np.ceil(batch_size / chunk_size))
                for chunk in range(num_chunks):
                    # take chunks of size chunk_size, plus overlap due to
                    # max_lags
                    indx_beg = np.max([chunk * chunk_size - max_lags, 0])
                    indx_end = np.min([(chunk + 1) * chunk_size + max_lags, batch_size])
                    outputs = model(predictors[indx_beg:indx_end])
                    slc = (indx_beg + max_lags, indx_end - max_lags)
                    predictors[dataset][data['batch_indx'].item(), slice(*slc), :] = \
                        outputs[max_lags:-max_lags].cpu().detach().numpy()
            else:
                outputs = model(predictors)
                slc = (max_lags, -max_lags)
                predictions[dataset][data['batch_indx'].item(), slice(*slc), :] = \
                    outputs[max_lags:-max_lags].cpu().detach().numpy()

    # save latents separately for each dataset
    for i, dataset in enumerate(data_generator.datasets):
        # get save name which includes lab/expt/animal/session
        # sess_id = str(
        #     '%s_%s_%s_%s_latents.pkl' % (
        #         dataset.lab, dataset.expt, dataset.animal,
        #         dataset.session))
        if filename is None:
            sess_id = 'predictions.pkl'
            filename = os.path.join(
                model.hparams['results_dir'], 'test_tube_data',
                model.hparams['experiment_name'], 'version_%i' % model.version,
                sess_id)
        # save out array in pickle file
        pickle.dump({
            'predictions': predictions[i],
            'trials': data_generator.batch_indxs[i]},
            open(filename, 'wb'))


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
    from data.transforms import Threshold
    from behavenet.models import NN, LSTM

    # ###########################
    # ### Get Best Experiment ###
    # ###########################

    # expt_dir contains version_%i directories
    hparams['results_dir'] = os.path.join(
        hparams['tt_save_path'], hparams['lab'], hparams['expt'],
        hparams['animal'], hparams['session'])
    expt_dir = os.path.join(
        hparams['results_dir'], 'test_tube_data', hparams['experiment_name'])
    best_version = get_best_model_version(expt_dir)
    best_model_file = os.path.join(expt_dir, best_version, 'best_val_model.pt')

    # copy over hparams from best model
    hparams_file = os.path.join(expt_dir, best_version, 'meta_tags.pkl')
    with open(hparams_file, 'rb') as f:
        hparams = pickle.load(f)

    # ###########################
    # ### LOAD DATA GENERATOR ###
    # ###########################

    # TODO: this code is now copied several times; put in own function?
    ids = {
        'lab': hparams['lab'],
        'expt': hparams['expt'],
        'animal': hparams['animal'],
        'session': hparams['session']}

    if hparams['neural_thresh'] > 0 and hparams['neural_type'] == 'spikes':
        neural_transforms = Threshold(
            threshold=hparams['neural_thresh'],
            bin_size=hparams['neural_bin_size'])
    else:
        neural_transforms = None  # neural_region
    neural_kwargs = None

    # get model-specific signals/transforms/load_kwargs
    if hparams['model_name'] == 'neural-ae':
        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'ae'

        signals = ['neural', 'ae']

        ae_transforms = None

        ae_dir = os.path.join(
            hparams['results_dir'], 'test_tube_data',
            hparams['ae_experiment_name'])

        ae_kwargs = {  # TODO: base_dir + ids (here or in data generator?)
            'model_dir': ae_dir,
            'model_version': hparams['ae_version']}

        transforms = [neural_transforms, ae_transforms]
        load_kwargs = [neural_kwargs, ae_kwargs]

        hparams['output_size'] = hparams['n_ae_latents']

    elif hparams['model_name'] == 'neural-arhmm':
        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'arhmm'

        signals = ['neural', 'arhmm']

        arhmm_transforms = None

        arhmm_dir = os.path.join(
            hparams['results_dir'], 'test_tube_data',
            hparams['arhmm_experiment_name'])
        arhmm_kwargs = {
            'model_dir': arhmm_dir,
            'model_version': hparams['arhmm_version']}

        transforms = [neural_transforms, arhmm_transforms]
        load_kwargs = [neural_kwargs, arhmm_kwargs]

        hparams['output_size'] = hparams['n_arhmm_latents']

    else:
        raise ValueError(
            '"%s" is an invalid model_name' % hparams['model_name'])

    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], ids,
        signals=signals, transforms=transforms, load_kwargs=load_kwargs,
        device=hparams['device'], as_numpy=hparams['as_numpy'],
        batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed'])
    hparams['input_size'] = data_generator.datasets[0].dims[hparams['input_signal']][2]

    # ####################
    # ### CREATE MODEL ###
    # ####################

    if hparams['model_name'] == 'neural-ae':
        hparams['noise_dist'] = 'gaussian'
    elif hparams['model_name'] == 'neural-arhmm':
        hparams['noise_dist'] = 'categorical'
    else:
        raise ValueError(
            '"%s" is an invalid model_name' % hparams['model_name'])

    if hparams['model_type'] == 'ff' or hparams['model_type'] == 'linear':
        model = NN(hparams)
    elif hparams['model_type'] == 'lstm':
        model = LSTM(hparams)
    else:
        raise ValueError('"%s" is an invalid model_type' % hparams['model_type'])

    # load best model params
    model.version = int(best_version[8:])  # omg this is awful
    model.load_state_dict(torch.load(best_model_file))
    model.to(hparams['device'])
    model.eval()

    # push data through model
    export_predictions(data_generator, model)
