import numpy as np
from behavenet.data.utils import get_best_model_and_data


def export_states(hparams, exp, data_generator, model, filename=None):
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

    # initialize container for states
    states = [[] for _ in range(data_generator.n_datasets)]
    for i, dataset in enumerate(data_generator.datasets):
        trial_len = dataset.trial_len
        n_trials = dataset.n_trials
        states[i] = np.full(
            shape=(n_trials, trial_len),
            fill_value=np.nan)

    # partially fill container (gap trials will be included as nans)
    dtypes = ['train', 'val', 'test']
    for dtype in dtypes:
        data_generator.reset_iterators(dtype)
        for i in range(data_generator.n_tot_batches[dtype]):
            data, dataset = data_generator.next_batch(dtype)

            # process batch,
            y = data['ae'][0]
            batch_size = y.shape[0]

            curr_states = model.most_likely_states(y)
            states[dataset][data['batch_indx'].item(), :] = curr_states

    # save states separately for each dataset
    for i, dataset in enumerate(data_generator.datasets):
        if filename is None:
            sess_id = 'states.pkl'
            filename = os.path.join(
                hparams['results_dir'], 'test_tube_data',
                hparams['experiment_name'], 'version_%i' % exp.version,
                sess_id)
        # save out array in pickle file
        pickle.dump({
            'states': states[i],
            'trials': data_generator.batch_indxs[i]},
            open(filename, 'wb'))


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
    latents = [[] for _ in range(data_generator.n_datasets)]
    for i, dataset in enumerate(data_generator.datasets):
        trial_len = dataset.trial_len
        n_trials = dataset.n_trials
        latents[i] = np.full(
            shape=(n_trials, trial_len, model.hparams['n_ae_latents']),
            fill_value=np.nan)

    # partially fill container (gap trials will be included as nans)
    dtypes = ['train', 'val', 'test']
    for dtype in dtypes:
        data_generator.reset_iterators(dtype)
        for i in range(data_generator.n_tot_batches[dtype]):
            data, dataset = data_generator.next_batch(dtype)

            # process batch, perhaps in chunks if full batch is too large
            # to fit on gpu
            chunk_size = 200
            y = data['images'][0]
            batch_size = y.shape[0]
            if batch_size > chunk_size:
                # split into chunks
                n_chunks = int(np.ceil(batch_size / chunk_size))
                for chunk in range(n_chunks):
                    # take chunks of size chunk_size, plus overlap due to
                    # max_lags
                    indx_beg = chunk * chunk_size
                    indx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                    curr_latents, _, _ = model.encoding(
                        y[indx_beg:indx_end])
                    latents[dataset][data['batch_indx'].item(), indx_beg:indx_end, :] = \
                        curr_latents.cpu().detach().numpy()
            else:
                curr_latents, _, _ = model.encoding(y)
                latents[dataset][data['batch_indx'].item(), :, :] = \
                    curr_latents.cpu().detach().numpy()

    # save latents separately for each dataset
    for i, dataset in enumerate(data_generator.datasets):
        if filename is None or data_generator.n_datasets > 1:
            # get save name which includes lab/expt/animal/session
            if data_generator.n_datasets > 1:
                sess_id = str(
                    '%s_%s_%s_%s_latents.pkl' % (
                        dataset.lab, dataset.expt, dataset.animal,
                        dataset.session))
            else:
                sess_id = 'latents.pkl'
            filename = os.path.join(
                model.hparams['results_dir'], 'test_tube_data',
                model.hparams['experiment_name'], 'version_%i' % model.version,
                sess_id)
        # save out array in pickle file
        print(
            'saving latents %i of %i:\n%s' %
            (i + 1, data_generator.n_datasets, filename))
        latents_dict = {'latents': latents[i], 'trials': dataset.batch_indxs}
        with open(filename, 'wb') as f:
            pickle.dump(latents_dict, f)


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
    predictions = [[] for _ in range(data_generator.n_datasets)]
    for i, dataset in enumerate(data_generator.datasets):
        trial_len = dataset.trial_len
        n_trials = dataset.n_trials
        predictions[i] = np.full(
            shape=(n_trials, trial_len, model.hparams['output_size']),
            fill_value=np.nan)

    # partially fill container (gap trials will be included as nans)
    max_lags = model.hparams['n_max_lags']
    dtypes = ['train', 'val', 'test']
    for dtype in dtypes:
        data_generator.reset_iterators(dtype)
        for i in range(data_generator.n_tot_batches[dtype]):
            data, dataset = data_generator.next_batch(dtype)

            predictors = data[model.hparams['input_signal']][0]
            targets = data[model.hparams['output_signal']][0]

            # process batch, perhaps in chunks if full batch is too large
            # to fit on gpu
            chunk_size = 200
            batch_size = targets.shape[0]
            if batch_size > chunk_size:
                # split into chunks
                n_chunks = int(np.ceil(batch_size / chunk_size))
                for chunk in range(n_chunks):
                    # take chunks of size chunk_size, plus overlap due to
                    # max_lags
                    indx_beg = np.max([chunk * chunk_size - max_lags, 0])
                    indx_end = np.min([(chunk + 1) * chunk_size + max_lags, batch_size])
                    outputs, _ = model(predictors[indx_beg:indx_end])
                    slc = (indx_beg + max_lags, indx_end - max_lags)
                    predictions[dataset][data['batch_indx'].item(), slice(*slc), :] = \
                        outputs[max_lags:-max_lags].cpu().detach().numpy()
            else:
                outputs, _ = model(predictors)
                slc = (max_lags, -max_lags)
                predictions[dataset][data['batch_indx'].item(), slice(*slc), :] = \
                    outputs[max_lags:-max_lags].cpu().detach().numpy()

    # save latents separately for each dataset
    for i, dataset in enumerate(data_generator.datasets):
        if filename is None or data_generator.n_datasets > 1:
            # get save name which includes lab/expt/animal/session
            if data_generator.n_datasets > 1:
                sess_id = str(
                    '%s_%s_%s_%s_predictions.pkl' % (
                        dataset.lab, dataset.expt, dataset.animal,
                        dataset.session))
            else:
                sess_id = 'predictions.pkl'
            filename = os.path.join(
                model.hparams['results_dir'], 'test_tube_data',
                model.hparams['experiment_name'], 'version_%i' % model.version,
                sess_id)
        # save out array in pickle file
        print(
            'saving latents %i of %i to %s' %
            (i + 1, data_generator.n_datasets, filename))
        predictions_dict = {
            'predictions': predictions[i], 'trials': dataset.batch_indxs}
        with open(filename, 'wb') as f:
            pickle.dump(predictions_dict, f)


def export_latents_best(hparams):
    """
    Export predictions for the best decoding model in a test tube experiment.
    Predictions are saved in the corresponding model directory.

    Args:
        hparams (dict):
    """

    from behavenet.models import AE
    model, data_generator = get_best_model_and_data(hparams, AE)
    export_latents(data_generator, model)


def export_predictions_best(hparams):
    """
    Export predictions for the best decoding model in a test tube experiment.
    Predictions are saved in the corresponding model directory.

    Args:
        hparams (dict):
    """

    from behavenet.models import Decoder
    model, data_generator = get_best_model_and_data(hparams, Decoder)
    export_predictions(data_generator, model)


def get_reconstruction(model, inputs):
    """
    Reconstruct an image from either image or latent inputs

    Args:
        model: pt Model
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

    if input_type == 'images':
        ims_recon, _ = model(inputs)
    else:
        # TODO: how to incorporate maxpool layers for decoding only?
        ims_recon = model.decoding(inputs, None, None)
    ims_recon = ims_recon.cpu().detach().numpy()

    return ims_recon
