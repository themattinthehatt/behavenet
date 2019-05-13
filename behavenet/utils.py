import numpy as np


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
    states = [[] for _ in range(data_generator.num_datasets)]
    for i, dataset in enumerate(data_generator.datasets):
        trial_len = dataset.trial_len
        num_trials = dataset.num_trials
        states[i] = np.full(
            shape=(num_trials, trial_len),
            fill_value=np.nan)

    # partially fill container (gap trials will be included as nans)
    dtypes = ['train', 'val', 'test']
    for dtype in dtypes:
        data_generator.reset_iterators(dtype)
        for i in range(data_generator.num_tot_batches[dtype]):
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
            filename = os.path.join(
                model.hparams['results_dir'], 'test_tube_data',
                model.hparams['experiment_name'], 'version_%i' % model.version,
                'latents.pkl')
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
        # get save name which includes lab/expt/animal/session
        # sess_id = str(
        #     '%s_%s_%s_%s_latents.pkl' % (
        #         dataset.lab, dataset.expt, dataset.animal,
        #         dataset.session))
        if filename is None:
            filename = os.path.join(
                model.hparams['results_dir'], 'test_tube_data',
                model.hparams['experiment_name'], 'version_%i' % model.version,
                'predictions.pkl')
        # save out array in pickle file
        pickle.dump({
            'predictions': predictions[i],
            'trials': data_generator.batch_indxs[i]},
            open(filename, 'wb'))
