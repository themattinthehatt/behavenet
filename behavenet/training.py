# from behavenet.messages import hmm_expectations, hmm_sample
from behavenet.core import expected_log_likelihood, EarlyStopping, log_sum_exp
from tqdm import tqdm
import torch
from torch import nn, optim
import numpy as np
import math
import copy
import os
import pickle
import time


class FitMethod(object):

    def __init__(self, model, metric_strs):
        self.model = model
        self.metrics = {}
        dtype_strs = ['train', 'val', 'test', 'curr']
        for dtype in dtype_strs:
            self.metrics[dtype] = {}
            for metric in metric_strs:
                self.metrics[dtype][metric] = 0

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def calc_loss(self, data, device):
        raise NotImplementedError

    def get_loss(self, dtype):
        return self.metrics[dtype]['loss']

    def create_metric_row(self, dtype, epoch, batch, dataset, trial, **kwargs):
        raise NotImplementedError

    def reset_metrics(self, dtype):
        for key in self.metrics[dtype].keys():
            self.metrics[dtype][key] = 0

    def update_metrics(self, dtype):
        for key in self.metrics[dtype].keys():
            self.metrics[dtype][key] += self.metrics['curr'][key]
            self.metrics['curr'][key] = 0


class VAELoss(FitMethod):

    def __init__(self, model):
        metric_strs = ['batches', 'nll', 'mse', 'prior', 'loss']
        super().__init__(model, metric_strs)

    def calc_loss(self, data, device):
        raise NotImplementedError

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None):
        raise NotImplementedError
        # val_row = {
        #     'epoch': i_epoch,
        #     'batch_nb': i_train,
        #     'tng_err': train_loss / (i_train + 1),
        #     'val_err': val_loss / data_generator.num_tot_batches['val'],
        #     'val_NLL': val_NLL / data_generator.num_tot_batches['val'],
        #     'val_KL': val_KL / data_generator.num_tot_batches['val'],
        #     'val_MSE': val_MSE / data_generator.num_tot_batches['val'],
        #     'best_val_epoch': best_val_epoch}
        # test_row = {
        #     'epoch': i_epoch,
        #     'batch_nb': i_train,
        #     'test_err': test_loss / data_generator.num_tot_batches['test'],
        #     'test_NLL': test_NLL / data_generator.num_tot_batches['test'],
        #     'test_KL': test_KL / data_generator.num_tot_batches['test'],
        #     'test_MSE': test_MSE / data_generator.num_tot_batches['test'],
        #     'best_val_epoch': best_val_epoch}


class AELoss(FitMethod):

    def __init__(self, model):
        metric_strs = ['batches', 'loss']
        super().__init__(model, metric_strs)

    def calc_loss(self, data, device):

        y = data['images'][0]

        y_mu, _ = self.model(y)

        # define loss
        loss = torch.mean((y - y_mu)**2)
        # compute gradients
        loss.backward()

        # store current metrics (normalize by trial size)
        self.metrics['curr']['loss'] = loss.item() / y.shape[0]
        self.metrics['curr']['batches'] = 1

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None):
        if dtype == 'train':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': self.metrics['train']['loss'] / self.metrics['train']['batches']
            }
        elif dtype == 'val':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': self.metrics['train']['loss'] / self.metrics['train']['batches'],
                'val_loss': self.metrics['val']['loss'] / self.metrics['val']['batches'],
                'best_val_epoch': best_epoch}
        elif dtype == 'test':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'test_loss': self.metrics['test']['loss'] / self.metrics['test']['batches']}
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        return metric_row


class EMLoss(FitMethod):

    def __init__(self, model):
        metric_strs = ['batches', 'nll', 'prior']
        super().__init__(model, metric_strs)

    def calc_loss(self, data, device):
        raise NotImplementedError

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None):
        raise NotImplementedError
        # val_row = {
        #     'epoch': i_epoch,
        #     'batch_nb': i_train,
        #     'tng_err': train_loss / (i_train + 1),
        #     'val_err': val_loss / data_generator.num_tot_batches['val'],
        #     'val_NLL': val_NLL / data_generator.num_tot_batches['val'],
        #     'val_KL': val_KL / data_generator.num_tot_batches['val'],
        #     'val_MSE': val_MSE / data_generator.num_tot_batches['val'],
        #     'best_val_epoch': best_val_epoch}
        # test_row = {
        #     'epoch': i_epoch,
        #     'batch_nb': i_train,
        #     'test_err': test_loss / data_generator.num_tot_batches['test'],
        #     'test_NLL': test_NLL / data_generator.num_tot_batches['test'],
        #     'test_prior': test_prior / data_generator.num_tot_batches['test'],
        #     'best_val_epoch': best_val_epoch}


class SVILoss(FitMethod):

    def __init__(self, model, variational_posterior):

        metric_strs = ['batches', 'ell', 'prior', 'log_likelihood', 'log_q']
        super().__init__(model, metric_strs)

        assert variational_posterior is not None
        self.variational_posterior = variational_posterior

    def get_parameters(self):
        model_parameters = list(filter(
            lambda p: p.requires_grad, self.model.parameters()))
        var_post_parameters = list(filter(
            lambda p: p.requires_grad, self.variational_posterior.parameters()))
        return model_parameters + var_post_parameters

    def calc_loss(self, data, device):
        raise NotImplementedError

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None):
        raise NotImplementedError
        # test_row = {'epoch': i_epoch, 'batch_nb': i_train,
        #          'tng_err': train_loss / i_train}
        # val_row = {
        #     'epoch': i_epoch,
        #     'batch_nb': i_train,
        #     'tng_err': train_loss / (i_train + 1),
        #     'val_err': val_loss / data_generator.num_tot_batches['val'],
        #     'val_ell': val_ell / data_generator.num_tot_batches['val'],
        #     'val_prior': val_prior / data_generator.num_tot_batches['val'],
        #     'val_log_likelihood': val_log_likelihood /
        #                           data_generator.num_tot_batches['val'],
        #     'val_log_q': val_log_q / data_generator.num_tot_batches['val'],
        #     'best_val_epoch': best_val_epoch}
        # test_row = {
        #     'epoch': i_epoch,
        #     'batch_nb': i_train,
        #     'test_err': test_loss / data_generator.num_tot_batches['test'],
        #     'test_ell': test_ell / data_generator.num_tot_batches['test'],
        #     'test_prior': test_prior / data_generator.num_tot_batches['test'],
        #     'test_log_likelihood': test_log_likelihood / data_generator.num_tot_batches['test'],
        #     'test_log_q': test_log_q / data_generator.num_tot_batches['test'],
        #     'best_val_epoch': best_val_epoch}


def fit(hparams, model, data_generator, exp, method="em", variational_posterior=None):
    """
    Args:
        hparams:
        model:
        data_generator:
        exp:
        method:
        variational_posterior:
    """

    # Check inputs
    if method == 'em':
        loss = EMLoss(model)
    elif method == 'svi':
        loss = SVILoss(model, variational_posterior)
    elif method == 'vae':
        loss = VAELoss(model)
    elif method == 'ae':
        loss = AELoss(model)
    else:
        raise ValueError('"%s" is an invalid fitting method' % method)

    # Optimizer set-up
    optimizer = torch.optim.Adam(
        loss.get_parameters(), lr=hparams['learning_rate'])

    # Early stopping set-up
    best_val_loss = math.inf
    best_val_epoch = None
    nb_epochs_since_check = 0

    # enumerate batches on which validation metrics should be recorded
    val_check_batch = np.linspace(
        data_generator.num_tot_batches['train'] * hparams['val_check_interval'],
        data_generator.num_tot_batches['train'] * hparams['max_nb_epochs'],
        int(hparams['max_nb_epochs'] / hparams['val_check_interval'])).astype('int')

    if hparams['enable_early_stop']:
        raise NotImplementedError
        # early_stop = EarlyStopping(
        #     min_fraction=hparams['early_stop_fraction'],
        #     patience=hparams['early_stop_patience'])
    should_stop = False

    i_epoch = 0
    for i_epoch in range(hparams['max_nb_epochs']):

        loss.reset_metrics('train')
        data_generator.reset_iterators('train')
        model.train()

        for i_train in tqdm(range(data_generator.num_tot_batches['train'])):

            # Zero out gradients. Don't want gradients from previous iterations
            optimizer.zero_grad()

            # Get next minibatch and put it on the device
            data, dataset = data_generator.next_batch('train')

            # Call the appropriate loss function
            loss.calc_loss(data, hparams['device'])
            loss.update_metrics('train')

            # Step (evaluate untrained network on epoch 0)
            if i_epoch > 0:
                optimizer.step()

            # Check validation according to schedule
            curr_batch = (i_train + 1) + i_epoch * data_generator.num_tot_batches['train']
            if np.any(curr_batch == val_check_batch):

                loss.reset_metrics('val')
                data_generator.reset_iterators('val')
                model.eval()

                for i_val in range(data_generator.num_tot_batches['val']):

                    # Get next minibatch and put it on the device
                    data, dataset = data_generator.next_batch('val')

                    # Call the appropriate loss function
                    loss.calc_loss(data, hparams['device'])
                    loss.update_metrics('val')

                # Save best val model
                if loss.get_loss('val') < best_val_loss:
                    best_val_loss = loss.get_loss('val')
                    filepath = os.path.join(
                        hparams['tt_save_path'], 'test_tube_data',
                        hparams['experiment_name'],
                        'version_' + str(exp.version),
                        'best_val_model.pt')
                    torch.save(model.state_dict(), filepath)
                    model.hparams = None
                    best_val_model = copy.deepcopy(model)
                    model.hparams = hparams
                    best_val_model.hparams = hparams
                    best_val_epoch = i_epoch

                if hparams['enable_early_stop']:
                    pass
                    # stop_train = early_stop.on_val_check(
                    #     i_epoch, loss.get_loss('val') / i_val)
                    # met_min_epochs = i_epoch > hparams['min_nb_epochs']
                    # should_stop = stop_train and met_min_epochs
                    # if should_stop:
                    #     break

                exp.log(loss.create_metric_row(
                    'val', i_epoch, i_train, dataset, None,
                    best_epoch=best_val_epoch))
                exp.save()

            elif (i_train + 1) % data_generator.num_tot_batches['train'] == 0:
                # export training metrics at end of epoch
                exp.log(loss.create_metric_row(
                    'train', i_epoch, i_train, dataset, None))
                exp.save()

        if should_stop:
            break

    # Compute test loss
    loss.reset_metrics('test')
    data_generator.reset_iterators('test')
    model.eval()

    for i_test in range(data_generator.num_tot_batches['test']):

        # Get next minibatch and put it on the device
        data, dataset = data_generator.next_batch('test')

        # Call the appropriate loss function
        loss.reset_metrics('test')
        loss.calc_loss(data, hparams['device'])
        loss.update_metrics('test')

        # calculate metrics for each batch
        exp.log(loss.create_metric_row(
            'test', i_epoch, i_test, dataset, data['batch_indx'].item()))
    exp.save()

    # save out best model
    filepath = os.path.join(
        hparams['tt_save_path'], 'test_tube_data', hparams['experiment_name'],
        'version_' + str(exp.version), 'last_model.pt')
    torch.save(model.state_dict(), filepath)

    # export latents
    if hparams['export_latents']:

        # initialize container for latents
        latents = [[] for _ in range(data_generator.num_datasets)]
        for i, dataset in enumerate(data_generator.datasets):
            trial_len = dataset.trial_len
            num_trials = dataset.num_trials
            latents[i] = np.full(
                shape=(num_trials, trial_len, hparams['n_latents']),
                fill_value=np.nan)

        # partially fill container (gap trials will not be included)
        dtypes = ['train', 'val', 'test']
        for dtype in dtypes:
            data_generator.reset_iterators(dtype)
            for i in range(data_generator.num_tot_batches[dtype]):
                data, dataset = data_generator.next_batch(dtype)
                # _, curr_latents = model(data[hparams['signals']][0])
                curr_latents, _, _ = model.encoding(data[hparams['signals']][0])
                latents[dataset][data['batch_indxs'].item(), :, :] = curr_latents

        # save latents separately for each dataset
        for i, dataset in enumerate(data_generator.datasets):
            # get save name which includes lab/expt/animal/session
            sess_id = str('%s_%s_%s_%s_latents.pkl')
            filepath = os.path.join(
                hparams['tt_save_path'], 'test_tube_data',
                hparams['experiment_name'], 'version_' + str(exp.version),
                sess_id)
            # save out array in pickle file
            pickle.dump(latents[i], filepath)
