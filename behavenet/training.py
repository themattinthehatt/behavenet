from behavenet.core import expected_log_likelihood, log_sum_exp
from behavenet.messages import hmm_expectations, hmm_sample
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

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch, **kwargs):
        if dtype == 'train':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': self.metrics['train']['loss'] /
                           self.metrics['train']['batches']
            }
        elif dtype == 'val':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': self.metrics['train']['loss'] /
                           self.metrics['train']['batches'],
                'val_loss': self.metrics['val']['loss'] /
                            self.metrics['val']['batches'],
                'best_val_epoch': best_epoch}
        elif dtype == 'test':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'test_loss': self.metrics['test']['loss'] /
                             self.metrics['test']['batches']}
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        return metric_row

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

        y = data[self.model.hparams['signals']][0]

        chunk_size = 200
        batch_size = y.shape[0]

        if batch_size > chunk_size:
            # split into chunks
            num_chunks = np.ceil(batch_size / chunk_size)
            loss_val = 0
            for chunk in range(num_chunks):
                indx_beg = chunk * chunk_size
                indx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                y_mu, _ = self.model(y[indx_beg:indx_end])
                loss = torch.mean((y - y_mu) ** 2)
                # compute gradients
                loss.backward()
                # get loss value (weighted by batch size)
                loss_val += loss.item() * (indx_end - indx_beg)
            loss_val /= y.shape[0]
        else:
            y_mu, _ = self.model(y)
            # define loss
            loss = torch.mean((y - y_mu)**2)
            # compute gradients
            loss.backward()
            # get loss value
            loss_val = loss.item()

        # store current metrics
        self.metrics['curr']['loss'] = loss_val
        self.metrics['curr']['batches'] = 1


class NLLLoss(FitMethod):

    def __init__(self, model):
        metric_strs = ['batches', 'loss']
        super().__init__(model, metric_strs)

        if self.model.hparams['noise_dist'] == 'gaussian':
            self._loss = nn.MSELoss()
        elif self.model.hparams['noise_dist'] == 'poisson':
            self._loss = nn.PoissonNLLLoss(log_input=False)
        elif self.model.hparams['noise_dist'] == 'categorical':
            self._loss = nn.CrossEntropyLoss()

    def calc_loss(self, data, device):
        """data is 1 x T x N"""

        predictors = data[self.model.hparams['input_signal']][0]
        targets = data[self.model.hparams['output_signal']][0]

        max_lags = self.model.hparams['n_max_lags']

        chunk_size = 200
        batch_size = targets.shape[0]

        if batch_size > chunk_size:
            # split into chunks
            num_chunks = int(np.ceil(batch_size / chunk_size))
            loss_val = 0
            for chunk in range(num_chunks):
                # take chunks of size chunk_size, plus overlap due to max_lags
                indx_beg = np.max([chunk * chunk_size - max_lags, 0])
                indx_end = np.min([(chunk + 1) * chunk_size + max_lags, batch_size])
                outputs = self.model(predictors[indx_beg:indx_end])
                # define loss on allowed window of data
                loss = self._loss(
                    outputs[max_lags:-max_lags],
                    targets[max_lags:-max_lags])
                # compute gradients
                loss.backward()
                # get loss value (weighted by batch size)
                loss_val += loss.item() * outputs[max_lags:-max_lags].shape[0]
            loss_val /= targets.shape[0]
        else:
            outputs = self.model(predictors)
            # define loss on allowed window of data
            loss = self._loss(
                outputs[max_lags:-max_lags],
                targets[max_lags:-max_lags])
            # compute gradients
            loss.backward()
            # get loss value
            loss_val = loss.item()

        # store current metrics
        self.metrics['curr']['loss'] = loss_val
        self.metrics['curr']['batches'] = 1


class EMLoss(FitMethod):

    def __init__(self, model):
        metric_strs = ['batches', 'nll', 'prior']
        super().__init__(model, metric_strs)

    def calc_loss(self, data, device):
        ae = data['ae'][0]
        neural = data['neural'][0]

        low_d = self.model.get_low_d(ae)
        log_prior = self.model.log_prior()
        log_pi0 = self.model.log_pi0(low_d)
        log_Ps = self.model.log_transition_proba(low_d)
        lls = self.model.log_dynamics_proba(low_d)

        with torch.no_grad():
            expectations = hmm_expectations(log_pi0, log_Ps, lls, device)
        
        prior = log_prior #/ int(nb_tng_batches)
        likelihood = expected_log_likelihood(expectations, log_pi0, log_Ps, lls)

        elp = prior + likelihood
        if np.isnan(elp.item()):
            raise Exception("Expected log probability is not finite")

        loss = -elp / low_d.shape[0] / low_d.shape[1]

        loss.backward()
        loss_val = loss.item()

        self.metrics['curr']['nll'] = -likelihood.item() / low_d.shape[0] / low_d.shape[1]
        self.metrics['curr']['prior'] = prior.item() / low_d.shape[0] / low_d.shape[1]
        self.metrics['curr']['batches'] = 1

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None):
        if dtype == 'train':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_nll': self.metrics['train']['nll'] / self.metrics['train']['batches'],
                'tr_prior': self.metrics['train']['prior'] / self.metrics['train']['batches']
            }
        elif dtype == 'val':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_nll': self.metrics['train']['nll'] / self.metrics['train']['batches'],
                'val_nll': self.metrics['val']['nll'] / self.metrics['val']['batches'],
                'tr_prior': self.metrics['train']['prior'] / self.metrics['train']['batches'],
                'val_prior': self.metrics['val']['prior'] / self.metrics['val']['batches'],
                'best_val_epoch': best_epoch}
        elif dtype == 'test':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'test_nll': self.metrics['test']['nll'] / self.metrics['test']['batches'],
                'test_prior': self.metrics['test']['prior'] / self.metrics['test']['batches']}
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        return metric_row

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


class EarlyStopping(object):
    """Stop training when a monitored quantity has stopped improving"""

    def __init__(self, history=0, min_epochs=10):
        """
        Args:
            history (int): number of previous checks to average over when
                checking for increase in loss
            min_epochs (int): minimum number of epochs for training
        """

        self.history = history
        self.min_epochs = min_epochs

        # keep track of `history` most recent losses
        self.prev_losses = np.full(self.history, fill_value=np.nan)
        self.best_epoch = 0
        self.best_loss = np.inf
        self.stopped_epoch = 0
        self.should_stop = False

    def on_val_check(self, epoch, curr_loss):

        prev_mean = np.nanmean(self.prev_losses)
        self.prev_losses = np.roll(self.prev_losses, 1)
        self.prev_losses[0] = curr_loss
        curr_mean = np.nanmean(self.prev_losses)

        # update best loss and epoch that it happened at
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss
            self.best_epoch = epoch

        # check if smoothed loss is starting to increase; exit training if so
        if epoch > self.min_epochs and curr_mean >= prev_mean:
            print('\n== early stop criteria met; exiting train loop ==')
            print('training epochs: %d' % epoch)
            print('end cost: %04f' % curr_loss)
            print('best epoch: %i' % self.best_epoch)
            print('best cost: %04f\n' % self.best_loss)
            self.stopped_epoch = epoch
            self.should_stop = True


# class EarlyStopping(object):
#     """Stop training when a monitored quantity has stopped improving"""
#
#     def __init__(self, min_fraction=1.0, patience=0, min_epochs=10):
#         """
#         Args:
#             min_fraction (float): minimum change in the monitored quantity
#                 to qualify as an improvement, i.e. change of less than
#                 min_fraction * best val loss will count as no improvement.
#             patience (int): number of epochs with no improvement after which
#                 training will be stopped.
#             min_epochs (int): minimum number of epochs for training
#         """
#
#         self.min_fraction = min_fraction
#         self.patience = patience
#         self.min_epochs = min_epochs
#         self.wait = 0
#         self.stopped_epoch = 0
#         self.best = np.inf
#         self.should_stop = False
#
#     def on_val_check(self, epoch, val_loss):
#
#         stop_training = False
#
#         if val_loss < self.min_fraction * self.best:
#             self.best = val_loss
#             self.wait = 0
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 self.stopped_epoch = epoch
#                 stop_training = True
#
#         met_min_epochs = epoch > self.min_epochs
#
#         self.should_stop = stop_training and met_min_epochs


def fit(
        hparams, model, data_generator, exp, method='em',
        variational_posterior=None):
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
    elif method == 'nll':
        loss = NLLLoss(model)
    else:
        raise ValueError('"%s" is an invalid fitting method' % method)

    # Optimizer set-up
    optimizer = torch.optim.Adam(
        loss.get_parameters(),
        lr=hparams['learning_rate'],
        weight_decay=hparams['l2_reg'])

    # enumerate batches on which validation metrics should be recorded
    best_val_loss = math.inf
    best_val_epoch = None
    best_val_model = None
    val_check_batch = np.linspace(
        data_generator.num_tot_batches['train'] * hparams['val_check_interval'],
        data_generator.num_tot_batches['train'] * hparams['max_nb_epochs'],
        int(hparams['max_nb_epochs'] / hparams['val_check_interval'])).astype('int')

    # early stopping set-up
    if hparams['enable_early_stop']:
        early_stop = EarlyStopping(
            history=hparams['history'],
            min_epochs=hparams['min_nb_epochs'])

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
                        hparams['results_dir'], 'test_tube_data',
                        hparams['experiment_name'],
                        'version_%i' % exp.version,
                        'best_val_model.pt')
                    torch.save(model.state_dict(), filepath)

                    model.hparams = None
                    best_val_model = copy.deepcopy(model)
                    model.hparams = hparams
                    best_val_model.hparams = hparams
                    best_val_epoch = i_epoch

                # TODO: should this be here or outside batch loop?
                if hparams['enable_early_stop']:
                    early_stop.on_val_check(i_epoch, loss.get_loss('val'))
                    if early_stop.should_stop:
                        break

                exp.log(loss.create_metric_row(
                    'val', i_epoch, i_train, dataset, None,
                    best_epoch=best_val_epoch))
                exp.save()

            elif (i_train + 1) % data_generator.num_tot_batches['train'] == 0:
                # export training metrics at end of epoch
                exp.log(loss.create_metric_row(
                    'train', i_epoch, i_train, dataset, None))
                exp.save()

        # if hparams['enable_early_stop']:
        #     early_stop.on_val_check(i_epoch, loss.get_loss('val'))
        #     if early_stop.should_stop:
        #         break

    # save out last model
    filepath = os.path.join(
        hparams['results_dir'], 'test_tube_data', hparams['experiment_name'],
        'version_%i' % exp.version, 'last_model.pt')
    torch.save(model.state_dict(), filepath)

    # Compute test loss
    if method == 'em':
        test_loss = EMLoss(best_val_model)
    elif method == 'svi':
        test_loss = SVILoss(best_val_model, variational_posterior)
    elif method == 'vae':
        test_loss = VAELoss(best_val_model)
    elif method == 'ae':
        test_loss = AELoss(best_val_model)
    elif method == 'nll':
        test_loss = NLLLoss(best_val_model)
    else:
        raise ValueError('"%s" is an invalid fitting method' % method)

    test_loss.reset_metrics('test')
    data_generator.reset_iterators('test')
    best_val_model.eval()

    for i_test in range(data_generator.num_tot_batches['test']):

        # Get next minibatch and put it on the device
        data, dataset = data_generator.next_batch('test')

        # Call the appropriate loss function
        test_loss.reset_metrics('test')
        test_loss.calc_loss(data, hparams['device'])
        test_loss.update_metrics('test')

        # calculate metrics for each batch
        exp.log(test_loss.create_metric_row(
            'test', i_epoch, i_test, dataset, data['batch_indx'].item()))
    exp.save()

    # export latents
    # TODO: export decoder predictions?
    if method == 'ae' and hparams['export_latents']:

        # initialize container for latents
        latents = [[] for _ in range(data_generator.num_datasets)]
        for i, dataset in enumerate(data_generator.datasets):
            trial_len = dataset.trial_len
            num_trials = dataset.num_trials
            latents[i] = np.full(
                shape=(num_trials, trial_len, hparams['n_latents']),
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
                batch_size = data[hparams['signals'][0]].shape[0]
                y = data[hparams['signals']][0]
                if batch_size > chunk_size:
                    # split into chunks
                    num_chunks = int(np.ceil(batch_size / chunk_size))
                    for chunk in range(num_chunks):
                        # take chunks of size chunk_size, plus overlap due to
                        # max_lags
                        indx_beg = chunk * chunk_size
                        indx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                        curr_latents, _, _ = best_val_model.encoding(
                            y[indx_beg:indx_end])
                        latents[dataset][data['batch_indx'].item(), indx_beg:indx_end, :] = \
                            curr_latents.cpu().detach().numpy()
                else:
                    curr_latents, _, _ = best_val_model.encoding(y)
                    latents[dataset][data['batch_indx'].item(), :, :] = \
                        curr_latents.cpu().detach().numpy()

        # save latents separately for each dataset
        for i, dataset in enumerate(data_generator.datasets):
            # get save name which includes lab/expt/animal/session
            sess_id = str(
                '%s_%s_%s_%s_latents.pkl' % (
                    dataset.lab, dataset.expt, dataset.animal,
                    dataset.session))
            filepath = os.path.join(
                hparams['results_dir'], 'test_tube_data',
                hparams['experiment_name'], 'version_%i' % exp.version,
                sess_id)
            # save out array in pickle file
            pickle.dump({'latents': latents[i]}, open(filepath, 'wb'))
