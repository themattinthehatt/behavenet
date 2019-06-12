from tqdm import tqdm
import torch
from torch import nn, optim
import numpy as np
import math
import copy
import os
import pickle
import time
from sklearn.metrics import r2_score, accuracy_score
from behavenet.core import expected_log_likelihood, log_sum_exp
from behavenet.utils import export_latents, export_predictions
from behavenet.losses import GaussianNegLogProb
# from behavenet.messages import hmm_expectations, hmm_sample


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
        return self.metrics[dtype]['loss'] / self.metrics[dtype]['batches']

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None,
            *args, **kwargs):
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
            if self.metrics['curr'][key] is not None:
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
        #     'batch_n': i_train,
        #     'tng_err': train_loss / (i_train + 1),
        #     'val_err': val_loss / data_generator.num_tot_batches['val'],
        #     'val_NLL': val_NLL / data_generator.num_tot_batches['val'],
        #     'val_KL': val_KL / data_generator.num_tot_batches['val'],
        #     'val_MSE': val_MSE / data_generator.num_tot_batches['val'],
        #     'best_val_epoch': best_val_epoch}
        # test_row = {
        #     'epoch': i_epoch,
        #     'batch_n': i_train,
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

        if 'masks' in data:
            masks = data['masks'][0]
        else:
            masks = None

        chunk_size = 200
        batch_size = y.shape[0]

        if batch_size > chunk_size:
            # split into chunks
            num_chunks = int(np.ceil(batch_size / chunk_size))
            loss_val = 0
            for chunk in range(num_chunks):
                indx_beg = chunk * chunk_size
                indx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                y_mu, _ = self.model(y[indx_beg:indx_end])
                if masks is not None:
                    loss = torch.mean((
                        (y[indx_beg:indx_end] - y_mu) ** 2) *
                        masks[indx_beg:indx_end])
                else:
                    loss = torch.mean((y[indx_beg:indx_end] - y_mu) ** 2)
                # compute gradients
                loss.backward()
                # get loss value (weighted by batch size)
                loss_val += loss.item() * (indx_end - indx_beg)
            loss_val /= y.shape[0]
        else:
            y_mu, _ = self.model(y)
            # define loss
            if masks is not None:
                loss = torch.mean(((y - y_mu)**2) * masks)
            else:
                loss = torch.mean((y - y_mu) ** 2)
            # compute gradients
            loss.backward()
            # get loss value
            loss_val = loss.item()

        # store current metrics
        self.metrics['curr']['loss'] = loss_val
        self.metrics['curr']['batches'] = 1


class NLLLoss(FitMethod):

    def __init__(self, model):
        metric_strs = ['batches', 'loss', 'r2', 'fc']
        super().__init__(model, metric_strs)

        if self.model.hparams['noise_dist'] == 'gaussian':
            self._loss = nn.MSELoss()
        elif self.model.hparams['noise_dist'] == 'gaussian-full':
            self._loss = GaussianNegLogProb()  # model holds precision mat
        elif self.model.hparams['noise_dist'] == 'poisson':
            self._loss = nn.PoissonNLLLoss(log_input=False)
        elif self.model.hparams['noise_dist'] == 'categorical':
            self._loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                '"%s" is not a valid noise dist' % self.model.hparams['noise_dist'])

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
            outputs_all = []
            loss_val = 0
            for chunk in range(num_chunks):
                # take chunks of size chunk_size, plus overlap due to max_lags
                indx_beg = np.max([chunk * chunk_size - max_lags, 0])
                indx_end = np.min([(chunk + 1) * chunk_size + max_lags, batch_size])
                outputs, precision = self.model(predictors[indx_beg:indx_end])
                # define loss on allowed window of data
                if self.model.hparams['noise_dist'] == 'gaussian-full':
                    loss = self._loss(
                        outputs[max_lags:-max_lags],
                        targets[indx_beg:indx_end][max_lags:-max_lags],
                        precision[max_lags:-max_lags])
                else:
                    loss = self._loss(
                        outputs[max_lags:-max_lags],
                        targets[indx_beg:indx_end][max_lags:-max_lags])
                # compute gradients
                loss.backward()
                # get loss value (weighted by batch size)
                loss_val += loss.item() * outputs[max_lags:-max_lags].shape[0]
                outputs_all.append(
                    outputs[max_lags:-max_lags].cpu().detach().numpy())
            loss_val /= targets.shape[0]
            outputs_all = np.concatenate(outputs_all, axis=0)
        else:
            outputs, precision = self.model(predictors)
            # define loss on allowed window of data
            if self.model.hparams['noise_dist'] == 'gaussian-full':
                loss = self._loss(
                    outputs[max_lags:-max_lags],
                    targets[max_lags:-max_lags],
                    precision[max_lags:-max_lags])
            else:
                loss = self._loss(
                    outputs[max_lags:-max_lags],
                    targets[max_lags:-max_lags])
            # compute gradients
            loss.backward()
            # get loss value
            loss_val = loss.item()
            outputs_all = outputs[max_lags:-max_lags].cpu().detach().numpy()

        if self.model.hparams['noise_dist'] == 'gaussian' \
                or self.model.hparams['noise_dist'] == 'gaussian-full':
            # use variance-weighted r2s to ignore small-variance latents
            r2 = r2_score(
                targets[max_lags:-max_lags].cpu().detach().numpy(),
                outputs_all,
                multioutput='variance_weighted')
            fc = None
        elif self.model.hparams['noise_dist'] == 'poisson':
            raise NotImplementedError
        elif self.model.hparams['noise_dist'] == 'categorical':
            r2 = None
            fc = accuracy_score(
                targets[max_lags:-max_lags].cpu().detach().numpy(),
                np.argmax(outputs_all, axis=1))
        else:
            raise ValueError(
                '"%s" is not a valid noise_dist' %
                self.model.hparams['noise_dist'])

        # store current metrics
        self.metrics['curr']['loss'] = loss_val
        self.metrics['curr']['r2'] = r2
        self.metrics['curr']['fc'] = fc
        self.metrics['curr']['batches'] = 1

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None,
            **kwargs):
        if dtype == 'train':
            norm = self.metrics['train']['batches']
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': self.metrics['train']['loss'] / norm,
                'tr_r2': self.metrics['train']['r2'] / norm,
                'tr_fc': self.metrics['train']['fc'] / norm}
        elif dtype == 'val':
            norm_tr = self.metrics['train']['batches']
            norm_val = self.metrics['val']['batches']
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': self.metrics['train']['loss'] / norm_tr,
                'tr_r2': self.metrics['train']['r2'] / norm_tr,
                'tr_fc': self.metrics['train']['fc'] / norm_tr,
                'val_loss': self.metrics['val']['loss'] / norm_val,
                'val_r2': self.metrics['val']['r2'] / norm_val,
                'val_fc': self.metrics['val']['fc'] / norm_val,
                'best_val_epoch': best_epoch}
        elif dtype == 'test':
            norm = self.metrics['test']['batches']
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'test_loss': self.metrics['test']['loss'] / norm,
                'test_r2': self.metrics['test']['r2'] / norm,
                'test_fc': self.metrics['test']['fc'] / norm}
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        return metric_row


class EMLoss(FitMethod):

    def __init__(self, model):
        metric_strs = ['batches', 'nll', 'prior']
        super().__init__(model, metric_strs)

    def calc_loss(self, data, device):
        ae = data['ae'][0]

        if 'neural' in data.keys():
            inputs = data['neural'][0]
        else:
            inputs=None

        low_d = self.model.get_low_d(ae)
        log_prior = self.model.log_prior()
        log_pi0 = self.model.log_pi0(low_d)
        log_Ps = self.model.log_transition_proba(inputs)
        lls = self.model.log_dynamics_proba(low_d, inputs)

        with torch.no_grad():
            expectations = hmm_expectations(log_pi0, log_Ps, lls, device)
        
        prior = log_prior #/ int(n_tng_batches)
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
        # test_row = {'epoch': i_epoch, 'batch_n': i_train,
        #          'tng_err': train_loss / i_train}
        # val_row = {
        #     'epoch': i_epoch,
        #     'batch_n': i_train,
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
        #     'batch_n': i_train,
        #     'test_err': test_loss / data_generator.num_tot_batches['test'],
        #     'test_ell': test_ell / data_generator.num_tot_batches['test'],
        #     'test_prior': test_prior / data_generator.num_tot_batches['test'],
        #     'test_log_likelihood': test_log_likelihood / data_generator.num_tot_batches['test'],
        #     'test_log_q': test_log_q / data_generator.num_tot_batches['test'],
        #     'best_val_epoch': best_val_epoch}


class EarlyStopping(object):
    """Stop training when a monitored quantity has stopped improving"""

    def __init__(self, history=10, min_epochs=10):
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
        if epoch > max(self.min_epochs, self.history) and curr_mean >= prev_mean:
            print('\n== early stopping criteria met; exiting train loop ==')
            print('training epochs: %d' % epoch)
            print('end cost: %04f' % curr_loss)
            print('best epoch: %i' % self.best_epoch)
            print('best cost: %04f\n' % self.best_loss)
            self.stopped_epoch = epoch
            self.should_stop = True


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
        weight_decay=hparams.get('l2_reg', 0),
        amsgrad=True)

    # enumerate batches on which validation metrics should be recorded
    best_val_loss = math.inf
    best_val_epoch = None
    best_val_model = None
    val_check_batch = np.linspace(
        data_generator.num_tot_batches['train'] * hparams['val_check_interval'],
        data_generator.num_tot_batches['train'] * (hparams['max_n_epochs']+1),
        int((hparams['max_n_epochs']+1) / hparams['val_check_interval'])).astype('int')

    # early stopping set-up
    if hparams['enable_early_stop']:
        early_stop = EarlyStopping(
            history=hparams['early_stop_history'],
            min_epochs=hparams['min_n_epochs'])

    model.version = exp.version  # for exporting latents
    i_epoch = 0
    for i_epoch in range(hparams['max_n_epochs']+1): # the 0th epoch has no training so we cycle through hparams['max_n_epochs'] of training epochs

        loss.reset_metrics('train')
        data_generator.reset_iterators('train')

        for i_train in tqdm(range(data_generator.num_tot_batches['train'])):

            model.train()

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

                exp.log(loss.create_metric_row(
                    'val', i_epoch, i_train, dataset, None,
                    best_epoch=best_val_epoch))
                exp.save()

            elif (i_train + 1) % data_generator.num_tot_batches['train'] == 0:
                # export training metrics at end of epoch
                exp.log(loss.create_metric_row(
                    'train', i_epoch, i_train, dataset, None))
                exp.save()

        if hparams['enable_early_stop']:
            early_stop.on_val_check(i_epoch, loss.get_loss('val'))
            if early_stop.should_stop:
                break

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
    if method == 'ae' and hparams['export_latents']:
        export_latents(data_generator, best_val_model)
    elif method == 'nll' and hparams['export_predictions']:
        export_predictions(data_generator, best_val_model)
