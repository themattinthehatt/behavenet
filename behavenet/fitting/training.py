import math
import copy
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import r2_score, accuracy_score
from behavenet.fitting.eval import export_latents
from behavenet.fitting.eval import export_predictions

# TODO: move to fitting module
# TODO: use epoch number as rng seed so that batches are served in a controllable way?
# TODO: make it easy to finish training if unexpectedly stopped


class FitMethod(object):
    """
    Base method for defining model losses and tracking loss metrics.

    Loss metrics are tracked for the aggregate dataset (potentially spanning
    multiple sessions) as well as session-specific metrics for easier
    downstream analyses.
    """

    def __init__(self, model, metric_strs, n_datasets=1):
        """
        Args:
            model (pt Model):
            metric_strs (list of strs): names of metrics to be tracked
            n_datasets (int): total number of datasets (sessions) served by
                data generator
        """
        self.model = model
        self.metrics = {}
        self.n_datasets = n_datasets
        dtype_strs = ['train', 'val', 'test', 'curr']

        # aggregate metrics over all datasets
        for dtype in dtype_strs:
            self.metrics[dtype] = {}
            for metric in metric_strs:
                self.metrics[dtype][metric] = 0

        # separate metrics by dataset
        if self.n_datasets > 1:
            self.metrics_by_dataset = []
            for dataset in range(self.n_datasets):
                self.metrics_by_dataset.append({})
                for dtype in dtype_strs:
                    self.metrics_by_dataset[dataset][dtype] = {}
                    for metric in metric_strs:
                        self.metrics_by_dataset[dataset][dtype][metric] = 0
        else:
            self.metrics_by_dataset = None

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def calc_loss(self, data, **kwargs):
        raise NotImplementedError

    def get_loss(self, dtype):
        """return loss aggregated over all datasets"""
        return self.metrics[dtype]['loss'] / self.metrics[dtype]['batches']

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None,
            by_dataset=False, *args, **kwargs):
        """
        Export metrics and other data (e.g. epoch) for logging train progress.

        Args:
            dtype (str): 'train' | 'val' | 'test'
            epoch (int): current training epoch
            batch (int): current training batch
            dataset (int): dataset id for current batch
            trial (int or NoneType): trial id within the current dataset
            best_epoch (int): best current training epoch
            by_dataset (bool, optional): `True` to return metrics for a
                specific dataset, `False` to return metrics aggregated over
                multiple datasets

        Returns:
            (dict)
        """

        if by_dataset and self.n_datasets > 1:
            loss = self.metrics_by_dataset[dataset][dtype]['loss'] \
                      / self.metrics_by_dataset[dataset][dtype]['batches']
        else:
            dataset = -1
            loss = self.metrics[dtype]['loss'] / self.metrics[dtype]['batches']

        if dtype == 'train':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': loss}
        elif dtype == 'val':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'val_loss': loss,
                'best_val_epoch': best_epoch}
        elif dtype == 'test':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'test_loss': loss}
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        return metric_row

    def reset_metrics(self, dtype):
        """Reset all metrics"""
        # reset aggregate metrics
        for key in self.metrics[dtype].keys():
            self.metrics[dtype][key] = 0
        # reset separated metrics
        if self.n_datasets > 1:
            for dataset in range(self.n_datasets):
                for key in self.metrics_by_dataset[dataset][dtype].keys():
                    self.metrics_by_dataset[dataset][dtype][key] = 0

    def update_metrics(self, dtype, dataset=None):
        """Update metrics for a specific dtype/dataset"""
        for key in self.metrics[dtype].keys():
            if self.metrics['curr'][key] is not None:
                # update aggregate methods
                self.metrics[dtype][key] += self.metrics['curr'][key]
                # update separated metrics
                if dataset is not None and self.n_datasets > 1:
                    self.metrics_by_dataset[dataset][dtype][key] += \
                        self.metrics['curr'][key]
                # reset current metrics
                self.metrics['curr'][key] = 0


class AELoss(FitMethod):
    """MSE loss for (non-variational) autoencoders"""

    def __init__(self, model, n_datasets=1):
        metric_strs = ['batches', 'loss']
        super().__init__(model, metric_strs, n_datasets=n_datasets)

    def calc_loss(self, data, dataset=0, **kwargs):
        """
        Calculate MSE loss for autoencoder. The batch is split into chunks if
        larger than a hard-coded `chunk_size` to keep memory requirements low;
        gradients are accumulated across all chunks before a gradient step is
        taken.

        Args:
            data (dict):
            dataset (int, optional)
        """

        y = data['images'][0]

        if 'masks' in data:
            masks = data['masks'][0]
        else:
            masks = None

        chunk_size = 200
        batch_size = y.shape[0]

        if batch_size > chunk_size:
            # split into chunks
            n_chunks = int(np.ceil(batch_size / chunk_size))
            loss_val = 0
            for chunk in range(n_chunks):
                indx_beg = chunk * chunk_size
                indx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                y_mu, _ = self.model(y[indx_beg:indx_end], dataset=dataset)
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
            y_mu, _ = self.model(y, dataset=dataset)
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
    """Negative log-likelihood loss for supervised models (en/decoders)"""

    def __init__(self, model, n_datasets=1):

        if n_datasets > 1:
            raise ValueError('NLLLoss only supports single datasets')

        metric_strs = ['batches', 'loss', 'r2', 'fc']
        super().__init__(model, metric_strs, n_datasets=n_datasets)

        # choose loss based on noise distribution of the model
        if self.model.hparams['noise_dist'] == 'gaussian':
            self._loss = nn.MSELoss()
        elif self.model.hparams['noise_dist'] == 'gaussian-full':
            from behavenet.fitting.losses import GaussianNegLogProb
            self._loss = GaussianNegLogProb()  # model holds precision mat
        elif self.model.hparams['noise_dist'] == 'poisson':
            self._loss = nn.PoissonNLLLoss(log_input=False)
        elif self.model.hparams['noise_dist'] == 'categorical':
            self._loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                '"%s" is not a valid noise dist' % self.model.hparams['noise_dist'])

    def calc_loss(self, data, **kwargs):
        """
        Calculate negative log-likelihood loss for supervised models, i.e.
        encoders and decoders. The batch is split into chunks if larger than a
        hard-coded `chunk_size` to keep memory requirements low; gradients are
        accumulated across all chunks before a gradient step is taken.

        Args:
            data (dict): signals are 1 x T x N
        """

        predictors = data[self.model.hparams['input_signal']][0]
        targets = data[self.model.hparams['output_signal']][0]

        max_lags = self.model.hparams['n_max_lags']

        chunk_size = 200
        batch_size = targets.shape[0]

        if batch_size > chunk_size:
            # split into chunks
            n_chunks = int(np.ceil(batch_size / chunk_size))
            outputs_all = []
            loss_val = 0
            for chunk in range(n_chunks):
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
            by_dataset=False, *args, **kwargs):

        norm = self.metrics[dtype]['batches']
        loss = self.metrics[dtype]['loss'] / norm
        r2 = self.metrics[dtype]['r2'] / norm
        fc = self.metrics[dtype]['fc'] / norm
        if dtype == 'train':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': loss,
                'tr_r2': r2,
                'tr_fc': fc}
        elif dtype == 'val':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'val_loss': loss,
                'val_r2': r2,
                'val_fc': fc,
                'best_val_epoch': best_epoch}
        elif dtype == 'test':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'test_loss': loss,
                'test_r2': r2,
                'test_fc': fc}
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        return metric_row


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
        if epoch > max(self.min_epochs, self.history) \
                and curr_mean >= prev_mean:
            print('\n== early stopping criteria met; exiting train loop ==')
            print('training epochs: %d' % epoch)
            print('end cost: %04f' % curr_loss)
            print('best epoch: %i' % self.best_epoch)
            print('best cost: %04f\n' % self.best_loss)
            self.stopped_epoch = epoch
            self.should_stop = True


def fit(hparams, model, data_generator, exp, method='em'):
    """
    Args:
        hparams (dict):
        model (pt Model):
        data_generator (ConcatSessionsGenerator object):
        exp (testube.Experiment object):
        method (str): 'ae' | 'nll'
    """

    # check inputs
    if method == 'ae':
        loss = AELoss(model, n_datasets=data_generator.n_datasets)
    elif method == 'nll':
        loss = NLLLoss(model, n_datasets=data_generator.n_datasets)
    else:
        raise ValueError('"%s" is an invalid fitting method' % method)

    # optimizer set-up
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
        data_generator.n_tot_batches['train'] * hparams['val_check_interval'],
        data_generator.n_tot_batches['train'] * (hparams['max_n_epochs']+1),
        int((hparams['max_n_epochs']+1) / hparams['val_check_interval'])).astype('int')

    # early stopping set-up
    if hparams['enable_early_stop']:
        early_stop = EarlyStopping(
            history=hparams['early_stop_history'],
            min_epochs=hparams['min_n_epochs'])
    else:
        early_stop = None

    i_epoch = 0
    for i_epoch in range(hparams['max_n_epochs'] + 1):
        # Note: the 0th epoch has no training (randomly initialized model is
        # evaluated) so we cycle through `max_n_epochs` training epochs

        if hparams['max_n_epochs'] < 10:
            print('epoch %i/%i' % (i_epoch, hparams['max_n_epochs']))
        elif hparams['max_n_epochs'] < 100:
            print('epoch %02i/%02i' % (i_epoch, hparams['max_n_epochs']))
        elif hparams['max_n_epochs'] < 1000:
            print('epoch %03i/%03i' % (i_epoch, hparams['max_n_epochs']))
        elif hparams['max_n_epochs'] < 10000:
            print('epoch %04i/%04i' % (i_epoch, hparams['max_n_epochs']))
        elif hparams['max_n_epochs'] < 100000:
            print('epoch %05i/%05i' % (i_epoch, hparams['max_n_epochs']))
        else:
            print('epoch %i/%i' % (i_epoch, hparams['max_n_epochs']))

        loss.reset_metrics('train')
        data_generator.reset_iterators('train')

        for i_train in tqdm(range(data_generator.n_tot_batches['train'])):

            model.train()

            # zero out gradients. Don't want gradients from previous iterations
            optimizer.zero_grad()

            # get next minibatch and put it on the device
            data, dataset = data_generator.next_batch('train')

            # call the appropriate loss function
            loss.calc_loss(data, dataset=dataset)
            loss.update_metrics('train', dataset=dataset)

            # step (evaluate untrained network on epoch 0)
            if i_epoch > 0:
                optimizer.step()

            # check validation according to schedule
            curr_batch = \
                (i_train + 1) + i_epoch * data_generator.n_tot_batches['train']
            if np.any(curr_batch == val_check_batch):

                loss.reset_metrics('val')
                data_generator.reset_iterators('val')
                model.eval()

                for i_val in range(data_generator.n_tot_batches['val']):

                    # get next minibatch and put it on the device
                    data, dataset = data_generator.next_batch('val')

                    # call the appropriate loss function
                    loss.calc_loss(data, dataset=dataset)
                    loss.update_metrics('val', dataset=dataset)

                # save best val model
                if loss.get_loss('val') < best_val_loss:
                    best_val_loss = loss.get_loss('val')
                    filepath = os.path.join(
                        hparams['expt_dir'], 'version_%i' % exp.version,
                        'best_val_model.pt')
                    torch.save(model.state_dict(), filepath)

                    model.hparams = None
                    best_val_model = copy.deepcopy(model)
                    model.hparams = hparams
                    best_val_model.hparams = hparams
                    best_val_epoch = i_epoch

                # export aggregated metrics on train/val data
                exp.log(loss.create_metric_row(
                    'train', i_epoch, i_train, -1, trial=-1,
                    by_dataset=False, best_epoch=best_val_epoch))
                exp.log(loss.create_metric_row(
                    'val', i_epoch, i_train, -1, trial=-1,
                    by_dataset=False, best_epoch=best_val_epoch))
                # export individual session metrics on train/val data
                if data_generator.n_datasets > 1:
                    for dataset in range(data_generator.n_datasets):
                        exp.log(loss.create_metric_row(
                            'train', i_epoch, i_train, dataset, trial=-1,
                            by_dataset=True, best_epoch=best_val_epoch))
                        exp.log(loss.create_metric_row(
                            'val', i_epoch, i_train, dataset, trial=-1,
                            by_dataset=True, best_epoch=best_val_epoch))
                exp.save()

            elif (i_train + 1) % data_generator.n_tot_batches['train'] == 0:
                # export training metrics at end of epoch

                # export aggregated metrics on train/val data
                exp.log(loss.create_metric_row(
                    'train', i_epoch, i_train, -1, trial=-1,
                    by_dataset=False, best_epoch=best_val_epoch))
                # export individual session metrics on train/val data
                if data_generator.n_datasets > 1:
                    for dataset in range(data_generator.n_datasets):
                        exp.log(loss.create_metric_row(
                            'train', i_epoch, i_train, dataset, trial=-1,
                            by_dataset=True, best_epoch=best_val_epoch))
                exp.save()

        if hparams['enable_early_stop']:
            early_stop.on_val_check(i_epoch, loss.get_loss('val'))
            if early_stop.should_stop:
                break

    # save out last model
    if hparams.get('save_last_model', False):
        filepath = os.path.join(
            hparams['expt_dir'], 'version_%i' % exp.version, 'last_model.pt')
        torch.save(model.state_dict(), filepath)

    # compute test loss
    if method == 'ae':
        test_loss = AELoss(
            best_val_model, n_datasets=data_generator.n_datasets)
    elif method == 'nll':
        test_loss = NLLLoss(
            best_val_model, n_datasets=data_generator.n_datasets)
    else:
        raise ValueError('"%s" is an invalid fitting method' % method)

    test_loss.reset_metrics('test')
    data_generator.reset_iterators('test')
    best_val_model.eval()

    for i_test in range(data_generator.n_tot_batches['test']):

        # get next minibatch and put it on the device
        data, dataset = data_generator.next_batch('test')

        # call the appropriate loss function
        test_loss.reset_metrics('test')
        test_loss.calc_loss(data, dataset=dataset)
        test_loss.update_metrics('test', dataset=dataset)

        # calculate metrics for each *batch* (rather than whole dataset)
        exp.log(test_loss.create_metric_row(
            'test', i_epoch, i_test, dataset, trial=data['batch_indx'].item(),
            by_dataset=True))

    exp.save()

    # export latents
    if method == 'ae' and hparams['export_latents']:
        print('exporting latents')
        export_latents(data_generator, best_val_model)
    elif method == 'nll' and hparams['export_predictions']:
        print('exporting predictions')
        export_predictions(data_generator, best_val_model)
