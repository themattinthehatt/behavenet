"""Functions and classes for fitting PyTorch models with stochastic gradient descent."""

import copy
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import r2_score, accuracy_score
from behavenet.fitting.eval import export_latents
from behavenet.fitting.eval import export_predictions

# TODO: make it easy to finish training if unexpectedly stopped
# TODO: save models at prespecified intervals (check ae recon as a func of epoch w/o retraining)

# to ignore imports for sphix-autoapidoc
__all__ = ['Logger', 'EarlyStopping', 'fit']


class Logger(object):
    """Base method for logging loss metrics.

    Loss metrics are tracked for the aggregate dataset (potentially spanning multiple sessions) as
    well as session-specific metrics for easier downstream plotting.
    """

    def __init__(self, n_datasets=1):
        """

        Parameters
        ----------
        n_datasets : :obj:`int`
            total number of datasets (sessions) served by data generator

        """
        self.metrics = {}
        self.n_datasets = n_datasets
        dtype_strs = ['train', 'val', 'test', 'curr']

        # aggregate metrics over all datasets
        for dtype in dtype_strs:
            self.metrics[dtype] = {}

        # separate metrics by dataset
        self.metrics_by_dataset = []
        if self.n_datasets > 1:
            for dataset in range(self.n_datasets):
                self.metrics_by_dataset.append({})
                for dtype in dtype_strs:
                    self.metrics_by_dataset[dataset][dtype] = {}

    def get_loss(self, dtype):
        """Return loss aggregated over all datasets.

        Parameters
        ----------
        dtype : :obj:`str`
            datatype to calculate loss for (e.g. 'train', 'val', 'test')

        """
        return self.metrics[dtype]['loss'] / self.metrics[dtype]['batches']

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None, by_dataset=False):
        """Export metrics and other data (e.g. epoch) for logging train progress.

        Parameters
        ----------
        dtype : :obj:`str`
            'train' | 'val' | 'test'
        epoch : :obj:`int`
            current training epoch
        batch : :obj:`int`
            current training batch
        dataset : :obj:`int`
            dataset id for current batch
        trial : :obj:`int` or :obj:`NoneType`
            trial id within the current dataset
        best_epoch : :obj:`int`, optional
            best current training epoch
        by_dataset : :obj:`bool`, optional
            :obj:`True` to return metrics for a specific dataset, :obj:`False` to return metrics
            aggregated over multiple datasets

        Returns
        -------
        :obj:`dict`
            aggregated metrics for current epoch/batch

        """

        if dtype == 'train':
            prefix = 'tr'
        elif dtype == 'val':
            prefix = 'val'
        elif dtype == 'test':
            prefix = 'test'
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        metric_row = {
            'epoch': epoch,
            'batch': batch,
            'trial': trial}

        if dtype == 'val':
            metric_row['best_val_epoch'] = best_epoch

        if by_dataset and self.n_datasets > 1:
            norm = self.metrics_by_dataset[dataset][dtype]['batches']
            for key, val in self.metrics_by_dataset[dataset][dtype].items():
                if key == 'batches':
                    continue
                metric_row['%s_%s' % (prefix, key)] = val / norm
        else:
            dataset = -1
            norm = self.metrics[dtype]['batches']
            for key, val in self.metrics[dtype].items():
                if key == 'batches':
                    continue
                metric_row['%s_%s' % (prefix, key)] = val / norm

        metric_row['dataset'] = dataset

        return metric_row

    def reset_metrics(self, dtype):
        """Reset all metrics.

        Parameters
        ----------
        dtype : :obj:`str`
            datatype to reset metrics for (e.g. 'train', 'val', 'test')

        """
        # reset aggregate metrics
        for key in self.metrics[dtype].keys():
            self.metrics[dtype][key] = 0
        # reset separated metrics
        for m in self.metrics_by_dataset:
            for key in m[dtype].keys():
                m[dtype][key] = 0

    def update_metrics(self, dtype, loss_dict, dataset=None):
        """Update metrics for a specific dtype/dataset.

        Parameters
        ----------
        dtype : :obj:`str`
            dataset type to update metrics for (e.g. 'train', 'val', 'test')
        loss_dict : :obj:`dict`
            key-value pairs correspond to all quantities that should be logged throughout training;
            dictionary returned by `loss` attribute of BehaveNet models
        dataset : :obj:`int` or :obj:`NoneType`, optional
            if :obj:`NoneType`, updates the aggregated metrics; if :obj:`int`, updates the
            associated dataset/session

        """
        metrics = {**loss_dict, 'batches': 1}  # append `batches` to loss_dict

        for key, val in metrics.items():

            # define metric for the first time if necessary
            if key not in self.metrics[dtype]:
                self.metrics[dtype][key] = 0

            # update aggregate methods
            self.metrics[dtype][key] += val

            # update separated metrics
            if dataset is not None and self.n_datasets > 1:
                if key not in self.metrics_by_dataset[dataset][dtype]:
                    self.metrics_by_dataset[dataset][dtype][key] = 0
                self.metrics_by_dataset[dataset][dtype][key] += val


class EarlyStopping(object):
    """Stop training when a monitored quantity has stopped improving.

    Adapted from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=10, min_epochs=10, delta=0):
        """

        Parameters
        ----------
        patience : :obj:`int`, optional
            number of previous checks to average over when checking for increase in loss
        min_epochs : :obj:`int`, optional
            minimum number of epochs for training
        delta : :obj:`float`, optional
            minimum change in monitored quantity to qualify as an improvement

        """

        self.patience = patience
        self.min_epochs = min_epochs
        self.delta = delta

        # keep track of `history` most recent losses
        # self.prev_losses = np.full(self.history, fill_value=np.nan)
        self.counter = 0
        self.best_epoch = 0
        self.best_loss = np.inf
        self.stopped_epoch = 0
        self.should_stop = False

    def on_val_check(self, epoch, curr_loss):
        """Check to see if loss has begun to increase on validation data for current epoch.

        Rather than returning the results of the check, this method updates the class attribute
        :obj:`should_stop`, which is checked externally by the fitting function.

        Parameters
        ----------
        epoch : :obj:`int`
            current epoch
        curr_loss : :obj:`float`
            current loss

        """

        # prev_mean = np.nanmean(self.prev_losses)
        # self.prev_losses = np.roll(self.prev_losses, 1)
        # self.prev_losses[0] = curr_loss
        # curr_mean = np.nanmean(self.prev_losses)

        # update best loss and epoch that it happened at
        if curr_loss < self.best_loss - self.delta:
            self.best_loss = curr_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        # check if smoothed loss is starting to increase; exit training if so
        if epoch > self.min_epochs and self.counter >= self.patience:
            print('\n== early stopping criteria met; exiting train loop ==')
            print('training epochs: %d' % epoch)
            print('end cost: %04f' % curr_loss)
            print('best epoch: %i' % self.best_epoch)
            print('best cost: %04f\n' % self.best_loss)
            self.stopped_epoch = epoch
            self.should_stop = True


def fit(hparams, model, data_generator, exp, method='ae'):
    """Fit pytorch models with stochastic gradient descent and early stopping.

    Training parameters such as min epochs, max epochs, and early stopping hyperparameters are
    specified in :obj:`hparams`.

    For more information on how model losses are calculated, see the classes that inherit from
    :class:`FitMethod`.

    For more information on how early stopping is implemented, see the class
    :class:`EarlyStopping`.

    Training progess is monitored by calculating the model loss on both training data and
    validation data. The training loss is calculated each epoch, and the validation loss is
    calculated according to the :obj:`hparams` key :obj:`'val_check_interval'`. For example, if
    :obj:`val_check_interval=5` then the validation loss is calculated every 5 epochs. If
    :obj:`val_check_interval=0.5` then the validation loss is calculated twice per epoch - after
    the first half of the batches have been processed, then again after all batches have been
    processed.

    Monitored metrics are saved in a csv file in the model directory. This logging is handled by
    the :obj:`testtube` package.

    At the end of training, model outputs (such as latents for autoencoder models, or predictions
    for decoder models) can optionally be computed and saved using the :obj:`hparams` keys
    :obj:`'export_latents'` or :obj:`'export_predictions'`, respectively.

    Parameters
    ----------
    hparams : :obj:`dict`
        model/training specification
    model : :obj:`PyTorch` model
        model to fit
    data_generator : :obj:`ConcatSessionsGenerator` object
        data generator to serve data batches
    exp : :obj:`test_tube.Experiment` object
        for logging training progress
    method : :obj:`str`
        specifies the type of loss - 'ae' | 'ae-msp' | 'nll' | 'conv-decoder'

    """

    # optimizer setup
    optimizer = torch.optim.Adam(
        model.get_parameters(), lr=hparams['learning_rate'], weight_decay=hparams.get('l2_reg', 0),
        amsgrad=True)

    # logging setup
    logger = Logger(n_datasets=data_generator.n_datasets)

    # early stopping setup
    if hparams['enable_early_stop']:
        early_stop = EarlyStopping(
            patience=hparams['early_stop_history'], min_epochs=hparams['min_n_epochs'])
    else:
        early_stop = None

    # enumerate batches on which validation metrics should be recorded
    best_val_loss = np.inf
    best_val_epoch = None
    best_val_model = None
    val_check_batch = np.linspace(
        data_generator.n_tot_batches['train'] * hparams['val_check_interval'],
        data_generator.n_tot_batches['train'] * (hparams['max_n_epochs']+1),
        int((hparams['max_n_epochs'] + 1) / hparams['val_check_interval'])).astype('int')

    # set random seeds for training
    if hparams.get('rng_seed_train', None) is None:
        rng_train = np.random.randint(0, 10000)
    else:
        rng_train = int(hparams['rng_seed_train'])
    torch.manual_seed(rng_train)
    np.random.seed(rng_train)

    expt_dir = os.path.join(hparams['expt_dir'], 'version_%i' % exp.version)

    i_epoch = 0
    best_model_saved = False
    for i_epoch in range(hparams['max_n_epochs'] + 1):
        # Note: the 0th epoch has no training (randomly initialized model is evaluated) so we cycle
        # through `max_n_epochs` training epochs

        print_epoch(i_epoch, hparams['max_n_epochs'])

        # control how data is batched to that models can be restarted from a particular epoch
        torch.manual_seed(rng_train + i_epoch)  # order of trials within sessions
        np.random.seed(rng_train + i_epoch)  # order of sessions

        logger.reset_metrics('train')
        data_generator.reset_iterators('train')

        for i_train in tqdm(range(data_generator.n_tot_batches['train'])):

            model.train()

            # zero out gradients. Don't want gradients from previous iterations
            optimizer.zero_grad()

            # get next minibatch and put it on the device
            data, dataset = data_generator.next_batch('train')

            # call the appropriate loss function
            loss_dict = model.loss(data, dataset=dataset, accumulate_grad=True)
            logger.update_metrics('train', loss_dict, dataset=dataset)

            # step (evaluate untrained network on epoch 0)
            if i_epoch > 0:
                optimizer.step()

            # check validation according to schedule
            curr_batch = (i_train + 1) + i_epoch * data_generator.n_tot_batches['train']
            if np.any(curr_batch == val_check_batch):

                logger.reset_metrics('val')
                data_generator.reset_iterators('val')
                model.eval()

                for i_val in range(data_generator.n_tot_batches['val']):

                    # get next minibatch and put it on the device
                    data, dataset = data_generator.next_batch('val')

                    # call the appropriate loss function
                    loss_dict = model.loss(data, dataset=dataset, accumulate_grad=False)
                    logger.update_metrics('val', loss_dict, dataset=dataset)

                # save best val model
                if logger.get_loss('val') < best_val_loss:
                    best_val_loss = logger.get_loss('val')
                    model.save(os.path.join(expt_dir, 'best_val_model.pt'))
                    best_model_saved = True

                    model.hparams = None
                    best_val_model = copy.deepcopy(model)
                    model.hparams = hparams
                    best_val_model.hparams = hparams
                    best_val_epoch = i_epoch

                # export aggregated metrics on val data
                exp.log(logger.create_metric_row(
                    'val', i_epoch, i_train, -1, trial=-1,
                    by_dataset=False, best_epoch=best_val_epoch))
                # export individual session metrics on val data
                if data_generator.n_datasets > 1:
                        exp.log(logger.create_metric_row(
                            'val', i_epoch, i_train, dataset, trial=-1,
                            by_dataset=True, best_epoch=best_val_epoch))
                exp.save()

            # export training metrics at end of epoch
            if (i_train + 1) % data_generator.n_tot_batches['train'] == 0:

                # export aggregated metrics on train data
                exp.log(logger.create_metric_row(
                    'train', i_epoch, i_train, -1, trial=-1,
                    by_dataset=False, best_epoch=best_val_epoch))
                # export individual session metrics on train/val data
                if data_generator.n_datasets > 1:
                    for dataset in range(data_generator.n_datasets):
                        exp.log(logger.create_metric_row(
                            'train', i_epoch, i_train, dataset, trial=-1,
                            by_dataset=True, best_epoch=best_val_epoch))
                exp.save()

        if hparams['enable_early_stop']:
            early_stop.on_val_check(i_epoch, logger.get_loss('val'))
            if early_stop.should_stop:
                break

    # save out last model as best model if no best model saved
    if not best_model_saved:
        model.save(os.path.join(expt_dir, 'best_val_model.pt'))

        model.hparams = None
        best_val_model = copy.deepcopy(model)
        model.hparams = hparams
        best_val_model.hparams = hparams

    # save out last model
    if hparams.get('save_last_model', False):
        model.save(os.path.join(expt_dir, 'last_model.pt'))

    # compute test loss
    logger.reset_metrics('test')
    data_generator.reset_iterators('test')
    best_val_model.eval()

    for i_test in range(data_generator.n_tot_batches['test']):

        # get next minibatch and put it on the device
        data, dataset = data_generator.next_batch('test')

        # call the appropriate loss function
        logger.reset_metrics('test')
        loss_dict = model.loss(data, dataset=dataset, accumulate_grad=False)
        logger.update_metrics('test', loss_dict, dataset=dataset)

        # calculate metrics for each *batch* (rather than whole dataset)
        exp.log(logger.create_metric_row(
            'test', i_epoch, i_test, dataset, trial=data['batch_idx'].item(), by_dataset=True))

    exp.save()

    # export latents
    if (method == 'ae' or method == 'ae-msp') and hparams['export_latents']:
        print('exporting latents')
        export_latents(data_generator, best_val_model)
    elif method == 'nll' and hparams['export_predictions']:
        print('exporting predictions')
        export_predictions(data_generator, best_val_model)
    elif method == 'conv-decoder' and hparams['export_predictions']:
        print('warning! exporting predictions not currently implemented for convolutional decoder')


def print_epoch(curr, total):
    """Pretty print epoch number."""
    if total < 10:
        print('epoch %i/%i' % (curr, total))
    elif total < 100:
        print('epoch %02i/%02i' % (curr, total))
    elif total < 1000:
        print('epoch %03i/%03i' % (curr, total))
    elif total < 10000:
        print('epoch %04i/%04i' % (curr, total))
    elif total < 100000:
        print('epoch %05i/%05i' % (curr, total))
    else:
        print('epoch %i/%i' % (curr, total))


class FitMethod(object):
    """Base method for defining model losses and tracking loss metrics.

    Loss metrics are tracked for the aggregate dataset (potentially spanning multiple sessions) as
    well as session-specific metrics for easier downstream plotting.
    """

    def __init__(self, model, metric_strs, n_datasets=1):
        """

        Parameters
        ----------
        model : :obj:`PyTorch` model
        metric_strs : :obj:`list` of :obj:`strs`
            names of metrics to be tracked, e.g. 'epoch', 'batch', 'train_loss', etc.
        n_datasets : :obj:`int`
            total number of datasets (sessions) served by data generator

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
        """Get all model parameters that have gradient updates turned on."""
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def calc_loss(self, data, **kwargs):
        """Calculate loss on data."""
        raise NotImplementedError

    def get_loss(self, dtype):
        """Return loss aggregated over all datasets.

        Parameters
        ----------
        dtype : :obj:`str`
            datatype to calculate loss for (e.g. 'train', 'val', 'test')

        """
        return self.metrics[dtype]['loss'] / self.metrics[dtype]['batches']

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None,
            by_dataset=False, *args, **kwargs):
        """Export metrics and other data (e.g. epoch) for logging train progress.

        Parameters
        ----------
        dtype : :obj:`str`
            'train' | 'val' | 'test'
        epoch : :obj:`int`
            current training epoch
        batch : :obj:`int`
            current training batch
        dataset : :obj:`int`
            dataset id for current batch
        trial : :obj:`int` or :obj:`NoneType`
            trial id within the current dataset
        best_epoch : :obj:`int`, optional
            best current training epoch
        by_dataset : :obj:`bool`, optional
            :obj:`True` to return metrics for a specific dataset, :obj:`False` to return metrics
            aggregated over multiple datasets

        Returns
        -------
        :obj:`dict`
            aggregated metrics for current epoch/batch

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
        """Reset all metrics.

        Parameters
        ----------
        dtype : :obj:`str`
            datatype to reset metrics for (e.g. 'train', 'val', 'test')

        """
        # reset aggregate metrics
        for key in self.metrics[dtype].keys():
            self.metrics[dtype][key] = 0
        # reset separated metrics
        if self.n_datasets > 1:
            for dataset in range(self.n_datasets):
                for key in self.metrics_by_dataset[dataset][dtype].keys():
                    self.metrics_by_dataset[dataset][dtype][key] = 0

    def update_metrics(self, dtype, dataset=None):
        """Update metrics for a specific dtype/dataset.

        Parameters
        ----------
        dtype : :obj:`str`
            dataset type to update metrics for (e.g. 'train', 'val', 'test')
        dataset : :obj:`int` or :obj:`NoneType`, optional
            if :obj:`NoneType`, updates the aggregated metrics; if :obj:`int`, updates the
            associated dataset/session

        """
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
    """MSE loss for non-variational autoencoders."""

    def __init__(self, model, n_datasets=1):
        metric_strs = ['batches', 'loss']
        super().__init__(model, metric_strs, n_datasets=n_datasets)

    def calc_loss(self, data, dataset=0, **kwargs):
        """Calculate MSE loss for autoencoder.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : :obj:`dict`
            batch of data; keys should include 'images' and 'masks', if necessary
        dataset : :obj:`int`, optional
            used for session-specific io layers

        """

        if self.model.hparams['device'] == 'cuda':
            data = {key: val.to('cuda') for key, val in data.items()}

        y = data['images'][0]

        if 'masks' in data:
            masks = data['masks'][0]
        else:
            masks = None

        if self.model.hparams['model_class'] == 'cond-ae':
            labels = data['labels'][0]
            if self.model.hparams['conditional_encoder']:
                # continuous labels transformed into 2d one-hot array as input to encoder
                labels_2d = data['labels_sc'][0]
            else:
                labels_2d = None
        else:
            labels = None
            labels_2d = None

        chunk_size = 200
        batch_size = y.shape[0]

        if batch_size > chunk_size:
            # split into chunks
            n_chunks = int(np.ceil(batch_size / chunk_size))
            loss_val = 0
            for chunk in range(n_chunks):
                idx_beg = chunk * chunk_size
                idx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                y_in = y[idx_beg:idx_end]
                labels_in = labels[idx_beg:idx_end] if labels is not None else None
                labels_2d_in = labels_2d[idx_beg:idx_end] if labels_2d is not None else None
                y_mu, _ = self.model(
                    y_in, dataset=dataset, labels=labels_in, labels_2d=labels_2d_in)
                if masks is not None:
                    loss = torch.mean(((y[idx_beg:idx_end] - y_mu) ** 2) * masks[idx_beg:idx_end])
                else:
                    loss = torch.mean((y[idx_beg:idx_end] - y_mu) ** 2)
                # compute gradients
                loss.backward()
                # get loss value (weighted by batch size)
                loss_val += loss.item() * (idx_end - idx_beg)
            loss_val /= y.shape[0]
        else:
            y_mu, _ = self.model(y, dataset=dataset, labels=labels, labels_2d=labels_2d)
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


class ConvDecoderLoss(AELoss):
    """MSE loss for convolutional decoders."""

    def __init__(self, model, n_datasets=1):
        super().__init__(model, n_datasets=n_datasets)

    def calc_loss(self, data, dataset=0, **kwargs):
        """Calculate MSE loss for convolutional decoder.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : :obj:`dict`
            batch of data; keys should include 'labels', 'images' and 'masks', if necessary
        dataset : :obj:`int`, optional
            used for session-specific io layers

        """

        if self.model.hparams['device'] == 'cuda':
            data = {key: val.to('cuda') for key, val in data.items()}

        y = data['images'][0]
        x = data['labels'][0]

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
                idx_beg = chunk * chunk_size
                idx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                y_mu = self.model(x[idx_beg:idx_end], dataset=dataset)
                if masks is not None:
                    loss = torch.mean(((y[idx_beg:idx_end] - y_mu) ** 2) * masks[idx_beg:idx_end])
                else:
                    loss = torch.mean((y[idx_beg:idx_end] - y_mu) ** 2)
                # compute gradients
                loss.backward()
                # get loss value (weighted by batch size)
                loss_val += loss.item() * (idx_end - idx_beg)
            loss_val /= y.shape[0]
        else:
            y_mu = self.model(x, dataset=dataset)
            # define loss
            if masks is not None:
                loss = torch.mean(((y - y_mu) ** 2) * masks)
            else:
                loss = torch.mean((y - y_mu) ** 2)
            # compute gradients
            loss.backward()
            # get loss value
            loss_val = loss.item()

        # store current metrics
        self.metrics['curr']['loss'] = loss_val
        self.metrics['curr']['batches'] = 1


class AEMSPLoss(FitMethod):
    """MSE + MSP loss for non-variational autoencoders."""

    def __init__(self, model, n_datasets=1):
        metric_strs = ['batches', 'loss', 'loss_mse', 'loss_msp', 'labels_r2']
        super().__init__(model, metric_strs, n_datasets=n_datasets)

    def calc_loss(self, data, dataset=0, **kwargs):
        """Calculate MSE loss for autoencoder with additional matrix subspace projection loss.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : :obj:`dict`
            batch of data; keys should include 'images', 'labels', and 'masks', if necessary
        dataset : :obj:`int`, optional
            used for session-specific io layers

        """

        if self.model.hparams['device'] == 'cuda':
            data = {key: val.to('cuda') for key, val in data.items()}

        y = data['images'][0]

        if 'masks' in data:
            masks = data['masks'][0]
        else:
            masks = None

        labels = data['labels'][0]
        if self.model.hparams['conditional_encoder']:
            # continuous labels transformed into 2d one-hot array as input to encoder
            labels_2d = data['labels_sc'][0]
        else:
            labels_2d = None

        chunk_size = 200
        batch_size = y.shape[0]

        if batch_size > chunk_size:
            # split into chunks
            n_chunks = int(np.ceil(batch_size / chunk_size))
            loss_val = 0
            loss_mse_val = 0
            loss_msp_val = 0
            for chunk in range(n_chunks):

                # push data through model
                idx_beg = chunk * chunk_size
                idx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                y_in = y[idx_beg:idx_end]
                labels_in = labels[idx_beg:idx_end] if labels is not None else None
                labels_2d_in = labels_2d[idx_beg:idx_end] if labels_2d is not None else None

                y_mu, z, labels_pred = self.model(y_in, dataset=dataset, labels_2d=labels_2d_in)

                # mse loss
                if masks is not None:
                    loss_mse = torch.mean(((y_in - y_mu) ** 2) * masks[idx_beg:idx_end])
                else:
                    loss_mse = torch.mean((y_in - y_mu) ** 2)

                # msp loss
                loss_msp = torch.mean((labels_in - labels_pred) ** 2) + torch.mean(
                    (z - torch.matmul(labels_pred, self.model.projection.weight)) ** 2)
                # ^NOTE: transpose on projection weights implicitly performed due to layer def

                # combine
                loss = loss_mse + self.model.hparams['msp_weight'] * loss_msp

                # compute gradients
                loss.backward()

                # get loss value (weighted by batch size)
                loss_val += loss.item() * (idx_end - idx_beg)
                loss_mse_val += loss_mse.item() * (idx_end - idx_beg)
                loss_msp_val += loss_msp.item() * (idx_end - idx_beg)

            loss_val /= y.shape[0]
            loss_mse_val /= y.shape[0]
            loss_msp_val /= y.shape[0]
            r2 = np.nan  # TODO
        else:
            y_mu, z, labels_pred = self.model(y, dataset=dataset, labels_2d=labels_2d)
            # mse loss
            if masks is not None:
                loss_mse = torch.mean(((y - y_mu) ** 2) * masks)
            else:
                loss_mse = torch.mean((y - y_mu) ** 2)
            # msp loss
            loss_msp = torch.mean((labels - labels_pred) ** 2) + torch.mean(
                    (z - torch.matmul(labels_pred, self.model.projection.weight)) ** 2)
            # ^NOTE: transpose on projection weights implicitly performed due to layer def
            # combine
            loss = loss_mse + self.model.hparams['msp_weight'] * loss_msp
            # compute gradients
            loss.backward()
            # get loss value
            loss_val = loss.item()
            loss_mse_val = loss_mse.item()
            loss_msp_val = loss_msp.item()

            # use variance-weighted r2s to ignore small-variance latents
            r2 = r2_score(
                labels.cpu().detach().numpy(),
                labels_pred.cpu().detach().numpy(),
                multioutput='variance_weighted')

        # store current metrics
        self.metrics['curr']['loss'] = loss_val
        self.metrics['curr']['loss_mse'] = loss_mse_val
        self.metrics['curr']['loss_msp'] = loss_msp_val
        self.metrics['curr']['labels_r2'] = r2
        self.metrics['curr']['batches'] = 1

    def create_metric_row(
            self, dtype, epoch, batch, dataset, trial, best_epoch=None, by_dataset=False,
            *args, **kwargs):
        """Export metrics and other data (e.g. epoch) for logging train progress.

        Parameters
        ----------
        dtype : :obj:`str`
            'train' | 'val' | 'test'
        epoch : :obj:`int`
            current training epoch
        batch : :obj:`int`
            current training batch
        dataset : :obj:`int`
            dataset id for current batch
        trial : :obj:`int` or :obj:`NoneType`
            trial id within the current dataset
        best_epoch : :obj:`int`, optional
            best current training epoch
        by_dataset : :obj:`bool`, optional
            :obj:`True` to return metrics for a specific dataset, :obj:`False` to return metrics
            aggregated over multiple datasets

        Returns
        -------
        :obj:`dict`
            aggregated metrics for current epoch/batch

        """

        norm = self.metrics[dtype]['batches']
        loss = self.metrics[dtype]['loss'] / norm
        loss_mse = self.metrics[dtype]['loss_mse'] / norm
        loss_msp = self.metrics[dtype]['loss_msp'] / norm
        labels_r2 = self.metrics[dtype]['labels_r2'] / norm
        if dtype == 'train':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'tr_loss': loss,
                'tr_loss_mse': loss_mse,
                'tr_loss_msp': loss_msp,
                'tr_r2': labels_r2}
        elif dtype == 'val':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'val_loss': loss,
                'val_loss_mse': loss_mse,
                'val_loss_msp': loss_msp,
                'val_r2': labels_r2,
                'best_val_epoch': best_epoch}
        elif dtype == 'test':
            metric_row = {
                'epoch': epoch,
                'batch': batch,
                'dataset': dataset,
                'trial': trial,
                'test_loss': loss,
                'test_loss_mse': loss_mse,
                'test_loss_msp': loss_msp,
                'test_r2': labels_r2}
        else:
            raise ValueError("%s is an invalid data type" % dtype)

        return metric_row


class NLLLoss(FitMethod):
    """Negative log-likelihood loss for supervised models (en/decoders)."""

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
            raise ValueError('"%s" is not a valid noise dist' % self.model.hparams['noise_dist'])

    def calc_loss(self, data, **kwargs):
        """Calculate negative log-likelihood loss for supervised models.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : :obj:`dict`
            signals are of shape (1, time, n_channels)

        """

        if self.model.hparams['device'] == 'cuda':
            data = {key: val.to('cuda') for key, val in data.items()}

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
                idx_beg = np.max([chunk * chunk_size - max_lags, 0])
                idx_end = np.min([(chunk + 1) * chunk_size + max_lags, batch_size])
                outputs, precision = self.model(predictors[idx_beg:idx_end])
                # define loss on allowed window of data
                if self.model.hparams['noise_dist'] == 'gaussian-full':
                    loss = self._loss(
                        outputs[max_lags:-max_lags],
                        targets[idx_beg:idx_end][max_lags:-max_lags],
                        precision[max_lags:-max_lags])
                else:
                    loss = self._loss(
                        outputs[max_lags:-max_lags],
                        targets[idx_beg:idx_end][max_lags:-max_lags])
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
            self, dtype, epoch, batch, dataset, trial, best_epoch=None, by_dataset=False,
            *args, **kwargs):
        """Export metrics and other data (e.g. epoch) for logging train progress.

        Parameters
        ----------
        dtype : :obj:`str`
            'train' | 'val' | 'test'
        epoch : :obj:`int`
            current training epoch
        batch : :obj:`int`
            current training batch
        dataset : :obj:`int`
            dataset id for current batch
        trial : :obj:`int` or :obj:`NoneType`
            trial id within the current dataset
        best_epoch : :obj:`int`, optional
            best current training epoch
        by_dataset : :obj:`bool`, optional
            :obj:`True` to return metrics for a specific dataset, :obj:`False` to return metrics
            aggregated over multiple datasets

        Returns
        -------
        :obj:`dict`
            aggregated metrics for current epoch/batch

        """

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