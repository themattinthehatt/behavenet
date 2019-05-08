from behavenet.core import expected_log_likelihood, log_sum_exp
from behavenet.utils import export_latents, export_predictions
# from behavenet.messages import hmm_expectations, hmm_sample
from tqdm import tqdm
import numpy as np
import math
import copy
import os
import pickle
import time
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from behavenet.training import EarlyStopping
from behavenet.training import FitMethod


class FitMethodTF(FitMethod):

    def __init__(self, model, metric_strs, optimizer, next_batch, objective):

        super().__init__(model, metric_strs)

        self.optimizer = optimizer
        self.next_batch = next_batch
        self.objective = objective

        # set up this loss object so that we can accumulate gradients and take
        # gradient steps like pytorch
        # https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow

        # Retrieve all trainable variables you defined in your graph
        self.tvs = tf.trainable_variables()

        # self.accum_vars = [
        #     tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for
        #     tv in self.tvs]
        # variables for accumulating the gradient
        self.accum_vars = [
            tf.Variable(tf.zeros_like(tv), trainable=False) for tv in self.tvs]

        # op to zero out the gradient
        self.zero_grad_op = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]

        # op to compute the gradients
        self.gvs = self.optimizer.compute_gradients(objective, self.tvs)

        # op to accumulate gradients
        self.accum_ops = [
            self.accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.gvs)]

        # op to take training step
        self.step_op = self.optimizer.apply_gradients(
            [(self.accum_vars[i], gv[1]) for i, gv in enumerate(self.gvs)])

    def zero_grad(self, sess):
        sess.run(self.zero_grad_op)

    def calc_loss(self, data, sess):

        y = np.transpose(
            data[self.model.hparams['signals']][0].cpu().detach().numpy(),
            [0, 2, 3, 1])

        chunk_size = 200
        batch_size = y.shape[0]

        if batch_size > chunk_size:
            # split into chunks
            num_chunks = int(np.ceil(batch_size / chunk_size))
            loss_val = 0
            for chunk in range(num_chunks):
                indx_beg = chunk * chunk_size
                indx_end = np.min([(chunk + 1) * chunk_size, batch_size])

                feed_dict = {self.next_batch: y[indx_beg:indx_end]}
                # accumulate gradients
                self.backward(sess, feed_dict)
                # get loss value (weighted by batch size)
                loss_val_ = sess.run(self.objective, feed_dict=feed_dict)
                loss_val += loss_val_ * (indx_end - indx_beg)

                # compute loss and gradients
                # loss_val_, _ = sess.run(
                #     [self.objective, self.gvs], feed_dict=feed_dict)
                #
            loss_val /= y.shape[0]
        else:
            feed_dict = {self.next_batch: y}
            # accumulate gradients
            self.backward(sess, feed_dict=feed_dict)
            # compute loss
            loss_val_ = sess.run(self.objective, feed_dict=feed_dict)
            loss_val = loss_val_

        # store current metrics
        self.metrics['curr']['loss'] = loss_val
        self.metrics['curr']['batches'] = 1

    def backward(self, sess, feed_dict):
        sess.run(self.accum_ops, feed_dict=feed_dict)

    def step(self, sess):
        # Define the training step (part with variable value update)
        sess.run(self.step_op)


class AELoss(FitMethodTF):

    def __init__(self, model, optimizer, next_batch, objective):
        metric_strs = ['batches', 'loss']
        super().__init__(model, metric_strs, optimizer, next_batch, objective)


def fit(
        hparams, model, data_generator, exp, method='ae'):
    """
    Args:
        hparams:
        model:
        data_generator:
        exp:
        method:
    """

    # Optimizer set-up
    # optimizer = tf.train.AdamOptimizer(hparams['learning_rate']).minimize(
    #     loss.get_loss())
    optimizer = tf.train.AdamOptimizer(hparams['learning_rate'])

    # get model input/output
    next_batch = tf.placeholder(
        dtype=tf.float32,
        shape=(
            None,
            model.hparams['y_pixels'],
            model.hparams['x_pixels'],
            model.hparams['n_input_channels']))
    model.forward(next_batch)

    if method == 'ae':
        objective = tf.reduce_mean(tf.squared_difference(
            next_batch, model.y))
        loss = AELoss(model, optimizer, next_batch, objective)
    else:
        raise ValueError('"%s" is an invalid fitting method' % method)

    # enumerate batches on which validation metrics should be recorded
    best_val_loss = math.inf
    best_val_epoch = None
    best_val_model = None
    val_check_batch = np.linspace(
        data_generator.num_tot_batches['train'] * hparams['val_check_interval'],
        data_generator.num_tot_batches['train'] * (hparams['max_nb_epochs']+1),
        int((hparams['max_nb_epochs']+1) / hparams['val_check_interval'])).astype('int')

    # early stopping set-up
    if hparams['enable_early_stop']:
        early_stop = EarlyStopping(
            history=hparams['early_stop_history'],
            min_epochs=hparams['min_nb_epochs'])

    saver = tf.train.Saver()
    use_gpu = True
    if use_gpu:
        sess_config = tf.ConfigProto(
            device_count={'GPU': 1},
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
        sess_config.gpu_options.allow_growth = True
        # https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    model.version = exp.version  # for exporting latents
    i_epoch = 0
    for i_epoch in range(hparams['max_nb_epochs']+1): # the 0th epoch has no training so we cycle through hparams['max_nb_epochs'] of training epochs

        loss.reset_metrics('train')
        data_generator.reset_iterators('train')

        for i_train in tqdm(range(data_generator.num_tot_batches['train'])):

            # Zero out gradients. Don't want gradients from previous iterations
            loss.zero_grad(sess)

            # Get next minibatch and put it on the device
            data, dataset = data_generator.next_batch('train')

            # Call the appropriate loss function
            loss.calc_loss(data, sess)
            loss.update_metrics('train')

            # Step (evaluate untrained network on epoch 0)
            if i_epoch > 0:
                loss.step(sess)

            # Check validation according to schedule
            curr_batch = (i_train + 1) + i_epoch * data_generator.num_tot_batches['train']
            if np.any(curr_batch == val_check_batch):

                loss.reset_metrics('val')
                data_generator.reset_iterators('val')

                for i_val in range(data_generator.num_tot_batches['val']):

                    # Get next minibatch and put it on the device
                    data, dataset = data_generator.next_batch('val')

                    # Call the appropriate loss function
                    loss.calc_loss(data, sess)
                    loss.update_metrics('val')

                # Save best val model
                if loss.get_loss('val') < best_val_loss:
                    best_val_loss = loss.get_loss('val')
                    filepath = os.path.join(
                        hparams['results_dir'], 'test_tube_data',
                        hparams['experiment_name'],
                        'version_%i' % exp.version,
                        'best_val_model.pt')
                    saver.save(sess, filepath)

                    # model.hparams = None
                    # best_val_model = copy.deepcopy(model)
                    # model.hparams = hparams
                    # best_val_model.hparams = hparams
                    best_val_model = filepath
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
    saver.save(sess, filepath)

    # Compute test loss
    saver.restore(sess, best_val_model)

    loss.reset_metrics('test')
    data_generator.reset_iterators('test')

    for i_test in range(data_generator.num_tot_batches['test']):

        # Get next minibatch and put it on the device
        data, dataset = data_generator.next_batch('test')

        # Call the appropriate loss function
        loss.reset_metrics('test')
        loss.calc_loss(data, sess)
        loss.update_metrics('test')

        # calculate metrics for each batch
        exp.log(loss.create_metric_row(
            'test', i_epoch, i_test, dataset, data['batch_indx'].item()))
    exp.save()

    # export latents
    # if method == 'ae' and hparams['export_latents']:
    #     export_latents(data_generator, best_val_model)
    # elif method == 'nll' and hparams['export_predictions']:
    #     export_predictions(data_generator, best_val_model)
