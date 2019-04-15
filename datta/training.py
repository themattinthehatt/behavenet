from datta.messages import hmm_expectations, hmm_sample
from datta.core import expected_log_likelihood, EarlyStopping, log_sum_exp
from tqdm import tqdm
import torch
from torch import nn, optim
import numpy as np
import math
import copy
import time 

def vae_loss(model, data,random_draw=1):

    total_loss=0
    NLL_loss=0
    KL_loss=0
    MSE_loss=0
    for i_session in range(data['depth'].shape[0]):

        y = data['depth'][i_session].unsqueeze(1)

        y_mu, y_var, h_mu, h_var = model(y,random_draw)

        NLL = -torch.sum((-0.5 * math.log(2 * math.pi) - 0.5 * torch.log(y_var)-0.5 * (y - y_mu)**2 / y_var)) #*mask[i_session] ) # cable pixels will be multiplied with mask 0 and not included in sum

        KLD = -.5*torch.sum(1+torch.log(h_var)-h_mu.pow(2)-h_var)

        MSE = torch.mean((y-y_mu)**2)
        if model.hparams.use_KL == 1:
            loss  =  (NLL+KLD)/y_mu.shape[0] 
        elif model.hparams.use_KL == 0:
            loss = (NLL)/y_mu.shape[0]

        loss.backward()

        # Collect losses
        total_loss += loss.item()/data['depth'].shape[0]
        NLL_loss += NLL.item()/y_mu.shape[0]/data['depth'].shape[0]
        KL_loss += KLD.item()/y_mu.shape[0]/data['depth'].shape[0]
        MSE_loss += MSE.item()/data['depth'].shape[0]
    return loss, total_loss, NLL_loss, KL_loss, MSE_loss

def ae_loss(model, data):

    total_loss=0
    NLL_loss=0
    MSE_loss=0
    for i_session in range(data['depth'].shape[0]):

        y = data['depth'][i_session].unsqueeze(1)

        y_mu, y_var, h_mu, = model(y)

        NLL = -torch.sum((-0.5 * math.log(2 * math.pi) - 0.5 * torch.log(y_var)-0.5 * (y - y_mu)**2 / y_var)) #*mask[i_session] ) # cable pixels will be multiplied with mask 0 and not included in sum

        MSE = torch.mean((y-y_mu)**2)

        if model.hparams.loss_type == 'mse':
            loss  =  MSE
        elif model.hparams.loss_type == 'nll':
            loss = (NLL)/y_mu.shape[0]
        else:
            raise NotImplementedError

        loss.backward()

        # Collect losses
        total_loss += loss.item()/data['depth'].shape[0]
        NLL_loss += NLL.item()/y_mu.shape[0]/data['depth'].shape[0]
        MSE_loss += MSE.item()/data['depth'].shape[0]

    return loss, total_loss, NLL_loss, MSE_loss

def em_loss(model, data, nb_tng_batches,device):
    """
    L(theta) = E_q(z) [log p(x, z; theta) - log q(z | x) | x]
    """

    if model.hparams.low_d_type == 'vae':
        this_data = data['depth'].unsqueeze(2)
    elif model.hparams.low_d_type == 'pca':
        this_data = data['pca_score']
    else:
        raise NotImplementedError


    total_loss=0
    prior_loss=0
    NLL_loss=0
    for i_session in range(this_data.shape[0]):

        # Get the sufficient statistics from the model and data
        low_d = model.get_low_d(this_data[i_session])
        log_prior = model.log_prior()
        log_pi0 = model.log_pi0(low_d)
        log_Ps = model.log_transition_proba(low_d)
        lls = model.log_dynamics_proba(low_d)

        # Run message passing to compute expected states for this minibatch
        with torch.no_grad():
            expectations = hmm_expectations(log_pi0, log_Ps, lls, device)

        # Compute the expected log probability
        # TODO: Figure this out...

        prior = log_prior / int(nb_tng_batches*this_data.shape[0])
        likelihood = expected_log_likelihood(expectations, log_pi0, log_Ps, lls)

        elp = prior + likelihood
       # elp = log_prior / int(nb_tng_batches)
      #  elp += expected_log_likelihood(expectations, log_pi0, log_Ps, lls)
        if np.isnan(elp.item()):
            raise Exception("Expected log probability is not finite")

        # Loss is negative expected log probability
        loss = -elp / low_d.shape[0] / low_d.shape[1]/this_data.shape[0]

        loss.backward()

        # Collect losses
        total_loss += loss.item()
        prior_loss += -prior.item()/this_data.shape[0]/low_d.shape[0] / low_d.shape[1]
        NLL_loss += -likelihood.item()/this_data.shape[0]/low_d.shape[0] / low_d.shape[1]

    return loss, total_loss, prior_loss, NLL_loss


def svi_loss(model, data, variational_posterior, nb_tng_batches, device, N_samples=1):
    """
    L(theta, phi) = E_q(x; phi)[ 
                        E_q(z | x) [log p(x, z; theta) - log q(z | x) | x] 
                    + log p(y | x, theta) - log q(x; phi)]

    The middle line is an "inner ELBO" over the discrete states z.  

    We use the importance weighted Monte Carlo estimate of L.  
    """

    total_loss=0
    prior_loss=0
    ell_loss=0
    log_likelihood_loss=0
    log_q_loss=0
    for i_session in range(data['depth'].shape[0]):

        this_data = data['depth'][i_session].unsqueeze(1) # T x 1 x sqrt(P) x sqrt(P)

        elbos = torch.zeros(N_samples)
        for smpl in range(N_samples):
            # Sample the variational posterior
            states = variational_posterior.sample(this_data) # T x H

            # Get the sufficient statistics from the model and data
            log_prior = model.log_prior()
            log_pi0 = model.log_pi0(states)
            log_Ps = model.log_transition_proba(states)
            log_dynamics_proba = model.log_dynamics_proba(states)

            # Run message passing to compute expected discrete states for this minibatch
            # Don't compute gradients wrt states/model parameters backward through `expectations`.
            with torch.no_grad():
                expectations = hmm_expectations(log_pi0, log_Ps, log_dynamics_proba, device)

            # Compute the inner ELBO, E_q(z | x) [log p(x, z; theta) - log q(z | x) | x]
            ell = expected_log_likelihood(expectations, log_pi0, log_Ps, log_dynamics_proba)
            if np.isnan(ell.item()):
                raise Exception("Expected log likelihood is not finite")

            # Compute the emission likelihood for these states log p(y | x, theta)
            log_likelihood = torch.sum(model.log_emission_proba(this_data, states))

            # Compute the variational entropy
            log_q = variational_posterior.log_proba(this_data, states)

            # Loss is negative expected log probability
            prior = log_prior / int(nb_tng_batches*data['depth'].shape[0])
            elbos[smpl] = prior + ell + log_likelihood - log_q
        
        # Compute the importance weighted estimate of the ELBO
        if N_samples > 1:
            elbo = log_sum_exp(elbos) - torch.log(torch.tensor(N_samples).float())
        else:
            elbo = elbos[0]

        # Normalize for optimization 
        loss = -elbo / this_data.shape[0] / this_data.shape[2]/this_data.shape[3]/data['depth'].shape[0]
        loss.backward()

        # Collect losses
        total_loss += loss.item()
        prior_loss += -prior.item() / this_data.shape[0] / this_data.shape[2]/this_data.shape[3]/data['depth'].shape[0]
        ell_loss += -ell.item() / this_data.shape[0] / this_data.shape[2]/this_data.shape[3]/data['depth'].shape[0]
        log_likelihood_loss += -log_likelihood.item() / this_data.shape[0] / this_data.shape[2]/this_data.shape[3]/ data['depth'].shape[0]
        log_q_loss += log_q.item() / this_data.shape[0] / this_data.shape[2]/this_data.shape[3]/data['depth'].shape[0]

    return loss, total_loss, prior_loss, ell_loss, log_likelihood_loss, log_q_loss


def fit(hparams, model, data_generator,  exp, 
        method="em",
        variational_posterior=None):
    

    # Check inputs
    assert method in ("em", "svi", "vae","ae")
    if method == "svi":
        assert variational_posterior is not None

    # Extract parameters
    if method == "svi":
        parameters = list(filter(lambda p: p.requires_grad, model.parameters())) + list(filter(lambda p: p.requires_grad, variational_posterior.parameters()))
    else:
        parameters = filter(lambda p: p.requires_grad, model.parameters())  

    optimizer = torch.optim.Adam(parameters, lr=hparams.learning_rate)

    # Early stopping set-up
    best_val_loss = math.inf
    nb_epochs_since_check = 0

    if hparams.val_check_interval < 1:
        val_check_batch = np.linspace(data_generator.n_max_train_batches*hparams.val_check_interval,data_generator.n_max_train_batches,1/hparams.val_check_interval).astype('int')
    elif hparams.val_check_interval % 1 ==0 :
        val_check_batch = data_generator.n_max_train_batches*hparams.val_check_interval
    else:
        raise Exception('ERROR: val check interval should be below 1 or an integer')
    
    if hparams.enable_early_stop:
        early_stop = EarlyStopping(min_fraction=hparams.early_stop_fraction, patience=hparams.early_stop_patience)
    should_stop = False

    for i_epoch in range(hparams.max_nb_epochs):

        train_loss = 0
        model.train()
        for i_train in tqdm(range(data_generator.n_max_train_batches)):

            # Zero out gradients. Don't want gradients from previous iterations
            optimizer.zero_grad()

            # Get next minibatch and put it on the device
            data = data_generator.next_train_batch()

            # Call the appropriate loss function
            if method == "em":
                # TO DO: are gradients working how we want? think so based on toy examples
                loss, total_loss, prior, NLL = em_loss(model, data, data_generator.n_max_train_batches,hparams.device)
            elif method == "svi":
                loss, total_loss, _, _, _, _ = svi_loss(model, data, variational_posterior, data_generator.n_max_train_batches,hparams.device)
            elif method == "vae":
                loss, total_loss, _, _, _ = vae_loss(model, data)
            elif method == "ae":
                loss, total_loss, _, _ = ae_loss(model, data)
            else:
                raise Exception("Invalid loss function!")

            train_loss += total_loss

            # Step
            if i_epoch > 0:
                optimizer.step()

            # Check validation according to schedule

            if np.any((i_train+1)+nb_epochs_since_check*data_generator.n_max_train_batches==val_check_batch):
                
                data_generator.reset_iterators('val')
                model.eval()
                for i_val in range(data_generator.n_max_val_batches):

                    # Get next minibatch and put it on the device
                    data = data_generator.next_val_batch()

                    # Call the appropriate loss function
                    if method == "em":
                        if i_val == 0:
                            val_loss=0
                            val_NLL=0
                            val_prior=0
                        loss, total_loss, prior_loss, NLL = em_loss(model, data, data_generator.n_max_train_batches,hparams.device)
                        val_NLL += NLL
                        val_prior += prior_loss
                        val_loss += total_loss
                    elif method == "svi":
                        if i_val == 0:
                            val_loss=0
                            val_ell=0
                            val_prior=0
                            val_log_likelihood=0
                            val_log_q=0
                        loss, total_loss, prior_loss, ell_loss, log_likelihood_loss, log_q_loss = svi_loss(model, data, variational_posterior, data_generator.n_max_train_batches,hparams.device)
                        val_loss += total_loss
                        val_ell += ell_loss
                        val_prior += prior_loss
                        val_log_likelihood += log_likelihood_loss
                        val_log_q += log_q_loss
                    elif method == "vae":
                        if i_val == 0:
                            val_loss=0
                            val_NLL=0
                            val_KL = 0
                            val_MSE = 0
                        loss, total_loss, NLL, KL, MSE = vae_loss(model, data)
                        val_NLL += NLL
                        val_KL += KL
                        val_MSE += MSE
                        val_loss += total_loss
                    elif method == "ae":
                        if i_val == 0:
                            val_loss=0
                            val_NLL=0
                            val_MSE = 0
                        loss, total_loss, NLL, MSE = ae_loss(model, data)
                        val_NLL += NLL
                        val_MSE += MSE
                        val_loss += total_loss
                    else:
                        raise Exception("Invalid loss function!")


                # Save best val model
                if val_loss/data_generator.n_max_val_batches < best_val_loss:
                    best_val_loss = val_loss/data_generator.n_max_val_batches
                    filepath = hparams.tt_save_path + '/test_tube_data/' + hparams.model_name + '/version_' + str(exp.version) + '/best_val_model.pt'
                    torch.save(model.state_dict(),filepath)
                    model.hparams=None
                    best_val_model = copy.deepcopy(model)
                    model.hparams=hparams
                    best_val_model.hparams=hparams
                    best_val_epoch = i_epoch

                if hparams.enable_early_stop:
                    stop_train = early_stop.on_val_check(i_epoch,val_loss/i_val)
                    met_min_epochs = i_epoch > hparams.min_nb_epochs
                    should_stop = stop_train and met_min_epochs
                    if should_stop:
                        break
                # if np.mod(i_epoch,5)==0:
                #     filepath = hparams.tt_save_path + '/test_tube_data/' + hparams.model_name + '/version_' + str(exp.version) + '/model_epoch_'+str(i_epoch)+'.pt'
                #     torch.save(model.state_dict(),filepath)

                nb_epochs_since_check=0
                if method == "vae":
                    val_row = {'epoch': i_epoch, 'batch_nb': i_train, 'tng_err': train_loss/(i_train+1), 'val_err':val_loss/data_generator.n_max_val_batches,'val_NLL':val_NLL/data_generator.n_max_val_batches,'val_KL':val_KL/data_generator.n_max_val_batches,'val_MSE':val_MSE/data_generator.n_max_val_batches,'best_val_epoch':best_val_epoch}
                elif method == "ae":
                    val_row = {'epoch': i_epoch, 'batch_nb': i_train, 'tng_err': train_loss/(i_train+1), 'val_err':val_loss/data_generator.n_max_val_batches,'val_NLL':val_NLL/data_generator.n_max_val_batches,'val_MSE':val_MSE/data_generator.n_max_val_batches,'best_val_epoch':best_val_epoch}
                elif method == "em":
                    val_row = {'epoch': i_epoch, 'batch_nb': i_train, 'tng_err': train_loss/(i_train+1), 'val_err':val_loss/data_generator.n_max_val_batches,'val_NLL':val_NLL/data_generator.n_max_val_batches,'val_prior':val_prior/data_generator.n_max_val_batches,'best_val_epoch':best_val_epoch}
                elif method == 'svi':
                    val_row = {'epoch': i_epoch, 'batch_nb': i_train, 'tng_err': train_loss/(i_train+1), 'val_err':val_loss/data_generator.n_max_val_batches,'val_ell':val_ell/data_generator.n_max_val_batches,'val_prior':val_prior/data_generator.n_max_val_batches,'val_log_likelihood':val_log_likelihood/data_generator.n_max_val_batches,'val_log_q':val_log_q/data_generator.n_max_val_batches,'best_val_epoch':best_val_epoch}
 
                exp.add_metric_row(val_row)
                exp.save()

            elif (i_train+1) % data_generator.n_max_train_batches ==0:

                exp.add_metric_row({'epoch': i_epoch, 'batch_nb': i_train, 'tng_err': train_loss/i_train})
                exp.save()
                nb_epochs_since_check+=1

        if should_stop:
            break


    # Compute test loss
    data_generator.reset_iterators('test')   
    model.eval()

    for i_test in range(data_generator.n_max_test_batches):

        # Get next minibatch and put it on the device
        data = data_generator.next_test_batch()

        # Call the appropriate loss function
        if method == "em":
            if i_test == 0:
                test_loss=0
                test_NLL=0
                test_prior = 0
            loss, total_loss, prior_loss, NLL  = em_loss(best_val_model, data, data_generator.n_max_train_batches,hparams.device)
            test_NLL += NLL
            test_prior += prior_loss
            test_loss += total_loss
        elif method == "svi":
            if i_test == 0:
                test_loss=0
                test_ell=0
                test_prior=0
                test_log_likelihood=0
                test_log_q=0
            loss, total_loss, prior_loss, ell_loss, log_likelihood_loss, log_q_loss = svi_loss(best_val_model, data, variational_posterior, data_generator.n_max_train_batches,hparams.device)
            test_loss += total_loss
            test_ell += ell_loss
            test_prior += prior_loss
            test_log_likelihood += log_likelihood_loss
            test_log_q += log_q_loss
        elif method == "vae":
            if i_test == 0:
                test_loss=0
                test_NLL=0
                test_KL = 0
                test_MSE = 0
            loss, total_loss, NLL, KL, MSE = vae_loss(best_val_model, data, random_draw=0)
            test_NLL += NLL
            test_KL += KL
            test_MSE += MSE
            test_loss += total_loss
        elif method == "ae":
            if i_test == 0:
                test_loss=0
                test_NLL=0
                test_MSE = 0
            loss, total_loss, NLL, MSE = vae_loss(best_val_model, data)
            test_NLL += NLL
            test_MSE += MSE
            test_loss += total_loss
        else:
            raise Exception("Invalid loss function!")

   
    if method == "vae":
        test_row = {'epoch': i_epoch, 'batch_nb': i_train, 'test_err':test_loss/data_generator.n_max_test_batches,'test_NLL':test_NLL/data_generator.n_max_test_batches,'test_KL':test_KL/data_generator.n_max_test_batches,'test_MSE':test_MSE/data_generator.n_max_test_batches,'best_val_epoch':best_val_epoch}
    elif method == "ae":
        test_row = {'epoch': i_epoch, 'batch_nb': i_train, 'test_err':test_loss/data_generator.n_max_test_batches,'test_NLL':test_NLL/data_generator.n_max_test_batches,'test_MSE':test_MSE/data_generator.n_max_test_batches,'best_val_epoch':best_val_epoch}
    elif method == "em":
        test_row = {'epoch': i_epoch, 'batch_nb': i_train, 'test_err':test_loss/data_generator.n_max_test_batches,'test_NLL':test_NLL/data_generator.n_max_test_batches,'test_prior':test_prior/data_generator.n_max_test_batches,'best_val_epoch':best_val_epoch}
    elif method == 'svi':
        test_row = {'epoch': i_epoch, 'batch_nb': i_train, 'test_err':test_loss/data_generator.n_max_test_batches,'test_ell':test_ell/data_generator.n_max_test_batches,'test_prior':test_prior/data_generator.n_max_test_batches,'test_log_likelihood':test_log_likelihood/data_generator.n_max_test_batches,'test_log_q':test_log_q/data_generator.n_max_test_batches,'best_val_epoch':best_val_epoch}
  
    exp.add_metric_row(test_row)
    exp.save()

    filepath = hparams.tt_save_path + '/test_tube_data/' + hparams.model_name + '/version_' + str(exp.version) + '/last_model.pt'
    torch.save(model.state_dict(),filepath)
