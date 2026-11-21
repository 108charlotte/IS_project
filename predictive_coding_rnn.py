# I created this file to work on everything from scratch with the PCRNN, rather than using my existing files, in order to stay focused on what's happening and keep everything in the same place. Because of this, functionality may later be migrated out of this file and into the file it was originally intended to be in. 
# https://github.com/Monadillo/pcn-intro/blob/main/pcn_cifar10_notebook.ipynb

import torch
import torch.nn as nn
import numpy as np
import copy


# TODO: add energy

# architecture
# layers
class PCNLayer(nn.Module): 
    def __init__(self, in_dim, out_dim, activation_fn=torch.relu, activation_deriv=lambda a: (a > 0).float()): # I may turn this into tanh
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim)) # layer weights
        nn.init.xavier_uniform_(self.W) # initialize W_ls with xavier init
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv

    def forward(self, x_above): 
        a = x_above @ self.W.T # matrix multiplication
        x_hat = self.activation_fn(a)
        print(f"{type(x_hat)=}")
        return x_hat, a

    
# boxes (PCN)
class PredictiveCodingNetwork(nn.Module):
    def __init__(self, dims, output_dim, activation_fn=torch.relu, activation_deriv=lambda a: (a > 0).float()): 
        super().__init__()
        self.dims = dims # number of layers it will have
        self.L = len(dims)-1
        # inputs_latents[1] = first layer, inputs_latents[2] = second layer, etc. 
        self.layers = nn.ModuleList([
            PCNLayer(in_dim=dims[l+1], out_dim=dims[l], activation_fn=activation_fn, activation_deriv=activation_deriv)
            for l in range(self.L)
        ])
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)

    def init_latents(self, batch_size): 
        return [
            torch.randn(batch_size, d, requires_grad=False)
            for d in self.dims[1:]
        ]
    
    def predict_latents(self, layer, inputs_latents_below): # l+1 inputs latents
        x_hat, a = layer(inputs_latents_below)
        return x_hat, a

    def compute_errors(self, inputs_latents): 
        errors, gain_modulated_errors = [], [] # gain modulated errors are errors scaled by activations, so there's a nonlinearity in the error I think? 
        all_activations = []
        for l, layer in enumerate(self.layers): 
            x_hat, a = self.predict_latents(layer, inputs_latents[l+1])
            all_activations.append(a)
            err = inputs_latents[l] - x_hat # actual latent minus predicted latent (eps latent)
            gm_err = err*layer.activation_deriv(a)
            errors.append(err)
            gain_modulated_errors.append(gm_err)
        return errors, gain_modulated_errors, all_activations
    
    def calculate_latent_grad(self, error_latent, gain_mod_error_last_latent, weights_last_latent): 
        # error at the layer - the weight at the last layer times the weighted error at the last layer
        grad_xl = error_latent - gain_mod_error_last_latent @ weights_last_latent
        return grad_xl
    
    # math notes: 
    # T means to transpose a matrix (swap rows and cols)
    # circle w/ a dot inside is the Hadamard product (element-wise multiplication)
    # together: multiply the error for each layer by its corresponding activation, passed through the derivative of the activation
    # derivative tells you how sensitive a neuron's output is to changes in its input - things that have large derivatives will therefore be adapted more, smaller derivatives adapted less
    # same with error - things with larger error will be adapted more, smaller error less
    # this is basic backpropagation
    '''
    def calculate_weight_grad(self, error_latent, activation_latent, ) # T means transpose a matrix; gradient for the weights is 
        grad_wl = -(error_latent)
    '''

    def infer(self, inputs_latents, eta_infer): # x-input = x-batch
        working_inputs_latents = copy.deepcopy(inputs_latents) # bc I'm worried that I'll accidentally update the original latent_states list otherwise; used https://stackoverflow.com/questions/17873384/how-to-deep-copy-a-list
        errors, gain_modulated_errors, all_activations = self.compute_errors(inputs_latents)

        # update latents based on errors
        for l in range(1, self.L + 1): 
            if l == self.L: # top layer
                error_above = torch.zeros_like(inputs_latents[l]) # needs to be the right dimensions
            else: 
                error_above = errors[l] # if its a middle layer (bottom layer is also a middle layer bc input is below it), then it looks at the actual error above
            grad_xl = self.calculate_latent_grad(error_above, gain_modulated_errors[l-1], self.layers[l-1].W)
            # TODO: fix how I'm handling input latents' size, bc its dimensions are all weird bc of batching
            working_inputs_latents[l] -=  eta_infer * grad_xl

        return working_inputs_latents, errors, gain_modulated_errors, all_activations

    def inference(self, T_infer, x_batch, last_activations, eta_infer): 
        with torch.no_grad(): 
            inputs_latents = [x_batch] + [last_activations] + self.init_latents(len(x_batch))
            all_errors, all_gm_errors, all_activations = [], [], []

            for t in range(1, T_infer + 1): 
                inputs_latents, errors, gain_modulated_errors, activations = self.infer(inputs_latents, eta_infer)
                all_errors.append(errors)
                all_gm_errors.append(gain_modulated_errors)
                all_activations.append(activations) # creates a nested list

        return inputs_latents, all_errors, all_gm_errors, all_activations
    
    def learn(self, x_batch, y_batch, inputs_latents, eta_learn): 
        with torch.no_grad(): # handling weight updates manually; don't want pytorch automatic updates to interfere
            all_errors, all_gm_errors = [], []
            for l in range(self.L): 
                errors, gm_errors = self.compute_errors(inputs_latents)
                # below code from code sample provided in pcn paper (https://arxiv.org/html/2506.06332v1)
                # TODO: figure out why they're dividing by batch length
                grad_wl = -(gm_errors[l].T @ inputs_latents[l+1]) / len(x_batch)  # -gm_errors represents the hadamard product of the error at each layer with the derivative of the activation function applied to the activation
                self.layers[l].W -= eta_learn * grad_wl

            # updating readout based on y_batch (this is the only time that actual y should be used)
            top_latent_state = inputs_latents[self.L]
            y_hat = self.readout(top_latent_state)
            eps_L = y_batch - y_hat
            grad_w_readout = eps_L @ top_latent_state
            self.readout.weight -= eta_learn * grad_w_readout
            
            all_errors.append(eps_L)
            all_gm_errors.append(eps_L * 1) # bc the derivative of the linear readout layer is just 1

        return inputs_latents, all_errors, all_gm_errors # I'm not sure if/how these will be useful, but I do want something I can use to measure the accuracy of training

# TODO: make defense for why I chose two PCN layers
class PredictiveCodingRNN(nn.ModuleList): 
    def __init__(self, n_time_steps, nn_dims, nn_output_dim, activation_fn=torch.relu, activation_deriv=lambda a: (a > 0).float()): # nn_dims will be 2 for me, plus 1 output layer
        super().__init__()
        # n_time_steps = the number of PCNs needed
        self.steps = nn.ModuleList([PredictiveCodingNetwork(nn_dims, nn_output_dim, activation_fn, activation_deriv) for _ in range(n_time_steps)])
    
    def _calculate_energies(self, errors, eps_sup, batch_size): 
        supervised_energy = 0.5 * eps_sup.pow(2).sum().item() / batch_size
        latent_energy = 0.5 * sum(e.pow(2).sum().item() for e in errors) / batch_size
        return supervised_energy, latent_energy
    
    def forward_train(self, x_batch, y_batch, T_infer, eta_infer, eta_learn): # eta_infer is learning rate
        '''
        Goes through each PCN in the RNN and trains it to predict its elements of y_batch given its elements of x_batch
        '''
        # TODO: pass activations to the next step
        # x_batch is a sequence of pd decisions
        # go through each PCN and train it separately, with the others frozen
        last_activations = [] # TODO: figure out what to do for the first step where there are no past activations
        errors_latents, gm_errors_latents, errors_learn, gm_errors_learn = [], [], [], []
        for i, pcn in enumerate(self.steps): 
            this_pcn_x = x_batch[i] # NOTE: x_batch needs to have the format: d1 is step, d2 is sequence of timesteps, d3 is player decision + computer decision
            this_pcn_y = y_batch[i] # same note applies here
            inputs_latents, all_errors_latents, all_gm_errors_latents, activations = pcn.inference(T_infer, this_pcn_x, last_activations, eta_infer)
            inputs_latents, all_errors_learn, all_gm_errors_learn = pcn.learn(this_pcn_x, this_pcn_y, inputs_latents, eta_learn)
            last_activations = activations

            errors_latents.append(all_errors_latents)
            gm_errors_latents.append(all_gm_errors_latents)
            errors_learn.append(all_errors_learn)
            gm_errors_learn.append(all_gm_errors_learn)

        return errors_latents, gm_errors_latents, errors_learn, gm_errors_learn

    def forward_test(self, x_batch, T_infer, eta_infer): 
        all_y_preds = []
        last_activations = []
        for i, pcn in enumerate(self.steps): 
            this_pcn_x = x_batch[i] # needs to have the same format as x_batch in forward_train
            inputs_latents, all_errors_latents, all_gm_errors_latents = pcn.inference(T_infer, this_pcn_x, last_activations, eta_infer)
            last_latent = inputs_latents[len(inputs_latents)-1]
            y_pred = pcn.readout(last_latent)
            all_y_preds.append(y_pred)
        return all_y_preds
