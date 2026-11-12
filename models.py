# https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import torch
from torch import nn

class NeuralNet(nn.Module): 
    def __init__(self, inp=1, l_1=8, out=1): # outputs between 0 and 1 where 0 means more likely to choose decision 1 and 1 means more likely to choose decision 2
        super(NeuralNet, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(inp, l_1), 
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(l_1, out), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        result = self.layer_2(self.layer_1(x))
        return result

class RNN(nn.Module): 
    def __init__(self, inp, l_1, out): 
        super(RNN, self).__init__()
        self.rnn = nn.RNN(inp, l_1)
        self.to_out = nn.Linear(l_1, out)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, tensor): 
        rnn_out, hidden = self.rnn(tensor)
        output = self.to_out(hidden[-1])
        return output

'''
Notes from paper: 
- hierarchical RNN
- each layer has latent variables and updates its state based on top-down predictions and bottom-up prediction errors (like predictive coding)
- variational Bayesian framework optimizes parameters to minimize surprise for observed sequences (loss btw actual and expected predictions)
- there's a lower layer which inputs the player's action, their resulting payoff, the opponent's action, and the opponent's payoff (the "sensory outcomes")
- there's a higher layer which stores the latent state? 
- the top-down prediction uses the higher layer's latent state to predict the lowest-level inputs ("sensory outcomes"), then the actual outcomes are used in a bottom-up update, to update the latent state
'''

class PredictiveCodingRNN(nn.Module): 
    def __init__(self, inp, l_1, out): 
        super(PredictiveCodingRNN, self).__init__()
        self.lower_layer = nn.Sequential(
            nn.RNN(inp, l_1)
        )
        self.upper_layer = nn.Sequential(
            nn.RNN(l_1, out)
        )
    
    def forward(self, tensor): 
        rnn_out, hidden = self.lower_layer(tensor)
        output = self.upper_layer(hidden[-1])
        return output
    
    def backward(self, predictions, actual): 
        pass

# random practice to understand PC NNs
# each layer represents the data at a different level of abstraction: raw sensory inputs --> higher-order features
# predictions flow top-down and errors flow bottom-up

class PCNLayer(nn.Module): 
    def __init__(self, in_dim, out_dim, activation_fn=torch.relu, activation_deriv=lambda a: (a>0).float()): 
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv
    
    def forward(self, x):
        a = x @ self.W.T # matrix multiplication operator
        x_pred = self.activation_fn(a)
        return x_pred, a

class PredictiveCodingNetwork(nn.Module): 
    def __init__(self, dims, output_dim): 
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1 # number of latent layers
        self.layers = nn.ModuleList([
            PCNLayer(in_dim=dims[l+1], out_dim=dims[l]) for l in range(self.L)
        ])
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)
    
    def init_latents(self, batch_size, device): 
        return [
            torch.randn(batch_size, d, device=device, requires_grad=False) for d in self.dims[1:]
        ]

    def compute_errors(self, inputs_latents): 
        errors = []
        gain_modulated_errors = []
        for l, layer in enumerate(self.layers): 
            x_pred, a = layer(inputs_latents[l+1])
            err = inputs_latents[l] - x_pred
            gm_error = err * layer.activation_deriv(a) # WHAT DOES GM_ERROR MEAN
            errors.append(err)
            gain_modulated_errors.append(gm_error)
        return errors, gain_modulated_errors