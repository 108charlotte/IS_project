# https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import torch
from torch import nn

class NeuralNet(nn.Module): 
    def __init__(self, inp=1, l_1=8, out=2): # outputs between 0 and 1 where 0 means more likely to choose decision 1 and 1 means more likely to choose decision 2
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

class PredictiveCodingRNN(nn.Module): 
    def __init__(self, inp, l_1, out): 
        super(PredictiveCodingRNN, self).__init__()
        self.rnn = nn.RNN(inp, l_1)
        self.to_out = nn.Linear(l_1, out)
    
    def forward(self, tensor): 
        rnn_out, hidden = self.rnn(tensor)
        output = self.to_out(hidden[-1])
        return output