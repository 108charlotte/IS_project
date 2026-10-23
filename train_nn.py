import pandas as pd
from data_cleaning import cols_btw
import torch
from training_functions import train_nn
from models import NeuralNet
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data_with_computer_decisions.csv")

# retrieving personality columns
ni_cols = cols_btw('NI01', 'NI16')
nis = df[ni_cols]

pp01_cols = cols_btw('PP01_01', 'PP01_40')
pp01 = df[pp01_cols]

# converting to tensors
ni_tensors = []
for i in range(len(nis)): 
    ni_tensors.append(torch.tensor(nis.iloc[i].values))

pp01_tensors = []
for i in range(len(pp01)): 
    pp01_tensors.append(torch.tensor(pp01.iloc[i].values))

# getting player decisions and calculating the "average" decision a player makes across their 30 decisions
player_decision_cols = cols_btw('G101', 'G130')
player_decisions = df[player_decision_cols]
avg_decisions = []
for i in range(len(player_decisions)): 
    avg_decisions.append(torch.tensor(np.mean(player_decisions.iloc[i, :])))
# something to look into: does personality play a bigger impact at certain moments, ex. the beginning or middle of a game? 

'''
notes: 
- significant negative corelation btw fearless dominance & cooperation
'''

def display_results(training_func_out): 
    print("will create this function later")

# TODO: update rnn code to also use this function
ni_train, ni_val_test, pp01_train, pp01_val_test, avg_decisions_train, avg_decisions_val_test = train_test_split(ni_tensors,pp01_tensors, avg_decisions, test_size=0.2, train_size=0.8)
ni_val, ni_test, pp01_val, pp01_test, avg_decisions_val, avg_decisions_test = train_test_split(ni_val_test, pp01_val_test, avg_decisions_val_test, test_size=0.5, train_size=0.5)

nn_ni = NeuralNet(16)
# training with ni as the input, which has 16 values
out = train_nn(nn_ni, ni_train, avg_decisions_train, ni_val, avg_decisions_val)
display_results(out)

nn_pp01 = NeuralNet(40)
# training with pp01 as the input, which has 40 values
train_nn(nn_pp01, pp01_train, avg_decisions_train, pp01_val, avg_decisions_val)
display_results(out)

nn_2channel = NeuralNet(inp=40+16)
ni_train_tensor = torch.stack(ni_train) # now a 2D tensor
pp01_train_tensor = torch.stack(pp01_train)
combined_train = torch.cat((ni_train_tensor, pp01_train_tensor), dim=1)

ni_val_tensor = torch.stack(ni_val)
pp01_val_tensor = torch.stack(pp01_val)
combined_val = torch.cat((ni_val_tensor, pp01_val_tensor), dim=1)
# training with both ni and pp01 as inputs
train_nn(nn_2channel, combined_train, avg_decisions_train, combined_val, avg_decisions_val)