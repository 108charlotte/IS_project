import pandas as pd
from data_cleaning import cols_btw
import torch
from training_functions import train_rnn
from models import PredictiveCodingRNN

df = pd.read_csv("data_with_computer_decisions.csv")

# retrieving player decision columns
player_decision_cols = cols_btw('G101', 'G130')
player_decisions = df[player_decision_cols]
# print("player decisions")
# print(player_decisions)

# retrieving computer decision columns
computer_decision_cols = cols_btw("CD_01", "CD_30")
computer_decisions = df[computer_decision_cols]
# print("computer decisions")
# print(computer_decisions)


# computer decision as input, player decision as output --> repeat

# converting data to pytorch tensor format
# each row is a continuous series of training examples - need to be done together
# each column is a new training example

player_decision_tensors = []
for i in range(len(player_decisions)): 
    player_decision_tensors.append(torch.tensor(player_decisions.iloc[i].values))

computer_decision_tensors = []
for i in range(len(computer_decisions)): 
    computer_decision_tensors.append(torch.tensor(computer_decisions.iloc[i].values))

# TODO: randomize this
train_player_decisions = player_decision_tensors[:164]
train_computer_decisions = computer_decision_tensors[:164]

val_player_decisions = player_decision_tensors[164:185]
val_computer_decisions = computer_decision_tensors[164:185]

test_player_decisions = player_decision_tensors[185:206]
test_computer_decisions = computer_decision_tensors[185:206]

rnn = PredictiveCodingRNN(2, 8, 2)
train_rnn(rnn, train_computer_decisions, train_player_decisions, val_computer_decisions, val_player_decisions) # passing in lists of 1D tensors

# model validation
