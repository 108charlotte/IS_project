import pandas as pd
from data_cleaning import cols_btw
import torch
from train_test import train_rnn
from models import RNN
from sklearn.model_selection import train_test_split

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

# retrieving player score cols
player_score_cols = cols_btw("VI03_01", "VI03_30")
player_scores = df[player_score_cols]

# calculating payoff by taking new score - old score
player_payoffs = player_scores.diff(axis=1).iloc[:, 1:]
print(player_payoffs)

# same thing, but for computers
computer_score_cols = cols_btw("VI05_01", "VI05_30")
computer_scores = df[computer_score_cols]
computer_payoffs = computer_scores.diff(axis=1).iloc[:, 1:] # axis = 1 calculates differences btw cols

# computer decision as input, player decision as output --> repeat

# converting data to pytorch tensor format
# each row is a continuous series of training examples - need to be done together
# each column is a new training example

player_decision_tensors = []
for i in range(len(player_decisions)): 
    player_decision_tensors.append(torch.tensor(player_decisions.iloc[i].values))

player_payoff_tensors = []
for i in range(len(player_payoffs)): 
    player_payoff_tensors.append(torch.tensor(player_payoffs.iloc[i].values))

computer_decision_tensors = []
for i in range(len(computer_decisions)): 
    computer_decision_tensors.append(torch.tensor(computer_decisions.iloc[i].values))

computer_payoff_tensors = []
for i in range(len(computer_payoffs)): 
    computer_payoff_tensors.append(torch.tensor(computer_payoffs.iloc[i].values))

def split_into_train_val_test(to_split, test_size=0.1, val_size=0.1): # pass in an list of lists
    split_data = train_test_split(*to_split, test_size=test_size+val_size)
    train = split_data[::2]
    test_val = split_data[1::2]
    split_test_val = train_test_split(*test_val, test_size=test_size)
    val = split_test_val[::2]
    test = split_test_val[1::2]
    return train, val, test

all_trains, all_vals, all_tests = split_into_train_val_test((computer_decision_tensors, player_decision_tensors, computer_payoff_tensors, player_payoff_tensors))

# key: 
# 0 = player_decisions
# 1 = computer_decisions
# 2 = player_payoffs
# 3 = computer_payoffs

from baseline_functions import get_majority_baseline
get_majority_baseline(all_trains[1], all_vals[1])

rnn = RNN(2, 8, 2) # list of their decisions, list of their opponent's decisions, list of their payoffs, list of their opponent's payoffs
train_rnn(rnn, all_trains, all_vals) # passing in lists of 1D tensors

# model validation