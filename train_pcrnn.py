from predictive_coding_rnn import PredictiveCodingRNN
import pandas as pd
from data_cleaning import cols_btw
import torch
from sklearn.model_selection import train_test_split

pcrnn = PredictiveCodingRNN(29, [5*2, 5*1], 5*29*1)

batch_size = 5

# creating x and y batches w/ size batch_size
df = pd.read_csv("data_with_computer_decisions.csv")
player_decision_cols = cols_btw('G101', 'G130')
player_decisions = df[player_decision_cols]

computer_decision_cols = cols_btw('CD_01', 'CD_30')
computer_decisions = df[computer_decision_cols]

player_decision_tensors = []
for i in range(len(player_decisions)): 
    player_decision_tensors.append(torch.tensor(player_decisions.iloc[i].values))

computer_decision_tensors = []
for i in range(len(computer_decisions)): 
    computer_decision_tensors.append(torch.tensor(computer_decisions.iloc[i].values))

def split_into_train_val_test(to_split, test_size=0.1, val_size=0.1): # pass in an list of lists
    split_data = train_test_split(*to_split, test_size=test_size+val_size)
    train = split_data[::2]
    test_val = split_data[1::2]
    split_test_val = train_test_split(*test_val, test_size=test_size/(test_size + val_size))
    val = split_test_val[::2]
    test = split_test_val[1::2]
    return train, val, test

all_trains, all_vals, all_tests = split_into_train_val_test((computer_decision_tensors, player_decision_tensors))

all_trains_tuples = []
for i, item in enumerate(all_trains[0]): 
    all_trains_tuples.append((item, all_trains[1][i]))

inputs = []
labels = []

for computer, player in all_trains_tuples: 
    x = [computer[:-1], player[:-1]]
    y = [player[2:]]
    print(f"{x=}, {y=}")
    inputs.append(x)
    labels.append(y)

# TODO: I couldn't find one of the printed decision series in the data. I need to figure out what caused that/what's going on

# TODO: unknown from the paper who made the first move. I'm assuming the player, but I know this could be wrong since I'm making an assumption

x_batches = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
y_batches = [labels[i:i+batch_size] for i in range(0, len(labels), batch_size)]

def train(pcrnn, n_epochs, x_batches, y_batches, batch_size, eta_infer, eta_learn, T_infer, T_learn): 
    pcrnn.train()
    energy_history, supervised_energy_history = [], []

    for epoch in range(n_epochs): 
        print(f"Epoch {epoch+1} / {n_epochs}")
        epoch_energies, epoch_supervised_energies = [], []

        for x_batch, y_batch in zip(x_batches, y_batches): 
            print(f"{x_batch=}, {y_batch=}")
            energies, supervised_energies = pcrnn.forward_train(x_batch, y_batch, batch_size, T_infer, T_learn, eta_infer, eta_learn)
        
            epoch_energies.append(energies)
            epoch_supervised_energies.append(supervised_energies)
        
        energy_history.append(epoch_energies)
        supervised_energy_history.append(epoch_supervised_energies)
    
    return energy_history, supervised_energy_history

e_hist, se_hist = train(pcrnn, 20, x_batches, y_batches, batch_size, 0.01, 0.0001, 5, 5)

print(f"{e_hist=}, {se_hist=}")