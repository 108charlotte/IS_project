from pcrnn import PredictiveCodingRNN
import pandas as pd
from data_cleaning import cols_btw
import torch
from sklearn.model_selection import train_test_split

batch_size = 5
# TODO: confirm that dimensions are all correct
pcrnn = PredictiveCodingRNN(29, [2, 10, 5], 1) # 2 is input, 10 is 1st latent state, 5 is next latent state, 1 is output

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

# structure of all_trains: 
# - 1st dimension (tuple) is cd versus pd
# - 2nd dimension (list of tensors) are scenarios (these comprise batches)
# - 3rd dimension (tensors) are sequences of decisions

# first, I need to stack pd and cd separately into their own tensors
all_trains = [torch.stack(decision_type) for decision_type in all_trains]

# next, combine cd and pd into tensor dimensions
all_trains = torch.stack(all_trains).float() # so its the right type for the rnn

print(f"{all_trains.shape=}")

# TODO: I couldn't find one of the printed decision series in the data. I need to figure out what caused that/what's going on
# NOTE: the paper doesn't clarify who made the first move. I'm assuming the player, but I know this could be wrong since I'm making an assumption

# x_batch (and therefore also all_trains) needs to have the format: d1 is step, d2 is sequence of timesteps, d3 is player decision + computer decision
all_trains = all_trains.permute(2,1,0)

print(f"{all_trains.shape=}")

x_batches = [all_trains[:-1, i:i+batch_size, :] for i in range(0, all_trains.shape[1], batch_size)] # purposefully doesn't include the last decision
# TODO: remove computer decision from y_batches
y_batches = [all_trains[1:, i:i+batch_size, 1:] for i in range(0, all_trains.shape[1], batch_size)] # purposefully doesn't include the first decision or computer decisions (only player decisions)


def train(pcrnn, n_epochs, x_batches, y_batches, eta_infer, eta_learn, T_infer): 
    pcrnn.train()
    error_history, gm_error_history = {'infer': [], 'learn': []}, {'infer': [], 'learn': []}

    for epoch in range(n_epochs): 
        print(f"Epoch {epoch+1} / {n_epochs}")
        ep_e_infer, ep_gme_infer, ep_e_learn, ep_gme_learn = [], [], [], []

        for x_batch, y_batch in zip(x_batches, y_batches): 
            # print(f"{x_batch=}, {y_batch=}")
            e_latents, gme_latents, e_learn, gme_learn = pcrnn.forward_train(x_batch, y_batch, T_infer, eta_infer, eta_learn)

            ep_e_infer.append(e_latents)
            ep_gme_infer.append(gme_latents)
            ep_e_learn.append(e_learn)
            ep_gme_learn.append(gme_learn)
        
        error_history['infer'].append(ep_e_infer)
        gm_error_history['infer'].append(ep_gme_infer)
        error_history['learn'].append(ep_e_learn)
        gm_error_history['learn'].append(ep_gme_learn)
    
    return error_history, gm_error_history

e_hist, gme_hist = train(pcrnn, 20, x_batches, y_batches, 0.01, 0.0001, 5)

print(f"{e_hist=}, {gme_hist=}")