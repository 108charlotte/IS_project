from pcrnn import PredictiveCodingRNN
import pandas as pd
from data_cleaning import cols_btw
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

batch_size = 5
# TODO: confirm that dimensions are all correct
pcrnn = PredictiveCodingRNN(29, [2, 10, 5], 1) # 2 is input, 10 is 1st latent state, 5 is next latent state, 1 is output

# creating x and y batches w/ size batch_size
df = pd.read_csv("data_with_computer_decisions.csv")
player_decision_cols = cols_btw('G101', 'G130')
player_decisions = df[player_decision_cols]
player_decisions -= 1 # want decisions represented as 0 or 1, not 1 or 2

computer_decision_cols = cols_btw('CD_01', 'CD_30')
computer_decisions = df[computer_decision_cols]
computer_decisions -= 1 # want decisions represented as 0 or 1, not 1 or 2

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

# TODO: I couldn't find one of the printed decision series in the data. I need to figure out what caused that/what's going on
# NOTE: the paper doesn't clarify who made the first move. I'm assuming the player, but I know this could be wrong since I'm making an assumption

# x_batch (and therefore also all_trains) needs to have the format: d1 is step, d2 is sequence of timesteps, d3 is player decision + computer decision

def split_segments_to_tensor_pcrnn_format(split_segment): 
    # first, I need to stack pd and cd separately into their own tensors
    result = [torch.stack(decision_type) for decision_type in split_segment]
    # next, combine cd and pd into tensor dimensions
    result = torch.stack(result).float() # so its the right type for the rnn
     # x_batch (and therefore also all_trains) needs to have the format: d1 is step, d2 is sequence of timesteps, d3 is player decision + computer decision
    result = result.permute(2, 1, 0)
    return result

all_trains = split_segments_to_tensor_pcrnn_format(all_trains)
all_vals = split_segments_to_tensor_pcrnn_format(all_vals)
all_tests = split_segments_to_tensor_pcrnn_format(all_tests)

def get_x_y_batches(data, b): # b is batch size
    x_batches = [data[:-1, i:i+b, :] for i in range(0, data.shape[1], b)] # purposefully doesn't include the last decision
    y_batches = [data[1:, i:i+b, 1:] for i in range(0, data.shape[1], b)] # purposefully doesn't include the first decision or computer decisions (only player decisions)
    return x_batches, y_batches

train_x, train_y = get_x_y_batches(all_trains, batch_size)
val_x, val_y = get_x_y_batches(all_vals, batch_size)

train_mean = torch.cat([y.flatten() for y in train_y]).mean()
val_targets = torch.cat([y.flatten() for y in val_y])
baseline_loss = torch.mean(torch.abs(val_targets - train_mean))
print("Training mean baseline:", train_mean.item())
print("Baseline validation loss:", baseline_loss.item())

def train(pcrnn, n_epochs, x_batches, y_batches, x_val, y_val, eta_infer, eta_learn, T_infer): 
    pcrnn.train()
    error_history, gm_error_history = {'infer': [], 'learn': []}, {'infer': [], 'learn': []}
    overall_val_errors, all_accuracies = [], []
    all_supervised_errors = []

    for epoch in range(n_epochs): 
        # print(f"Epoch {epoch+1} / {n_epochs}")
        ep_e_infer, ep_gme_infer, ep_e_learn, ep_gme_learn = [], [], [], []


        # training
        for x_batch, y_batch in zip(x_batches, y_batches): 
            # print(f"{x_batch=}, {y_batch=}")
            e_latents, gme_latents, e_learn, gme_learn = pcrnn.forward_train(x_batch, y_batch, T_infer, eta_infer, eta_learn)

            ep_e_infer.append(e_latents)
            ep_gme_infer.append(gme_latents)
            ep_e_learn.append(e_learn)
            ep_gme_learn.append(gme_learn)
        
        error_history['infer'].append(np.mean(ep_e_infer))
        gm_error_history['infer'].append(np.mean(ep_gme_infer))
        error_history['learn'].append(np.mean(ep_e_learn))
        gm_error_history['learn'].append(np.mean(ep_gme_learn))

        # testing
        all_val_preds = []
        all_val_targets = []

        for x_val_batch, y_val_batch in zip(x_val, y_val): 
            y_hat_batch = pcrnn.forward_test(x_val_batch, T_infer, eta_infer)
            y_hat_batch = torch.stack(y_hat_batch)

            all_val_preds.append(y_hat_batch)
            all_val_targets.append(y_val_batch)

            # print("Prediction min/max:",
            # y_hat_batch.min().item(), y_hat_batch.max().item())

            # print("Prediction mean:", y_hat_batch.mean().item())
            # print("Actual mean:", y_val_batch.mean().item())

        all_val_preds = torch.cat(all_val_preds, dim=1) # along columns, so they each become their own column
        all_val_targets = torch.cat(all_val_targets, dim=1) # also along columns

        val_losses = torch.abs(all_val_targets - all_val_preds)
        avg_val_loss = torch.mean(val_losses).item()
        overall_val_errors.append(avg_val_loss)

        binary_preds = (all_val_preds >= 0.5).float()

        accuracy = (binary_preds == all_val_targets).float().mean().item() # .float converts from bool to numerical value, which is then averaged with mean and the value is taken with .item
        print(f"{accuracy=}")
        all_accuracies.append(accuracy)

        print(f"Epoch {epoch+1} / {n_epochs}. VAL: val loss: {avg_val_loss:.5}, val accuracy: {accuracy}; TRAIN: avg infer error: {np.mean(ep_e_infer):.5}, avg gme infer: {np.mean(ep_gme_infer):.5}, avg error learn: {np.mean(ep_e_learn):.5}, avg gme learn: {np.mean(ep_gme_learn):.5}")
        print("----------------------------------------------------------------------------------------------------------------------------------")
    
    return error_history, gm_error_history, overall_val_errors, all_accuracies

n_epochs = 50
T_infer = 20
eta_infer = 0.001
eta_learn = 0.0001

e_hist, gme_hist, val_errors, accuracies = train(pcrnn, n_epochs, train_x, train_y, val_x, val_y, eta_infer, eta_learn, T_infer)

# chart code written by ChatGPT (very repetitive, didn't want to spend time on it)
epochs = range(1, len(e_hist['infer'])+1)
fig, axes = plt.subplots(3, 2, figsize=(12, 8))

# Inference error
axes[0, 0].plot(epochs, e_hist['infer'], marker='o')
axes[0, 0].set_title("Error History (Infer)")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Error")
axes[0, 0].grid(True)

# Inference GME error
axes[0, 1].plot(epochs, gme_hist['infer'], marker='o')
axes[0, 1].set_title("GME Error History (Infer)")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("GME Error")
axes[0, 1].grid(True)

# Learning error
axes[1, 0].plot(epochs, e_hist['learn'], marker='o')
axes[1, 0].set_title("Error History (Learn)")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Error")
axes[1, 0].grid(True)

# Learning GME error
axes[1, 1].plot(epochs, gme_hist['learn'], marker='o')
axes[1, 1].set_title("GME Error History (Learn)")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("GME Error")
axes[1, 1].grid(True)

# Validation Error
axes[2, 0].plot(epochs, val_errors, marker='o')
axes[2, 0].set_title("Validation Errors")
axes[2, 0].set_xlabel("Epoch")
axes[2, 0].set_ylabel("Error")
axes[2, 0].grid(True)

# Validation Accuracy
axes[2, 1].plot(epochs, accuracies, marker='o')
axes[2, 1].set_title("Validation Accuracies")
axes[2, 1].set_xlabel("Epoch")
axes[2, 1].set_ylabel("Accuracy")
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()