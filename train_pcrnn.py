from pcrnn import PredictiveCodingRNN
import pandas as pd
from data_cleaning import cols_btw
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from visualization import display_pcrnn_results
from baseline_functions import get_majority_baseline_accuracy

batch_size = 5 # needed for splitting data
pcrnn_dims = [8+5+10, 8, 5, 1] # used for computer and player pcrnn
# TODO: confirm that dimensions are all correct

# creating x and y batches w/ size batch_size
df = pd.read_csv("data_with_computer_decisions.csv")
player_decision_cols = cols_btw('G101', 'G130')
player_decisions = df[player_decision_cols]
player_decisions -= 1 # want decisions represented as 0 or 1, not 1 or 2

computer_decision_cols = cols_btw('CD_01', 'CD_30')
computer_decisions = df[computer_decision_cols]
computer_decisions -= 1 # want decisions represented as 0 or 1, not 1 or 2

# this code was written by ChatGPT: utility functions and models for debugging the model only predicting hte majority class
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Convert your DataFrames to numpy arrays: shape (num_games, 30)
P = player_decisions.values
C = computer_decisions.values

# 1. Split by GAME first (prevents leakage between rounds of the same game)
P_train, P_test, C_train, C_test = train_test_split(
    P, C, test_size=0.2, random_state=42
)


def make_windows(p_arr, c_arr, k=5):
  X, y = [], []
  for game_idx in range(len(p_arr)):
    for t in range(k, 30):
      # Past k player moves + past k computer moves -> 2*k features
      past_p = p_arr[game_idx, t - k : t]
      past_c = c_arr[game_idx, t - k : t]
      X.append(np.concatenate([past_p, past_c]))
      y.append(p_arr[game_idx, t])  # Target: player's move at round t
  return np.array(X), np.array(y)


# 2. Create windows (try k=3, 5, or 10)
X_train, y_train = make_windows(P_train, C_train, k=5)
X_test, y_test = make_windows(P_test, C_test, k=5)

# 3. Train and evaluate
clf = LogisticRegression()
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
print("Test ROC-AUC:", roc_auc_score(y_test, probs))


def collect_latents(pcrnn, x_batches, y_batches, T_infer, eta_infer):
    all_latents = []
    all_targets = []

    pcrnn.eval()

    with torch.no_grad():
        for x_batch, y_batch in zip(x_batches, y_batches):
            batch_size = x_batch[0].shape[0]
            last_activations = pcrnn.pcn.init_latents(batch_size)

            for i in range(pcrnn.n_time_steps):
                x_input = torch.cat([x_batch[i]] + last_activations, dim=1)
                inputs_latents, _, _, _ = pcrnn.pcn.inference(
                    T_infer,
                    x_input,
                    eta_infer
                )

                last_activations = inputs_latents[1:]

                all_latents.append(inputs_latents[-1])
                all_targets.append(y_batch[i])

    return torch.cat(all_latents), torch.cat(all_targets)

# my code again
def make_x_y_windows(x_batch, y_batch, w_len=5): # for PCRNN input, creates windows of the past 5 decisions to pass as input to each pcrnn layer
    # 1st dimension of x_batch is decision number, next is sample (adding these together creates batches)
    x_windows, y_windows = [], []
    for i in range(x_batch.shape[0]-w_len+1): # i goes from 0 to 29-w_len
        five_decision_batch = x_batch[i:i+w_len] # dims: w_len by batch size by 2; gives player and computer decisions for i:i+1_len
        # keep each 2 decisions next to each other (like if you were to push the 2 datasets together, interlocking them)
        # first, dim 1 needs to be batch, dim 2 needs to be 5, and dim 3 should be 2 - so that they're next to each other, and then flatten starting at dim=1 into dim=2
        five_decision_batch = five_decision_batch.permute(1, 0, 2)
        five_decision_batch = five_decision_batch.flatten(start_dim=1)

        x_windows.append(five_decision_batch) # doesn't include the last decision: the next player decision - TODO: make sure that everything is ordered right for this
        y_windows.append(y_batch[i+w_len-1]) # next thing after i+w_len

    return torch.stack(x_windows), torch.stack(y_windows)


player_decision_tensors = []
for i in range(len(player_decisions)): 
    player_decision_tensors.append(torch.tensor(player_decisions.iloc[i].values))

computer_decision_tensors = []
for i in range(len(computer_decisions)): 
    computer_decision_tensors.append(torch.tensor(computer_decisions.iloc[i].values))

def split_into_train_val_test(to_split, test_size=0.1, val_size=0.1, random_seed=None): # pass in an list of lists
    split_data = train_test_split(*to_split, test_size=test_size+val_size, random_state=random_seed)
    train = split_data[::2]
    test_val = split_data[1::2]
    split_test_val = train_test_split(*test_val, test_size=test_size/(test_size + val_size), random_state=random_seed)
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
     # x_batch (and therefore also all_trains) needs to have the format: d1 is step, d2 is sequence of timesteps (creates a batch when combined), d3 is player decision + computer decision
    result = result.permute(2, 1, 0)
    return result

all_trains = split_segments_to_tensor_pcrnn_format(all_trains)
all_vals = split_segments_to_tensor_pcrnn_format(all_vals)
all_tests = split_segments_to_tensor_pcrnn_format(all_tests)

def get_x_y_batches(data, b, w_len=5): # b is batch size
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

def train(pcrnn, n_epochs, x_batches, y_batches, x_val, y_val, eta_infer, eta_learn, T_infer, eta_learn_readout, with_print=True, report_every=1): 
    pcrnn.train()
    error_history, gm_error_history = {'infer': [], 'learn': []}, {'infer': [], 'learn': []}
    all_val_errors, all_accuracies = [], []

    for epoch in range(n_epochs): 
        ep_e_infer, ep_gme_infer, ep_e_learn, ep_gme_learn = [], [], [], []

        # training
        for x_batch, y_batch in zip(x_batches, y_batches): 
            # print(f"{x_batch=}, {y_batch=}")
            e_latents, gme_latents, e_learn, gme_learn = pcrnn.forward_train(x_batch, y_batch, T_infer, eta_infer, eta_learn, eta_learn_readout)
            # print("Readout weight change:",
            #     torch.norm(readout_after - readout_before).item())
            # print(f"Bias change: {torch.norm(bias_after-bias_before).item()}")
            
            ep_e_infer.append(e_latents)
            ep_gme_infer.append(gme_latents)
            ep_e_learn.append(e_learn)
            ep_gme_learn.append(gme_learn)
        
        error_history['infer'].append(np.mean(ep_e_infer))
        gm_error_history['infer'].append(np.mean(ep_gme_infer))
        error_history['learn'].append(np.mean(ep_e_learn))
        gm_error_history['learn'].append(np.mean(ep_gme_learn))

        # validation
        all_val_preds = []
        all_val_targets = []

        for x_val_batch, y_val_batch in zip(x_val, y_val): 
            y_hat_batch = pcrnn.forward_test(x_val_batch, T_infer, eta_infer)
            y_hat_batch = torch.stack(y_hat_batch)

            all_val_preds.append(y_hat_batch)
            all_val_targets.append(y_val_batch)

        all_val_preds = torch.cat(all_val_preds, dim=1) # along columns, so they each become their own column
        all_val_targets = torch.cat(all_val_targets, dim=1) # also along columns

        pred_flat = all_val_preds.detach().flatten()
        target_flat = all_val_targets.detach().flatten()

        correlation = torch.corrcoef(
            torch.stack([pred_flat, target_flat])
        )[0, 1]

        binary_preds = (pred_flat >= 0.5).float()
        val_losses = torch.abs(all_val_targets - all_val_preds)
        avg_val_loss = torch.mean(val_losses).item() # not MSE, just absolute error so it can be compared with the baseline
        all_val_errors.append(avg_val_loss)

        binary_preds = (all_val_preds >= 0.5).float()

        accuracy = (binary_preds == all_val_targets).float().mean().item() # .float converts from bool to numerical value, which is then averaged with mean and the value is taken with .item
        all_accuracies.append(accuracy)

        if with_print and epoch % report_every == 0: print(f"Epoch {epoch+1}/{n_epochs}, VAL: val loss: {avg_val_loss:.5}, val accuracy: {accuracy:.5}; TRAIN: avg infer error: {np.mean(ep_e_infer):.5}, avg gme infer: {np.mean(ep_gme_infer):.5}, avg error learn: {np.mean(ep_e_learn):.5}, avg gme learn: {np.mean(ep_gme_learn):.5}")


    return error_history, gm_error_history, all_val_errors, all_accuracies

def train_pcrnn(pcrnn, n_epochs, T_infer, eta_infer, eta_learn, batch_size, with_print, eta_learn_readout, computer=False, random_seed=None, report_every=1): 
    if not computer: 
        all_trains, all_vals, all_tests = split_into_train_val_test((computer_decision_tensors, player_decision_tensors), random_seed=random_seed)
    else: # flipping which is which
        all_trains, all_vals, all_tests = split_into_train_val_test((player_decision_tensors, computer_decision_tensors), random_seed=random_seed)

    # structure of all_trains: 
    # - 1st dimension (tuple) is cd versus pd
    # - 2nd dimension (list of tensors) are scenarios (these comprise batches)
    # - 3rd dimension (tensors) are sequences of decisions

    # x_batch (and therefore also all_trains) needs to have the format: d1 is step, d2 is sequence of timesteps, d3 is player decision + computer decision

    all_trains = split_segments_to_tensor_pcrnn_format(all_trains)
    all_vals = split_segments_to_tensor_pcrnn_format(all_vals)
    all_tests = split_segments_to_tensor_pcrnn_format(all_tests)

    train_x, train_y = get_x_y_batches(all_trains, batch_size)
    val_x, val_y = get_x_y_batches(all_vals, batch_size)
    test_x, test_y = get_x_y_batches(all_tests, batch_size) # TODO: add windows for test? 

    train_windows = [make_x_y_windows(x, y) for x, y in zip(train_x, train_y)]
    val_windows = [make_x_y_windows(x, y) for x, y in zip(val_x, val_y)]

    train_x = [item[0] for item in train_windows]
    train_y = [item[1] for item in train_windows]

    val_x = [item[0] for item in val_windows]
    val_y = [item[1] for item in val_windows]

    test_windows = [make_x_y_windows(x, y) for x, y in zip(test_x, test_y)]

    test_x = [item[0] for item in test_windows]
    test_y = [item[1] for item in test_windows]

    # flatten the y inside of each batch of train_y, and join all of those together (so that u can process together)
    val_targets = torch.cat([y.flatten() for y in val_y])
    train_targets = torch.cat([y.flatten() for y in train_y])
    train_mean = train_targets.mean()
    baseline_loss = torch.mean(torch.abs(val_targets - train_mean))
    baseline_accuracy = get_majority_baseline_accuracy(train_targets, val_targets)

    test_targets = torch.cat([y.flatten() for y in test_y])
    baseline_test_loss = torch.mean(torch.abs(test_targets - train_mean))
    baseline_test_accuracy = get_majority_baseline_accuracy(train_targets, test_targets)

    out = train(pcrnn, n_epochs, train_x, train_y, val_x, val_y, eta_infer, eta_learn, T_infer, eta_learn_readout, with_print, report_every=report_every)

    return out, float(baseline_loss.item()), baseline_accuracy, baseline_test_loss, baseline_test_accuracy, all_tests # so that I have both player and computer decisions; will be useful when testing
    # print("Training mean baseline:", train_mean.item())
    # print("Baseline validation loss:", baseline_loss.item())

# TODO: finish this function (it currently isn't functional)
def full_forward_test(player_rnn, computer_rnn, player_first_decision, computer_first_decision, computer_T_infer, player_T_infer, computer_eta_infer, player_eta_infer):
    all_player_decision_preds = []
    all_player_latent_errors = []
    all_player_latent_gme = []
    all_computer_decision_preds = []
    all_computer_latent_errors = []
    all_computer_latent_gme = []

    last_decisions = [player_first_decision, computer_first_decision] # TODO: figure out if this should be a tensor
    p_last_activations = player_rnn.pcn.init_latents(1)
    c_last_activations = computer_rnn.pcn.init_latents(1)
    if player_rnn.n_time_steps != computer_rnn.n_time_steps: 
        print("WARNING: TIMESTEP NUMBER MISMATCH between player rnn and computer rnn")

    for i in range(player_rnn.n_time_steps): 
        # makes player prediction
        p_il, p_el, p_gme_l, p_a = player_rnn.pcn.inference(player_T_infer, last_decisions, p_last_activations, player_eta_infer) # I'm assuming the player goes first
        p_last_activations = p_il[1:]
        p_last_latent = p_il[len(p_il)-1]
        p_pred = player_rnn.pcn.readout(p_last_latent)
        p_pred = player_rnn.pcn.readout_activation(p_pred)
        all_player_decision_preds.append(p_pred)
        all_player_latent_errors.append(p_el)
        all_player_latent_gme.append(p_gme_l)

        # makes computer prediction
        last_decisions = [last_decisions[0], p_pred]
        c_il, c_el, c_gme_l, c_a = computer_rnn.pcn.inference(computer_T_infer, last_decisions, c_last_activations, computer_eta_infer)
        c_last_activations = c_il[1:] # TODO: I can't remember why I'm indexing this like this...oh I think its so that it doesn't include the input
        c_last_latent = c_il[len(c_il)-1]
        c_pred = computer_rnn.pcn.readout(c_last_latent)
        c_pred = computer_rnn.pcn.readout_activation(c_pred)
        all_computer_decision_preds.append(c_pred)
        all_computer_latent_errors.append(c_el)
        all_computer_latent_gme.append(c_gme_l)
        last_decisions = [c_pred, p_pred]

    return all_player_decision_preds, all_computer_decision_preds

print("\nModel Training Details")
report_every = int(input("Report every X epochs (integers only): "))
p_c_b = input("Would you like to train just the player decision prediction model, just the computer decision prediction model, or both (p, c, or b)? ")
num_times_p, num_times_c = 0, 0

# gather all data first, so that the person running the code can just let it run after the beginning

if p_c_b in ['p', 'b']: 
    print("\nHyperparameters for player model: ")
    p_n_epochs = int(input("Number of epochs for training: "))
    p_T_infer = int(input("T infer (number of inference loops): "))
    p_eta_infer = float(input("Inference rate: "))
    p_eta_learn = float(input("Learning rate: "))
    p_eta_learn_readout = float(input("Readout learning rate: "))

    print("\nTraining info: ")
    num_times_p = int(input("How many times would you like to train the player model? "))

    print()

if p_c_b in ['c', 'b']: 
    print("\nHyperparameters for computer model: ")
    c_n_epochs = int(input("Number of epochs for training: "))
    c_T_infer = int(input("T infer (number of inference loops): "))
    c_eta_infer = float(input("Inference rate: "))
    c_eta_learn = float(input("Learning rate: "))
    c_eta_learn_readout = float(input("Readout learning rate: "))

    print("\nTraining info: ")
    num_times_c = int(input("How many times would you like to train the computer model? "))

if p_c_b in ['p', 'b']: 
    # train player model
    p_outs, p_mean_baselines, p_baseline_accs = [], [], []

    p_test = False
    if num_times_p == 1: 
        individual_test = input("Would you like to test the player model individually (y/n)? ")
        if individual_test == 'y': p_test = True

    for i in range(num_times_p): 
        print(f"\nStarting Player Model Training {i+1}/{num_times_p}")
        player_pcrnn = PredictiveCodingRNN(29, pcrnn_dims, 1) # 17 bc 2 decisions and then 15 latents
        p_out, p_mean_baseline, p_baseline_acc, p_t_b_l, p_t_b_a, p_tests = train_pcrnn(player_pcrnn, p_n_epochs, p_T_infer, p_eta_infer, p_eta_learn, batch_size, eta_learn_readout=p_eta_learn_readout, with_print=True, report_every=report_every)
        p_outs.append(p_out)
        p_mean_baselines.append(p_mean_baseline)
        p_baseline_accs.append(p_baseline_acc)

        '''latents, targets = collect_latents(
            player_pcrnn,
            train_x,
            train_y,
            p_T_infer,
            p_eta_infer
        )

        class_0 = latents[targets.squeeze() == 0]
        class_1 = latents[targets.squeeze() == 1]

        print("Class 0 count:", len(class_0))
        print("Class 1 count:", len(class_1))

        print("Class 0 latent mean:", class_0.mean(dim=0))
        print("Class 1 latent mean:", class_1.mean(dim=0))

        print("Difference between class means:",
            (class_0.mean(dim=0) - class_1.mean(dim=0)).abs())
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

        X_latents = latents.numpy()
        y_targets = targets.numpy().flatten()

        latent_clf = LogisticRegression()
        latent_clf.fit(X_latents, y_targets)

        latent_probs = latent_clf.predict_proba(X_latents)[:, 1]
        latent_preds = latent_clf.predict(X_latents)

        print("Latent logistic regression accuracy:",
            accuracy_score(y_targets, latent_preds))

        print("Latent logistic regression balanced accuracy:",
            balanced_accuracy_score(y_targets, latent_preds))

        print("Latent logistic regression ROC-AUC:",
            roc_auc_score(y_targets, latent_probs))
        '''

    # average lowest loss and avg lowest accuracy, along w/ standard deviation for both
    p_final_losses = []
    p_final_accuracies = []
    for o in p_outs: 
        p_final_losses.append(o[2][-1])
        p_final_accuracies.append(o[3][-1])
    
    if p_test: 
        p_test_x, p_test_y = get_x_y_batches(p_tests, batch_size)
        all_y_preds = []
        all_y_targets = []

        with torch.no_grad(): 
            # needs to handle batching correctly 
            for x_test_batch, y_test_batch in zip(p_test_x, p_test_y): 
                preds = torch.stack(player_pcrnn.forward_test(x_test_batch, p_T_infer, p_eta_infer))
                all_y_preds.append(preds)
                all_y_targets.append(y_test_batch)
            
        all_y_preds = torch.cat(all_y_preds, dim=1) # columns smashed together
        all_y_targets = torch.cat(all_y_targets, dim=1)

        avg_test_loss = torch.mean(torch.abs(all_y_targets - all_y_preds)).item()
        
        binary_preds = (all_y_preds >= 0.5).float()
        test_accuracy = (binary_preds == all_y_targets).float().mean().item() # .float converts from bool to numerical value, which is then averaged with mean and the value is taken with .item

        print("\nPrediction diagnostics:")

        print("Predicted class counts:",
            torch.unique(binary_preds, return_counts=True))

        print("Actual class counts:",
            torch.unique(all_y_targets, return_counts=True))

        print("Fraction predicted as 1:",
            binary_preds.mean().item())

        print("Fraction actually 1:",
            all_y_targets.mean().item())

    print(f"\nResults of player PCRNN from {num_times_p} random training runs: ")
    print(f"Average final loss: {np.mean(p_final_losses)}, w/ standard deviation {np.std(p_final_losses, ddof=1)}") #TODO: confirm if ddof should be 1 here, I think bc these are only a few random samples from many different random samples there could have been that I should use it but I want to make sure
    print(f"vs average loss baseline: {np.mean(p_mean_baselines)}")
    print(f"Average final accuracy: {np.mean(p_final_accuracies)}, w/ standard deviation: {np.std(p_final_accuracies, ddof=1)}")
    print(f"Versus average baseline accuracy: {np.mean(p_baseline_accs)}")

    if p_test: 
        print(f"\nTest results from player PCRNN after {num_times_p} random training runs: ")
        print(f"Accuracy: {test_accuracy} vs baseline accuracy: {p_t_b_a}")
        print(f"Test loss: {avg_test_loss} vs baseline loss: {p_t_b_l}")
        print()

if p_c_b in ['c', 'b']: 
    # train computer model
    c_outs, c_mean_baselines, c_baseline_accs = [], [], []

    for i in range(num_times_c): 
        print(f"\nStarting Computer Model Training {i+1}/{num_times_c}")
        computer_pcrnn = PredictiveCodingRNN(29, pcrnn_dims, 1)
        c_out, c_mean_baseline, c_baseline_acc, c_t_b_l, c_t_b_a, c_tests = train_pcrnn(computer_pcrnn, c_n_epochs, c_T_infer, c_eta_infer, c_eta_learn, batch_size, eta_learn_readout=c_eta_learn_readout, with_print=True, report_every=report_every)
        c_outs.append(c_out)
        c_mean_baselines.append(c_mean_baseline)
        c_baseline_accs.append(c_baseline_acc)

    # average lowest loss and avg lowest accuracy, along w/ standard deviation for both
    c_lowest_losses = []
    c_highest_accuracies = []
    for o in c_outs: 
        c_lowest_losses.append(min(o[2]))
        c_highest_accuracies.append(max(o[3]))

    print(f"\nResults of computer PCRNN from {num_times_c} random training runs: ")
    print(f"Average lowest loss: {np.mean(c_lowest_losses)}, w/ standard deviation {np.std(c_lowest_losses, ddof=1)}") #TODO: confirm if ddof should be 1 here, I think bc these are only a few random samples from many different random samples there could have been that I should use it but I want to make sure
    print(f"vs average loss baseline: {np.mean(c_mean_baselines)}")
    print(f"Average highest accuracy: {np.mean(c_highest_accuracies)}, w/ standard deviation: {np.std(c_highest_accuracies, ddof=1)}")
    print(f"Versus average baseline accuracy: {np.mean(c_baseline_accs)}")

# potentially test models
if p_c_b == 'b' and num_times_p == 1 and num_times_c == 1: 
    print()
    to_test = input("INPUT REQUIRED: Would you like to test these PCRNNs (y/n)? ")
    if to_test == 'y': 
        with torch.no_grad(): 
            player_pcrnn.eval()
            computer_pcrnn.eval()
            # test - all of the variables I'm using will be defined if the conditions above are true
            # I have both c_test and p_test to test these models on
            c_test_first_decisions = c_tests[1] # for each batch (dim 1), contains a player decision and a computer decision (dim 2)
            # need c_test to be rearranged to player format (flip computer and player decision positions)
            c_test_first_decisions[:,[1, 0]] = c_test_first_decisions[:, [0, 1]]
            p_test_first_decisions = p_tests[1] # same as above

            criterion = torch.nn.MSELoss()
            p_losses = [[], []]
            p_accuracies = [[], []]

            c_losses = [[], []]
            c_accuracies = [[], []]

            for d in c_test_first_decisions: 
                pp_1, cp_1 = full_forward_test(player_pcrnn, computer_pcrnn, d[0], d[1], c_T_infer, p_T_infer, c_eta_infer, p_eta_infer)
                # pp_1 should be compared to c_tests[2:, 1], and cp_1 should be compared to c_tests[2:, 0]
                print(f"{pp_1=}")
                print(f"{c_tests[2:, 1]=}")
                p_loss_1 = criterion(pp_1, c_tests[2:, 1])
                p_preds = [(p >= 0.5).float() for p in pp_1]
                p_accuracy_1 = (p_preds == c_tests[2:, 1]).float().mean()

                c_loss_1 = criterion(cp_1, c_tests[2:, 0])
                c_preds = [(c >= 0.5).float() for c in cp_1]
                c_accuracy_1 = (cp_1 == c_tests[2:, 0]).float().mean()

                p_losses[0].append(p_loss_1)
                p_accuracies[0].append(p_accuracy_1)
                c_losses[0].append(c_loss_1)
                c_accuracies[0].append(c_accuracy_1)

            for d in p_test_first_decisions: 
                pp_2, cp_2 = full_forward_test(player_pcrnn, computer_pcrnn, d[0], d[1], c_T_infer, p_T_infer, c_eta_infer, p_eta_infer)
                p_loss_2 = criterion(pp_2, p_tests[2:, 1])
                p_accuracy_2 = (pp_2 == p_tests[2:, 1]).float().mean()

                c_loss_2 = criterion(cp_2, p_tests[2:, 0])
                c_accuracy_2 = (cp_2 == p_tests[2:, 0]).float().mean()

                p_losses[1].append(p_loss_2)
                p_accuracies[1].append(p_accuracy_2)
                c_losses[1].append(c_loss_2)
                c_accuracies[1].append(c_accuracy_2)

        # display test results
        print("\nTest results: ")
        print("Note: the models are each only predicting the next decision when given the correct past sequence (regardless of their predictions for past decisions)")

# summarize results for player model and display
if p_c_b in ['p', 'b']: 
    p_inference_errors = [p_out[0]['infer'] for p_out in p_outs]
    p_inference_gm_errors = [p_out[1]['infer'] for p_out in p_outs]
    p_learning_errors = [p_out[0]['learn'] for p_out in p_outs]
    p_learning_gm_errors = [p_out[1]['learn'] for p_out in p_outs]

    p_val_errors = [p_out[2] for p_out in p_outs]
    p_val_accs = [p_out[3] for p_out in p_outs]

    display_pcrnn_results(p_n_epochs, p_inference_errors, p_inference_gm_errors, p_learning_errors, p_learning_gm_errors, p_val_errors, p_val_accs)

# summarize results for computer model and display
if p_c_b in ['c', 'b']: 
    c_inference_errors = [c_out[0]['infer'] for c_out in c_outs]
    c_inference_gm_errors = [c_out[1]['infer'] for c_out in c_outs]
    c_learning_errors = [c_out[0]['learn'] for c_out in c_outs]
    c_learning_gm_errors = [c_out[1]['learn'] for c_out in c_outs]

    c_val_errors = [c_out[2] for c_out in c_outs]
    c_val_accs = [c_out[3] for c_out in c_outs]

    display_pcrnn_results(c_n_epochs, c_inference_errors, c_inference_gm_errors, c_learning_errors, c_learning_gm_errors, c_val_errors, c_val_accs)