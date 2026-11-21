import pandas as pd
from data_cleaning import cols_btw
import torch
from training_functions import train_nn
from models import NeuralNet
import numpy as np
from sklearn.model_selection import train_test_split
import json
from tqdm.auto import tqdm
from testing_functions import get_avg_baseline

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

def display_results(training_func_out): # want to add optional charts later
    output = json.dumps(training_func_out, indent=4, sort_keys=True)
    print(output)


ni_train, ni_val_test, pp01_train, pp01_val_test, avg_decisions_train, avg_decisions_val_test = train_test_split(ni_tensors,pp01_tensors, avg_decisions, test_size=0.2, train_size=0.8)
ni_val, ni_test, pp01_val, pp01_test, avg_decisions_val, avg_decisions_test = train_test_split(ni_val_test, pp01_val_test, avg_decisions_val_test, test_size=0.5, train_size=0.5)

# for nn_2channel
ni_train_tensor = torch.stack(ni_train) # now a 2D tensor
pp01_train_tensor = torch.stack(pp01_train)
combined_train = torch.cat((ni_train_tensor, pp01_train_tensor), dim=1)

ni_val_tensor = torch.stack(ni_val)
pp01_val_tensor = torch.stack(pp01_val)
combined_val = torch.cat((ni_val_tensor, pp01_val_tensor), dim=1)

# experiment section
batch_sizes = list(range(5, 20, 5))
epochs = list(range(20, 30, 5))
learning_rates = [0.001, 0.005, 0.01, 0.015]
random_seeds = [2, 4, 6, 9, 42, 109]

print(f"testing the following: \n",
      f"- {batch_sizes=}\n", 
      f"- {epochs=}\n", 
      f"- {learning_rates=}\n", 
      f"- {random_seeds=}")

full_loop = input("Full loop? (y/n): ")
one_run = True if full_loop == 'n' else False

if one_run: 
    nn_ni = NeuralNet(16)
    # training with ni as the input, which has 16 values
    out = train_nn(nn_ni, ni_train, avg_decisions_train, ni_val, avg_decisions_val)
    display_results(out)

    nn_pp01 = NeuralNet(40)
    # training with pp01 as the input, which has 40 values
    out = train_nn(nn_pp01, pp01_train, avg_decisions_train, pp01_val, avg_decisions_val)
    display_results(out)

    nn_2channel = NeuralNet(inp=40+16)
    # training with both ni and pp01 as inputs
    out = train_nn(nn_2channel, combined_train, avg_decisions_train, combined_val, avg_decisions_val)
    display_results(out)

else: 
    model_n = input("Which model? Enter 'n' for ni, 'p' for pp01, and '2' for combined: ")

    lowest_test_loss = np.inf
    lowest_test_loss_hyperparams = []
    best_test_loss_imp = -np.inf
    best_test_loss_imp_hyperparams = []

    print("\n====TRAINING====")

    all_results = []
    with tqdm(total=len(batch_sizes)*len(epochs)*len(learning_rates)) as pbar:
        for batch_size in batch_sizes: 
            for epoch_n in epochs: 
                for learning_rate in learning_rates: 
                    # I only want to look at the average test loss and improvement across multiple random seeds so things are standardized rather than thinking one model is better because it performed better on a certain random seed
                    all_test_losses_each_epoch = []
                    all_test_loss_improvements = []
                    avg_test_losses_each_epoch = [] # calculate using the above once loop is over
                    avg_test_loss_improvement = np.inf

                    for random_seed in random_seeds: 
                        if model_n == 'n': 
                            model = NeuralNet(16)
                            out = train_nn(model, ni_train, avg_decisions_train, ni_val, avg_decisions_val, n_epoch=epoch_n, n_batch_size=batch_size, learning_rate=learning_rate, random_seed=random_seed)
                        elif model_n == 'p': 
                            model = NeuralNet(40)
                            out = train_nn(model, pp01_train, avg_decisions_train, pp01_val, avg_decisions_val, n_epoch=epoch_n, n_batch_size=batch_size, learning_rate=learning_rate, random_seed=random_seed)
                        elif model_n == '2': # model_n == '2'
                            model = NeuralNet(16+40)
                            out = train_nn(model, combined_train, avg_decisions_train, combined_val, avg_decisions_val, n_epoch=epoch_n, n_batch_size=batch_size, learning_rate=learning_rate, random_seed=random_seed)

                        losses = out['test_losses']
                        all_test_losses_each_epoch.append(losses)
                        all_test_loss_improvements.append(out['avg_test_loss_improvement'])

                    for epoch in range(epoch_n//5): 
                        all_losses_this_epoch = [all_test_losses_each_epoch[i][epoch] for i in range(len(all_test_losses_each_epoch))]
                        avg_test_loss_this_epoch = np.mean(all_losses_this_epoch)
                        avg_test_losses_each_epoch.append(avg_test_loss_this_epoch)
                    
                    avg_test_loss_improvement = np.mean(all_test_loss_improvements)
                    
                    result_dict = {'epoch_avg_losses': avg_test_losses_each_epoch, 'avg_test_loss_improvement': avg_test_loss_improvement, 'batch_size': batch_size, 'total_epoch_n': epoch_n,'learning_rate': learning_rate, 'best_loss_occurred_at_epoch': ((avg_test_losses_each_epoch.index(min(avg_test_losses_each_epoch))+1)*5)}
                    if min(avg_test_losses_each_epoch) < lowest_test_loss: 
                        lowest_test_loss = min(avg_test_losses_each_epoch)
                        lowest_test_loss_hyperparams = result_dict
                    if avg_test_loss_improvement < best_test_loss_imp: 
                        best_test_loss_imp = avg_test_loss_improvement
                        best_test_loss_imp_hyperparams = result_dict
                    
                    all_results.append(result_dict)
                    pbar.update(1)
                    print("\n\n")
    
    print("\n====REPORT====")

    print(f"Best test loss: {lowest_test_loss}")
    print(f"Hyperparams: ")
    display_results(lowest_test_loss_hyperparams)
    print("")
    
    print(f"Best test loss improvement: {best_test_loss_imp}")
    print(f"Hyperparams: ")
    display_results(best_test_loss_imp_hyperparams)


print("\n====FOR COMPARISON====")
get_avg_baseline(avg_decisions_train, avg_decisions_val)