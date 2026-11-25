import pandas as pd
from data_cleaning import cols_btw
import torch
from train_test import train_nn
from models import NeuralNet
import numpy as np
from sklearn.model_selection import train_test_split
import json
from tqdm.auto import tqdm
from baseline_functions import get_avg_baseline
from visualization import display_results

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

report_every = 5

def train_model(character_identifier, epochs, batch_size, lr, report_every, with_print, random_seed=None): 
    ni_train, ni_val_test, pp01_train, pp01_val_test, avg_decisions_train, avg_decisions_val_test = train_test_split(ni_tensors,pp01_tensors, avg_decisions, test_size=0.3, train_size=0.7, random_state=random_seed)
    ni_val, ni_test, pp01_val, pp01_test, avg_decisions_val, avg_decisions_test = train_test_split(ni_val_test, pp01_val_test, avg_decisions_val_test, test_size=0.5, train_size=0.5, random_state=random_seed)

    # for nn_2channel
    ni_train_tensor = torch.stack(ni_train) # now a 2D tensor
    pp01_train_tensor = torch.stack(pp01_train)
    combined_train = torch.cat((ni_train_tensor, pp01_train_tensor), dim=1)

    ni_val_tensor = torch.stack(ni_val)
    pp01_val_tensor = torch.stack(pp01_val)
    combined_val = torch.cat((ni_val_tensor, pp01_val_tensor), dim=1)

    match character_identifier: 
        case 'n': 
            out = train_nn(NeuralNet(16), ni_train, avg_decisions_train, ni_val, avg_decisions_val, n_epoch=epochs, n_batch_size=batch_size, learning_rate=lr, random_seed=random_seed, report_every=report_every, with_print=with_print)
        case 'p': 
            out = train_nn(NeuralNet(40), pp01_train, avg_decisions_train, pp01_val, avg_decisions_val, n_epoch=epochs, n_batch_size=batch_size, learning_rate=lr, random_seed=random_seed, report_every=report_every, with_print=with_print)
        case 'c': 
            out = train_nn(NeuralNet(16+40), combined_train, avg_decisions_train, combined_val, avg_decisions_val, n_epoch=epochs, n_batch_size=batch_size, learning_rate=lr, random_seed=random_seed, report_every=report_every, with_print=with_print)
        case _: 
            print(f"INVALID MODEL CHARACTER IDENTIFIER")
            out = {}

    return out, get_avg_baseline(avg_decisions_train, avg_decisions_val)


loop = input("Full loop (l) or defined config (d): ")

if loop == 'd': 
    model_input = input("NI, PP01, or combined? (n, p, c): ")
    epochs = int(input("Number of epochs (must be an integer): "))
    learning_rate = float(input("Learning rate (must be a float): "))
    batch_size = int(input("Batch size (must be an integer): "))
    report_every = int(input("Report every X epochs (must be an integer): "))

    if model_input not in ['n', 'p', 'c']: 
        print("Invalid model type. Please enter n, p, or c")
    else: 
        out, baseline = train_model(model_input, epochs, batch_size, learning_rate, report_every, True)
        print(f"Baseline validation MSE: {baseline}")
        display_results(out, report_every, True) # want to see plots
else: 
    # comprehensive optimal configuration finder loop
    epochs = list(range(0, 15, 2))
    learning_rates = [0.0005, 0.001, 0.005, 0.01]
    batch_sizes = list(range(1, 15, 3))
    random_seeds = [2, 4, 6, 9, 42, 109]
    models = ['n', 'p', 'c']

    print(f"testing the following: \n",
        f"- {batch_sizes=}\n", 
        f"- {epochs=}\n", 
        f"- {learning_rates=}\n", 
        f"- {random_seeds=}\n"
        f"- {models=}")

    # one cell for each configuration, allowing me to do calculations with everything later
    # each cell's entries are averaged across random seeds, so there's no dimension for random seeds
    results_tensor = torch.full(
        (
        len(models),
        len(epochs), 
        len(learning_rates), 
        len(batch_sizes), 
        5, # output of train will include 4 items (which will each get their own slot in the tensor): losses, test_losses, avg_loss_improvement, avg_test_loss_improvement + baseline results
        max(epochs) # greatest number of epochs - for losses and test losses, slots that aren't used will be NaN, all slots will have the same avg loss improvement and avg test loss improvement
        ), torch.nan
    )

    print(f"Training: ")
    with tqdm(total=len(models)*len(epochs)*len(learning_rates)*len(batch_sizes)) as pbar: 
        for m, model in enumerate(models): 
            for e, epoch in enumerate(epochs): 
                for l, lr in enumerate(learning_rates): 
                    for b, batch_size in enumerate(batch_sizes): 
                        # each component of out will be averaged across random seeds and then inserted into the proper place of the results tensor
                        avg_loss = None
                        avg_test_loss = None
                        avg_loss_improvement = 0
                        avg_test_loss_improvement = 0
                        avg_baseline = 0

                        for rs in random_seeds: 
                            # function will automatically create a new neural net each time
                            out, baseline = train_model(model, epoch, batch_size, lr, report_every, False, rs)
                            if avg_loss == None: 
                                avg_loss = [0] * len(out['losses'])
                                avg_test_loss = [0] * len(out['test_losses'])
                            for i, loss in enumerate(out['losses']): avg_loss[i] += loss
                            for i, test_loss in enumerate(out['test_losses']): avg_test_loss[i] += test_loss
                            avg_loss_improvement += out['avg_loss_improvement']
                            avg_test_loss_improvement += out['avg_test_loss_improvement']
                            avg_baseline += baseline

                        avg_loss = [loss / len(random_seeds) for loss in avg_loss]
                        avg_test_loss = [test_loss / len(random_seeds) for test_loss in avg_test_loss]
                        avg_loss_improvement /= len(random_seeds)
                        avg_test_loss_improvement /= len(random_seeds)
                        avg_baseline /= len(random_seeds)
                        
                        # see comments in results_tensor to understand what's happening here
                        for i, item in enumerate(avg_loss): 
                            results_tensor[m, e, l, b, 0, i] = avg_loss[i]

                        for i, item in enumerate(avg_test_loss): 
                            results_tensor[m, e, l, b, 1, (i+1)*report_every - 1] = avg_test_loss[i] # tensor indexing is based on 0, and I want the losses placed at their epochs so I don't have to calculate the epochs later
                        
                        results_tensor[m, e, l, b, 2, :] = avg_loss_improvement
                        results_tensor[m, e, l, b, 3, :] = avg_test_loss_improvement
                        results_tensor[m, e, l, b, 4, :] = avg_baseline

                        pbar.update(1)

        # results/calculations: 
        # dim 1 = models, dim 2 = epochs, dim 3 = learning rates, dim 4 = batch sizes, dim 5 = output types, dim 6 = epoch numbers those outputs occur at

        five_lowest_test_losses = [np.inf, np.inf, np.inf, np.inf, np.inf]
        five_lowest_test_loss_hyperparams = [{}, {}, {}, {}, {}]

        five_best_test_loss_improvements = [np.inf, np.inf, np.inf, np.inf, np.inf]
        five_best_test_loss_improvements_hyperparams = [{}, {}, {}, {}, {}]

        for m, model in enumerate(models): 
            for e, epoch in enumerate(epochs): 
                for l, lr in enumerate(learning_rates): 
                    for b, batch_size in enumerate(batch_sizes): 
                        out = results_tensor[m, e, l, b, :]
                        for i, loss in enumerate(out[1]): 
                            if not torch.isnan(loss) and loss < max(five_lowest_test_losses): 
                                max_index = np.argmax(five_lowest_test_losses)
                                five_lowest_test_losses[max_index] = loss.item()
                                five_lowest_test_loss_hyperparams[max_index] = {
                                    'model': model, 
                                    'num_epochs': epoch, 
                                    'learning_rate': lr, 
                                    'batch_size': batch_size, 
                                    'occured_at_epoch': i+1, 
                                    'baseline': out[4, 0].item()
                                }
                        if out[3][0].item() < max(five_best_test_loss_improvements): # homogenous throughout, doesn't matter what I choose
                            max_index = np.argmax(five_best_test_loss_improvements)
                            five_best_test_loss_improvements[max_index] = out[3][0].item()
                            five_best_test_loss_improvements_hyperparams[max_index] = {
                                    'model': model, 
                                    'num_epochs': epoch, 
                                    'learning_rate': lr, 
                                    'batch_size': batch_size, 
                                }
    
        # ordering for reporting: 
        order_1 = np.argsort(five_lowest_test_losses)
        five_lowest_test_losses = np.array(five_lowest_test_losses)[order_1].tolist()
        five_lowest_test_loss_hyperparams = np.array(five_lowest_test_loss_hyperparams)[order_1].tolist()

        order_2 = np.argsort(five_best_test_loss_improvements)
        five_best_test_loss_improvements = np.array(five_best_test_loss_improvements)[order_2].tolist()
        five_best_test_loss_improvements_hyperparams = np.array(five_best_test_loss_improvements_hyperparams)[order_2].tolist()

        print("\nReport: ")

        for i in range(5):
            print(f"\n{i+1} lowest validation loss:")
            print(f"Loss: {five_lowest_test_losses[i]:.7} vs baseline: {five_lowest_test_loss_hyperparams[i]['baseline']:.7}")
            print(five_lowest_test_loss_hyperparams[i])

        for i in range(5): 
            print(f"\n{i+1} best loss improvement:")
            print(f"Improvement: {five_best_test_loss_improvements[i]}")
            print(five_best_test_loss_improvements_hyperparams[i])
