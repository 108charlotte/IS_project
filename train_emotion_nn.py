import pandas as pd
from data_cleaning import cols_btw
import torch
from train_test import train_nn, test_nn
from models import NeuralNet
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from tqdm.auto import tqdm
from baseline_functions import get_avg_baseline, get_majority_baseline
import matplotlib.pyplot as plt
from visualization import display_results

df = pd.read_csv("Fields.csv")

# from codebook: 
'''
ID: participant identifier
round: round 1, 2, and sometimes 3
option: participant's choice that round
session: session ID number
group: group ID number for two participants in matched pair in each session
other_split: partner's choice for that round
PANAS scores
question_#: revised life orientation test question
age: self-reported
language: native language
sex: self-reported gender
fluency: self-reported fluency in english
'''

# Need to match rows with each other so that I can see effects of past emotions on future decisions
panas_cols = df[["optionAfraid", 'optionAggravated', 'optionAngry','optionAppreciative','optionAshamed','optionBelievable','optionCheerful', 'optionCompassionate','optionContemptuous','optionContent','optionDisgusted','optionEmbarrassed','optionGuilty','optionHappy','optionHopeful','optionProud','optionRegretful','optionSad','optionSurprised','optionTriumphant']]
option_col = df['option']
player_id_col = df['ID']
round_col = df['round']

input_data = []
result = []
participant_ids = []

# can always toss 1st round decision and last round emotion
for index, row in df.iterrows(): 
    # if round # is strictly increasing from the last row, then its the same person
    if index != 0 and df.loc[index, 'ID'] == df.loc[index-1, 'ID']: 
        # print(f"{index} is the same person as {index-1}")
        # first round emotion: 
        emotions = panas_cols.iloc[index-1, :]
        # second round decision: 
        decision = 2 if option_col[index] == 'split' else 1 # 1 = betray, 2 = cooperate

        input_data.append(torch.tensor(emotions.to_numpy()))
        result.append(torch.tensor(decision))
        participant_ids.append(df.loc[index, 'ID'])

def train_emotion_nn_ksplit(ksplits, epochs, batch_size, lr, report_every, with_print): 
    # https://scikit-learn.org/stable/modules/cross_validation.html
    gfk = GroupKFold(n_splits=ksplits)
    losses = []
    accuracies = []
    test_losses = []
    test_accuracies = []

    test_avg_baselines = []
    test_majority_baselines = []

    for train_val_index, test_index in gfk.split(input_data, result, participant_ids): 
        input_train_val = [input_data[i] for i in train_val_index]
        result_train_val = [result[i] for i in train_val_index]

        input_test = [input_data[i] for i in test_index]
        result_test = [result[i] for i in test_index]
        train_val_p_ids = np.array(participant_ids)[train_val_index]

        splitter_2 = GroupShuffleSplit(n_splits=1, test_size=0.2)
        train_index, val_index = next(splitter_2.split(input_train_val, result_train_val, train_val_p_ids))

        input_train = [input_train_val[i] for i in train_index]
        result_train = [result_train_val[i] for i in train_index]

        input_val = [input_train_val[i] for i in val_index]
        result_val = [result_train_val[i] for i in val_index]

        e_nn = NeuralNet(20)
        out = train_nn(e_nn, input_train, result_train, input_val, result_val, n_epoch=epochs, n_batch_size=batch_size, learning_rate=lr, report_every=report_every, with_print=with_print, accuracy=True)

        test_loss, test_accuracy = test_nn(e_nn, input_test, result_test, True)

        test_avg_baseline = get_avg_baseline(result_train, result_test)
        test_majority_baseline = get_majority_baseline(result_train, result_test)
        
        losses.append(np.mean(out['losses']))
        accuracies.append(np.mean(out['accuracies']))
        test_avg_baselines.append(test_avg_baseline)
        test_majority_baselines.append(test_majority_baseline)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    return losses, accuracies, test_avg_baselines, test_majority_baselines, test_losses, test_accuracies


def train_emotion_nn(epochs, batch_size, lr, report_every, with_print, random_seed=None): 
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
    splitter_1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed) # TODO: potentially add a random state here
    splitter_2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_seed)

    # only one fold (for now): just use next(): https://stackoverflow.com/questions/68152473/python-split-to-train-test-val-using-groupshufflesplit
    train_index, val_test_index = next(splitter_1.split(input_data, result, participant_ids))
    input_train = [input_data[i] for i in train_index]
    result_train = [result[i] for i in train_index]

    input_val_test = [input_data[i] for i in val_test_index]
    result_val_test = [result[i] for i in val_test_index]
    val_test_p_ids = np.array(participant_ids)[val_test_index]

    val_index, test_index = next(splitter_2.split(input_val_test, result_val_test, val_test_p_ids))
    input_val = [input_val_test[i] for i in val_index]
    result_val = [result_val_test[i] for i in val_index]
    input_test = [input_val_test[i] for i in test_index]
    result_test = [result_val_test[i] for i in test_index]

    e_nn = NeuralNet(20)
    out = train_nn(e_nn, input_train, result_train, input_val, result_val, n_epoch=epochs, n_batch_size=batch_size, learning_rate=lr, random_seed=random_seed, report_every=report_every, with_print=with_print, accuracy=True)

    val_avg_baseline = get_avg_baseline(result_train, result_val)
    val_majority_baseline = get_majority_baseline(result_train, result_val)

    test_avg_baseline = get_avg_baseline(result_train, result_test)
    test_majority_baseline = get_majority_baseline(result_train, result_test)

    return e_nn, out, [val_avg_baseline, val_majority_baseline, test_avg_baseline, test_majority_baseline], input_test, result_test


loop = input("Full loop (l) or defined config (d): ")
report_every = int(input("Report every X epochs (must be an integer): "))

if loop == 'd': 
    print("\nHyperparameters: ")
    epochs = int(input("Number of epochs (must be an integer): "))
    learning_rate = float(input("Learning rate (must be a float): "))
    batch_size = int(input("Batch size (must be an integer): "))

    print() # need a new line, so its not as cluttered and so the user clearly knows that they're now entering model hyperparameters
    print("Training/testing details: ")
    cross_val_or_random = input("Cross val or random splits (c/r)? ")

    if cross_val_or_random == 'c': 
        num_cross_val = int(input(("How many folds would you like? ")))
        losses, accuracies, avg_baselines, majority_baselines, test_losses, test_accuracies = train_emotion_nn_ksplit(num_cross_val, epochs, batch_size, learning_rate, report_every, True)
        
        print("Cross validation results: ")
        print(f"Average test loss: {np.mean(test_losses)} vs average baseline: {np.mean(avg_baselines)} vs majority baseline: {np.mean(majority_baselines)}")
        print(f"Average test accuracy: {np.mean(test_accuracies)}, standard deviation: {np.std(test_accuracies, ddof=1)}") # divides by the population -1 I think

        out = {
            'losses': losses, 
            'test_losses': test_losses, 
            'accuracies': accuracies
        }
        display_results(out, report_every, True, True, False)

    else: 
        num_times = int(input("How many times do you want to randomly run this model (>=1, repeated random group holdout)? "))
        test = False
        if num_times == 1: 
            train_or_test = input("Would you like to test this model after (y/n): ")
            if train_or_test == 'y': test = True
        
        all_outs = []
        all_avg_bs = []
        all_maj_bs = []

        for _ in range(num_times): 
            model, out, baselines, test_i, test_r = train_emotion_nn(epochs, batch_size, learning_rate, report_every, True)
            all_outs.append(out)
            all_avg_bs.append(baselines[0])
            all_maj_bs.append(baselines[1])
        
        # avg all lowest losses
        lowest_avg = 0
        for out in all_outs: 
            lowest_avg += min(out['losses'])
        lowest_avg /= len(all_outs)
        print(f"Lowest loss: {lowest_avg:.7} vs avg baseline: {np.mean(all_avg_bs):.7} vs majority baseline: {np.mean(all_avg_bs):.7}")
        
        if test: # only one model, out, etc. bc you can only test if you only train once
            avg_test_loss, avg_acc = test_nn(model, test_i, test_r, True)

            print(f"Average test loss: {avg_test_loss}, test accuracy: {avg_acc}")
            print(f"Average test baseline: {baselines[2]}, majority test baseline: {baselines[3]}")

        display_results(all_outs[0] if num_times == 1 else all_outs, report_every, charts=True, accuracy=True, multiple= num_times > 1) # want to see plots and display accuracy
else: 
    # comprehensive optimal configuration finder loop
    epochs = list(range(5, 50, 5))
    learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.05]
    batch_sizes = list(range(5, 25, 3))
    random_seeds = [2, 4, 6, 9, 42, 109]

    print(f"testing the following: \n",
        f"- {epochs=}\n", 
        f"- {batch_sizes=}\n", 
        f"- {learning_rates=}\n", 
        f"- {random_seeds=}\n"
        )

    # one cell for each configuration, allowing me to do calculations with everything later
    # each cell's entries are averaged across random seeds, so there's no dimension for random seeds
    results_tensor = torch.full(
        (
        len(epochs), 
        len(learning_rates), 
        len(batch_sizes), 
        6, # output of train will include 4 items (which will each get their own slot in the tensor): losses, test_losses, avg_loss_improvement, avg_test_loss_improvement + 2 baselines (majority and avg train)
        max(epochs) # greatest number of epochs - for losses and test losses, slots that aren't used will be NaN, all slots will have the same avg loss improvement and avg test loss improvement
        ), torch.nan
    )

    print(f"Training: ")
    with tqdm(total=len(epochs)*len(learning_rates)*len(batch_sizes)) as pbar: 
        for e, epoch in enumerate(epochs): 
            for l, lr in enumerate(learning_rates): 
                for b, batch_size in enumerate(batch_sizes):
                    # each component of out will be averaged across random seeds and then inserted into the proper place of the results tensor
                    avg_loss = None
                    avg_test_loss = None
                    avg_loss_improvement = 0
                    avg_test_loss_improvement = 0
                    avg_majority_baseline = 0
                    avg_avg_baseline = 0

                    for rs in random_seeds: 
                        nn = NeuralNet(20)
                        model, out, avg_b, maj_b, _, _ = train_emotion_nn(epochs=epoch, batch_size=batch_size, lr=lr, report_every=report_every, with_print=False, random_seed=rs)
                        if avg_loss == None: 
                            avg_loss = [0] * len(out['losses'])
                            avg_test_loss = [0] * len(out['test_losses'])
                        for i, loss in enumerate(out['losses']): avg_loss[i] += loss
                        for i, test_loss in enumerate(out['test_losses']): avg_test_loss[i] += test_loss
                        avg_loss_improvement += out['avg_loss_improvement']
                        avg_test_loss_improvement += out['avg_test_loss_improvement']
                        avg_majority_baseline += maj_b
                        avg_avg_baseline += avg_b

                    avg_loss = [loss / len(random_seeds) for loss in avg_loss]
                    avg_test_loss = [test_loss / len(random_seeds) for test_loss in avg_test_loss]
                    avg_loss_improvement /= len(random_seeds)
                    avg_test_loss_improvement /= len(random_seeds)
                    avg_majority_baseline /= len(random_seeds)
                    avg_avg_baseline /= len(random_seeds)
                    
                    # see comments in results_tensor to understand what's happening here
                    for i, item in enumerate(avg_loss): 
                        results_tensor[e, l, b, 0, i] = avg_loss[i]

                    for i, item in enumerate(avg_test_loss): 
                        results_tensor[e, l, b, 1, (i+1)*report_every - 1] = avg_test_loss[i] # tensor indexing is based on 0, and I want the losses placed at their epochs so I don't have to calculate the epochs later
                    
                    results_tensor[e, l, b, 2, :] = avg_loss_improvement
                    results_tensor[e, l, b, 3, :] = avg_test_loss_improvement
                    results_tensor[e, l, b, 4, :] = avg_majority_baseline
                    results_tensor[e, l, b, 5, :] = avg_avg_baseline

                    pbar.update(1)

        # results/calculations: 
        # dim 1 = epochs, dim 2 = learning rates, dim 3 = batch sizes, dim 4 = output types, dim 5 = epoch numbers those outputs occur at

        five_lowest_test_losses = [np.inf, np.inf, np.inf, np.inf, np.inf]
        five_lowest_test_loss_hyperparams = [{}, {}, {}, {}, {}]

        five_best_test_loss_improvements = [np.inf, np.inf, np.inf, np.inf, np.inf]
        five_best_test_loss_improvements_hyperparams = [{}, {}, {}, {}, {}]

        for e, epoch in enumerate(epochs): 
            for l, lr in enumerate(learning_rates): 
                for b, batch_size in enumerate(batch_sizes): 
                    out = results_tensor[e, l, b, :]
                    for i, loss in enumerate(out[1]): 
                        if not torch.isnan(loss) and loss < max(five_lowest_test_losses): 
                            max_index = np.argmax(five_lowest_test_losses)
                            five_lowest_test_losses[max_index] = loss.item()
                            five_lowest_test_loss_hyperparams[max_index] = {
                                'num_epochs': epoch, 
                                'learning_rate': lr, 
                                'batch_size': batch_size, 
                                'occured_at_epoch': i+1, 
                                'avg_baseline': out[5][0], 
                                'majority_baseline': out[4][0]
                            }
                    if out[3][0].item() < max(five_best_test_loss_improvements): # homogenous throughout, doesn't matter what I choose
                        max_index = np.argmax(five_best_test_loss_improvements)
                        five_best_test_loss_improvements[max_index] = out[3][0].item()
                        five_best_test_loss_improvements_hyperparams[max_index] = {
                                'num_epochs': epoch, 
                                'learning_rate': lr, 
                                'batch_size': batch_size
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
            print(f"Loss: {five_lowest_test_losses[i]:.7} vs average baseline: {five_lowest_test_loss_hyperparams[i]['avg_baseline']:.7} vs majority baseline: {five_lowest_test_loss_hyperparams[i]['majority_baseline']:.7}")
            print(five_lowest_test_loss_hyperparams[i])

        for i in range(5): 
            print(f"\n{i+1} best loss improvement:")
            print(f"Improvement: {five_best_test_loss_improvements[i]}")
            print(five_best_test_loss_improvements_hyperparams[i])