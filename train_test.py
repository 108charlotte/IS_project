from models import PredictiveCodingRNN, NeuralNet
import torch
from torch import nn
import random
import numpy as np
from scipy.stats import linregress # https://stackoverflow.com/questions/9538525/calculating-slopes-in-numpy-or-scipy

def test_nn(model, test_input, test_labels, accuracy=False, criterion=nn.MSELoss()): 
    test_losses = []
    accuracies = []
    model.eval()
    with torch.no_grad(): 
        for i in range(len(test_input)): 
            input_tensor = test_input[i].to(torch.float32)
            output = model(input_tensor).squeeze() # gives me the dimensions I want
            target = (test_labels[i].float()-1).squeeze() # TODO: I realize this sort of data transformation (subtracting one from all decision labels) shouldn't be happening here, it should be happening in the training files
            loss = float(criterion(output, target))
            test_losses.append(loss)

            if accuracy: 
                pred = 1.0 if output > 0.5 else 0.0
                acc = (pred == target).float().mean().item()
                accuracies.append(acc)
    if accuracy: 
        return np.mean(test_losses), np.mean(accuracies)
    return np.mean(test_losses), None

def train_nn(model, input_data=None, targets=None, test_data=None, test_targets=None, n_epoch=40, n_batch_size=10, report_every=5, learning_rate=0.001, criterion=nn.MSELoss(), random_seed=None, with_print=True, accuracy=False): # seed is for batching
    random.seed(random_seed) # for batching
    current_loss = 0
    all_losses = []
    all_test_losses = []
    all_accuracies = []
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if with_print: print(f"Training on dataset with n={len(input_data)}, dims={(np.asarray(input_data)).shape}")
    
    for k in range(1, n_epoch + 1): 
        model.train()
        model.zero_grad()
        batches = list(range(len(input_data)))
        random.shuffle(batches)
        n_batches = int(np.ceil(len(batches) / n_batch_size)) # rounding up to the total # of batches, since if there's any extra few data points those should go in their own batch
        batches = np.array_split(batches, n_batches)
        # print(f"batches: {batches}")

        for idx, batch in enumerate(batches): 
            # print(f"batch: {batch}")
            batch_losses = []
            for item in batch: 
                # print(f"{input_data[batch]=}")
                input_tensor = input_data[item].to(torch.float32)
                # squeezing both to make them scalar-shaped tensors
                output = model(input_tensor).squeeze()
                target = (targets[item].float()-1).squeeze()

                loss = criterion(output, target)
                batch_losses.append(loss)

            # get mean batch loss for backprop
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            current_loss += batch_loss.item() * len(batch) # so it includes the losses for all samples
        
        all_losses.append(current_loss / len(input_data))
        current_loss = 0
        
        if k % report_every == 0: 
            # calculate accuracy on validation set
            avg_test_batch_loss, acc_avg = test_nn(model, test_data, test_targets, accuracy, criterion) # NOTE: this is called test data and test_nn, but I'm passing in validation data here for training, not test data
            all_test_losses.append(avg_test_batch_loss)
            if accuracy: all_accuracies.append(acc_avg)
            
            to_print_string = f"{k} ({k / n_epoch:.3f}): \t avg batch loss = {all_losses[-1]:.5f} \t avg batch test loss = {all_test_losses[-1]:.5f}"
            if accuracy: to_print_string += f"\t avg batch acc = {all_accuracies[-1]}"
            if with_print: print(to_print_string) # NOTE: accuracies and test losses may not update every epoch like all_losses does
    
    if with_print: 
        to_print = "done training, "
        if len(all_losses) > 1: 
            to_print += f"\t avg improvement in losses: {linregress(list(range(len(all_losses))), all_losses).slope:.4f}"
        if len(all_test_losses) > 1: 
            to_print += f"\t avg improvement in test losses: {linregress(list(range(len(all_test_losses))), all_test_losses).slope:.4f}"
        print(to_print)

    result = {
        'losses': all_losses, 
        'test_losses': all_test_losses, 
    }

    to_print = "done training, "
    if len(all_losses) > 1: 
        avg_loss_improvement = linregress(list(range(len(all_losses))), all_losses).slope
        result['avg_loss_improvement'] = avg_loss_improvement
        to_print += f"\t avg improvement in losses: {avg_loss_improvement:.4f}"

    if len(all_test_losses) > 1: 
        avg_test_loss_improvement = linregress(list(range(len(all_test_losses))), all_test_losses).slope
        result['avg_test_loss_improvement'] = avg_test_loss_improvement
        to_print += f"\t avg improvement in test losses: {avg_test_loss_improvement:.4f}"

    if with_print: print(to_print)

    if accuracy: 
        result['accuracies'] = all_accuracies

    return result


# ORIGINAL RNN CODE
# not given payoff information - its not trying to maximize its payoff, its trying to mimic how behavior develops
def create_input_tensor(computer_decisions, player_decisions, player, decision): 
    # combines computer decision & most recent player decision
    input_tensor = torch.stack((computer_decisions[player][decision], torch.tensor(0) if decision == 0 else player_decisions[player][decision-1]), ).reshape(1,2) # decision=0 represents no previous decision
    input_tensor = input_tensor.to(torch.float32)
    return input_tensor

def unpack_rnn_input(data): 
    # d = decisions, p = payoffs
    return {'cd': data[0], 
            'pd': data[1], 
            'cp': data[2], 
            'pp': data[3], 
        }

# need to have train/val computer and player payoffs
# need to not have to pass so much input stuff in
def train_rnn(rnn, train_data, val_data, n_epoch=50, n_batch_size=10, report_every=5, learning_rate=0.001, criterion=nn.CrossEntropyLoss()): 
    current_loss = 0
    all_losses = []
    all_accuracies = []
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    u_train = unpack_rnn_input(train_data)
    u_val = unpack_rnn_input(val_data)

    print(f"Training on dataset with n={len(u_train['pd'])}")

    for k in range(1, n_epoch + 1): 
        rnn.train()
        rnn.zero_grad()

        batches = list(range(len(u_train['cd'])))
        random.shuffle(batches)
        n_batches = int(np.ceil(len(batches) / n_batch_size)) # rounding up to the total # of batches, since if there's any extra few data points those should go in their own batch
        batches = np.array_split(batches, n_batches)
        # print(f"batches: {batches}")

        for idx, batch in enumerate(batches): 
            batch_loss = 0
            for player in batch: # single decision
                for decision in range(len(u_train['cd'][player])): 
                    input_tensor = create_input_tensor(u_train['cd'], u_train['pd'], player, decision)
                    output = rnn.forward(input_tensor)
                    # print(f"{output=}")
                    loss = criterion(output, u_train['pd'][player][decision].long()-1) # make correct datatype for loss calc
                    batch_loss += loss
            
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
            current_loss += batch_loss.item() / len(batch)
        
        all_losses.append(current_loss / len(batches))
        
        if k % report_every == 0: 
            # calculate accuracy on validation set
            rnn.eval()
            with torch.no_grad(): 
                batch_acc = []
                for player in range(len(u_val['cd'])): # this setup is different and may be wrong. TODO: check this
                    for decision in range(len(u_val['cd'][player])): 
                        input_tensor = create_input_tensor(u_val['cd'], u_val['pd'], player, decision)
                        output = rnn(input_tensor)
                        # print(f"output: {output}")
                        player_decision_pred = output.argmax(dim=0).item() # gets the index of the highest likelihood
                        # print(f"player decision pred: {player_decision_pred}")
                        correct = (player_decision_pred == (u_val['pd'][player][decision]).long() - 1)
                        batch_acc.append(1 if correct else 0)
                
                # print("batch acc: ")
                # print(batch_acc)

                epoch_acc = float(np.mean(batch_acc))
                all_accuracies.append(epoch_acc) # TODO: FIX ACCURACIES IDENTICAL
                print(f"{k} ({k / n_epoch:.3f}): \t average batch loss = {all_losses[-1]:.5f} \t epoch accuracy = {epoch_acc:.5f}")

        current_loss = 0

    print(f"done training, \t avg improvement in losses: {linregress(list(range(len(all_losses))), all_losses).slope:.4f} \t avg improvement in test accuracies: {linregress(list(range(len(all_accuracies))), all_accuracies).slope:.4f}")
    return {'num_batches': len(u_train['cd']), 'all_losses': all_losses}