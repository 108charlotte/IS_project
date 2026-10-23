from models import PredictiveCodingRNN, NeuralNet
import torch
from torch import nn
import random
import numpy as np
import time
from scipy.stats import linregress # https://stackoverflow.com/questions/9538525/calculating-slopes-in-numpy-or-scipy

# TODO: fix batching bc I think its wrong
# TODO: fix test losses unexpectedly higher than train losses, even for the 1st value
def train_nn(nn, input_data=[], targets=[], test_data=[], test_targets=[], n_epoch=30, n_batch_size=10, report_every=5, learning_rate=0.2, criterion=nn.MSELoss()): 
    current_loss = 0
    all_losses = []
    all_test_losses = []
    nn.train()
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)

    print(f"Training on dataset with n={len(input_data)}, dims={(np.asarray(input_data)).shape}")
    
    for k in range(1, n_epoch + 1): 
        nn.train()
        nn.zero_grad()
        batches = []
        batches = list(range(len(input_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)
        # print(f"batches: {batches}")


        for idx, batch in enumerate(batches): 
            # print(f"batch: {batch}")
            batch_loss = 0
            for item in batch: 
                # print(f"{input_data[batch]=}")
                input_tensor = input_data[item].to(torch.float32)
                output = nn.forward(input_tensor)
                pred = output[output.argmax(dim=0).item()] # TODO: fix argmax here
                # print(f"{output=}, {pred=}, {targets[item]=}")
                loss = criterion(pred, targets[item].long()-1)
                batch_loss += loss
                
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            current_loss += batch_loss.item() / np.size(batch)
        
        all_losses.append(current_loss / len(batches))
        
        if k % report_every == 0: 
            # calculate accuracy on validation set
            nn.eval()
            with torch.no_grad(): 
                test_batch_losses = []
                for item in range(len(test_data)): 
                    # print(f"batch: {batch}")
                    input_tensor = test_data[item].to(torch.float32)
                    output = nn.forward(input_tensor)
                    pred = output[output.argmax(dim=0).item()] # TODO: fix argmax here, it should be able to get me all of this automatically
                    loss = criterion(pred, test_targets[item].long()-1)
                    test_batch_losses.append(loss)
                
                # print("batch test loss: ")
                # print(np.mean(test_batch_losses))

                epoch_test_loss = float(np.mean(test_batch_losses))
                all_test_losses.append(epoch_test_loss) 
                
            print(f"{k} ({k / n_epoch:.0%}): \t average batch loss = {all_losses[-1]} \t average batch test loss = {all_test_losses[-1]}")

        current_loss = 0

    print(f"done training,\t avg improvement in losses: {linregress(list(range(len(all_losses))), all_losses).slope} \t avg improvement in test losses: {linregress(list(range(len(all_test_losses))), all_test_losses).slope}")


def create_input_tensor(computer_decisions, player_decisions, player, decision): 
    input_tensor = torch.stack((computer_decisions[player][decision], torch.tensor(0) if decision == 0 else player_decisions[player][decision-1])).reshape(1,2) # decision=0 represents no previous decision
    input_tensor = input_tensor.to(torch.float32)
    return input_tensor

def train_rnn(rnn, computer_decisions, player_decisions, val_computer_decisions, val_player_decisions, n_epoch=30, n_batch_size=1, report_every=5, learning_rate=0.01, criterion=nn.CrossEntropyLoss()): 
    current_loss = 0
    all_losses = []
    all_accuracies = []
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    print(f"Training on dataset with n={len(computer_decisions)}")

    for k in range(1, n_epoch + 1): 
        rnn.train()
        rnn.zero_grad()

        batches = list(range(len(computer_decisions)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)
        print(f"batches: {batches}")

        for idx, batch in enumerate(batches): 
            batch_loss = 0
            for player in batch: # single decision
                for decision in range(len(computer_decisions[player])): 
                    input_tensor = create_input_tensor(computer_decisions, player_decisions, player, decision)
                    output = rnn.forward(input_tensor)
                    loss = criterion(output, player_decisions[player][decision].float()-1) # make correct datatype for loss calc
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
                for player in range(len(val_computer_decisions)): # this setup is different and may be wrong. TODO: check this
                    for decision in range(len(val_computer_decisions[player])): 
                        input_tensor = create_input_tensor(val_computer_decisions, val_player_decisions, player, decision)
                        output = rnn(input_tensor)
                        player_decision_pred = output.argmax(dim=0).item() # find the position of the largest value - the one the model predicted was the most likely
                        correct = (player_decision_pred == (val_player_decisions[player][decision]).long() - 1)
                        batch_acc.append(1 if correct else 0)
                
                print("batch acc: ")
                print(batch_acc)

                epoch_acc = float(np.mean(batch_acc))
                all_accuracies.append(epoch_acc) # TODO: FIX ACCURACIES IDENTICAL, maybe model isn't being saved? 
                print(f"{k} ({k / n_epoch:.0%}): \t average batch loss = {all_losses[-1]} \t epoch accuracy = {epoch_acc}")

        current_loss = 0

    print(f"done training, all losses: {all_losses}, all accuracies: {all_accuracies}")

    return {'num_batches': len(computer_decisions), 'all_losses': all_losses}