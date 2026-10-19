from models import PredictiveCodingRNN, NeuralNet
import torch
from torch import nn
import random
import numpy as np
import time

def train_nn(nn, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=nn.CrossEntropyLoss()): 
    current_loss = 0
    all_losses = []
    nn.train()
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)


def train_rnn(rnn, computer_decisions, player_decisions, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.1, criterion=nn.NLLLoss()): 
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"Training on dataset with n={len(computer_decisions)}")

    for k in range(1, n_epoch + 1): 
        rnn.zero_grad()

        batches = list(range(len(computer_decisions)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches): 
            batch_loss = 0
            for player in batch: # single decision
                for decision in range(len(computer_decisions[player])): 
                    input_tensor = torch.stack((computer_decisions[player][decision], torch.tensor(0) if decision == 0 else player_decisions[player][decision-1])).reshape(1,2) # decision=0 represents no previous decision
                    input_tensor = input_tensor.to(torch.float32)
                    output = rnn.forward(input_tensor)
                    loss = criterion(output, player_decisions[player][decision].long()-1) # make correct datatype for loss calc
                    batch_loss += loss
            
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
            current_loss += batch_loss.item() / len(batch)
        
        all_losses.append(current_loss / len(batches))
        if k % report_every == 0: 
            print(f"{k} ({k / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    print(f"done training, all losses: {all_losses}")

    return all_losses