from models import CustomRNN, NeuralNet
import torch
from torch import nn
import random
import numpy as np
import time

def train_nn(nn, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=nn.NLLLoss()): 
    current_loss = 0
    all_losses = []
    nn.train()
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)


def train_rnn(rnn, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=nn.NLLLoss()): 
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"Training on dataset with n={len(training_data)}")

    for k in range(1, n_epoch + 1): 
        rnn.zero_grad()

        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches): 
            batch_loss = 0
            for i in batch: 
                # get the tensor I need from training_data[i]
                output = rnn.forward()
                loss = criterion(output, _____)
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

    return all_losses