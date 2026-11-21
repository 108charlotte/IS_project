import pandas as pd
from data_cleaning import cols_btw
import torch
from training_functions import train_nn
from models import NeuralNet
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from testing_functions import get_avg_baseline

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

input_train, input_val_test, result_train, result_val_test = train_test_split(input_data, result, test_size=0.2, train_size=0.8)
input_val, input_test, result_val, result_test = train_test_split(input_val_test, result_val_test, test_size=0.5, train_size=0.5)

nn = NeuralNet(20)
out = train_nn(nn, input_train, result_train, input_val, result_val)
print(f"{out=}")