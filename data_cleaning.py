import pandas as pd
import re

df = pd.read_csv('data_pdf_faces_2017-03-19_16-36(2).csv')
# remove the first row (an extra info row)
df = df.drop(0)

# https://www.geeksforgeeks.org/python/python-extract-numbers-from-string/
def cols_btw(start, end): # "columns between"
    '''
    Returns a list of all columns between a designated start column and a designated end column (inclusive), based on their column numbers. 
    
    :param start: Initial column value. Must have the same non-numeric portion as the column end value. 
    :param end: Final column value. Has a different number (greater) than the start column, but must be otherwise identical. 
    '''
    
    def find_chars(string, excluding): 
        '''
        Returns a string but minus the last occurrence of the "excluding" string
        
        :param string: The string that should be returned without the last occurrence of "excluding" in it
        :param excluding: The string to remove the last occurrence of from "string"
        '''
        
        index = string.rfind(excluding)
        if index == -1: # excluding not found
            print(f"WARNING: column number {excluding=} could not be found in string {string=}")
            return string
        return string[:index] + string[index+len(excluding):]

    start_nums = re.findall(r'-?\d+', start)
    start_n = [int(x) for x in start_nums][-1]
    start_chars = ''
    formatted_start_n = str(start_n).zfill(2)
    start_chars = find_chars(start, formatted_start_n)

    print(f"{start_n=}")
    print(f"{start_chars=}")

    end_nums = re.findall(r'-?\d+', end)
    end_n = [int(x) for x in end_nums][-1]
    formatted_end_n = str(end_n).zfill(2)
    end_chars = find_chars(end, formatted_end_n)

    if end_chars != start_chars: print(f"WARNING: Start column chars: {start_chars} are not the same as end column chars: {end_chars}. This function will not work as intended. Please break up this span into multiple column sections. ")

    to_return = []
    for n in range(start_n, end_n+1): 
        to_return.append(start_chars + str(n).zfill(2))

    return to_return

# retrieving player decision columns
decision_cols = cols_btw('G101', 'G130')
decisions = df[decision_cols]
print(decisions)

# retrieving player score columns to examine relationship btw decisions and scores
player_score_cols = cols_btw("VI03_01", "VI03_30")
player_scores = df[player_score_cols]
print(player_scores)

# retrieving computer score columns
computer_score_cols = cols_btw('VI05_01', 'VI05_30')
computer_scores = df[computer_score_cols]
print(computer_scores)

# retrieving narcissistic indices
ni_cols = cols_btw('NI01', 'NI16')
nis = df[ni_cols]
print(nis)

# retrieving PP01 values
ppo1_cols = cols_btw('PP01_01', 'PP01_40')
ppo1 = df[ppo1_cols]
print(ppo1)