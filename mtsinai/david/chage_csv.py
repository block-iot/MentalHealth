import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat



"""

There are a few things you need to change.

1. Data + Survey == combined full data set. 

2. Week1 has some differences in questions. The other weeks should have approximately the same questions.
 
Find differences weeks (2 - 10) Make all columns standarlized. Find differences week 1 and other weeks.
Any columns that look stagnant you can add them on to the other weeks as well.


*There are some differences in column names as well* so just be aware of that.

Create a function so that you could always pull in dataset into pandas data frame
"""


def GetDataFrame():

    numeric_data_mat = loadmat('../data/covid_t1/numeric_data_t1.mat')
    numeric_data_df = pd.read_csv("../data/csv/week1_csv/numeric_data.csv", header=None)

    numeric_headers = []

    for i in range(len(numeric_data_mat['numeric_data_headers'][0])):
        numeric_headers.append(numeric_data_mat['numeric_data_headers'][0][i][0])

    numeric_data_df.columns = numeric_headers

    return numeric_data_df

df = GetDataFrame()

# Given you have a DataFrame called df with columns 'sds_1', 'sds_2', ... 'sds_20'

df['sds_score'] = (
    df['sds_1'] + 
    df['sds_3'] + df['sds_4'] +
    5 - df['sds_5'] + 5 - df['sds_6'] +
    df['sds_7'] + df['sds_8'] +
    df['sds_9'] + df['sds_10'] +
    5 - df['sds_11'] + 5 - df['sds_12'] +
    df['sds_13'] + 5 - df['sds_14'] +
    df['sds_15'] + 5 -  df['sds_16'] +
    5 - df['sds_17'] + 5 - df['sds_18'] +
    df['sds_19'] + 5 -  df['sds_20']
)



# Write the DataFrame to a CSV file
df.to_csv('output.csv', index=False)
