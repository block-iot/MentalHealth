import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat


# def GetDataFrame():

#     numeric_data_mat = loadmat('../data/original/covid_t1/numeric_data_t1.mat')
#     numeric_data_df = pd.read_csv("../data/extra/output.csv", header=None)

#     numeric_headers = []

#     for i in range(len(numeric_data_mat['numeric_data_headers'][0])):
#         numeric_headers.append(numeric_data_mat['numeric_data_headers'][0][i][0])

#     numeric_data_df.columns = numeric_headers

#     return numeric_data_df

"""
Where is sds_2 for csv1?
Where is stai_t_score for csv 3?

"""

def calc_scores(vals):

    pass


#WHERE IS TIMESTAMP COL?
# stai_s_score/columns, stai_t_score/columns
def filter_df(numeric_data_df, predict):

    cols = numeric_data_df.loc[:,numeric_data_df.isnull().any()].columns.tolist()

    remove_cols = []
    for col in cols:
        if col not in predict:
            remove_cols.append(col)

    remove_cols.append("sds_score")
    remove_cols.append("stai_s_score")

    # remove_cols.append("stai_t_score")
  
    remove_cols = list(set(remove_cols))
    numeric_data_df.drop(remove_cols, axis=1, inplace=True)  
    
    numeric_data_df = numeric_data_df.select_dtypes(include=['float64', 'int64'])
    return numeric_data_df.dropna(subset=list(predict))



class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 40) #Adjust output to size of len(predict)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


predict = ["stais1", "stais2", "stais3","stais4","stais5","stais6","stais7","stais8","stais9","stais10","stais11","stais12","stais13",
          "stais14","stais15","stais16","stais17","stais18","stais19","stais20", "sds_1", "sds_2", "sds_3","sds_4", "sds_5", "sds_6","sds_7","sds_8","sds_9","sds_10","sds_11","sds_12","sds_13",
          "sds_14","sds_15","sds_16","sds_17","sds_18","sds_19","sds_20"]


curWeek = "3"
numeric_data_df = pd.read_csv("../data/clean/nd_t"+ curWeek + "_clean.csv")

numeric_data_df = filter_df(numeric_data_df, set(predict))


X = numeric_data_df.drop(predict, axis=1).values

y = numeric_data_df[predict].values


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)



#Conver to Tensor Flow and DataLoaders 
batch_size = 64

train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)



model = NeuralNet(X.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 38
for epoch in range(epochs):

    model.train()

    for inputs, labels in train_loader:
        print(inputs.shape, labels.shape)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss.item():.4f} - Validation Loss: {val_loss/len(val_loader):.4f}")


model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader):.4f}")
