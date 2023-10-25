import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat


def GetDataFrame():

    numeric_data_mat = loadmat('../data/covid_t1/numeric_data_t1.mat')
    numeric_data_df = pd.read_csv("../data/extra/output.csv", header=None)

    numeric_headers = []

    for i in range(len(numeric_data_mat['numeric_data_headers'][0])):
        numeric_headers.append(numeric_data_mat['numeric_data_headers'][0][i][0])

    numeric_data_df.columns = numeric_headers

    return numeric_data_df


#WHERE IS TIMESTAMP COL?
# stai_s_score/columns, stai_t_score/columns
def filter_df(numeric_data_df):


    remove_cols = numeric_data_df.loc[:,numeric_data_df.isnull().any()].columns.tolist()

    if "sds_s_score" in remove_cols:
        remove_cols.remove("sds_s_score")

    if "stai_s_score" in remove_cols:
        remove_cols.remove("stai_s_score")

    additionalCols = ["stai_t_score"]
    for i in range(1, 21):
        if i != 2:
            additionalCols.extend(["sds_" + str(i), "stais" + str(i), "stait" + str(i)])

    remove_cols.extend(additionalCols)
    remove_cols = list(set(remove_cols))
    numeric_data_df.drop(remove_cols, axis=1, inplace=True)  
    
     

    numeric_data_df = numeric_data_df.select_dtypes(include=['float64', 'int64'])
    return numeric_data_df.dropna(subset=['sds_score', 'stai_s_score'])





class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



numeric_data_df = GetDataFrame()

numeric_data_df = filter_df(numeric_data_df)


X = numeric_data_df.drop(['sds_score', 'stai_s_score'], axis=1).values

y = numeric_data_df[['sds_score', 'stai_s_score']].values


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


model = NeuralNet(266)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 38
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
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

