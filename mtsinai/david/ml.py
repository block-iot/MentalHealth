import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import loadmat


def GetDataFrame():

    numeric_data_mat = loadmat('../data/covid_t1/numeric_data.mat')
    numeric_data_df = pd.read_csv("../data/week1_csv/numeric_data.csv", header=None)

    numeric_headers = []

    for i in range(len(numeric_data_mat['numeric_data_headers'][0])):
        numeric_headers.append(numeric_data_mat['numeric_data_headers'][0][i][0])

    numeric_data_df.columns = numeric_headers

    return numeric_data_df


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression output
        return x


numeric_data_df = GetDataFrame()

# Splitting the data
X = numeric_data_df[["age", "gender", "edu_level", "job"]].values
y = numeric_data_df["stai_s_score"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Instantiate the network, loss, and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training loop
epochs = 500
for epoch in range(epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = net(X_train_tensor)
    
    # Compute loss
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Print statistics
    if epoch % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print('Finished Training')

# Evaluate on the test set
with torch.no_grad():
    test_outputs = net(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")
