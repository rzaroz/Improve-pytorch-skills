import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
import torch.nn.functional as f
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv("../datasets/wustl-ehms-2020_with_attacks_categories.csv")

# Drop bullshits
df = df.drop(["Dir", "Flgs", "SrcAddr", "DstAddr", "DstMac"], axis=1)

# Encode object columns
object_columns = ["Sport","SrcMac"]
for obj in object_columns:
    df[obj] = LabelEncoder().fit_transform(df[obj])


# Define X and Y
X = df.drop(["Label", "Attack Category"], axis=1)
y = df["Attack Category"]

# Scale X
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
input_size = X.shape[1]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
output_size = len(Counter(y).keys())

# Spit data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Convert to tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define model
class Model(nn.Module):

    def __init__(self, feature_size, out_results):
        super().__init__()
        self.input_size = feature_size
        self.output_size = out_results

        self.input_layer = nn.Linear(self.input_size, 60)
        self.hidden_layer_1 = nn.Linear(60, 75)
        self.hidden_layer_2 = nn.Linear(75, 90)
        self.output_layer = nn.Linear(90, self.output_size)

    def forward(self, x):
        x = f.relu(self.input_layer(x))
        x = f.relu(self.hidden_layer_1(x))
        x = f.relu(self.hidden_layer_2(x))
        x = f.relu(self.output_layer(x))

        return x


model = Model(input_size, output_size)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Train
epochs = 500
losses = []
for e in range(epochs):
    y_pred = model.forward(X_train_tensor)

    # Measure loss
    loss = loss_function(y_pred, y_train_tensor)
    losses.append(loss.detach().numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 10 == 0:
        print(f"Loss : {loss.detach().numpy()} | Epoch: {e}")

plt.plot(losses)
plt.xlabel("Loss")
plt.show()

# Test model
y_pred = model(X_test_tensor)
y_pred = torch.argmax(y_pred, dim=1).numpy()

acc = accuracy_score(y_test, y_pred)

print(acc)