import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['v_i'] = data['v_i'].apply(eval)
    data['xi_i'] = data['xi_i'].apply(eval)
    
    # Prepare features and target
    X_linear = np.hstack([np.vstack(data['v_i']), np.vstack(data['xi_i'])])
    y = data['Q_i'].values
    
    # One-hot encoding for 'i' column
    encoder = OneHotEncoder(sparse=False)
    i_encoded = encoder.fit_transform(data[['i']])
    X_with_i = np.hstack([i_encoded, X_linear])
    
    # Split data
    X_train_linear, X_test_linear, X_train_with_i, X_test_with_i, y_train, y_test = train_test_split(
        X_linear, X_with_i, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_linear = scaler.fit_transform(X_train_linear)
    X_test_linear = scaler.transform(X_test_linear)
    X_train_with_i = scaler.fit_transform(X_train_with_i)
    X_test_with_i = scaler.transform(X_test_with_i)
    
    return X_train_linear, X_test_linear, X_train_with_i, X_test_with_i, y_train, y_test

# Linear Regression
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Decision Tree
def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# FeedForward Neural Network using PyTorch
class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

def train_nn(X_train, y_train, epochs=100, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForwardNN(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

# Plotting function
def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = 'training_data.csv'
    X_train_linear, X_test_linear, X_train_with_i, X_test_with_i, y_train, y_test = load_and_preprocess_data(file_path)
    
    # Linear Regression
    linear_model = train_linear_regression(X_train_linear, y_train)
    y_pred_linear = linear_model.predict(X_test_linear)
    plot_results(y_test, y_pred_linear, 'Linear Regression')
    
    # Decision Tree
    tree_model = train_decision_tree(X_train_with_i, y_train)
    y_pred_tree = tree_model.predict(X_test_with_i)
    plot_results(y_test, y_pred_tree, 'Decision Tree')
    
    # Neural Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = train_nn(X_train_with_i, y_train)
    nn_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_with_i).to(device)
        y_pred_nn = nn_model(X_test_tensor).cpu().numpy()
    plot_results(y_test, y_pred_nn, 'FeedForward Neural Network')