import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import ast
from sklearn.decomposition import PCA
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_saved_data(file_name='integrated_data.csv', load_format='csv'):
    if load_format == 'csv':
        df = pd.read_csv(file_name)
    elif load_format == 'json':
        df = pd.read_json(file_name)
    else:
        print("Unsupported format. Please use 'csv' or 'json'.")
        return None

    # Convert string representations of lists back to actual lists
    for col in ['xi_i', 'v_i']:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x))

    return df.to_dict(orient='records')

def analyze_model(model, test_loader, scaler_qi):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i_batch, xi_batch, vi_batch, qi_batch in test_loader:
            i_batch, xi_batch, vi_batch = i_batch.to(device), xi_batch.to(device), vi_batch.to(device)
            outputs = model(i_batch, xi_batch, vi_batch)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(qi_batch.numpy())
    
    # Inverse transform the predictions and targets
    all_predictions = scaler_qi.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
    all_targets = scaler_qi.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    
    # Model size and sparsity
    total_params = sum(p.numel() for p in model.parameters())
    #zero_params = sum(p.numel() for p in model.parameters() if p.data.sum().item() == 0)
    threshold = 1e-5  # 예시 임계값
    zero_params = sum((torch.abs(p) < threshold).sum().item() for p in model.parameters())
    sparsity = zero_params / total_params
    
    print(f"Model Size (number of parameters): {total_params}")
    print(f"Model Sparsity: {sparsity:.4f}")
    
    # Actual vs Predicted plot (y=x 그래프)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--', lw=2)
    plt.xlabel("Actual Q_i")
    plt.ylabel("Predicted Q_i")
    plt.title("Actual vs Predicted Q_i")
    plt.savefig('[NNE_Plot]_actual_vs_predicted_y_equals_x.png')
    plt.close()
    
    # Residual plot
    residuals = all_targets - all_predictions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=all_targets, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Actual Q_i")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig('[NNE_Plot]_residual_plot.png')
    plt.close()

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig('[NNE_Plot]_loss_history.png')
    plt.close()


def preprocess_data(data):
    i_data = np.array([item['i'] for item in data])
    xi_data = np.array([item['xi_i'] for item in data])
    vi_data = np.array([item['v_i'] for item in data])
    qi_data = np.array([item['q_i'] for item in data])

    scaler_xi = StandardScaler()
    scaler_vi = StandardScaler()
    scaler_qi = StandardScaler()

    xi_data_normalized = scaler_xi.fit_transform(xi_data)
    vi_data_normalized = scaler_vi.fit_transform(vi_data)

    # qi_data를 2차원 배열로 변환
    qi_data = qi_data.reshape(-1, 1)
    qi_data_normalized = scaler_qi.fit_transform(qi_data)

    return i_data, xi_data_normalized, vi_data_normalized, qi_data_normalized, scaler_xi, scaler_vi, scaler_qi


class SimpleFeedforwardNN(nn.Module):
    def __init__(self, input_dim_xi, input_dim_vi):
        super(SimpleFeedforwardNN, self).__init__()
        
        self.fc1_i = nn.Linear(1, 16)  # New layer for i_input
        self.fc1_xi = nn.Linear(input_dim_xi, 64)
        self.fc1_vi = nn.Linear(input_dim_vi, 64)
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.fc2 = nn.Linear(64 + 64 + 16, 64)  # Adjusted input size
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, i_input, xi_input, vi_input):
        i_input = i_input.float()  # Convert i_input to float
        i_out = self.relu(self.fc1_i(i_input.unsqueeze(1)))  # Unsqueeze to match dimensions
        xi_out = self.relu(self.fc1_xi(xi_input))
        vi_out = self.relu(self.fc1_vi(vi_input))
        
        combined = torch.cat([i_out, xi_out, vi_out], dim=1)
        combined = self.dropout(combined)  # Apply dropout
        out = self.relu(self.fc2(combined))
        out = self.fc3(out)
        
        return out

def train_model(model, train_loader, val_loader, epochs=1000, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i_batch, xi_batch, vi_batch, qi_batch in train_loader:
            i_batch, xi_batch, vi_batch, qi_batch = i_batch.to(device), xi_batch.to(device), vi_batch.to(device), qi_batch.to(device)

            optimizer.zero_grad()
            outputs = model(i_batch, xi_batch, vi_batch)
            loss = criterion(outputs, qi_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xi_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i_batch, xi_batch, vi_batch, qi_batch in val_loader:
                i_batch, xi_batch, vi_batch, qi_batch = i_batch.to(device), xi_batch.to(device), vi_batch.to(device), qi_batch.to(device)

                outputs = model(i_batch, xi_batch, vi_batch)
                loss = criterion(outputs, qi_batch)
                val_loss += loss.item() * xi_batch.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save losses
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), 'NN_model/best_model.pth')  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    plot_training_history(train_losses, val_losses)
    return model

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig('[NNE_Plot]_simple_nn_loss_history.png')
    plt.close()

def main():
    data_directory = 'NN_training_datasets'
    integrated_data = load_saved_data(file_name='integrated_data.csv', load_format='csv')
    i_data, xi_data, vi_data, qi_data, scaler_xi, scaler_vi, scaler_qi = preprocess_data(integrated_data)
    
    input_dim_xi = xi_data.shape[1]
    input_dim_vi = vi_data.shape[1]

    # The SimpleFeedforwardNN constructor no longer requires period_count
    model = SimpleFeedforwardNN(input_dim_xi, input_dim_vi).to(device)

    X = list(zip(i_data, xi_data, vi_data))
    y = qi_data

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(
        torch.tensor(np.array([x[0] for x in X_train]), dtype=torch.long),
        torch.tensor(np.array([x[1] for x in X_train]), dtype=torch.float32),
        torch.tensor(np.array([x[2] for x in X_train]), dtype=torch.float32),
        torch.tensor(np.array(y_train), dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(np.array([x[0] for x in X_val]), dtype=torch.long),
        torch.tensor(np.array([x[1] for x in X_val]), dtype=torch.float32),
        torch.tensor(np.array([x[2] for x in X_val]), dtype=torch.float32),
        torch.tensor(np.array(y_val), dtype=torch.float32)
    )

    test_dataset = TensorDataset(
        torch.tensor(np.array([x[0] for x in X_test]), dtype=torch.long),
        torch.tensor(np.array([x[1] for x in X_test]), dtype=torch.float32),
        torch.tensor(np.array([x[2] for x in X_test]), dtype=torch.float32),
        torch.tensor(np.array(y_test), dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    trained_model = train_model(model, train_loader, val_loader)
    
    analyze_model(trained_model, test_loader, scaler_qi)
    
    torch.save(trained_model.state_dict(), 'NN_model/simple_feedforward_model.pth')

if __name__ == "__main__":
    main()