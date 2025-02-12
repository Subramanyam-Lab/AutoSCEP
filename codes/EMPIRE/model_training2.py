
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
import ast
import joblib
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pandas as pd
import os
import onnx


def preprocessing_data(data):
    x_i = np.vstack(data['x_i'].apply(ast.literal_eval))    
    v_i = np.vstack(data['v_i'].apply(ast.literal_eval))
    xi_i = np.vstack(data['xi_i'].apply(ast.literal_eval))
    periods = data['period'].values
    # operational_cost = data['operational_cost'].values
    # shed_cost = data['shed_cost'].values
    # y = operational_cost + shed_cost

    y = data['scaled_Q_i'].values

    X_trains, y_trains = [], []
    X_tests, y_tests = [], []
    periods_train, periods_test = [], []

    # Ensure the directory for saving scalers exists
    scaler_dir = 'saved_models/scalers'
    os.makedirs(scaler_dir, exist_ok=True)

    for period in np.unique(periods):
        period_mask = (periods == period)

        period_X = np.hstack([
            x_i[period_mask],
            v_i[period_mask], 
            xi_i[period_mask]
        ])
        # period_X = np.hstack([
        #     v_i[period_mask], 
        #     xi_i[period_mask]
        # ])
        period_y = y[period_mask]

        X_train, X_test, y_train, y_test = train_test_split(
            period_X, period_y, test_size=0.2, random_state=42
        )

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()  
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))  

        joblib.dump(scaler_X, os.path.join(scaler_dir, f'scaler_X_period_{period}.joblib'))
        joblib.dump(scaler_y, os.path.join(scaler_dir, f'scaler_y_period_{period}.joblib'))

        X_trains.append(X_train_scaled)
        y_trains.append(y_train_scaled)
        X_tests.append(X_test_scaled)
        y_tests.append(y_test_scaled)

    return X_trains, y_trains, X_tests, y_tests


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_rates):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create layers based on the provided architecture
        for size, dropout_rate in zip(layer_sizes, dropout_rates):
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = size
            
        # Add final output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return train_losses, val_losses


def plot_residuals_by_period(y_tests, y_preds, periods):
    num_periods = len(periods)
    cols = 3  # Number of columns for the subplot grid
    rows = (num_periods + cols - 1) // cols  # Calculate rows based on periods
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, (y_test, y_pred, period) in enumerate(zip(y_tests, y_preds, periods)):
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()
        
        residuals = y_test - y_pred
        
        # Check shapes
        print(f"Period {period}: y_pred shape: {y_pred.shape}, residuals shape: {residuals.shape}")
        if y_pred.shape != residuals.shape:
            print(f"Warning: Shapes do not match in Period {period}. Skipping this plot.")
            continue  # Skip plotting for this period
        
        # Plot residuals
        axes[i].scatter(y_pred, residuals, alpha=0.6, color='orange')
        axes[i].axhline(0, color='r', linestyle='--', label="Zero Residual Line")
        axes[i].set_title(f'Period {period}')
        axes[i].set_xlabel('Predicted Values')
        axes[i].set_ylabel('Residuals')
        axes[i].legend()
        axes[i].grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle('Residual Analysis by Period', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("saved_models/residual_errors.png")




def evaluate_models(models_list, X_trains, y_trains, X_tests, y_tests, model_name,
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    results = {}
    num_periods = len(models_list)
    cols = 3
    rows = math.ceil(num_periods / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    y_test_preds = []  # Collect predictions for each period
    
    for period, (model, X_train, y_train, X_test, y_test) in enumerate(zip(models_list, X_trains, y_trains, X_tests, y_tests), 1):
        model.eval()
        with torch.no_grad():
            # Convert data to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            # Ensure y_train and y_test are NumPy arrays and flattened
            y_train = np.array(y_train).flatten()
            y_test = np.array(y_test).flatten()
            
            # Train predictions and metrics
            y_train_pred = model(X_train_tensor).cpu().numpy().squeeze()
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            # Test predictions and metrics
            y_test_pred = model(X_test_tensor).cpu().numpy().squeeze()
            
            # Ensure y_test_pred is flattened
            y_test_pred = y_test_pred.flatten()
            
            # Check shapes
            print(f"Period {period}: y_test shape: {y_test.shape}, y_test_pred shape: {y_test_pred.shape}")
            if y_test.shape != y_test_pred.shape:
                print(f"Warning: Shapes do not match in Period {period}. Adjusting shapes.")
                min_length = min(len(y_test), len(y_test_pred))
                y_test = y_test[:min_length]
                y_test_pred = y_test_pred[:min_length]
            
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            results[f"Period_{period}"] = {
                'train_mse': train_mse,
                'train_r2': train_r2,
                'test_mse': test_mse,
                'test_r2': test_r2
            }
            
            print(f"\nPeriod {period} Results:")
            print(f"Model: {model_name}")
            print(f"Train MSE: {train_mse:.4f}, Train R2: {train_r2:.4f}")
            print(f"Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")
            
            # Collect predictions
            y_test_preds.append(y_test_pred)
            
            # Scatter plot for this period
            ax = axes[period - 1]
            ax.scatter(y_test, y_test_pred, alpha=0.6, label=f'Period {period}')
            y_min = min(y_test.min(), y_test_pred.min())
            y_max = max(y_test.max(), y_test_pred.max())
            ax.plot([y_min, y_max], [y_min, y_max], 'r--', label='y=x')
            ax.set_title(f'Period {period}')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.legend()
            ax.grid(True)
        
    # Hide any unused subplots
    for i in range(num_periods, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f'{model_name} - Period Results', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("saved_models/actual_predict_nn.png")
    
    # Now pass the collected predictions to the plotting function
    plot_residuals_by_period(y_tests, y_test_preds, range(1, num_periods + 1))
    
    return results





def train_period_models_with_pytorch(X_trains, y_trains, device='cuda' if torch.cuda.is_available() else 'cpu'):
    models = []
    period_params = []
    
    # Define period-specific parameter grids
    period_specific_params = {
        # Early periods (1-3) - smaller networks, lower dropout
        'early': {
            'layer_sizes': [
                [32, 16, 8],
                [64, 32, 16]
            ],
            'dropout_rates': [
                [0.1, 0.1, 0.1],
                [0.2, 0.1, 0.1]
            ],
            'learning_rate': [0.001, 0.0005],
            'batch_size': [16, 32]
        },
        # Middle periods (4-6) - medium networks, moderate dropout
        'middle': {
            'layer_sizes': [
                [64, 32, 16],
                [128, 64, 32]
            ],
            'dropout_rates': [
                [0.2, 0.2, 0.1],
                [0.3, 0.2, 0.1]
            ],
            'learning_rate': [0.0005, 0.0001],
            'batch_size': [32, 64]
        },
        # Later periods (7-8) - larger networks, higher dropout
        'late': {
            'layer_sizes': [
                [128, 64, 32],
                [256, 128, 64]
            ],
            'dropout_rates': [
                [0.3, 0.3, 0.2],
                [0.4, 0.3, 0.2]
            ],
            'learning_rate': [0.0001, 0.00005],
            'batch_size': [64, 128]
        }
    }

    for period, (X_train, y_train) in enumerate(zip(X_trains, y_trains), 1):
        print(f"\nTuning and training model for Period {period}")
        
        # Select appropriate parameter grid based on period
        if period <= 3:
            param_grid = period_specific_params['early']
        elif period <= 6:
            param_grid = period_specific_params['middle']
        else:
            param_grid = period_specific_params['late']
        
        # Perform hyperparameter tuning for this period
        best_params = hyperparameter_tuning(X_train, y_train, param_grid, device)
        period_params.append(best_params)
        
        # Create dataset and dataloader with best parameters
        dataset = RegressionDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)
        
        # Initialize model with best hyperparameters
        input_size = X_train.shape[1]
        model = NeuralNetwork(
            input_size=input_size,
            layer_sizes=best_params['layer_sizes'],
            dropout_rates=best_params['dropout_rates']
        ).to(device)
        
        # Initialize optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.HuberLoss(reduction='mean', delta=1.0)
        early_stopping = EarlyStopping(patience=30)
        
        # Training loop
        model.train()
        for epoch in range(1000):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/100], Loss: {avg_loss:.4f}')
            
            # Early stopping check
            early_stopping(avg_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        models.append(model)

        # Save the trained model as .onnx
        model_path = os.path.join('saved_models', f'model_period_{period}.onnx')
        torch.onnx.export(
            model, 
            torch.randn(1, input_size).to(device),
            model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        print(f"Model for Period {period} saved as ONNX at {model_path}")
        
        # Print period-specific parameters used
        print(f"\nPeriod {period} Best Parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
    
    return models, period_params

def hyperparameter_tuning(X_train, y_train, param_grid, device='cuda' if torch.cuda.is_available() else 'cpu'):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = defaultdict(list)
    
    # Generate all combinations of hyperparameters
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    for params in param_combinations:
        print(f"\nTesting parameters: {params}")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            # Split data for this fold
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            # Create datasets and dataloaders
            train_dataset = RegressionDataset(X_train_fold, y_train_fold)
            val_dataset = RegressionDataset(X_val_fold, y_val_fold)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
            
            # Initialize model
            input_size = X_train.shape[1]
            model = NeuralNetwork(
                input_size=input_size,
                layer_sizes=params['layer_sizes'],
                dropout_rates=params['dropout_rates']
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.HuberLoss(reduction='mean', delta=1.0)
            early_stopping = EarlyStopping(patience=10)
            
            _, val_losses = train_and_validate(
                model, train_loader, val_loader, criterion, optimizer, device,
                epochs=100, early_stopping=early_stopping
            )
            
            fold_scores.append(min(val_losses))
        
        avg_score = np.mean(fold_scores)
        results['params'].append(params)
        results['scores'].append(avg_score)
        print(f"Average validation loss: {avg_score:.4f}")
    
    best_idx = np.argmin(results['scores'])
    return results['params'][best_idx]


if __name__ == '__main__':
    file_path = 'results_training_data/training_data3_4.csv'
    df = pd.read_csv(file_path)
    X_trains, y_trains, X_tests, y_tests = preprocessing_data(df)

    # Ensure the directory for saving models exists
    os.makedirs('saved_models', exist_ok=True)
    nn_models, period_params = train_period_models_with_pytorch(X_trains, y_trains)
    nn_results = evaluate_models(nn_models, X_trains, y_trains, X_tests, y_tests, 'Neural Network')

