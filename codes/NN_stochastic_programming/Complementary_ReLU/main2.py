import os
import json
import gurobipy as gp
from gurobipy import GRB, Model
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from gurobi_ml import add_predictor_constr
from pyomo.environ import *
import onnx
from onnx import load
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
# from onnx2torch import convert
import onnx
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler
import argparse 


######################################
# Model Architecture
######################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# class Psi_d(nn.Module):
#     def __init__(self, f_dim, m=64):
#         super(Psi_d, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(f_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, m)
#         )

#     def forward(self, f_i):
#         batch_size, n, _ = f_i.size()
#         f_flat = f_i.view(batch_size*n, -1)
#         emb = self.fc(f_flat)
#         emb = emb.view(batch_size, n, -1)
#         return emb

# class Psi_s(nn.Module):
#     def __init__(self, m=64, k=32):
#         super(Psi_s, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(m, 128),
#             nn.ReLU(),
#             nn.Linear(128, k)
#         )

#     def forward(self, emb_sum):
#         return self.fc(emb_sum)

# class Psi_v(nn.Module):
#     def __init__(self, k=32, h_dim=9, output_dim=1):
#         super(Psi_v, self).__init__()
#         input_dim = k + h_dim
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )

#     def forward(self, instance_emb, h_x_i, x_i, is_kip=True):
#         batch_size, n, hdim = h_x_i.size()
#         instance_emb_expanded = instance_emb.unsqueeze(1).repeat(1, n, 1)
#         concat_vec = torch.cat([instance_emb_expanded, h_x_i], dim=2)
#         if is_kip:
#             mask = (1 - x_i).unsqueeze(2)
#             concat_vec = concat_vec * mask
#         concat_flat = concat_vec.view(batch_size*n, -1)
#         out = self.fc(concat_flat)
#         out = out.view(batch_size, n)
#         return out

# class ValueFunctionApproximator(nn.Module):
#     def __init__(self, f_dim=7, h_dim=9, is_kip=True):
#         super(ValueFunctionApproximator, self).__init__()
#         self.m = 64
#         self.k = 32
#         self.is_kip = is_kip

#         self.psi_d = Psi_d(f_dim, m=self.m)
#         self.psi_s = Psi_s(m=self.m, k=self.k)
#         self.psi_v = Psi_v(k=self.k, h_dim=h_dim, output_dim=1)

#     def forward(self, f_i, h_x_i, c_i, x_i):
#         emb = self.psi_d(f_i)
#         emb_sum = torch.sum(emb, dim=1)
#         instance_emb = self.psi_s(emb_sum)
#         pred = self.psi_v(instance_emb, h_x_i, x_i, is_kip=self.is_kip)
#         final_output = torch.sum(pred * c_i, dim=1)
#         return final_output.unsqueeze(1)


import torch
import torch.nn as nn

class ValueFunctionApproximator(nn.Module):
    def __init__(self, f_dim=7, h_dim=9, output_dim=1, is_kip=True):
        super(ValueFunctionApproximator, self).__init__()
        self.is_kip = is_kip

        self.fc_emb = nn.Sequential(
            nn.Linear(f_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32) 
        )
        
        self.fc_pred = nn.Sequential(
            nn.Linear(32 + h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, f_i, h_x_i, c_i, x_i):
        batch_size, n, _ = f_i.size()
        
        f_flat = f_i.view(batch_size * n, -1)
        emb = self.fc_emb(f_flat)  # (batch_size * n, 32)
        emb = emb.view(batch_size, n, -1)  # (batch_size, n, 32)
        
        concat_vec = torch.cat([emb, h_x_i], dim=2)  # (batch_size, n, 32 + h_dim)

        if self.is_kip:
            mask = (1 - x_i).unsqueeze(2)  # (batch_size, n, 1)
            concat_vec = concat_vec * mask

        concat_flat = concat_vec.view(batch_size * n, -1)
        pred = self.fc_pred(concat_flat).view(batch_size, n)  # (batch_size, n)
        
        final_output = torch.sum(pred * c_i, dim=1)  # (batch_size,)
        
        return final_output.unsqueeze(1)  # (batch_size, 1)



######################################
# Model Train
######################################


# def train_model(model, train_loader, lr=0.01, num_epochs=100, patience=10):
#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     best_loss = float('inf')
#     patience_counter = 0
#     best_state = None
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0
#         for f_i_batch, h_x_i_batch, c_i_batch, x_i_batch, y_batch in train_loader:
#             f_i_batch, h_x_i_batch, c_i_batch, x_i_batch, y_batch = \
#                   f_i_batch.to(device), h_x_i_batch.to(device), c_i_batch.to(device), \
#                   x_i_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             outputs = model(f_i_batch, h_x_i_batch, c_i_batch, x_i_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * f_i_batch.size(0)
#         epoch_loss /= len(train_loader.dataset)

#         if epoch_loss < best_loss:
#             best_loss = epoch_loss
#             best_state = model.state_dict()
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print("Early stopping!")
#                 break

#         if (epoch+1) % 10 == 0:
#             print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

#     model.load_state_dict(best_state)
#     return model

def train_model(model, train_loader, val_loader, lr=0.01, num_epochs=1000, patience=200):
    model.to(device)
    criterion = nn.MSELoss()
    val_criterion = nn.L1Loss()  # Use MAE for validation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop
        for f_i_batch, h_x_i_batch, c_i_batch, x_i_batch, y_batch in train_loader:
            f_i_batch, h_x_i_batch, c_i_batch, x_i_batch, y_batch = (
                f_i_batch.to(device),
                h_x_i_batch.to(device),
                c_i_batch.to(device),
                x_i_batch.to(device),
                y_batch.to(device),
            )
            optimizer.zero_grad()
            outputs = model(f_i_batch, h_x_i_batch, c_i_batch, x_i_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * f_i_batch.size(0)

        epoch_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for f_i_batch, h_x_i_batch, c_i_batch, x_i_batch, y_batch in val_loader:
                f_i_batch, h_x_i_batch, c_i_batch, x_i_batch, y_batch = (
                    f_i_batch.to(device),
                    h_x_i_batch.to(device),
                    c_i_batch.to(device),
                    x_i_batch.to(device),
                    y_batch.to(device),
                )
                outputs = model(f_i_batch, h_x_i_batch, c_i_batch, x_i_batch)
                val_mae += val_criterion(outputs, y_batch).item() * f_i_batch.size(0)

        val_mae /= len(val_loader.dataset)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement in validation MAE!")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation MAE: {val_mae:.4f}")

    model.load_state_dict(best_state)
    return model


######################################
# Model Performance
######################################

def plot_actual_vs_pred(actual, pred, n):
    r2 = r2_score(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)

    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Create scatter plot
    plt.figure()
    plt.scatter(actual, pred, alpha=0.5)
    max_val = max(actual.max(), pred.max())
    min_val = min(actual.min(), pred.min())
    line_vals = np.linspace(min_val, max_val, 100)
    plt.plot(line_vals, line_vals, 'r--', label='y=x')

    # Add performance metrics to the plot
    metrics_text = f"R²: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, color="white"))

    # Add labels and title
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted for n={n}")
    plt.legend()

    # Save the plot
    plt.savefig(f"plots/actual_vs_pred_n{n}.png")
    plt.close()


######################################
# Instance and Feature Generation
######################################

def generate_kip_instance(n, k_ratio):
    # Vectorized instance generation
    profits = np.random.randint(1, 101, size=n)
    weights = np.random.randint(1, 101, size=n)
    k = int(np.ceil(k_ratio * n))
    b = int(np.ceil(((n - k) / (2 * n)) * np.sum(weights)))
    
    # Vectorized weight check and regeneration
    while np.any(weights > b):
        weights = np.random.randint(1, 101, size=n)
    
    return {
        "n": n,
        "k": k,
        "b": b,
        "profits": profits,  # Keep as numpy array
        "weights": weights   # Keep as numpy array
    }

def greedy_packing(profits, weights, capacity, forbidden=None):
    n = len(profits)
    if forbidden is None:
        forbidden = np.zeros(n, dtype=bool)
    ratio = profits/(weights+1e-9)
    order = np.argsort(-ratio)
    y = np.zeros(n, dtype=int)
    cap = capacity
    for i in order:
        if (not forbidden[i]) and (weights[i] <= cap):
            y[i] = 1
            cap -= weights[i]
    obj_val = np.sum(profits * y)
    return y, obj_val

def compute_greedy_dg_features(instance):
    n = instance['n']
    k = instance['k']
    b = instance['b']
    profits = np.array(instance['profits'])
    weights = np.array(instance['weights'])
    ratio = profits / (weights+1e-9)
    order = np.argsort(-ratio)
    x_dg = np.zeros(n, dtype=int)
    top_k = order[:k]
    x_dg[top_k] = 1
    forbidden = x_dg == 1
    y_dg, obj_dg = greedy_packing(profits, weights, b, forbidden=forbidden)
    return x_dg, y_dg, obj_dg

def compute_g_vfa(instance):
    n = instance['n']
    b = instance['b']
    profits = np.array(instance['profits'])
    weights = np.array(instance['weights'])
    y_g, _ = greedy_packing(profits, weights, b, forbidden=None)
    return y_g

def solve_follower(x, instance):
    # Pre-extract values to avoid dictionary lookups
    weights = instance['weights']
    profits = instance['profits']
    b = instance['b']

    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)  # Important: Use single thread for parallel processing
    model.setParam('Method', 1)   # Use dual simplex - faster for this type of problem
    
    # Create variables all at once
    y = model.addVars(len(weights), vtype=GRB.BINARY)
    
    # Add constraints more efficiently
    model.addConstr(gp.quicksum(weights[i]*y[i] for i in range(len(weights))) <= b)
    model.addConstrs((y[i] <= 1 - x[i] for i in range(len(weights))))
    
    # Set objective more efficiently
    model.setObjective(gp.quicksum(profits[i]*y[i] for i in range(len(weights))), GRB.MAXIMIZE)
    
    model.optimize()
    return model.objVal

def generate_data_for_instance(args):
    instance, num_samples = args
    n = instance['n']
    k = instance['k']
    profits = instance['profits']  # Already numpy array
    weights = instance['weights']  # Already numpy array
    
    # Pre-compute common values once
    ratio = profits/(weights+1e-9)
    max_ratio = ratio.max()
    x_dg, y_dg, obj_dg = compute_greedy_dg_features(instance)
    y_g = compute_g_vfa(instance)
    
    # Pre-allocate arrays
    f_i_list = np.zeros((num_samples, n, 7))
    h_x_i_list = np.zeros((num_samples, n, 9))
    c_i_list = np.zeros((num_samples, n))
    x_i_list = np.zeros((num_samples, n))
    y_list = np.zeros(num_samples)

    # Pre-compute repeated values
    k_n_ratio = k/n
    obj_dg_n = obj_dg/n
    
    # Fill constant features for all samples at once
    f_i_list[:, :, 0] = (ratio/max_ratio)[np.newaxis, :]
    f_i_list[:, :, 1] = profits[np.newaxis, :]
    f_i_list[:, :, 2] = weights[np.newaxis, :]
    f_i_list[:, :, 3] = k_n_ratio
    f_i_list[:, :, 4] = x_dg[np.newaxis, :]
    f_i_list[:, :, 5] = y_dg[np.newaxis, :]
    f_i_list[:, :, 6] = obj_dg_n
    
    # Fill c_i for all samples at once
    c_i_list[:] = profits

    for i in range(num_samples):
        # Generate x more efficiently
        x = np.zeros(n, dtype=int)
        subset_size = np.random.randint(0, k+1)
        x[np.random.choice(n, subset_size, replace=False)] = 1
        
        # Store x
        x_i_list[i] = x
        
        # Compute objective value
        y_list[i] = solve_follower(x, instance)
        
        # Create h_x_i efficiently
        h_x_i_list[i] = np.concatenate([
            f_i_list[i],
            x.reshape(-1, 1),
            y_g.reshape(-1, 1)
        ], axis=1)

    return f_i_list, h_x_i_list, c_i_list, x_i_list, y_list



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and Evaluate Model")
    parser.add_argument("--n_values", type=int, nargs='+', required=True, help="List of n values")
    parser.add_argument("--k_ratios", type=float, nargs='+', required=True, help="List of k_ratios")
    parser.add_argument("--instances_per_combination", type=int, required=True, help="Number of instances per combination")
    parser.add_argument("--num_samples_per_instance", type=int, required=True, help="Number of samples per instance")

    args = parser.parse_args()

    n = args.n_values[0]
    k_ratio = args.k_ratios[0]
    instances_per_combination = args.instances_per_combination
    num_samples_per_instance = args.num_samples_per_instance

    print(f"Processing n={n}, k_ratio={k_ratio:.2f}...")
            
    # Generate all instances first
    instances = [generate_kip_instance(n, k_ratio) 
            for _ in range(args.instances_per_combination)]
    
    # Prepare arguments for parallel processing
    process_args = [(inst, args.num_samples_per_instance) 
                for inst in instances]
    num_workers = min(cpu_count(), len(instances))
    print(f"num_workers: {num_workers}")
    # Process instances in parallel
    with Pool(num_workers) as pool:
        results = pool.map(generate_data_for_instance, process_args)
    
    # Combine results
    f_i_list = []
    h_x_i_list = []
    c_i_list = []
    x_i_list = []
    y_list = []
    
    for result in results:
        f_i_list.append(result[0])
        h_x_i_list.append(result[1])
        c_i_list.append(result[2])
        x_i_list.append(result[3])
        y_list.append(result[4])
    
    # Convert to final numpy arrays
    f_i_full = np.concatenate(f_i_list, axis=0)
    h_x_i_full = np.concatenate(h_x_i_list, axis=0)
    c_i_full = np.concatenate(c_i_list, axis=0)
    x_i_full = np.concatenate(x_i_list, axis=0)
    y_full = np.concatenate(y_list, axis=0)

    # Split train/test sets
    # train_indices, test_indices = train_test_split(np.arange(len(y_full)), test_size=0.2, random_state=42)
    # Split train/test/validation sets
    train_indices, temp_indices = train_test_split(np.arange(len(y_full)), test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)


    f_i_train = f_i_full[train_indices]
    h_x_i_train = h_x_i_full[train_indices]
    c_i_train = c_i_full[train_indices]
    x_i_train = x_i_full[train_indices]
    y_train = y_full[train_indices]

    f_i_val = f_i_full[val_indices]
    h_x_i_val = h_x_i_full[val_indices]
    c_i_val = c_i_full[val_indices]
    x_i_val = x_i_full[val_indices]
    y_val = y_full[val_indices]


    f_i_test = f_i_full[test_indices]
    h_x_i_test = h_x_i_full[test_indices]
    c_i_test = c_i_full[test_indices]
    x_i_test = x_i_full[test_indices]
    y_test = y_full[test_indices]

    # Convert to PyTorch tensors
    f_i_train_tensor = torch.tensor(f_i_train, dtype=torch.float32)
    h_x_i_train_tensor = torch.tensor(h_x_i_train, dtype=torch.float32)
    c_i_train_tensor = torch.tensor(c_i_train, dtype=torch.float32)
    x_i_train_tensor = torch.tensor(x_i_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    f_i_val_tensor = torch.tensor(f_i_val, dtype=torch.float32)
    h_x_i_val_tensor = torch.tensor(h_x_i_val, dtype=torch.float32)
    c_i_val_tensor = torch.tensor(c_i_val, dtype=torch.float32)
    x_i_val_tensor = torch.tensor(x_i_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    train_dataset = TensorDataset(f_i_train_tensor, h_x_i_train_tensor, c_i_train_tensor, x_i_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(f_i_val_tensor, h_x_i_val_tensor, c_i_val_tensor, x_i_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train model
    model = ValueFunctionApproximator(f_dim=7, h_dim=9, is_kip=True).to(device)
    print(f"Training model for n={n}, k_ratio={k_ratio:.2f}...")
    # model = train_model(model, train_loader, lr=0.01, num_epochs=1000, patience=30)
    model = train_model(model, train_loader, val_loader, lr=0.01, num_epochs=1000)
    print(f"Model trained for n={n}, k_ratio={k_ratio:.2f}.")

    # Save model
    model_save_path = f"models/model_n{n}_k{k_ratio:.2f}.onnx"
    dummy_input_f = torch.randn(1, n, 7).to(device)
    dummy_input_h = torch.randn(1, n, 9).to(device)
    dummy_input_c = torch.randn(1, n).to(device)
    dummy_input_x = torch.randn(1, n).to(device)

    torch.onnx.export(
        model,
        (dummy_input_f, dummy_input_h, dummy_input_c, dummy_input_x),
        model_save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["f_i", "h_x_i", "c_i", "x_i"],
        output_names=["output"],
        dynamic_axes={"f_i": {0: "batch_size"}, "h_x_i": {0: "batch_size"}, "c_i": {0: "batch_size"},
                        "x_i": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Model saved to {model_save_path}")

    # Evaluate on the test set
    f_i_test_tensor = torch.tensor(f_i_test, dtype=torch.float32).to(device)
    h_x_i_test_tensor = torch.tensor(h_x_i_test, dtype=torch.float32).to(device)
    c_i_test_tensor = torch.tensor(c_i_test, dtype=torch.float32).to(device)
    x_i_test_tensor = torch.tensor(x_i_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        pred_test = model(f_i_test_tensor, h_x_i_test_tensor, c_i_test_tensor, x_i_test_tensor).cpu().numpy().flatten()

    # Inverse transform predictions
    pred_test_rescaled = pred_test.reshape(-1, 1).flatten()
    actual_test_rescaled = y_test.reshape(-1, 1).flatten()

    # Plot actual vs predicted
    plot_actual_vs_pred(actual_test_rescaled, pred_test_rescaled, f"n{n}_k{k_ratio:.2f}")
    print(f"Finished processing n={n}, k_ratio={k_ratio:.2f}. Plot saved.")

