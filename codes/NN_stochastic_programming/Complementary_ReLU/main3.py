#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import gurobipy as gp
from gurobipy import GRB
from multiprocessing import Pool, cpu_count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N_PROCESSES = int(os.environ.get("N_PROCESSES", "4"))
GUROBI_THREADS = int(os.environ.get("GUROBI_THREADS", "1"))


################################################################################
# 1. HELPER FUNCTIONS: Instance Generation, Solver, and Greedy Heuristics
################################################################################

def generate_kip_instance(n, k_ratio):
    """Generate a random Knapsack Interdiction Problem instance."""
    profits = np.random.randint(1, 101, size=n)
    weights = np.random.randint(1, 101, size=n)
    k = int(np.ceil(k_ratio * n))
    b = int(np.ceil(((n - k) / (2.0 * n)) * np.sum(weights)))

    # Ensure no item is heavier than b
    while np.any(weights > b):
        weights = np.random.randint(1, 101, size=n)

    return {
        "n": n,
        "k": k,
        "b": b,
        "profits": profits,
        "weights": weights
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
    n, k, b = instance["n"], instance["k"], instance["b"]
    profits, weights = instance["profits"], instance["weights"]
    ratio = profits / (weights + 1e-9)
    order = np.argsort(-ratio)

    x_dg = np.zeros(n, dtype=int)
    x_dg[order[:k]] = 1
    forbidden = x_dg == 1
    y_dg, obj_dg = greedy_packing(profits, weights, b, forbidden=forbidden)
    return x_dg, y_dg, obj_dg

def compute_g_vfa(instance):
    profits, weights, b = instance["profits"], instance["weights"], instance["b"]
    y_g, _ = greedy_packing(profits, weights, b, forbidden=None)
    return y_g

def solve_follower(x, instance):
    profits, weights, b = instance["profits"], instance["weights"], instance["b"]
    n = len(profits)

    model = gp.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", GUROBI_THREADS)
    model.setParam("Method", 1)  # dual simplex

    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    model.addConstr(gp.quicksum(weights[i] * y[i] for i in range(n)) <= b)
    model.addConstrs((y[i] <= 1 - x[i] for i in range(n)))

    model.setObjective(gp.quicksum(profits[i] * y[i] for i in range(n)), GRB.MAXIMIZE)
    model.optimize()
    return model.objVal

################################################################################
# 2. DATA GENERATION (FLAT FORMAT)
################################################################################

def generate_data_flat(instance, num_samples):
    n = instance["n"]
    k = instance["k"]
    profits = instance["profits"]
    weights = instance["weights"]

    # Precompute global info
    ratio = profits / (weights + 1e-9)
    max_ratio = ratio.max()
    x_dg, y_dg, obj_dg = compute_greedy_dg_features(instance)
    y_g = compute_g_vfa(instance)

    k_n_ratio = k / n
    obj_dg_n = obj_dg / n

    X = np.zeros((num_samples, n * 9), dtype=np.float32)
    y_vals = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        # Random interdiction
        x_i = np.zeros(n, dtype=int)
        subset_size = np.random.randint(0, k + 1)
        idx_chosen = np.random.choice(n, subset_size, replace=False)
        x_i[idx_chosen] = 1

        # Solve follower
        follower_obj = solve_follower(x_i, instance)
        y_vals[i] = follower_obj

        # Build features
        features_i = np.zeros((n, 9), dtype=np.float32)
        for j in range(n):
            features_i[j, 0] = ratio[j] / max_ratio
            features_i[j, 1] = profits[j]
            features_i[j, 2] = weights[j]
            features_i[j, 3] = k_n_ratio
            features_i[j, 4] = x_dg[j]
            features_i[j, 5] = y_dg[j]
            features_i[j, 6] = obj_dg_n
            features_i[j, 7] = x_i[j]
            features_i[j, 8] = y_g[j]

        # Flatten to shape (n*9,)
        X[i, :] = features_i.flatten()

    return X, y_vals

def generate_data_flat_for_instance(args):
    """Wrapper for parallel calls."""
    instance, num_samples = args
    return generate_data_flat(instance, num_samples)

################################################################################
# 3. SIMPLE FEEDFORWARD NETWORK (FLAT INPUT)
################################################################################

class SimpleFFN(nn.Module):
    """
    Takes a single 1D input of length (n * 9) and outputs a single scalar.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.model(x)

################################################################################
# 4. TRAINING LOOP (NO MINI-BATCHES)
################################################################################

def train_model_no_batch(model, X_train, y_train, X_val, y_val,
                         lr=1e-3, num_epochs=200, patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        pred_train = model(X_train)
        loss_train = criterion(pred_train, y_train)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            loss_val = criterion(pred_val, y_val)

        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {loss_train.item():.4f}, "
                  f"Val Loss: {loss_val.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

################################################################################
# 5. PLOTTING / EVALUATION
################################################################################

def plot_actual_vs_pred(y_true, y_pred, label_str="flat"):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("\nModel Performance Metrics:")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    grid = np.linspace(min_val, max_val, 100)
    plt.plot(grid, grid, 'r--', label='y = x')

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted ({label_str})")
    plt.legend()
    plt.tight_layout()

    os.makedirs("flat_plots", exist_ok=True)
    plot_path = f"flat_plots/actual_vs_pred_{label_str}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")

################################################################################
# 6. MAIN
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Flat Model")
    parser.add_argument("--n_values", type=int, nargs='+', required=True,
                        help="List of n values")
    parser.add_argument("--k_ratios", type=float, nargs='+', required=True,
                        help="List of k_ratios")
    parser.add_argument("--instances_per_combination", type=int, required=True,
                        help="Number of instances per combination")
    parser.add_argument("--num_samples_per_instance", type=int, required=True,
                        help="Number of samples per instance")
    args = parser.parse_args()

    # For simplicity, we take the first n and k_ratio from the user input
    n = args.n_values[0]
    k_ratio = args.k_ratios[0]
    instances_per_combination = args.instances_per_combination
    num_samples_per_instance = args.num_samples_per_instance

    print(f"Processing n={n}, k_ratio={k_ratio:.2f}")
    print(f"Instances per combination: {instances_per_combination}, "
          f"Samples per instance: {num_samples_per_instance}")

    # 1) Generate random instances
    instances = [generate_kip_instance(n, k_ratio) for _ in range(instances_per_combination)]

    # 2) Generate data in parallel
    process_args = [(inst, num_samples_per_instance) for inst in instances]
    num_workers = min(cpu_count(), len(instances))
    print(f"Using {num_workers} worker processes (pool).")

    with Pool(num_workers) as pool:
        results = pool.map(generate_data_flat_for_instance, process_args)

    # 3) Concatenate results (X_flat, y)
    X_list = []
    y_list = []
    for X_part, y_part in results:
        X_list.append(X_part)
        y_list.append(y_part)
    X_full = np.concatenate(X_list, axis=0)  # shape: (total_samples, n*9)
    y_full = np.concatenate(y_list, axis=0)  # shape: (total_samples,)

    # 4) Split into train/val/test
    indices = np.arange(len(y_full))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)
    # Now we have 60% train, 20% val, 20% test overall

    X_train = X_full[train_indices]
    y_train = y_full[train_indices]
    X_val = X_full[val_indices]
    y_val = y_full[val_indices]
    X_test = X_full[test_indices]
    y_test = y_full[test_indices]

    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # 5) Convert to torch Tensors (on CPU or GPU)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32, device=device).unsqueeze(-1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32, device=device).unsqueeze(-1)

    # 6) Build and Train Model (no mini-batches)
    input_dim = n * 9
    model = SimpleFFN(input_dim=input_dim, hidden_dim=128).to(device)
    print(f"Training model with input_dim={input_dim} ...")

    model = train_model_no_batch(
        model,
        X_train_t, y_train_t,
        X_val_t,   y_val_t,
        lr=1e-3,
        num_epochs=1000,
        patience=200
    )
    print("Training complete.")

    # 7) Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).cpu().numpy().flatten()

    # Plot actual vs predicted
    plot_actual_vs_pred(y_test, y_pred_test, label_str=f"n{n}_k{k_ratio:.2f}_flat")

    # 8) Export to ONNX
    os.makedirs("flat_models", exist_ok=True)
    onnx_path = f"flat_models/model_n{n}_k{k_ratio:.2f}_flat.onnx"

    dummy_input = torch.randn(1, input_dim, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"Model exported to {onnx_path}")
