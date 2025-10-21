import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import logging
import argparse
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import os
import time
from omlt.io.onnx import write_onnx_model_with_bounds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RegressionDataset(Dataset):
    def __init__(self, df, cost_threshold):
        safe_df = df[df['E_Q'] <= cost_threshold].copy()
        self.v_cols = [col for col in safe_df.columns if col.startswith('v_')]
        self.v_dim = len(self.v_cols)

        self.v_data = safe_df[self.v_cols].values.astype(np.float32)
        self.target_data = safe_df['E_Q'].values.astype(np.float32).reshape(-1, 1)
        
        self.v_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def setup_scalers(self, train_indices, file_prefix):
        self.v_scaler.fit(self.v_data[train_indices])
        self.target_scaler.fit(self.target_data[train_indices])
        
        joblib.dump(self.v_scaler, f'{file_prefix}_v_scaler.gz')
        joblib.dump(self.target_scaler, f'{file_prefix}_y_scaler.gz')
        logging.info("Save scalers done")

    def __len__(self):
        return len(self.v_data)

    def __getitem__(self, idx):
        v_norm = self.v_scaler.transform(self.v_data[idx].reshape(1, -1)).flatten()
        target_norm = self.target_scaler.transform(self.target_data[idx].reshape(1, -1)).flatten()
        return (torch.from_numpy(v_norm), torch.from_numpy(target_norm))

# Multi-layer perceptron
class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=8, hidden_dim2=4):
        super(RegressionNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )
    def forward(self, x):
        return self.net(x)

# Evaluate and plot regressors
def evaluate_and_plot_regressors(nn_model, lr_model, test_loader, dataset, device, file_prefix):
    nn_model.eval()
    
    all_v = []
    all_targets_norm = []
    with torch.no_grad():
        for v, targets in test_loader:
            all_v.append(v)
            all_targets_norm.append(targets)
    
    all_v_tensor = torch.cat(all_v).to(device)
    all_targets_norm = torch.cat(all_targets_norm)
    
    actuals = dataset.target_scaler.inverse_transform(all_targets_norm.numpy()).flatten()
    
    # MLP
    with torch.no_grad():
        nn_preds_norm = nn_model(all_v_tensor).cpu()
    nn_predictions = dataset.target_scaler.inverse_transform(nn_preds_norm.numpy()).flatten()
    
    nn_r2 = r2_score(actuals, nn_predictions)
    nn_mape = mean_absolute_percentage_error(actuals, nn_predictions) * 100
    logging.info("="*40); logging.info("MLP performance:")
    logging.info(f"  - R²: {nn_r2:.4f}, MAPE: {nn_mape:.2f}%")
    
    # Linear regression
    X_test_scaled = all_v_tensor.cpu().numpy()
    lr_preds_norm = lr_model.predict(X_test_scaled)
    lr_predictions = dataset.target_scaler.inverse_transform(lr_preds_norm).flatten()

    lr_r2 = r2_score(actuals, lr_predictions)
    lr_mape = mean_absolute_percentage_error(actuals, lr_predictions) * 100
    logging.info("="*40); logging.info("Linear regression performance:")
    logging.info(f"  - R²: {lr_r2:.4f}, MAPE: {lr_mape:.2f}%")
    logging.info("="*40)
    
    # Visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(actuals, nn_predictions, alpha=0.6, label=f'MLP (R²: {nn_r2:.4f})')
    plt.scatter(actuals, lr_predictions, alpha=0.6, label=f'LR (R²: {lr_r2:.4f})', marker='x')
    min_val, max_val = np.min(actuals), np.max(actuals)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    plt.xlabel("Actual E_Q"); plt.ylabel("Predicted E_Q")
    plt.title("Model Comparison"); plt.legend(); plt.grid(True)
    plt.savefig(f"{file_prefix}_regressor_comparison.png")
    logging.info(f"'{file_prefix}_regressor_comparison.png' saved")

    return nn_r2, nn_mape, lr_r2, lr_mape


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'models/adaptive'
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"All results will be saved in '{output_dir}' directory")
    np.random.seed(args.seed)
    
    data_file = f"aggregated_full_dataset_adaptive_{args.num_samples}_{args.seed}.csv"
    full_df = pd.read_csv(data_file)
    
    results_log = []
    
    sampled_df = full_df.copy()
    dataset = RegressionDataset(sampled_df, args.cost_threshold)
    file_prefix = os.path.join(output_dir, f"full_s{args.num_samples}_run{args.seed}")
    
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=args.seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.176, random_state=args.seed)
    
    dataset.setup_scalers(train_indices, file_prefix)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=args.batch_size)
    
    logging.info("\n" + "="*20 + " MLP training started " + "="*20)
    nn_model = RegressionNN(input_dim=dataset.v_dim).to(device)
    optimizer = optim.Adam(nn_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=1.0)
    best_val_loss = float('inf'); patience_counter = 0
    best_model_path = f'{file_prefix}_nn_regressor.pth'
    
    nn_start_time = time.time()
    for epoch in range(args.epochs):
        nn_model.train()
        for v, targets in train_loader:
            v, targets = v.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = nn_model(v)
            loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()
        
        nn_model.eval(); val_loss = 0.0
        with torch.no_grad():
            for v, targets in val_loader:
                v, targets = v.to(device), targets.to(device)
                val_loss += criterion(nn_model(v), targets).item() * v.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss; torch.save(nn_model.state_dict(), best_model_path); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience: logging.info(f"Early stopping at epoch {epoch+1}."); break
    nn_training_time = time.time() - nn_start_time
    logging.info(f"MLP training completed (time: {nn_training_time:.2f} seconds). Best model saved in '{best_model_path}'")

    logging.info("\n" + "="*20 + " Scikit-learn LR training started " + "="*20)
    X_train, y_train = dataset.v_data[train_indices], dataset.target_data[train_indices]
    X_train_scaled = dataset.v_scaler.transform(X_train)
    y_train_scaled = dataset.target_scaler.transform(y_train)

    K = 8.0  
    input_bounds = {int(i): (-float(K), float(K)) for i in range(dataset.v_dim)}

    lr_model = LinearRegression()
    lr_start_time = time.time()
    lr_model.fit(X_train_scaled, y_train_scaled)
    lr_training_time = time.time() - lr_start_time
    
    lr_joblib_path = f'{file_prefix}_lr.joblib'
    joblib.dump(lr_model, lr_joblib_path)
    logging.info(f"Scikit-learn LR training completed (time: {lr_training_time:.4f} seconds) and saved in '{lr_joblib_path}'")

    final_nn_model = RegressionNN(input_dim=dataset.v_dim).to(device)
    final_nn_model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_lr_model = joblib.load(lr_joblib_path)
    
    nn_r2, nn_mape, lr_r2, lr_mape = evaluate_and_plot_regressors(final_nn_model, final_lr_model, test_loader, dataset, device, file_prefix)

    results_log.append({
        'numsam': args.num_samples, 'seed': args.seed, 'model_type': 'NN', 'training_time_sec': nn_training_time,
        'r2_score': nn_r2, 'mape_percent': nn_mape
    })
    results_log.append({
        'numsam': args.num_samples, 'seed': args.seed, 'model_type': 'LR', 'training_time_sec': lr_training_time,
        'r2_score': lr_r2, 'mape_percent': lr_mape
    })
    
    onnx_path = f"{file_prefix}_nn_regressor.onnx"
    dummy_v = torch.randn(1, dataset.v_dim, device=device)
    torch.onnx.export(final_nn_model, dummy_v, onnx_path,input_names=['input'], output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, "output": {0: "batch_size"}})
    write_onnx_model_with_bounds(onnx_path, None, input_bounds)
    logging.info(f"NN Regressor ONNX model saved in '{onnx_path}'")
    
    log_df = pd.DataFrame(results_log)
    log_csv_path = os.path.join(output_dir, f'training_summary_adaptive_{args.num_samples}_{args.seed}_log.csv')
    log_df.to_csv(log_csv_path, index=False)
    logging.info(f"\nAll runs completed. Final results saved in '{log_csv_path}'")
    print("\n--- Final summary ---")
    print(log_df)
    print("----------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--num_samples', type=int, required=True, help='num_samples')
    parser.add_argument('--seed', type=int, required=True, help='seed') 
    parser.add_argument('--cost_threshold', type=float, default=1.5e12, help="cost threshold for preprocessing")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=20)
    
    args = parser.parse_args()
    main(args)