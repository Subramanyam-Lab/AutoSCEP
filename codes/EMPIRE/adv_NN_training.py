import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import ast
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def save_integrated_data(integrated_data, file_name='integrated_data.csv', save_format='csv'):
    df = pd.DataFrame(integrated_data)

    if save_format == 'csv':
        df.to_csv(file_name, index=False)
    elif save_format == 'json':
        df.to_json(file_name, orient='records')
    else:
        print("Unsupported format. Please use 'csv' or 'json'.")

    print(f"Data saved to {file_name}")

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

# Implement Self-Attention Mechanism
# class SelfAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(SelfAttention, self).__init__()
#         self.attn = nn.Linear(input_dim, input_dim)

#     def forward(self, x):
#         attn_weights = torch.softmax(self.attn(x), dim=-1)
#         return x * attn_weights
    
# # Implement Cross-Attention Mechanism
# class CrossAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(CrossAttention, self).__init__()
#         self.attn = nn.Linear(input_dim, input_dim)

#     def forward(self, x1, x2):
#         attn_weights = torch.softmax(self.attn(x1), dim=-1)
#         return x1 * attn_weights + x2


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
    print("total_params:" ,total_params)
    # zero_params = sum((p == 0).sum().item() for p in model.parameters())
    threshold = 1e-7  # 예시 임계값
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
    plt.savefig('[NN_Plot]_actual_vs_predicted_y_equals_x.png')
    plt.close()
    
    # Residual plot
    residuals = all_targets - all_predictions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=all_targets, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Actual Q_i")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig('[NN_Plot]_residual_plot.png')
    plt.close()

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig('[NN_Plot]_loss_history.png')
    plt.close()

def load_and_integrate_data(data_directory):
    integrated_data = []
    count = 0
    for filename in os.listdir(data_directory):
        if filename.startswith('xi_Q_i') and filename.endswith('.json'):
            xi_qi_file = os.path.join(data_directory, filename)
            vi_file = os.path.join(data_directory, f"x_v_i{filename[6:]}")
            
            if not os.path.exists(vi_file):
                print(f"Warning: Matching v_i file not found for {filename}")
                continue
            
            with open(xi_qi_file, 'r') as f1, open(vi_file, 'r') as f2:
                xi_qi_data = json.load(f1)
                vi_data = json.load(f2)

            for period in xi_qi_data.keys():
                i = int(period)
                xi_values = xi_qi_data[period]['xi_i']
                qi_values = xi_qi_data[period]['Q_i']

                # v_i 처리 (시나리오와 무관)
                vi_flat = [item for sublist in vi_data[period]['v_i'].values() for item in sublist.values()]
                
                scenario_dict = defaultdict(lambda: {"xi_i": [], "v_i": vi_flat, "q_i": None})
                
                # Process xi_i data
                for feature, values in xi_values.items():
                    for key, value in values.items():
                        try:
                            key_tuple = ast.literal_eval(key)
                            if isinstance(key_tuple, tuple):
                                scenario = key_tuple[-1]
                                scenario_dict[scenario]["xi_i"].append(value)
                        except (ValueError, SyntaxError):
                            print(f"Skipping invalid key: {key}")

                # Process q_i data
                for scenario, q_value in qi_values.items():
                    if scenario in scenario_dict:
                        scenario_dict[scenario]["q_i"] = q_value
                
                # Collect data for each scenario
                for scenario, data in scenario_dict.items():
                    data['i'] = i
                    integrated_data.append(data)
            
            print(f"{count}-th file loaded")
            count += 1
    
    return integrated_data


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    

def preprocess_data(data, latent_dim=64, epochs=50):
    i_data = np.array([item['i'] for item in data])
    xi_data = np.array([item['xi_i'] for item in data])
    vi_data = np.array([item['v_i'] for item in data])
    qi_data = np.array([item['q_i'] for item in data])

    # Ensure xi_data is 2D
    if len(xi_data.shape) == 1:
        print("xi_data is not 2D as expected. Exiting.")
        return None, None, None, None, None, None, None, None

    print(f"xi_data shape: {xi_data.shape}")

    # Normalize xi_data before training the autoencoder
    scaler_xi = StandardScaler()
    xi_data_normalized = scaler_xi.fit_transform(xi_data)

    # Define and train the autoencoder
    input_dim = xi_data_normalized.shape[1]
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Prepare data for training
    xi_tensor = torch.tensor(xi_data_normalized, dtype=torch.float32).to(device)
    dataset = TensorDataset(xi_tensor, xi_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train autoencoder
    autoencoder.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    # Use the encoder part of the trained autoencoder for feature extraction
    autoencoder.eval()
    with torch.no_grad():
        xi_data_reduced = autoencoder.encoder(xi_tensor).cpu().numpy()
    
    print(f"Reduced xi_data shape using Autoencoder: {xi_data_reduced.shape}")

    scaler_vi = StandardScaler()
    scaler_qi = StandardScaler()

    vi_data_normalized = scaler_vi.fit_transform(vi_data)
    qi_data = qi_data.reshape(-1, 1)
    qi_data_normalized = scaler_qi.fit_transform(qi_data)

    return i_data, xi_data_reduced, vi_data_normalized, qi_data_normalized, scaler_xi, scaler_vi, scaler_qi, autoencoder

# def preprocess_data(data):
#     i_data = np.array([item['i'] for item in data])
#     xi_data = np.array([item['xi_i'] for item in data])
#     vi_data = np.array([item['v_i'] for item in data])
#     qi_data = np.array([item['q_i'] for item in data])

#     # Ensure xi_data is 2D
#     if len(xi_data.shape) == 1:
#         print("xi_data is not 2D as expected. Exiting.")
#         return None, None, None, None, None, None, None, None

#     print(f"xi_data shape: {xi_data.shape}")

#     # PCA 컴포넌트 수를 데이터 차원에 맞게 조정
#     n_components = min(xi_data.shape[0], xi_data.shape[1], 64)
#     print(f"Using {n_components} components for PCA")

#     if n_components < xi_data.shape[1]:
#         pca = PCA(n_components=n_components)
#         xi_data_reduced = pca.fit_transform(xi_data)
#         print(f"Reduced xi_data shape: {xi_data_reduced.shape}")
#         print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
#     else:
#         print("No dimension reduction applied to xi_data")
#         xi_data_reduced = xi_data
#         pca = None

#     scaler_xi = StandardScaler()
#     scaler_vi = StandardScaler()
#     scaler_qi = StandardScaler()

#     xi_data_normalized = scaler_xi.fit_transform(xi_data_reduced)
#     vi_data_normalized = scaler_vi.fit_transform(vi_data)

#     # qi_data를 2차원 배열로 변환
#     qi_data = qi_data.reshape(-1, 1)
#     qi_data_normalized = scaler_qi.fit_transform(qi_data)

#     return i_data, xi_data_normalized, vi_data_normalized, qi_data_normalized, scaler_xi, scaler_vi, scaler_qi, pca


class EmpireModel(nn.Module):
    def __init__(self, input_dim_xi, input_dim_vi, period_count):
        super(EmpireModel, self).__init__()
        
        self.one_hot = nn.Embedding(period_count + 1, period_count)
        self.xi_dense = nn.Linear(input_dim_xi, 64)  
        self.vi_dense = nn.Linear(input_dim_vi, 64)
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.fc1 = nn.Linear(64 + 64 + period_count, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, i_input, xi_input, vi_input):
        i_one_hot = self.one_hot(i_input).squeeze(1)
        
        xi_out = torch.relu(self.xi_dense(xi_input))
        # xi_out_attn = self.self_attn(xi_out)

        vi_out = torch.relu(self.vi_dense(vi_input))
        # vi_out_attn = self.self_attn(vi_out)


        combined = torch.cat([xi_out, vi_out, i_one_hot], dim=1)
        combined = self.dropout(combined)  # Apply dropout
        out = torch.sigmoid(self.fc1(combined))
        # cross_attn = self.cross_attn(xi_out_attn, vi_out_attn)
        # combined = torch.cat([cross_attn, vi_out_attn, i_one_hot], dim=1)
        # out = torch.relu(self.fc1(combined))
        out = self.fc2(out)
        
        return out


def quantile_loss(prediction, target, quantile):
    error = target - prediction
    return torch.max((quantile - 1) * error, quantile * error).mean()


# def train_model(model, train_loader, val_loader, epochs=1000):
#     optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Added L2 regularization
#     # criterion = nn.SmoothL1Loss()
#     # Define quantile
#     quantile = 0.5  # Median

#     train_losses = []
#     val_losses = []
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for i_batch, xi_batch, vi_batch, qi_batch in train_loader:
#             i_batch, xi_batch, vi_batch, qi_batch = i_batch.to(device), xi_batch.to(device), vi_batch.to(device), qi_batch.to(device)

#             optimizer.zero_grad()
#             outputs = model(i_batch, xi_batch, vi_batch)
#             # loss = criterion(outputs, qi_batch)
#             loss = quantile_loss(outputs, qi_batch, quantile)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * xi_batch.size(0)

#         epoch_loss = running_loss / len(train_loader.dataset)
#         print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for i_batch, xi_batch, vi_batch, qi_batch in val_loader:
#                 i_batch, xi_batch, vi_batch, qi_batch = i_batch.to(device), xi_batch.to(device), vi_batch.to(device), qi_batch.to(device)

#                 outputs = model(i_batch, xi_batch, vi_batch)
#                 # loss = criterion(outputs, qi_batch)
#                 loss = quantile_loss(outputs, qi_batch, quantile)
#                 val_loss += loss.item() * xi_batch.size(0)

#         val_loss /= len(val_loader.dataset)
#         print(f"Validation Loss: {val_loss:.4f}")
#         train_losses.append(epoch_loss)
#         val_losses.append(val_loss)
#     plot_training_history(train_losses, val_losses)
#     return model

def train_model(model, train_loader, val_loader, epochs=1000, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    quantile = 0.5  # Median
    criterion = nn.SmoothL1Loss()
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
            # loss = quantile_loss(outputs, qi_batch, quantile)
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
                # loss = quantile_loss(outputs, qi_batch, quantile)
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

def main():
    data_directory = 'NN_training_datasets' 
    # integrated_data = load_and_integrate_data(data_directory)
    # save_integrated_data(integrated_data, file_name='integrated_data.csv', save_format='csv')
    integrated_data = load_saved_data(file_name='integrated_data.csv', load_format='csv')

    i_data, xi_data, vi_data, qi_data, scaler_xi, scaler_vi, scaler_qi, pca = preprocess_data(integrated_data)
    
    # 모델 초기화를 위한 파라미터 설정
    input_dim_xi = xi_data.shape[1]
    input_dim_vi = vi_data.shape[1]
    period_count = np.max(i_data)

    # 모델 초기화
    model = EmpireModel(input_dim_xi, input_dim_vi, period_count).to(device)

    # 데이터 분할 및 DataLoader 생성
    X = list(zip(i_data, xi_data, vi_data))
    y = qi_data

    # 먼저 테스트 세트 분리
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 남은 데이터를 훈련 세트와 검증 세트로 분할
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

    # 모델 학습
    trained_model = train_model(model, train_loader, val_loader)
    
    # 모델 분석
    analyze_model(trained_model, test_loader, scaler_qi)
    
    # 모델 및 전처리 객체 저장
    torch.save(trained_model.state_dict(), 'NN_model/trained_empire_model.pth')
    np.save('NN_model/scaler_xi.npy', scaler_xi)
    np.save('NN_model/scaler_vi.npy', scaler_vi)
    np.save('NN_model/scaler_qi.npy', scaler_qi)
    if pca:
        import joblib
        joblib.dump(pca, 'NN_model/pca_xi.joblib')
    else:
        print("No PCA model to save")

if __name__ == "__main__":
    main()