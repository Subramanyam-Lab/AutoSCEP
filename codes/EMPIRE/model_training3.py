import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import joblib
from mymodel import SimpleMLP
plt.rcParams.update({'font.size': 14})

# ---------------------------
# 1) Early Stopping 클래스
# ---------------------------

master_path = "scaler_pca_ad2"
num_fsd = 10000

class EarlyStopping:
    def __init__(self, patience=20, delta=0.0, path='scaler_pca2/best_model.pth', verbose=False):
        self.patience = patience

        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # 최적 모델 저장
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def build_label(df, approach=1, threshold=1.2e12):
    if 'C1' not in df.columns:
        raise ValueError("DataFrame에 'C1' 열이 존재해야 합니다 (1단계 비용).")
    if 'E_Q' not in df.columns:
        raise ValueError("DataFrame에 'E_Q' 열이 존재해야 합니다 (2단계 비용).")

    if approach == 1:
        # (1) 총비용 예측: label = C1 + E_Q
        df['label'] = df['C1'] + df['E_Q']
    elif approach == 2:
        # (2) 2단계 비용 + 페널티: label = E_Q + max(0, C1 - threshold)
        df['label'] = df['E_Q'] + np.maximum(0, df['C1'] - threshold)
    else:
        raise ValueError("approach는 1 또는 2 중 하나여야 합니다.")
    return df


def main(
    pca_csv='pca_results.csv',
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    epochs=200,
    batch_size=32,
    learning_rate=1e-3,
    dropout_rate=0.3,
    patience=10,  # EarlyStopping
    device=None,
):

    # 0) 디바이스 설정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------------------------
    # 4) 데이터 불러오기 & 전처리
    # ---------------------------
    df = pd.read_csv(pca_csv)

    df = df[df['E_Q'] < 1.1e12]

    # Plot and save the distribution of E_Q after filtering
    plt.figure(figsize=(8, 6))
    plt.hist(df['E_Q'], bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel('E_Q')
    plt.ylabel('Frequency')
    plt.title('Distribution of E_Q')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/E_Q_distribution_filtered.png')
    plt.close()


    df = build_label(df, approach=2)

    # 주성분(PC) 열만 골라서 X로, E_Q는 y로
    feature_cols = [col for col in df.columns if col.startswith('PC')]
    X_all = df[feature_cols].values   # (N, num_pc)
    y_all = df['E_Q'].values          # (N, )
    # y_all = df['label'].values

    # Train/Test 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all,
        test_size=test_size,
        random_state=random_state
    )
    # Train/Val 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=random_state
    )
    print(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # Target 정규화(Scaling)
    scaler_y = StandardScaler()
    # scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  # (N, 1)
    y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1))        # (N, 1)
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))       # (N, 1) - 평가 시 역변환
    
    # 텐서 변환
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)

    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    y_val_t   = torch.tensor(y_val_scaled,   dtype=torch.float32).to(device)
    # y_test_t  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device) 
    # (테스트용 y는 학습 시 필요없으므로 굳이 안 만들어도 됨)

    # DataLoader 구성 (Train/Val)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    # joblib.dump(scaler_y, f"scaler_pca2/scaler_y_{num_sce}_{num_sam}_2.joblib")
    joblib.dump(scaler_y, f"{master_path}/scaler_y_ad_{num_fsd}.joblib")

    # ---------------------------
    # 5) 모델/옵티마이저/손실함수
    # ---------------------------
    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim=input_dim, dropout_rate=dropout_rate).to(device)
    print(model)

    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    

    # Early Stopping 준비
    # early_stopper = EarlyStopping(patience=patience, path=f'scaler_pca2/best_model_{num_sce}_{num_sam}.pth', verbose=True)
    early_stopper = EarlyStopping(patience=patience, path=f'{master_path}/best_model_ad_{num_fsd}.pth', verbose=True)

    # ---------------------------
    # 6) 학습 루프
    # ---------------------------
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        running_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)  # (batch_size,1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_X.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        # Early Stopping 체크
        early_stopper(epoch_val_loss, model)

        print(f"Epoch [{epoch}/{epochs}] - "
              f"Train Loss: {epoch_train_loss:.6f}, "
              f"Val Loss: {epoch_val_loss:.6f}")

        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    # ---------------------------
    # 7) Best Model 로드
    # ---------------------------
    # torch.save(model.state_dict(), f'scaler_pca2/best_model_{num_sce}_{num_sam}_2.pth')
    # print(f"Full model saved as 'scaler_pca2/best_model_{num_sce}_{num_sam}_2.pth'")
    torch.save(model.state_dict(), f'{master_path}/best_model_ad_{num_fsd}.pth')
    
    # ---------------------------
    # 8) Train/Val Loss 시각화
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Train vs. Validation Loss')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'scaler_pca2/Train_Validation_Loss_{num_sce}_{num_sam}_2.png')
    plt.savefig(f'{master_path}/Train_Validation_Loss_ad_{num_fsd}.png')

    # ---------------------------
    # 9) Test 세트 최종 평가
    # ---------------------------
    model.eval()
    with torch.no_grad():
        y_pred_test_scaled = model(X_test_t).cpu().numpy()  # (N,1), 스케일된 예측
    # 역변환
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled).flatten()  # (N,)
    # y_pred_test = y_pred_test_scaled
    # 지표 계산
    r2  = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print("\n=== Final Evaluation on Test Set ===")
    print(f"R^2  : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.2f}%")
    

    # ---------------------------
    # 10) Actual vs Predicted Plot
    # ---------------------------
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.4, label='Prediction')
    # 1:1 선
    mn, mx = np.min([y_test, y_pred_test]), np.max([y_test, y_pred_test])
    plt.plot([mn, mx], [mn, mx], color='red', label='Ideal 1:1')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Neural Network (Test Set)\nR^2: {r2:.4f}, MAPE: {mape:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'scaler_pca2/Actual_Predicted_Plot_{num_sce}_{num_sam}_2.png')
    plt.savefig(f'{master_path}/Actual_Predicted_Plot_ad_{num_fsd}.png')


    # ---------------------------
    # 10.1) Residual Error Plot for Test Set
    # ---------------------------
    residuals_test = y_test - y_pred_test
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred_test, residuals_test, alpha=0.4, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Neural Network (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'scaler_pca2/Residual_Error_Plot_{num_sce}_{num_sam}_2.png')
    plt.savefig(f'{master_path}/Residual_Error_Plot_ad_{num_fsd}.png')


# ------------------------------------------------
# 12) Evaluate on the entire dataset (Optional)
# ------------------------------------------------
def evaluate_on_all_data(model, df, scaler_y, device):
        
    # 1) Separate features and target from the entire DataFrame
    feature_cols = [col for col in df.columns if col.startswith('PC')]

    # df = df[(df['E_Q'] > 0.5e12) & (df['E_Q'] < 1.2e12)]
    df = df[df['E_Q'] < 1.1e12]
    df = build_label(df, approach=2)

    
    X_all = df[feature_cols].values
    y_all = df['E_Q'].values
    # y_all = df['label'].values
    
    X_train_min = np.min(X_all, axis=0)
    X_train_max = np.max(X_all, axis=0)
    
    # 2) Scale the target using the SAME scaler_y as training
    # y_all_scaled = scaler_y.transform(y_all.reshape(-1, 1))
    # 3) Convert features to torch.tensor
    X_all_t = torch.tensor(X_all, dtype=torch.float32).to(device)

    # 4) Get predictions (scaled)
    model.eval()
    with torch.no_grad():
        y_all_pred_scaled = model(X_all_t).cpu().numpy()  # shape (N, 1)

    # 5) Inverse transform the predictions to the original scale
    y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled).flatten()
    # y_all_pred = y_all_pred_scaled
    # 6) Compute metrics
    r2  = r2_score(y_all, y_all_pred)
    mse = mean_squared_error(y_all, y_all_pred)
    mae = mean_absolute_error(y_all, y_all_pred)
    rmse = np.sqrt(mse)
    mape_all = np.mean(np.abs((y_all - y_all_pred) / y_all)) * 100

    print("\n=== Evaluation on the Entire Dataset ===")
    print(f"R^2  : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape_all:.2f}%")

    # 7) Plot Actual vs Predicted for the entire dataset
    plt.figure(figsize=(12, 6))
    plt.scatter(y_all, y_all_pred, alpha=0.4, label='Prediction')
    # 1:1 line
    mn, mx = np.min([y_all, y_all_pred]), np.max([y_all, y_all_pred])
    plt.plot([mn, mx], [mn, mx], color='red', label='Ideal 1:1')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Neural Network \nR^2: {r2:.4f}, MAPE: {mape_all:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'scaler_pca2/Actual_Predicted_Entire_Dataset_{numsce}_{num_sam}_2.png')
    plt.savefig(f'{master_path}/Actual_Predicted_Entire_Dataset_ad_{num_fsd}.png')


    # ---------------------------
    # Residual Error Plot for Entire Dataset
    # ---------------------------
    residuals_all = y_all - y_all_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_all_pred, residuals_all, alpha=0.4, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Neural Network')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'scaler_pca2/Residual_Error_Entire_Dataset_{numsce}_{num_sam}_2.png')
    plt.savefig(f'{master_path}/Residual_Error_Entire_Dataset_ad_{num_fsd}.png')


if __name__ == "__main__":
    file_path = f'scaler_pca5/pca_results_ad_{num_fsd}.csv'
    main(
        pca_csv=file_path,
        test_size=0.2,    # Train:80%, Test:20%
        val_size=0.2,     # 그 중 Train의 20%를 Validation으로 => (Train:64%, Val:16%, Test:20%)
        random_state=42,
        epochs=1000,
        batch_size=32,
        learning_rate=1e-4,
        dropout_rate=0.2,
        patience=50,      # EarlyStopping
        device=None
    )

    pca_obj = joblib.load(f'scaler_pca5/pca_ad_{num_fsd}.pkl')
    components = pca_obj.components_  # (n_pca, n_features)
    n_pca = components.shape[0]

    # 2) Load the best model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMLP(input_dim=n_pca).to(device)  
    model.load_state_dict(torch.load(f'{master_path}/best_model_ad_{num_fsd}.pth', map_location=device))
    model.eval()

    # 3) Load scaler_y
    # scaler_y = joblib.load(f"scaler_pca2/scaler_y_{numsce}_{numsam}_2.joblib")
    scaler_y = joblib.load(f"{master_path}/scaler_y_ad_{num_fsd}.joblib")

    # 4) Load the entire dataset (pca_results.csv) again
    df_full = pd.read_csv(file_path)

    # 5) Evaluate on the entire dataset
    evaluate_on_all_data(model, df_full, scaler_y, device)


    # ------------------------------------------------
    # Additional: Train and evaluate Linear Regression and Decision Tree
    # ------------------------------------------------
    print("\n=== Training Linear Regression and Decision Tree Models ===")
    # Load dataset and build label
    df_lr = pd.read_csv(file_path)
    # df_lr = df_lr[(df_lr['E_Q'] > 0.5e12) & (df_lr['E_Q'] < 1.2e12)]
    df_lr = df_lr[df_lr['E_Q'] < 1.1e12]
    df_lr = build_label(df_lr, approach=2)


    feature_cols = [col for col in df_lr.columns if col.startswith('PC')]
    X_lr = df_lr[feature_cols].values
    # y_lr = df_lr['label'].values
    y_lr = df_lr['E_Q'].values

    # Split the data into train and test sets (if not already done)
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_lr, y_lr, test_size=0.2, random_state=42
    )

    # Use the loaded scaler to transform the target values for training and testing
    y_train_lr_scaled = scaler_y.transform(y_train_lr.reshape(-1, 1))
    y_test_lr_scaled = scaler_y.transform(y_test_lr.reshape(-1, 1))

    # ----- Linear Regression -----
    from sklearn.linear_model import LinearRegression
    # Train the Linear Regression model on the original features and the scaled target
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_lr, y_train_lr_scaled.ravel())

    # Predict on the test set (predictions are in the scaled domain)
    y_pred_lr_scaled = lin_reg.predict(X_test_lr)
    # Inverse transform the predictions back to the original target scale
    y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1, 1)).flatten()

     # Save the Linear Regression model
    joblib.dump(lin_reg, f'{master_path}/linear_regression_model_{num_fsd}.joblib')

    r2_lr = r2_score(y_test_lr, y_pred_lr)
    mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    mae_lr = mean_absolute_error(y_test_lr, y_pred_lr)
    mape_lr = np.mean(np.abs((y_test_lr - y_pred_lr) / y_test_lr)) * 100

    print("\n--- Linear Regression Evaluation (Test Set) ---")
    print(f"R^2  : {r2_lr:.4f}")
    print(f"MSE  : {mse_lr:.4f}")
    print(f"RMSE : {rmse_lr:.4f}")
    print(f"MAE  : {mae_lr:.4f}")
    print(f"MAPE : {mape_lr:.2f}%")

    # Actual vs. Predicted Plot (Test Set) - Linear Regression
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_lr, y_pred_lr, alpha=0.4, label='Prediction')
    mn_lr, mx_lr = np.min([y_test_lr, y_pred_lr]), np.max([y_test_lr, y_pred_lr])
    plt.plot([mn_lr, mx_lr], [mn_lr, mx_lr], color='red', label='Ideal 1:1')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Linear Regression (Test Set)\nR^2: {r2_lr:.4f}, MAPE: {mape_lr:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/Actual_Predicted_Plot_LR_Test_{num_fsd}.png')

    # Residual Plot (Test Set) - Linear Regression
    residuals_lr_test = y_test_lr - y_pred_lr
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred_lr,  residuals_lr_test, alpha=0.4, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Linear Regression (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/Residual_Error_Plot_LR_Test_{num_fsd}.png')

    # Entire dataset evaluation for Linear Regression
    X_lr_scaled_all = X_lr
    y_pred_lr_all = lin_reg.predict(X_lr_scaled_all)
    y_pred_lr_all = scaler_y.inverse_transform(y_pred_lr_all.reshape(-1, 1)).flatten()
    
    r2_lr_all = r2_score(y_lr, y_pred_lr_all)
    mse_lr_all = mean_squared_error(y_lr, y_pred_lr_all)
    rmse_lr_all = np.sqrt(mse_lr_all)
    mae_lr_all = mean_absolute_error(y_lr, y_pred_lr_all)
    mape_lr_all = np.mean(np.abs((y_lr - y_pred_lr_all) / y_lr)) * 100

    print("\n--- Linear Regression Entire Dataset Evaluation ---")
    print(f"R^2  : {r2_lr_all:.4f}")
    print(f"MSE  : {mse_lr_all:.4f}")
    print(f"RMSE : {rmse_lr_all:.4f}")
    print(f"MAE  : {mae_lr_all:.4f}")
    print(f"MAPE : {mape_lr_all:.2f}%")

    # Actual vs. Predicted Plot (Entire Dataset) - Linear Regression
    plt.figure(figsize=(12, 6))
    plt.scatter(y_lr, y_pred_lr_all, alpha=0.4, label='Prediction')
    mn_lr_all, mx_lr_all = np.min([y_lr, y_pred_lr_all]), np.max([y_lr, y_pred_lr_all])
    plt.plot([mn_lr_all, mx_lr_all], [mn_lr_all, mx_lr_all], color='red', label='Ideal 1:1')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Linear Regression \nR^2: {r2_lr_all:.4f}, MAPE: {mape_lr_all:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/Actual_Predicted_Entire_Dataset_LR_{num_fsd}.png')

    # Residual Plot (Entire Dataset) - Linear Regression
    residuals_lr_all = y_lr - y_pred_lr_all
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred_lr_all, residuals_lr_all,  alpha=0.4, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/Residual_Error_Entire_Dataset_LR_{num_fsd}.png')

    #################################### ----- Decision Tree Regressor ----- #########################################
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train_lr, y_train_lr)
    y_pred_tree = tree_reg.predict(X_test_lr)

    # Save the Decision Tree model
    joblib.dump(tree_reg, f'{master_path}/decision_tree_model_{num_fsd}.joblib')

    r2_tree = r2_score(y_test_lr, y_pred_tree)
    mse_tree = mean_squared_error(y_test_lr, y_pred_tree)
    rmse_tree = np.sqrt(mse_tree)
    mae_tree = mean_absolute_error(y_test_lr, y_pred_tree)
    mape_tree = np.mean(np.abs((y_test_lr - y_pred_tree) / y_test_lr)) * 100

    print("\n--- Decision Tree Evaluation (Test Set) ---")
    print(f"R^2  : {r2_tree:.4f}")
    print(f"MSE  : {mse_tree:.4f}")
    print(f"RMSE : {rmse_tree:.4f}")
    print(f"MAE  : {mae_tree:.4f}")
    print(f"MAPE : {mape_tree:.2f}%")

    # Actual vs. Predicted Plot (Test Set) - Decision Tree
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_lr, y_pred_tree, alpha=0.4, label='Prediction')
    mn_tree, mx_tree = np.min([y_test_lr, y_pred_tree]), np.max([y_test_lr, y_pred_tree])
    plt.plot([mn_tree, mx_tree], [mn_tree, mx_tree], color='red', label='Ideal 1:1')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Decision Tree (Test Set)\nR^2: {r2_tree:.4f}, MAPE: {mape_tree:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/Actual_Predicted_Plot_Tree_Test_{num_fsd}.png')

    # Residual Plot (Test Set) - Decision Tree
    residuals_tree_test = y_test_lr - y_pred_tree
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred_tree, residuals_tree_test,  alpha=0.4, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Decision Tree (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/Residual_Error_Plot_Tree_Test_{num_fsd}.png')

    # Entire dataset evaluation for Decision Tree
    y_pred_tree_all = tree_reg.predict(X_lr)
    r2_tree_all = r2_score(y_lr, y_pred_tree_all)
    mse_tree_all = mean_squared_error(y_lr, y_pred_tree_all)
    rmse_tree_all = np.sqrt(mse_tree_all)
    mae_tree_all = mean_absolute_error(y_lr, y_pred_tree_all)
    mape_tree_all = np.mean(np.abs((y_lr - y_pred_tree_all) / y_lr)) * 100

    print("\n--- Decision Tree Entire Dataset Evaluation ---")
    print(f"R^2  : {r2_tree_all:.4f}")
    print(f"MSE  : {mse_tree_all:.4f}")
    print(f"RMSE : {rmse_tree_all:.4f}")
    print(f"MAE  : {mae_tree_all:.4f}")
    print(f"MAPE : {mape_tree_all:.2f}%")

    # Actual vs. Predicted Plot (Entire Dataset) - Decision Tree
    plt.figure(figsize=(12, 6))
    plt.scatter(y_lr, y_pred_tree_all, alpha=0.4, label='Prediction')
    mn_tree_all, mx_tree_all = np.min([y_lr, y_pred_tree_all]), np.max([y_lr, y_pred_tree_all])
    plt.plot([mn_tree_all, mx_tree_all], [mn_tree_all, mx_tree_all], color='red', label='Ideal 1:1')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Decision Tree \nR^2: {r2_tree_all:.4f}, MAPE: {mape_tree_all:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/Actual_Predicted_Entire_Dataset_Tree_{num_fsd}.png')

    # Residual Plot (Entire Dataset) - Decision Tree
    residuals_tree_all = y_lr - y_pred_tree_all
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred_tree_all, residuals_tree_all, alpha=0.4, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Decision Tree')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{master_path}/Residual_Error_Entire_Dataset_Tree_{num_fsd}.png')
