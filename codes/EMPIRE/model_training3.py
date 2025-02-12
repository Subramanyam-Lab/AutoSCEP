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

# ---------------------------
# 1) Early Stopping 클래스
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=20, delta=0.0, path='scaler_pca/best_model3.pth', verbose=False):
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


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden1=8, hidden2=4, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 첫 번째 은닉층 뒤 Dropout
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 두 번째 은닉층 뒤 Dropout
            nn.Linear(hidden2, 1)      # 회귀 -> 출력 1개
        )

    def forward(self, x):
        return self.net(x)

# def main(
#     pca_csv='pca_results.csv',
#     test_size=0.2,
#     val_size=0.2,
#     random_state=42,
#     epochs=200,
#     batch_size=32,
#     learning_rate=1e-3,
#     dropout_rate=0.3,
#     patience=10,  # EarlyStopping
#     device=None,
# ):

#     # 0) 디바이스 설정
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # ---------------------------
#     # 4) 데이터 불러오기 & 전처리
#     # ---------------------------
#     df = pd.read_csv(pca_csv)
#     # 주성분(PC) 열만 골라서 X로, E_Q는 y로
#     feature_cols = [col for col in df.columns if col.startswith('PC')]
#     X_all = df[feature_cols].values   # (N, num_pc)
#     y_all = df['E_Q'].values          # (N, )

#     # Train/Test 분할
#     X_train_val, X_test, y_train_val, y_test = train_test_split(
#         X_all, y_all,
#         test_size=test_size,
#         random_state=random_state
#     )
#     # Train/Val 분할
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_val, y_train_val,
#         test_size=val_size,
#         random_state=random_state
#     )
#     print(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

#     # Target 정규화(Scaling)
#     scaler_y = StandardScaler()
#     # scaler_y = MinMaxScaler()
#     y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  # (N, 1)
#     y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1))        # (N, 1)
#     y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))       # (N, 1) - 평가 시 역변환

#     # 텐서 변환
#     X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
#     X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
#     X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)

#     y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
#     y_val_t   = torch.tensor(y_val_scaled,   dtype=torch.float32).to(device)
#     # y_test_t  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device) 
#     # (테스트용 y는 학습 시 필요없으므로 굳이 안 만들어도 됨)

#     # DataLoader 구성 (Train/Val)
#     train_dataset = TensorDataset(X_train_t, y_train_t)
#     val_dataset   = TensorDataset(X_val_t,   y_val_t)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

#     joblib.dump(scaler_y, "scaler_pca/scaler_y3.joblib")

#     # ---------------------------
#     # 5) 모델/옵티마이저/손실함수
#     # ---------------------------
#     input_dim = X_train.shape[1]
#     model = SimpleMLP(input_dim=input_dim, dropout_rate=dropout_rate).to(device)
#     print(model)

#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

#     # Early Stopping 준비
#     early_stopper = EarlyStopping(patience=patience, path='scaler_pca/best_model3.pth', verbose=True)

#     # ---------------------------
#     # 6) 학습 루프
#     # ---------------------------
#     train_losses = []
#     val_losses = []

#     for epoch in range(1, epochs + 1):
#         # --- Training ---
#         model.train()
#         running_train_loss = 0.0
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             preds = model(batch_X)  # (batch_size,1)
#             loss = criterion(preds, batch_y)
#             loss.backward()
#             optimizer.step()
#             running_train_loss += loss.item() * batch_X.size(0)

#         epoch_train_loss = running_train_loss / len(train_dataset)
#         train_losses.append(epoch_train_loss)

#         # --- Validation ---
#         model.eval()
#         running_val_loss = 0.0
#         with torch.no_grad():
#             for batch_X, batch_y in val_loader:
#                 preds = model(batch_X)
#                 loss = criterion(preds, batch_y)
#                 running_val_loss += loss.item() * batch_X.size(0)

#         epoch_val_loss = running_val_loss / len(val_dataset)
#         val_losses.append(epoch_val_loss)

#         # Early Stopping 체크
#         early_stopper(epoch_val_loss, model)

#         print(f"Epoch [{epoch}/{epochs}] - "
#               f"Train Loss: {epoch_train_loss:.6f}, "
#               f"Val Loss: {epoch_val_loss:.6f}")

#         if early_stopper.early_stop:
#             print("Early stopping triggered!")
#             break

#     # ---------------------------
#     # 7) Best Model 로드
#     # ---------------------------
#     model.load_state_dict(torch.load('scaler_pca/best_model3.pth', map_location=device))

#     # ---------------------------
#     # 8) Train/Val Loss 시각화
#     # ---------------------------
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('MSE Loss')
#     plt.legend()
#     plt.title('Train vs. Validation Loss')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('scaler_pca/Train_Validation_Loss3.png')

#     # ---------------------------
#     # 9) Test 세트 최종 평가
#     # ---------------------------
#     model.eval()
#     with torch.no_grad():
#         y_pred_test_scaled = model(X_test_t).cpu().numpy()  # (N,1), 스케일된 예측
#     # 역변환
#     y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled).flatten()  # (N,)
#     # 지표 계산
#     r2  = r2_score(y_test, y_pred_test)
#     mse = mean_squared_error(y_test, y_pred_test)
#     mae = mean_absolute_error(y_test, y_pred_test)
#     rmse = np.sqrt(mse)

#     print("\n=== Final Evaluation on Test Set ===")
#     print(f"R^2  : {r2:.4f}")
#     print(f"MSE  : {mse:.4f}")
#     print(f"RMSE : {rmse:.4f}")
#     print(f"MAE  : {mae:.4f}")

#     # ---------------------------
#     # 10) Actual vs Predicted Plot
#     # ---------------------------
#     plt.figure(figsize=(6, 6))
#     plt.scatter(y_test, y_pred_test, alpha=0.4, label='Prediction')
#     # 1:1 선
#     mn, mx = np.min([y_test, y_pred_test]), np.max([y_test, y_pred_test])
#     plt.plot([mn, mx], [mn, mx], color='red', label='Ideal 1:1')
#     plt.xlabel('Actual E_Q')
#     plt.ylabel('Predicted E_Q')
#     plt.title('Actual vs Predicted (Test Set)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('scaler_pca/Actual_Predicted_Plot3.png')

#     # ---------------------------
#     # 11) ONNX 내보내기
#     # ---------------------------
#     onnx_path = 'scaler_pca/pytorch_regression3.onnx'
#     dummy_input = torch.randn(1, input_dim).to(device)
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         export_params=True,
#         opset_version=12,
#         do_constant_folding=True,
#         input_names=['input'],
#         output_names=['output']
#     )
#     print(f"\nONNX 모델이 '{onnx_path}'로 저장되었습니다.")



def main(
    pca_csv='pca_results.csv',
    target_col='E_Q',  # NEW: Specify the target variable (default is E_Q)
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
    # ---------------------------
    # 0) 디바이스 설정
    # ---------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if target_col == 'E_Q_ELSE':
        startswith_letter = 'V'
    else:
        startswith_letter = 'V'

    # ---------------------------
    # 1) 데이터 불러오기 & 전처리
    # ---------------------------
    df = pd.read_csv(pca_csv)
    # 주성분(PC) 열만 골라서 X로, target_col은 y로
    feature_cols = [col for col in df.columns if col.startswith(startswith_letter)]
    X_all = df[feature_cols].values   # (N, num_pc)
    y_all = df[target_col].values    # (N, )

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
    print(f"Data split for {target_col}: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # Target 정규화(Scaling)
    if target_col == 'E_Q_ELSE':
        scaler_y = StandardScaler()
    else:
        scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  # (N, 1)
    y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1))        # (N, 1)
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))       # (N, 1) - 평가 시 역변환

    # 텐서 변환
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)

    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    y_val_t   = torch.tensor(y_val_scaled,   dtype=torch.float32).to(device)

    # DataLoader 구성 (Train/Val)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Save scaler for the specific target variable
    scaler_path = f"scaler_pca/scaler_{target_col}.joblib"
    joblib.dump(scaler_y, scaler_path)

    # ---------------------------
    # 2) 모델/옵티마이저/손실함수
    # ---------------------------
    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim=input_dim, dropout_rate=dropout_rate).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # Early Stopping 준비
    early_stopper = EarlyStopping(patience=patience, path=f'scaler_pca/best_model_{target_col}.pth', verbose=True)

    # ---------------------------
    # 3) 학습 루프
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
    # 4) Best Model 로드
    # ---------------------------
    model.load_state_dict(torch.load(f'scaler_pca/best_model_{target_col}.pth', map_location=device))

    # ---------------------------
    # 5) Train/Val Loss 시각화
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title(f'Train vs. Validation Loss ({target_col})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scaler_pca/Train_Validation_Loss_{target_col}.png')

    # ---------------------------
    # 6) Test 세트 최종 평가
    # ---------------------------
    model.eval()
    with torch.no_grad():
        y_pred_test_scaled = model(X_test_t).cpu().numpy()  # (N,1), 스케일된 예측
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled).flatten()  # (N,)
    r2  = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)

    print(f"\n=== Final Evaluation on Test Set ({target_col}) ===")
    print(f"R^2  : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    # ---------------------------
    # 7) Actual vs Predicted Plot
    # ---------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.4, label='Prediction')
    mn, mx = np.min([y_test, y_pred_test]), np.max([y_test, y_pred_test])
    plt.plot([mn, mx], [mn, mx], color='red', label='Ideal 1:1')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f'Actual vs Predicted (Test Set - {target_col})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scaler_pca/Actual_Predicted_Plot_{target_col}.png')


    # ---------------------------
    # 11) ONNX 내보내기
    # ---------------------------
    onnx_path = f'scaler_pca/pytorch_regression_{target_col}.onnx'
    dummy_input = torch.randn(1, input_dim).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"\nONNX 모델이 '{onnx_path}'로 저장되었습니다.")





# ------------------------------------------------
# 12) Evaluate on the entire dataset (Optional)
# ------------------------------------------------
def evaluate_on_all_data(model, df, scaler_y, device, target_col):
    """
    Evaluate the final model on the entire dataset
    that was used for training + test.
    """

    if target_col == 'E_Q_ELSE':
        startswith_letter = 'V'
    else:
        startswith_letter = 'V'
    # 1) Separate features and target from the entire DataFrame
    feature_cols = [col for col in df.columns if col.startswith(startswith_letter)]
    X_all = df[feature_cols].values
    y_all = df[target_col].values

    # 2) Scale the target using the SAME scaler_y as training
    y_all_scaled = scaler_y.transform(y_all.reshape(-1, 1))

    # 3) Convert features to torch.tensor
    X_all_t = torch.tensor(X_all, dtype=torch.float32).to(device)

    # 4) Get predictions (scaled)
    model.eval()
    with torch.no_grad():
        y_all_pred_scaled = model(X_all_t).cpu().numpy()  # shape (N, 1)

    # 5) Inverse transform the predictions to the original scale
    y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled).flatten()

    # 6) Compute metrics
    r2  = r2_score(y_all, y_all_pred)
    mse = mean_squared_error(y_all, y_all_pred)
    mae = mean_absolute_error(y_all, y_all_pred)
    rmse = np.sqrt(mse)

    print("\n=== Evaluation on the Entire Dataset ===")
    print(f"R^2  : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    # 7) Plot Actual vs Predicted for the entire dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(y_all, y_all_pred, alpha=0.4, label='Prediction')
    # 1:1 line
    mn, mx = np.min([y_all, y_all_pred]), np.max([y_all, y_all_pred])
    plt.plot([mn, mx], [mn, mx], color='red', label='Ideal 1:1')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted (Entire Dataset)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scaler_pca/Actual_Predicted_Entire_Dataset_{target_col}.png')


if __name__ == "__main__":
    target_col_lst = ['E_Q_ELSE', 'LL_AMT']
    for target_col in target_col_lst:
        if target_col == 'E_Q_ELSE':
            file_path = 'scaler_pca/v_scl_results2.csv'
            input_dim = 616
        else: 
            file_path = 'scaler_pca/v_scl_results2.csv'
            input_dim = 616
        # main(
        #     pca_csv='scaler_pca/pca_results3.csv',
        #     test_size=0.2,    # Train:80%, Test:20%
        #     val_size=0.2,     # 그 중 Train의 20%를 Validation으로 => (Train:64%, Val:16%, Test:20%)
        #     random_state=42,
        #     epochs=300,
        #     batch_size=32,
        #     learning_rate=1e-4,
        #     dropout_rate=0.1,
        #     patience=20,      # EarlyStopping
        #     device=None       # 자동 설정
        # )

        main(
            pca_csv=file_path,
            target_col=target_col,  
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            epochs=300,
            batch_size=32,
            learning_rate=1e-4,
            dropout_rate=0.1,
            patience=20,  # EarlyStopping
            device=None
        )
        # 2) Load the best model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleMLP(input_dim=input_dim).to(device)  # Make sure input_dim matches your PCA output dimension
        model.load_state_dict(torch.load(f'scaler_pca/best_model_{target_col}.pth', map_location=device))
        model.eval()

        # 3) Load scaler_y
        scaler_y = joblib.load(f"scaler_pca/scaler_{target_col}.joblib")

        # 4) Load the entire dataset (pca_results.csv) again
        df_full = pd.read_csv(file_path)

        # 5) Evaluate on the entire dataset
        evaluate_on_all_data(model, df_full, scaler_y, device, target_col)

        total_count = len(df_full[target_col])
        zero_count = (df_full[target_col] == 0).sum()
        zero_percentage = (zero_count / total_count) * 100

        print(f"{target_col} zero_count : {zero_count}, zero_percentage : {zero_percentage}")
