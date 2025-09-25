import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

# Function to compute Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main(
    pca_csv='pca_results.csv',
    test_size=0.2,
    val_size=0.2,            # fraction of training data used for early stopping
    random_state=42,
    epochs=300,              # max_iter for MLPRegressor
    learning_rate=1e-4,
    hidden_layer_sizes=(8, 4),  # corresponds to two hidden layers
    patience=20,             # n_iter_no_change for early stopping
    num_sam = 5
):
    # ---------------------------
    # 1) Data Loading & Preprocessing
    # ---------------------------
    df = pd.read_csv(pca_csv)
    # Use only the principal component columns for features and 'E_Q' as target
    feature_cols = [col for col in df.columns if col.startswith('V')]
    X_all = df[feature_cols].values
    y_all = df['E_Q'].values

    # Split into Train and Test (we let MLPRegressor handle internal validation via early_stopping)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # Target scaling
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()  # flatten for sklearn
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Save the scaler for later use
    os.makedirs('scaler_pca2', exist_ok=True)
    joblib.dump(scaler_y, f"scaler_pca2/scaler_y_{num_sam}.joblib")

    # ---------------------------
    # 2) Define and Train the MLPRegressor Model
    # ---------------------------
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        learning_rate_init=learning_rate,
        max_iter=epochs,
        early_stopping=True,
        validation_fraction=val_size,
        n_iter_no_change=patience,
        verbose=True,
        random_state=random_state
    )
    print("Training MLPRegressor...")
    model.fit(X_train, y_train_scaled)
    
    # Save the best model using pickle so it can be loaded exactly later
    with open(f'scaler_pca2/best_model_{num_sam}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Best model saved as 'scaler_pca2/best_model_{num_sam}.pkl'.")

    # ---------------------------
    # 3) Plot Training Loss Curve (MLPRegressor provides loss_curve_)
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scaler_pca2/Train_Loss_Curve_{num_sam}.png')
    print("Training loss curve saved.")

    # ---------------------------
    # 4) Evaluate on Test Set
    # ---------------------------
    y_pred_test_scaled = model.predict(X_test)
    # Inverse transform the predictions
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()

    r2  = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)

    print("\n=== Final Evaluation on Test Set ===")
    print(f"R^2  : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # ---------------------------
    # 5) Actual vs Predicted Plot (Test Set)
    # ---------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.4, label='Prediction')
    mn, mx = np.min([y_test, y_pred_test]), np.max([y_test, y_pred_test])
    plt.plot([mn, mx], [mn, mx], color='red', label='Ideal 1:1')
    plt.xlabel('Actual E_Q')
    plt.ylabel('Predicted E_Q')
    plt.title(f'Actual vs Predicted (Test Set)\nR^2: {r2:.4f}, MAPE: {mape:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scaler_pca2/Actual_Predicted_Plot_{num_sam}.png')

    # ---------------------------
    # 6) Residual Error Plot (Test Set)
    # ---------------------------
    residuals_test = y_test - y_pred_test
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred_test, residuals_test, alpha=0.4, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted E_Q')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Error Plot (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scaler_pca2/Residual_Error_Plot_{num_sam}.png')

def evaluate_on_all_data(model, df, scaler_y, num_sam):
    # ---------------------------
    # Evaluate on the Entire Dataset
    # ---------------------------
    feature_cols = [col for col in df.columns if col.startswith('V')]
    X_all = df[feature_cols].values
    y_all = df['E_Q'].values

    y_all_scaled = scaler_y.transform(y_all.reshape(-1, 1)).ravel()

    y_all_pred_scaled = model.predict(X_all)
    y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).flatten()

    r2  = r2_score(y_all, y_all_pred)
    mse = mean_squared_error(y_all, y_all_pred)
    mae = mean_absolute_error(y_all, y_all_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_all, y_all_pred)

    print("\n=== Evaluation on the Entire Dataset ===")
    print(f"R^2  : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # Plot Actual vs Predicted for entire dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(y_all, y_all_pred, alpha=0.4, label='Prediction')
    mn, mx = np.min([y_all, y_all_pred]), np.max([y_all, y_all_pred])
    plt.plot([mn, mx], [mn, mx], color='red', label='Ideal 1:1')
    plt.xlabel('Actual E_Q')
    plt.ylabel('Predicted E_Q')
    plt.title(f'Actual vs Predicted (Entire Dataset)\nR^2: {r2:.4f}, MAPE: {mape:.2f}%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scaler_pca2/Actual_Predicted_Entire_Dataset_{num_sam}.png')

    # Residual Error Plot for Entire Dataset
    residuals_all = y_all - y_all_pred
    plt.figure(figsize=(6, 6))
    plt.scatter(y_all_pred, residuals_all, alpha=0.4, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted E_Q')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Error Plot (Entire Dataset)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scaler_pca2/Residual_Error_Entire_Dataset_{num_sam}.png')

if __name__ == "__main__":
    # Train and evaluate using MLPRegressor
    num_sam = 2500
    csv_file = f'scaler_pca2/v_scl_results_{num_sam}.csv'
    main(
        pca_csv= csv_file,
        test_size=0.2,          # 80% Train, 20% Test
        val_size=0.2,           # 20% of training data used for early stopping
        random_state=42,
        epochs=1000,
        learning_rate=1e-4,
        hidden_layer_sizes=(8, 4),
        patience=20,
        num_sam = num_sam
    )

    # ------------- Evaluate on the Entire Dataset -------------
    # Load the saved model and scaler
    with open(f'scaler_pca2/best_model_{num_sam}.pkl', 'rb') as f:
        model = pickle.load(f)
    scaler_y = joblib.load(f"scaler_pca2/scaler_y_{num_sam}.joblib")
    df_full = pd.read_csv(csv_file)
    evaluate_on_all_data(model, df_full, scaler_y, num_sam)
