import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
import ast
import joblib
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['v_i'] = data['v_i'].apply(ast.literal_eval)
    data['xi_i'] = data['xi_i'].apply(ast.literal_eval)
    data['period'] = data['period'].astype(float) 
    return data


def preprocessing_data(data):
    v_i = np.vstack(data['v_i'])
    xi_i = np.vstack(data['xi_i'])
    Q_i = data['Q_i'].values

    X_raw = np.hstack([v_i, xi_i])

    v_feature_names = [f'v_{i}' for i in range(v_i.shape[1])]
    xi_feature_names = [f'xi_{i}' for i in range(xi_i.shape[1])]
    feature_names = v_feature_names + xi_feature_names

    X_df = pd.DataFrame(X_raw, columns=feature_names)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(Q_i.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_scaled, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test, scaler_y


def ML_training(X_train, y_train, X_test, y_test,scaler_y):

    lr_pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    dt_pipe = make_pipeline(StandardScaler(), DecisionTreeRegressor(random_state=42, max_depth=5))
    rf_pipe = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100))
    gb_pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=100, learning_rate=0.1))

    # Train models
    models = {
        'Ridge Regression' : lr_pipe,
        'Decision Tree': dt_pipe,
        'Random Forest': rf_pipe,
        'GBoosting': gb_pipe
    }

    results = {}
    scaled_results = {}  # Store results in scaled space

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Get predictions in scaled space
        y_pred_scaled = model.predict(X_test)

        # Transform predictions and actual values back to original scale for interpretation
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        results[name] = {
            'mse': mean_squared_error(y_test_original, y_pred_original),
            'mae': mean_absolute_error(y_test_original, y_pred_original),
            'r2': r2_score(y_test_original, y_pred_original)
        }

        print(f"{name} Results (in original scale):")
        print(f"MSE: {results[name]['mse']:.4f}, MAE: {results[name]['mae']:.4f}, R²: {results[name]['r2']:.4f}")

    best_model_name = min(results, key=lambda x: results[x]['mse'])
    print(f"\nBest Overall Model (based on scaled MSE): {best_model_name}")
    print(f"Best Scaled MSE: {results[best_model_name]['mse']:.4f}")

    return models['Ridge Regression'], models['Decision Tree'], models['Random Forest'], models['GBoosting']

def save_models(trained_dt, trained_rf, trained_gb, scalers=None, path='models/'):
    Path(path).mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_dt, f'{path}decision_tree.joblib')
    joblib.dump(trained_rf, f'{path}random_forest.joblib')
    joblib.dump(trained_gb, f'{path}gradient_boosting.joblib')
    if scalers:
        joblib.dump(scalers, f'{path}scalers.joblib')

def load_models(path='models/'):
    trained_dt = joblib.load(f'{path}decision_tree.joblib')
    trained_rf = joblib.load(f'{path}random_forest.joblib')
    trained_gb = joblib.load(f'{path}gradient_boosting.joblib')
    return trained_dt, trained_rf, trained_gb