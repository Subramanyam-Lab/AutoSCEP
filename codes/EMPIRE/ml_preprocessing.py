import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import ast
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')  # To suppress any warnings for cleaner output
from gurobipy import GRB, quicksum  # Import Gurobi's quicksum
from pyomo.environ import value  # Import the value function
from gurobi_ml import add_predictor_constr
from gurobi_ml.sklearn import add_decision_tree_regressor_constr,add_linear_regression_constr
from gurobi_ml.sklearn import add_standard_scaler_constr



# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Filter out data points where i=1
    # data = data[data['i'] != 1]
    data['v_i'] = data['v_i'].apply(ast.literal_eval)
    data['xi_i'] = data['xi_i'].apply(ast.literal_eval)
    
    # Ensure necessary columns are present
    required_columns = {'i', 'v_i', 'xi_i', 'Q_i'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing columns in the data: {missing}")
    
    # Prepare features and target
    # X = np.hstack([np.vstack(data['v_i']), np.vstack(data['xi_i'])])
    X = np.vstack(data['v_i'])
    y = data['Q_i'].values
    
    return X, y

def ML_trainig(X_train, y_train, X_test, y_test, scaler_X, scaler_y):
    # Standardize features based on training data
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Standardize target
    # y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
    # y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    y_train_scaled = y_train
    y_test_scaled = y_test
    # ---- Linear Regression ----
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_scaled)
    y_pred_lr_scaled = lr.predict(X_test_scaled)
    # y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1,1)).flatten()
    y_pred_lr = y_pred_lr_scaled.reshape(-1,1)

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f'Linear Regression - MSE: {mse_lr:.4f}, MAE: {mae_lr:.4f}, R²: {r2_lr:.4f}')
    
    # ---- Decision Tree ----
    tree = DecisionTreeRegressor(random_state=SEED)
    tree.fit(X_train_scaled, y_train_scaled)
    y_pred_tree_scaled = tree.predict(X_test_scaled)
    # y_pred_tree = scaler_y.inverse_transform(y_pred_tree_scaled.reshape(-1,1)).flatten()
    y_pred_tree = y_pred_tree_scaled.reshape(-1,1)
    mse_tree = mean_squared_error(y_test, y_pred_tree)
    mae_tree = mean_absolute_error(y_test, y_pred_tree)
    r2_tree = r2_score(y_test, y_pred_tree)

    print(f'Decision Tree - MSE: {mse_tree:.4f}, MAE: {mae_tree:.4f}, R²: {r2_tree:.4f}')

    return lr,tree


def ML_embedding(instance, gurobi_model ,trained_lr,trained_dt, sorted_indices, pyomo_var_to_gurobi_var_ml, period,scaler_X, scaler_y):
    i = period
    # Add approximation variable
    y_approx = gurobi_model.addVar(lb=0, name=f'y_approx_{i}')
    gurobi_model.update()
    if i == 8:  # Final period
        # Get all y_approx variables
        y_approx_vars = []
        for v in gurobi_model.getVars():
            if v.VarName.startswith('y_approx_'):
                y_approx_vars.append(v)
        
        # Get existing objective
        existing_obj = gurobi_model.getObjective()
        
        # Create ML term
        ml_term = quicksum(y_approx_vars)
        
        # Combine objectives (you can adjust the weight α as needed)
        alpha = 1.0  # Weight for ML term
        combined_obj = existing_obj + alpha * ml_term

        # Set combined objective
        gurobi_model.setObjective(combined_obj, GRB.MINIMIZE)
        
    # Get ML input variables in correct order
    ml_input_vars = [pyomo_var_to_gurobi_var_ml[name] for name in sorted_indices]
    
    add_standard_scaler_constr(gurobi_model, scaler_X, ml_input_vars)
    # Add decision tree constraints
    add_decision_tree_regressor_constr(gurobi_model, trained_dt, ml_input_vars, y_approx)
    # add_linear_regression_constr(gurobi_model, trained_lr, ml_input_vars, y_approx)
    gurobi_model.update()
    
    return gurobi_model


def var_mapping(instance, solver):
    pyomo_var_to_gurobi_var = {}

    # 1. Generator installed capacities
    for (n,g) in instance.GeneratorsOfNode:
        for i in instance.PeriodActive: 
            var_inv = instance.genInvCap[n,g,i]
            var_installed = instance.genInstalledCap[n,g,i]
            gurobi_var_inv = solver._pyomo_var_to_solver_var_map[instance.genInvCap[n,g,i]]
            gurobi_var_installed = solver._pyomo_var_to_solver_var_map[instance.genInstalledCap[n,g,i]]
            pyomo_var_to_gurobi_var[var_inv.name] = gurobi_var_inv
            pyomo_var_to_gurobi_var[var_installed.name] = gurobi_var_installed


    # 2. Transmission installed capacities
    for (n1,n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            var_inv = instance.transmisionInvCap[n1,n2,i]
            var_installed = instance.transmissionInstalledCap[n1,n2,i]
            gurobi_var_inv = solver._pyomo_var_to_solver_var_map[var_inv] 
            gurobi_var_installed = solver._pyomo_var_to_solver_var_map[var_installed]
            pyomo_var_to_gurobi_var[var_inv.name] = gurobi_var_inv
            pyomo_var_to_gurobi_var[var_installed.name] = gurobi_var_installed

    # 2. Storage Power installed capacities
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            var_inv = instance.storPWInvCap[n,b,i]
            var_installed = instance.storPWInstalledCap[n,b,i]
            gurobi_var_inv = solver._pyomo_var_to_solver_var_map[var_inv]
            gurobi_var_installed = solver._pyomo_var_to_solver_var_map[var_installed]
            pyomo_var_to_gurobi_var[var_inv.name] = gurobi_var_inv
            pyomo_var_to_gurobi_var[var_installed.name] = gurobi_var_installed

            var_inv = instance.storENInvCap[n,b,i]
            var_installed = instance.storENInstalledCap[n,b,i]
            gurobi_var_inv = solver._pyomo_var_to_solver_var_map[var_inv]
            gurobi_var_installed = solver._pyomo_var_to_solver_var_map[var_installed]
            pyomo_var_to_gurobi_var[var_inv.name] = gurobi_var_inv
            pyomo_var_to_gurobi_var[var_installed.name] = gurobi_var_installed

    sorted_indices = sorted(pyomo_var_to_gurobi_var.keys())
    
    return sorted_indices, pyomo_var_to_gurobi_var



def selected_var_mapping(instance, solver, period):
    i = period
    selected_indices = {
        'Generation': {
            'nodes': ['Germany', 'France'],
            'types': ['Solar', 'Windonshore', 'GasCCGT', 'Bio']  
        },
        'Storage Power': {
            'nodes': ['Germany'],
            'types': ['Li-Ion_BESS']
        },
        'Storage Energy': {
            'nodes': ['Germany'],
            'types': ['Li-Ion_BESS']
        }
    }
    
    pyomo_var_to_gurobi_var = {}
    
    # 1. Generator installed capacities
    for (n,g) in instance.GeneratorsOfNode:
        if (n in selected_indices['Generation']['nodes'] and 
            g in selected_indices['Generation']['types']):
            var = instance.genInstalledCap[n,g,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[instance.genInstalledCap[n,g,i]]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 2. Transmission installed capacities
    for (n1,n2) in instance.BidirectionalArc:
            var = instance.transmissionInstalledCap[n1,n2,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 2. Storage Power installed capacities
    for (n,b) in instance.StoragesOfNode:
        if (n in selected_indices['Storage Power']['nodes'] and 
            b in selected_indices['Storage Power']['types']):
                var = instance.storPWInstalledCap[n,b,i]
                gurobi_var = solver._pyomo_var_to_solver_var_map[var]
                pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 3. Storage Energy installed capacities
    for (n,b) in instance.StoragesOfNode:
        if (n in selected_indices['Storage Energy']['nodes'] and 
            b in selected_indices['Storage Energy']['types']):
                var = instance.storENInstalledCap[n,b,i]
                gurobi_var = solver._pyomo_var_to_solver_var_map[var]
                pyomo_var_to_gurobi_var[var.name] = gurobi_var

    sorted_indices = sorted(pyomo_var_to_gurobi_var.keys())
    
    return sorted_indices, pyomo_var_to_gurobi_var


def get_gurobi_installed_cap_vars(instance, gurobi_model,pyomo_var_to_gurobi_var):

    # Recreate the lifetime constraint in Gurobi for generators
    for (n, g) in instance.GeneratorsOfNode:
        for i in instance.PeriodActive:
            startPeriod = 1
            genLifetime = value(instance.genLifetime[g])
            LeapYearsInvestment = value(instance.LeapYearsInvestment)
            if 1 + i - (genLifetime / LeapYearsInvestment) > startPeriod:
                startPeriod = int(1 + i - (genLifetime / LeapYearsInvestment))
            expr = (
                quicksum(
                    pyomo_var_to_gurobi_var[instance.genInvCap[n, g, j]]
                    for j in instance.PeriodActive
                    if j >= startPeriod and j <= i
                )
                - pyomo_var_to_gurobi_var[instance.genInstalledCap[n, g, i]]
                + value(instance.genInitCap[n, g, i])
            )
            gurobi_model.addConstr(expr == 0, name=f'lifetime_gen_{n}_{g}_{i}')

    # Assuming transmissionLifetime is a single value
    
    for (n1, n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            startPeriod = 1
            transmission_lifetime = value(instance.transmissionLifetime[n1,n2])  # Adjust accordingly
            LeapYearsInvestment = value(instance.LeapYearsInvestment)
            if 1 + i - (transmission_lifetime / LeapYearsInvestment) > startPeriod:
                startPeriod = int(1 + i - (transmission_lifetime / LeapYearsInvestment))
            expr = (
                quicksum(
                    pyomo_var_to_gurobi_var[instance.transmisionInvCap[n1, n2, j]]
                    for j in instance.PeriodActive
                    if j >= startPeriod and j <= i
                )
                - pyomo_var_to_gurobi_var[instance.transmissionInstalledCap[n1, n2, i]]
                + value(instance.transmissionInitCap[n1, n2, i])
            )
            gurobi_model.addConstr(expr == 0, name=f'lifetime_trans_{n1}_{n2}_{i}')

    # Storage Power Installed Capacity Constraints
    for (n, b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            startPeriod = 1
            lifetime = value(instance.storageLifetime[b])  # Lifetime specific to storage technology b
            LeapYearsInvestment = value(instance.LeapYearsInvestment)
            if 1 + i - (lifetime / LeapYearsInvestment) > startPeriod:
                startPeriod = int(1 + i - (lifetime / LeapYearsInvestment))
            expr = (
                quicksum(
                    pyomo_var_to_gurobi_var[instance.storPWInvCap[n, b, j]]
                    for j in instance.PeriodActive
                    if j >= startPeriod and j <= i
                )
                - pyomo_var_to_gurobi_var[instance.storPWInstalledCap[n, b, i]]
                + value(instance.storPWInitCap[n, b, i])
            )
            gurobi_model.addConstr(expr == 0, name=f'lifetime_storPW_{n}_{b}_{i}')

    # Storage Energy Installed Capacity Constraints
    for (n, b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            startPeriod = 1
            lifetime = value(instance.storageLifetime[b])  # Lifetime specific to storage technology b
            LeapYearsInvestment = value(instance.LeapYearsInvestment)
            if 1 + i - (lifetime / LeapYearsInvestment) > startPeriod:
                startPeriod = int(1 + i - (lifetime / LeapYearsInvestment))
            expr = (
                quicksum(
                    pyomo_var_to_gurobi_var[instance.storENInvCap[n, b, j]]
                    for j in instance.PeriodActive
                    if j >= startPeriod and j <= i
                )
                - pyomo_var_to_gurobi_var[instance.storENInstalledCap[n, b, i]]
                + value(instance.storENInitCap[n, b, i])
            )
            gurobi_model.addConstr(expr == 0, name=f'lifetime_storEN_{n}_{b}_{i}')

    gurobi_model.update()

    return gurobi_model