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
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).flatten()

    # ---- Linear Regression ----
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_scaled)
    y_pred_lr_scaled = lr.predict(X_test_scaled)
    y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1,1)).flatten()

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f'Linear Regression - MSE: {mse_lr:.4f}, MAE: {mae_lr:.4f}, R²: {r2_lr:.4f}')
    
    # ---- Decision Tree ----
    tree = DecisionTreeRegressor(random_state=SEED)
    tree.fit(X_train_scaled, y_train_scaled)
    y_pred_tree_scaled = tree.predict(X_test_scaled)
    y_pred_tree = scaler_y.inverse_transform(y_pred_tree_scaled.reshape(-1,1)).flatten()

    mse_tree = mean_squared_error(y_test, y_pred_tree)
    mae_tree = mean_absolute_error(y_test, y_pred_tree)
    r2_tree = r2_score(y_test, y_pred_tree)

    print(f'Decision Tree - MSE: {mse_tree:.4f}, MAE: {mae_tree:.4f}, R²: {r2_tree:.4f}')

    return lr,tree


def ML_embedding(gurobi_model, gurobi_inv_cap_vars, pyomo_var_to_gurobi_var_ml, regression, sorted_indices, instance, period):
    i = period
    # Build the original objective expression using Gurobi's methods

    # Build the objective function in Gurobi
    first_stage_expr_terms = []

    # first_stage_expr = quicksum(
    #     value(instance.discount_multiplier[i]) * (
    #         quicksum(
    #             value(instance.genInvCost[g, i]) * pyomo_var_to_gurobi_var[('genInvCap', n, g, i)] 
    #             for (n, g) in instance.GeneratorsOfNode
    #         )
    #         +
    #         quicksum(
    #             value(instance.transmissionInvCost[n1, n2, i]) * pyomo_var_to_gurobi_var[('transmisionInvCap', n1, n2, i)]
    #             for (n1, n2) in instance.BidirectionalArc
    #         )
    #         +
    #         quicksum(
    #             value(instance.storPWInvCost[b, i]) * pyomo_var_to_gurobi_var[('storPWInvCap', n, b, i)]
    #             +
    #             value(instance.storENInvCost[b, i]) * pyomo_var_to_gurobi_var[('storENInvCap', n, b, i)]
    #             for (n, b) in instance.StoragesOfNode
    #         )
    #     )
    # )
    discount_multiplier = value(instance.discount_multiplier[i])

    # Generator Investment Costs
    gen_inv_cost_terms = [
        discount_multiplier * value(instance.genInvCost[g, i]) * gurobi_inv_cap_vars[('genInvCap', n, g, i)]
        for (n, g) in instance.GeneratorsOfNode
    ]

    # Transmission Investment Costs
    trans_inv_cost_terms = [
        discount_multiplier * value(instance.transmissionInvCost[n1, n2, i]) * gurobi_inv_cap_vars[('transmisionInvCap', n1, n2, i)]
        for (n1, n2) in instance.BidirectionalArc
    ]

    # Storage Investment Costs
    stor_inv_cost_terms = [
        discount_multiplier * (
            value(instance.storPWInvCost[b, i]) * gurobi_inv_cap_vars[('storPWInvCap', n, b, i)] +
            value(instance.storENInvCost[b, i]) * gurobi_inv_cap_vars[('storENInvCap', n, b, i)]
        )
        for (n, b) in instance.StoragesOfNode
    ]

    # Collect all terms
    first_stage_expr_terms.extend(gen_inv_cost_terms)
    first_stage_expr_terms.extend(trans_inv_cost_terms)
    first_stage_expr_terms.extend(stor_inv_cost_terms)

    # Build the first-stage expression using Gurobi's quicksum
    first_stage_expr = quicksum(first_stage_expr_terms)

    # Add the variable for the approximated second-stage cost
    y_approx = gurobi_model.addVar(lb=-GRB.INFINITY, name=f'y_approx_{i}')

    gurobi_model.update()
    
    # Set the new objective function including 'y_approx'
    gurobi_model.setObjective(first_stage_expr + y_approx, GRB.MINIMIZE)

    # Build ml_input_vars based on sorted_indices
    ml_input_vars = [pyomo_var_to_gurobi_var_ml[name] for name in sorted_indices]

    # Add the surrogate constraint using the regression model
    pred_constr = add_predictor_constr(gurobi_model, regression, ml_input_vars, y_approx)

    # Update the Gurobi model
    gurobi_model.update()

    return gurobi_model

def var_mapping(instance, solver, period):
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
#            for i in instance.PeriodActive:
                var = instance.genInstalledCap[n,g,i]
                gurobi_var = solver._pyomo_var_to_solver_var_map[var]
                # pyomo_var_to_gurobi_var[(n,g,i)] = gurobi_var
                pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 2. Transmission installed capacities
    for (n1,n2) in instance.BidirectionalArc:
#        for i in instance.PeriodActive:
            var = instance.transmissionInstalledCap[n1,n2,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            # pyomo_var_to_gurobi_var[(n1,n2,i)] = gurobi_var
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 2. Storage Power installed capacities
    for (n,b) in instance.StoragesOfNode:
        if (n in selected_indices['Storage Power']['nodes'] and 
            b in selected_indices['Storage Power']['types']):
#            for i in instance.PeriodActive:
                var = instance.storPWInstalledCap[n,b,i]
                gurobi_var = solver._pyomo_var_to_solver_var_map[var]
                # pyomo_var_to_gurobi_var[(n,b,i, "Storage Power")] = gurobi_var
                pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 3. Storage Energy installed capacities
    for (n,b) in instance.StoragesOfNode:
        if (n in selected_indices['Storage Energy']['nodes'] and 
            b in selected_indices['Storage Energy']['types']):
#            for i in instance.PeriodActive:
                var = instance.storENInstalledCap[n,b,i]
                gurobi_var = solver._pyomo_var_to_solver_var_map[var]
                # pyomo_var_to_gurobi_var[(n,b,i, "Storage Energy")] = gurobi_var
                pyomo_var_to_gurobi_var[var.name] = gurobi_var

    sorted_indices = sorted(pyomo_var_to_gurobi_var.keys())
    print("dict_size:", len(pyomo_var_to_gurobi_var))
    
    return sorted_indices, pyomo_var_to_gurobi_var



def get_gurobi_inv_cap_vars(instance, gurobi_model, period):
    i = period
    # Create dictionaries to hold the Gurobi variables
    gurobi_inv_cap_vars = {}

    # 1. Generator Investment Capacity Variables
    for (n, g) in instance.GeneratorsOfNode:
        for i in instance.PeriodActive:
            var_name = f'genInvCap_{n}_{g}_{i}'
            gurobi_var = gurobi_model.addVar(lb=0, name=var_name)
            gurobi_inv_cap_vars[('genInvCap', n, g, i)] = gurobi_var

    # 2. Transmission Investment Capacity Variables
    for (n1, n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            var_name = f'transmisionInvCap_{n1}_{n2}_{i}'
            gurobi_var = gurobi_model.addVar(lb=0, name=var_name)
            gurobi_inv_cap_vars[('transmisionInvCap', n1, n2, i)] = gurobi_var

    # 3. Storage Power Investment Capacity Variables
    for (n, b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            var_name = f'storPWInvCap_{n}_{b}_{i}'
            gurobi_var = gurobi_model.addVar(lb=0, name=var_name)
            gurobi_inv_cap_vars[('storPWInvCap', n, b, i)] = gurobi_var

    # 4. Storage Energy Investment Capacity Variables
    for (n, b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            var_name = f'storENInvCap_{n}_{b}_{i}'
            gurobi_var = gurobi_model.addVar(lb=0, name=var_name)
            gurobi_inv_cap_vars[('storENInvCap', n, b, i)] = gurobi_var

    gurobi_model.update()

    return gurobi_inv_cap_vars


def get_gurobi_installed_cap_vars(instance, gurobi_model,gurobi_inv_cap_vars):

    # Recreate the lifetime constraint in Gurobi for generators
    for (n, g) in instance.GeneratorsOfNode:
        for i in instance.PeriodActive:
            startPeriod = 1
            if 1 + i - (instance.genLifetime[g] / instance.LeapYearsInvestment) > startPeriod:
                startPeriod = int(1 + i - instance.genLifetime[g] / instance.LeapYearsInvestment)
            expr = (
                quicksum(
                    gurobi_inv_cap_vars[('genInvCap', n, g, j)]
                    for j in instance.PeriodActive
                    if j >= startPeriod and j <= i
                )
                - pyomo_var_to_gurobi_var[instance.genInstalledCap[n, g, i]]
                + value(instance.genInitCap[n, g, i])
            )
            gurobi_model.addConstr(expr == 0, name=f'lifetime_gen_{n}_{g}_{i}')

    # Assuming transmissionLifetime is a single value
    transmission_lifetime = instance.transmissionLifetime  # Adjust accordingly

    for (n1, n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            startPeriod = 1
            if 1 + i - (transmission_lifetime / instance.LeapYearsInvestment) > startPeriod:
                startPeriod = int(1 + i - (transmission_lifetime / instance.LeapYearsInvestment))
            expr = (
                quicksum(
                    gurobi_inv_cap_vars[('transmisionInvCap', n1, n2, j)]
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
            lifetime = instance.storageLifetime[b]  # Lifetime specific to storage technology b
            if 1 + i - (lifetime / instance.LeapYearsInvestment) > startPeriod:
                startPeriod = int(1 + i - (lifetime / instance.LeapYearsInvestment))
            expr = (
                quicksum(
                    gurobi_inv_cap_vars[('storPWInvCap', n, b, j)]
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
            lifetime = instance.storageLifetime[b]  # Lifetime specific to storage technology b
            if 1 + i - (lifetime / instance.LeapYearsInvestment) > startPeriod:
                startPeriod = int(1 + i - (lifetime / instance.LeapYearsInvestment))
            expr = (
                quicksum(
                    gurobi_inv_cap_vars[('storENInvCap', n, b, j)]
                    for j in instance.PeriodActive
                    if j >= startPeriod and j <= i
                )
                - pyomo_var_to_gurobi_var[instance.storENInstalledCap[n, b, i]]
                + value(instance.storENInitCap[n, b, i])
            )
            gurobi_model.addConstr(expr == 0, name=f'lifetime_storEN_{n}_{b}_{i}')

    # Deactivate the constraints in Pyomo
    instance.installedCapDefinitionGen.deactivate()
    instance.installedCapDefinitionStorEN.deactivate()
    instance.installedCapDefinitionStorPOW.deactivate()
    instance.installedCapDefinitionTrans.deactivate()

    return gurobi_model




import logging
import pickle

def load_pca_data(input_dir='pca_results'):
    X_pca = np.load(f'{input_dir}/pca_vectors.npy')

    with open(f'{input_dir}/pca_model.pkl', 'rb') as f:
        pca_model = pickle.load(f)
    
    with open(f'{input_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(f'{input_dir}/pca_results.json', 'r') as f:
        pca_results = json.load(f)
    
    return X_pca, pca_model, scaler, pca_results