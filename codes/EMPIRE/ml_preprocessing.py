import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
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
from gurobi_ml.sklearn import add_decision_tree_regressor_constr,add_linear_regression_constr,add_mlp_regressor_constr
from gurobi_ml.sklearn import add_standard_scaler_constr, add_pipeline_constr
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import pickle



# Load and preprocess data
# def load_data(file_path):
#     data = pd.read_csv(file_path)
#     # Filter out data points where i=1
#     # data = data[data['i'] != 1]
#     data['v_i'] = data['v_i'].apply(ast.literal_eval)
#     data['xi_i'] = data['xi_i'].apply(ast.literal_eval)
    
#     # Ensure necessary columns are present
#     required_columns = {'i', 'v_i', 'xi_i', 'Q_i'}
#     if not required_columns.issubset(data.columns):
#         missing = required_columns - set(data.columns)
#         raise ValueError(f"Missing columns in the data: {missing}")



def load_data(file_path):
    data = pd.read_csv(file_path)
    # Filter out data points where i=1
    # data = data[data['i'] != 1]
    data['x'] = data['x'].apply(ast.literal_eval)
    data['xi'] = data['xi'].apply(ast.literal_eval)
    
    # Ensure necessary columns are present
    required_columns = {'s', 'x', 'xi', 'Q'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing columns in the data: {missing}")
    
    # # Prepare features and target
    # v_i = np.vstack(data['v_i'])
    # xi_i = np.vstack(data['xi_i'])
    # # X = np.hstack([np.vstack(data['v_i']), np.vstack(data['xi_i'])])
    # # X = np.vstack(data['v_i'])
    # y = data['Q_i'].values
    
    # return v_i,xi_i,y

    # Prepare features and target
    x = np.vstack(data['x'])
    xi = np.vstack(data['xi'])
    # X = np.hstack([np.vstack(data['v_i']), np.vstack(data['xi_i'])])
    # X = np.vstack(data['v_i'])
    y = data['Q'].values
    
    return x,xi,y



# def ML_embedding(instance, gurobi_model ,trained_lr,trained_dt, trained_mlp, 
#                 sorted_indices, pyomo_var_to_gurobi_var_ml, scaler_X, scaler_y,seed,SEED_range):

#     # Load the CSV file containing 'i' and 'xi_i' columns
#     xi_values_df = pd.read_csv('scenario3.csv')
#     # Filter xi_i values for the current period 'i' and seed 's'    
#     xi_values = xi_values_df[(xi_values_df['s'] == seed)]['xi']
#     # Convert xi_i_values from strings to lists using ast.literal_eval
#     xi_value = xi_values.apply(ast.literal_eval)

#     # Select a random xi_i vector among the values for period 'i'
#     xi_vector = xi_value.sample(n=1).values[0]  # This is a list of floats

#     # Create Gurobi variables fixed at xi_i_vector components
#     xi_i_vars = []
#     for idx, xi_component in enumerate(xi_i_vector):
#         xi_var = gurobi_model.addVar(lb=xi_component, ub=xi_component, name=f'xi_{i}_{idx}_{seed}')
#         xi_i_vars.append(xi_var)
#     gurobi_model.update()
#     # Add approximation variable
#     y_approx = gurobi_model.addVar(lb=0, name=f'y_approx_{i}_{seed}')
#     gurobi_model.update()
#     if i == 8:  # Final period
#         # Get all y_approx variables
#         y_approx_vars = []
#         for v in gurobi_model.getVars():
#             if v.VarName.startswith('y_approx_'):
#                 y_approx_vars.append(v)
        
#         # Get existing objective
#         existing_obj = gurobi_model.getObjective()
        
#         # Create ML term
#         # ml_term = quicksum(y_approx_vars)
#         ml_term = quicksum(value(instance.discount_multiplier[i+1]) * y_var for i, y_var in enumerate(y_approx_vars))
        
#         # Combine objectives (you can adjust the weight α as needed)
#         alpha = 1/(len(SEED_range))  # Weight for ML term
#         combined_obj = existing_obj + alpha * ml_term

#         # Set combined objective
#         gurobi_model.setObjective(combined_obj, GRB.MINIMIZE)
        
#     # Get ML input variables in correct order
#     ml_input_vars = [pyomo_var_to_gurobi_var_ml[name] for name in sorted_indices]
    
#     add_standard_scaler_constr(gurobi_model, scaler_X, ml_input_vars)
#     # Add decision tree constraints
#     extended_inputs = ml_input_vars + xi_i_vars  # Concatenate the lists
#     # add_decision_tree_regressor_constr(gurobi_model, trained_dt, ml_input_vars, y_approx)
#     add_decision_tree_regressor_constr(gurobi_model, trained_dt, extended_inputs, y_approx)
#     # add_linear_regression_constr(gurobi_model, trained_lr, extended_inputs, y_approx)
#     # add_mlp_regressor_constr(gurobi_model, trained_mlp, extended_inputs, y_approx)
#     gurobi_model.update()
    
#     return gurobi_model

def preprocessing_data(v_i,xi_i,y):
    X = np.hstack([v_i, xi_i])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test

# def ML_training(X_train, y_train, X_test, y_test):

#     lr = Pipeline([
#       ('scaler', StandardScaler()),
#       ('regressor', LinearRegression())
#     ])
#     # ---- Linear Regression ----
#     # lr = LinearRegression()
#     lr.fit(X_train, y_train)
#     y_pred_lr = lr.predict(X_test)

#     mse_lr = mean_squared_error(y_test, y_pred_lr)
#     mae_lr = mean_absolute_error(y_test, y_pred_lr)
#     r2_lr = r2_score(y_test, y_pred_lr)

#     print(f'Linear Regression - MSE: {mse_lr:.4f}, MAE: {mae_lr:.4f}, R²: {r2_lr:.4f}')
    
#     # ---- Decision Tree ----
#     # tree = DecisionTreeRegressor(random_state=42)
#     tree = Pipeline([
#       ('scaler', StandardScaler()),
#       ('regressor', DecisionTreeRegressor(random_state=42))
#     ])

#     tree.fit(X_train, y_train)
#     y_pred_tree = tree.predict(X_test)

#     mse_tree = mean_squared_error(y_test, y_pred_tree)
#     mae_tree = mean_absolute_error(y_test, y_pred_tree)
#     r2_tree = r2_score(y_test, y_pred_tree)

#     print(f'Decision Tree - MSE: {mse_tree:.4f}, MAE: {mae_tree:.4f}, R²: {r2_tree:.4f}')

#     mlp = Pipeline([
#       ('scaler', StandardScaler()),
#       ('regressor', MLPRegressor(
#           hidden_layer_sizes=(50, 25),
#           activation='relu',
#           alpha=0.2,
#           max_iter=500,
#           random_state=42
#       ))
#     ])

#     mlp.fit(X_train, y_train)
#     y_pred_mlp = mlp.predict(X_test)

#     mse_mlp = mean_squared_error(y_test, y_pred_mlp)
#     mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
#     r2_mlp = r2_score(y_test, y_pred_mlp)

#     print(f'MLP - MSE: {mse_mlp:.4f}, MAE: {mae_mlp:.4f}, R²: {r2_mlp:.4f}')

#     return lr,tree, mlp



def ML_training(X_train, y_train, X_test, y_test):
    # Linear Regression with Ridge (L2 regularization)
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))  # alpha는 regularization strength
    ])
    
    # Decision Tree with regularization parameters
    tree = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', DecisionTreeRegressor(
            min_samples_split=5,  # 과적합 방지
            min_samples_leaf=3,   # 과적합 방지
            max_depth=10,         # 트리 깊이 제한
            random_state=42
        ))
    ])

    # MLP with regularization
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            alpha=0.5,           # L2 regularization
            learning_rate='adaptive',
            early_stopping=True, # 조기 종료로 과적합 방지
            max_iter=500,
            random_state=42,
            validation_fraction=0.1
        ))
    ])

    # Train and evaluate models
    models = {
        'Linear Regression': lr,
        'Decision Tree': tree,
        'MLP': mlp
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f'{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')

    return lr, tree, mlp





# def ML_embedding(instance, gurobi_model ,trained_lr,trained_dt,trained_mlp, 
#             indices, pyomo_var_to_gurobi_var_ml, seed,SEED_range):

def ML_embedding(instance, gurobi_model ,trained_lr,trained_dt,trained_mlp, 
            indices, pyomo_var_to_gurobi_var_ml, seed,SEED_range):

    # Load stored PCA models and scalers for both x and xi
    with open('pca_results_x/pca_model.pkl', 'rb') as f:
        pca_x = pickle.load(f)
    with open('pca_results_x/scaler.pkl', 'rb') as f:
        scaler_x = pickle.load(f)

    # Load the CSV file containing 'i' and 'xi_i' columns
    xi_values_df = pd.read_csv('scenario5.csv')
    # Filter xi_i values for the current period 'i' and seed 's'    
    xi_values = xi_values_df[(xi_values_df['s'] == seed)]['xi']
    # Convert xi_i_values from strings to lists using ast.literal_eval
    xi_value = xi_values.apply(ast.literal_eval)

    # Select a random xi_i vector among the values for period 'i'
    xi_vector = xi_value.sample(n=1).values[0]  # This is a list of floats

    # Create Gurobi variables fixed at xi_i_vector components
    xi_vars = []
    for idx, xi_component in enumerate(xi_vector):
        xi_var = gurobi_model.addVar(lb=xi_component, ub=xi_component, name=f'xi_{idx}_{seed}')
        xi_vars.append(xi_var)
    gurobi_model.update()
    # Add approximation variable
    y_approx = gurobi_model.addVar(lb=0, name=f'y_approx_{seed}')
    # y_scaled_approx = gurobi_model.addVar(name=f'y_scaled_approx')
    gurobi_model.update()

    # Get x variables and transform them
    x_vars_original = [pyomo_var_to_gurobi_var_ml[name] for name in indices]
    
    # Get the PCA components count
    n_components_x = pca_x.n_components_
    
    # Create new variables for PCA-transformed x
    x_pca_vars = []
    for i in range(n_components_x):
        # These variables will be constrained to be the PCA transformation of x
        x_pca_var = gurobi_model.addVar(name=f'x_pca_{i}_{seed}')
        x_pca_vars.append(x_pca_var)
    gurobi_model.update()
    
    # Add constraints for PCA transformation of x
    # First standardize x using the scaler
    x_mean = scaler_x.mean_
    x_scale = scaler_x.scale_
    
    # Then apply PCA transformation
    components = pca_x.components_
    
    # For each PCA component
    for i in range(n_components_x):
        # Create the transformation expression
        scaled_expr = quicksum((x_vars_original[j] - x_mean[j]) / x_scale[j] * components[i,j] 
                             for j in range(len(x_vars_original)))
        # Add constraint
        gurobi_model.addConstr(x_pca_vars[i] == scaled_expr, name=f'pca_x_constr_{i}_{seed}')

    existing_obj = gurobi_model.getObjective()
    
    # Combine objectives (you can adjust the weight α as needed)
    alpha = 1/(len(SEED_range))  # Weight for ML term
    combined_obj = existing_obj + alpha * y_approx

    # Set combined objective
    gurobi_model.setObjective(combined_obj, GRB.MINIMIZE)

    gurobi_model.update()

    # Get ML input variables in correct order
    # ml_input_vars = [pyomo_var_to_gurobi_var_ml[name] for name in indices]
    # Add decision tree constraints
    extended_inputs = x_pca_vars + xi_vars  # Concatenate the lists
    
    # add_standard_scaler_constr(gurobi_model, scaler_X, extended_inputs)
    # add_decision_tree_regressor_constr(gurobi_model, trained_dt, ml_input_vars, y_approx)
    pred_constr = add_pipeline_constr(gurobi_model, trained_dt, extended_inputs, y_approx)
    
    print(pred_constr.print_stats())
    # add_linear_regression_constr(gurobi_model, trained_lr, extended_inputs, y_scaled_approx)    
        
    gurobi_model.update()
    
    return gurobi_model



# def selected_var_mapping(instance, solver):
#     selected_indices = {
#         'Generation': {
#             'nodes': ['Germany', 'France'],
#             'types': ['Solar', 'Windonshore', 'GasCCGT', 'Bio']  
#         },
#         'Storage Power': {
#             'nodes': ['Germany'],
#             'types': ['Li-Ion_BESS']
#         },
#         'Storage Energy': {
#             'nodes': ['Germany'],
#             'types': ['Li-Ion_BESS']
#         }
#     }

#     desired_data = {'Generation': ({'Germany' :['Solar', 'GasCCGT', 'Bio', 'Bio10cofiring']} , {'France' : ['Windonshore', 'Solar', 'GasCCGT', 'Bio']}, {'Denmark' : ['Solar', 'GasCCGT', 'Windonshore']}),
#     'Storage Power': ({'Germany':['Li-Ion_BESS']}, {'France':['Li-Ion_BESS']}, {'Denmark':['Li-Ion_BESS']}),
#     'Storage Enenrgy': ({'Germany':['Li-Ion_BESS']}, {'France':['Li-Ion_BESS']}, {'Denmark':['Li-Ion_BESS']})
#     }
    
#     pyomo_var_to_gurobi_var = {}
    
#     # 1. Generator installed capacities
#     for (n,g) in instance.GeneratorsOfNode:
#         for i in instance.PeriodActive:
#             if (n in selected_indices['Generation']['nodes'] and g in selected_indices['Generation']['types']):
#                 var = instance.genInstalledCap[n,g,i]
#                 gurobi_var = solver._pyomo_var_to_solver_var_map[instance.genInstalledCap[n,g,i]]
#                 pyomo_var_to_gurobi_var[var.name] = gurobi_var

#     # 3. Storage Power installed capacities
#     for (n,b) in instance.StoragesOfNode:
#         for i in instance.PeriodActive:
#             if (n in selected_indices['Storage Power']['nodes'] and b in selected_indices['Storage Power']['types']):
#                     var = instance.storPWInstalledCap[n,b,i]
#                     gurobi_var = solver._pyomo_var_to_solver_var_map[var]
#                     pyomo_var_to_gurobi_var[var.name] = gurobi_var

#     # 4. Storage Energy installed capacities
#     for (n,b) in instance.StoragesOfNode:
#         for i in instance.PeriodActive:
#             if (n in selected_indices['Storage Energy']['nodes'] and b in selected_indices['Storage Energy']['types']):
#                     var = instance.storENInstalledCap[n,b,i]
#                     gurobi_var = solver._pyomo_var_to_solver_var_map[var]
#                     pyomo_var_to_gurobi_var[var.name] = gurobi_var

#     # 2. Transmission installed capacities
#     for (n1,n2) in instance.BidirectionalArc:
#         for i in instance.PeriodActive:
#             var = instance.transmissionInstalledCap[n1,n2,i]
#             gurobi_var = solver._pyomo_var_to_solver_var_map[var]
#             pyomo_var_to_gurobi_var[var.name] = gurobi_var

#     indices = pyomo_var_to_gurobi_var.keys()
    
#     return indices, pyomo_var_to_gurobi_var


def selected_var_mapping(instance, solver):
    desired_data = {
        'Generation': [
            ('Germany', 'GasCCGT'),
            ('Denmark', 'GasCCGT'),
            ('France', 'GasCCGT'),
            ('Germany', 'Bio10cofiring'),
            ('Germany', 'Bio'),
            ('France', 'Bio'),
            ('Denmark', 'Windonshore'),
            ('France', 'Windonshore'),
            ('Germany', 'Solar'),
            ('Denmark', 'Solar'),
            ('France', 'Solar')
        ],
        'Storage Power': [
            ('Germany', 'Li-Ion_BESS'),
            ('Denmark', 'Li-Ion_BESS'),
            ('France', 'Li-Ion_BESS')
        ],
        'Storage Energy': [
            ('Germany', 'Li-Ion_BESS'),
            ('Denmark', 'Li-Ion_BESS'),
            ('France', 'Li-Ion_BESS')
        ]
    }
    
    pyomo_var_to_gurobi_var = {}
    
    # 1. Generator installed capacities
    for (n,g) in instance.GeneratorsOfNode:
        for i in instance.PeriodActive:
            if (n,g) in desired_data['Generation']:
                var = instance.genInstalledCap[n,g,i]
                gurobi_var = solver._pyomo_var_to_solver_var_map[instance.genInstalledCap[n,g,i]]
                pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 3. Storage Power installed capacities
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            if (n,b) in desired_data['Storage Power']:
                var = instance.storPWInstalledCap[n,b,i]
                gurobi_var = solver._pyomo_var_to_solver_var_map[var]
                pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 4. Storage Energy installed capacities
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            if (n,b) in desired_data['Storage Energy']:
                var = instance.storENInstalledCap[n,b,i]
                gurobi_var = solver._pyomo_var_to_solver_var_map[var]
                pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 2. Transmission installed capacities
    for (n1,n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            var = instance.transmissionInstalledCap[n1,n2,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    indices = pyomo_var_to_gurobi_var.keys()
    
    return indices, pyomo_var_to_gurobi_var


def input_var_mapping(instance, solver):
    
    pyomo_var_to_gurobi_var = {}
    
    # 1. Generator installed capacities
    for (n,g) in instance.GeneratorsOfNode:
        for i in instance.PeriodActive:
            var = instance.genInvCap[n,g,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 3. Storage Power installed capacities
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            var = instance.storPWInvCap[n,b,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 4. Storage Energy installed capacities
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            var = instance.storENInvCap[n,b,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 2. Transmission installed capacities
    for (n1,n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            var = instance.transmisionInvCap[n1,n2,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    indices = pyomo_var_to_gurobi_var.keys()
    print(indices)
    
    return indices, pyomo_var_to_gurobi_var
