#!/usr/bin/env python
# File & System imports
import os
import sys
import csv
import json
import pickle
import cloudpickle
import joblib
import logging
from pathlib import Path
import argparse
from datetime import datetime
from yaml import safe_load
import time
import ast
import warnings
warnings.filterwarnings('ignore')  # To suppress any warnings for cleaner output

# Data manipulation and analysis
import numpy as np
import pandas as pd
import yaml
from ast import literal_eval

# Machine Learning imports
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from skopt import BayesSearchCV

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

# Optimization imports
import gurobipy as grb
from gurobipy import GRB, quicksum, Model as GurobiModel
from gurobi_ml import add_predictor_constr
from gurobi_ml.sklearn import (
    add_decision_tree_regressor_constr,
    add_linear_regression_constr,
    add_mlp_regressor_constr,
    add_standard_scaler_constr,
    add_pipeline_constr
)
import gurobipy_pandas as gppd
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager

# Visualization
import matplotlib.pyplot as plt

# Custom modules
from reader import generate_tab_files
from first_stage_empire import run_first_stage
from NEUREMPIRE import run_empire
from scenario_random import generate_random_scenario
from Embed_Model_validation import empire_validation
from model_training import (
    load_data,
    preprocessing_data,
    ML_training,
    save_models,
    load_models
)


def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data




##############################
######### REAL ###############
##############################

def ML_embedding(instance, solver, gurobi_model, regression_model, seed, feature_names, n_v_features, n_xi_features):
    num_periods = len(instance.PeriodActive)
    logging.info(f"Number of periods: {num_periods}")

    # Load the CSV file containing 'i' and 'xi_i' columns
    xi_values_df = pd.read_csv('results/training_scenario.csv')

    # Variable to hold the total second-stage costs
    total_second_stage = gurobi_model.addVar(lb=0, name=f'total_second_stage_{seed}')
    total_second_stage_terms = {}
    xi_values_dict = {}
    ml_feats = {}
    regression_model = joblib.load(f'PCA_Scaler_Model/empire_model.onnx')
    

    for period in range(1, num_periods + 1):
        logging.info(f"Processing Period {period}")

        # Load the period-specific PCA, scaler, and regression model
        scaler_X = joblib.load(f'PCA_Scaler_Model/scaler_X_period_{period}.joblib')
        scaler_y = joblib.load(f'PCA_Scaler_Model/scaler_y_period_{period}.joblib')
        pca_v = joblib.load(f'PCA_Scaler_Model/PCA_model_v_i_period_{period}.pkl')
        pca_x = joblib.load(f'PCA_Scaler_Model/PCA_model_x_i_period_{period}.pkl')
        pca_xi = joblib.load(f'PCA_Scaler_Model/PCA_model_xi_i_period_{period}.pkl')
        

        # Extract PCA parameters for `v` and `x`
        mean_input = scaler_X.mean_
        scale_input = scaler_X.scale_
        mean_output = scaler_y.mean_
        scale_output = scaler_y.scale_
        components_v = pca_v.components_
        components_x = pca_x.components_
        components_xi = pca_xi.components_
        
        # Extract xi_values for the current period
        xi_vals = xi_values_df[
            (xi_values_df['seed'] == seed) &
            (xi_values_df['period'] == period)
        ]['xi_i']

        if xi_vals.empty:
            raise ValueError(f"No xi values found for seed {seed} and period {period}")

        xi_vector = xi_vals.apply(ast.literal_eval).sample(n=1).values[0]
        xi_vector = xi_vector[:n_xi_features]
        xi_values_dict[period] = xi_vector

        # Map Pyomo variables to Gurobi variables
        indices_v, pyomo_var_to_gurobi_var_v = all_var_mapping(instance, solver, period, 'v')
        v_vars = [pyomo_var_to_gurobi_var_v[name] for name in indices_v]
        indices_x, pyomo_var_to_gurobi_var_x = all_var_mapping(instance, solver, period, 'x')
        x_vars = [pyomo_var_to_gurobi_var_x[name] for name in indices_x]

        # Compute PCA-reduced variables using Gurobi expressions
        # For v_vars_reduced
        n_components_v = components_v.shape[0]
        v_vars_reduced = []
        for j in range(n_components_v):
            expr = grb.LinExpr()
            for i in range(len(v_vars)):
                coeff = components_v[j, i]
                expr.addTerms(coeff, v_vars[i])
            v_vars_reduced.append(expr)

        n_components_x = components_x.shape[0]
        x_vars_reduced = []
        for j in range(n_components_x):
            expr = grb.LinExpr()
            for i in range(len(x_vars)):
                coeff = components_x[j, i]
                expr.addTerms(coeff, x_vars[i])
            x_vars_reduced.append(expr)

        # Combine reduced variables
        input_vars = v_vars_reduced + x_vars_reduced

        # Apply scaling and shifting
        scaled_input_vars = []
        for i, var in enumerate(input_vars):
            scaled_var = var * scale_input[i] + mean_input[i]
            scaled_input_vars.append(scaled_var)
        # Append xi_vector (constants) to the input variables

        final_input_vars = scaled_input_vars + xi_vector.tolist()
        ml_feats[period] = (final_input_vars)

        # Approximation variable for the second-stage cost
        y_approx = gurobi_model.addVar(lb=0, name=f'y_approx_{seed}_{period}')
        gurobi_model.update()

        # Scale back y_approx using the saved scaler
        scaled_y_approx = y_approx * scale_output + mean_output

        # Accumulate total second-stage cost
        total_second_stage_terms[period] = (scaled_y_approx)

    
    feats_dict = {
    'period': list(range(1, num_periods + 1)),  
    'input': [],  
    'y_approx': [] 
    }

    for period in range(1, num_periods + 1):
        feats_dict['input'].append(ml_feats[period])  
        feats_dict['y_approx'].append(total_second_stage_terms[period])  


    feats = pd.DataFrame(feats_dict)
    feats.set_index('period', inplace=True)  

    print(feats)
    feats_for_ml = feats.drop('y_approx', axis=1)
    

    pred_constr = add_predictor_constr(gurobi_model, regression_model, feats_for_ml, feats[['y_approx']])
    y_approx_lst = feats['y_approx'].tolist()
    # After looping over periods, add total_second_stage constraint
    gurobi_model.addConstr(
        total_second_stage == grb.quicksum(y_approx_lst)
    )

    # Adjust the objective function
    existing_obj = gurobi_model.getObjective()
    alpha = 1
    # Set the combined objective function
    combined_obj = existing_obj + alpha * total_second_stage
    gurobi_model.setObjective(combined_obj, grb.GRB.MINIMIZE)

    gurobi_model.update()

    return gurobi_model

##############################
######### REAL ###############
##############################

















# def ML_embedding(instance, solver, gurobi_model, regression_model, seed, scaler_y,feature_names, n_v_features, n_xi_features):
#     num_periods = len(instance.PeriodActive)
#     logging.info(f"Number of periods: {num_periods}")

#     # Load the CSV file containing 'i' and 'xi_i' columns
#     xi_values_df = pd.read_csv('results/training_scenario.csv')
#     # Load saved PCA model
#     scaler = joblib.load('PCA_Scaler_Model/v_i_results_scaler.pkl')
#     pca = joblib.load('PCA_Scaler_Model/v_i_results_pca.pkl')

#     mean = scaler.mean_  
#     scale = scaler.scale_
#     components = pca.components_  

#     v_vars = {}
#     xi_values = {}
#     y_approx = {} 
#     v_vars_reduced = {}

#     for period in range(1, num_periods + 1):
#         xi_vals = xi_values_df[(xi_values_df['seed'] == seed) & 
#                                (xi_values_df['period'] == period)]['xi_i']

#         xi_vector = xi_vals.apply(ast.literal_eval).sample(n=1).values[0]
#         xi_vector = xi_vector[:n_xi_features]
#         xi_values[period] = xi_vector

#         indices, pyomo_var_to_gurobi_var_v = all_var_mapping(instance, solver, period)
#         v_vars[period] = [pyomo_var_to_gurobi_var_v[name] for name in indices]
#         # Create PCA transformed variables
#         v_vars_reduced[period] = []
#         for i in range(len(components)):
#             pca_var = gurobi_model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, 
#                                         name=f'v_pca_{period}_{i}')
#             v_vars_reduced[period].append(pca_var)

#         for i in range(len(components)):
#             gurobi_model.addConstr(
#                 grb.quicksum((v_vars[period][j] - mean[j]) / scale[j] * components[i][j] 
#                             for j in range(len(v_vars[period]))) == v_vars_reduced[period][i],
#                 name=f'pca_transform_{period}_{i}'
#             )

#         y_approx[period] = gurobi_model.addVar(lb=0, name=f'y_approx_{seed}_{period}')
    
#     gurobi_model.update() 

#     feats_dict = {
#         'period': list(range(1, num_periods + 1))  
#     }

#     for i in range(n_v_features):
#         feats_dict[f'v_{i}'] = []
#     for i in range(n_xi_features):
#         feats_dict[f'xi_{i}'] = []
    

#     feats_dict['y_approx'] = []

#     for period in range(1, num_periods + 1):
#         for i in range(n_v_features):
#             var = v_vars_reduced[period][i] if i < len(v_vars_reduced[period]) else 0
#             feats_dict[f'v_{i}'].append(var)
            
#         for i in range(n_xi_features):
#             xi_val = xi_values[period][i] if i < len(xi_values[period]) else 0
#             feats_dict[f'xi_{i}'].append(xi_val)
        

#         feats_dict['y_approx'].append(y_approx[period])

#     feats = pd.DataFrame(feats_dict)
#     feats.set_index('period', inplace=True)

#     print(feats)
#     feats_for_ml = feats.drop('y_approx', axis=1)
    
#     pred_constr = add_predictor_constr(gurobi_model, regression_model, feats_for_ml, feats[['y_approx']])
#     print(pred_constr.print_stats())

#     existing_obj = gurobi_model.getObjective()
#     alpha = 1
    
#     y_approx_lst = feats['y_approx'].tolist()
#     total_second_stage = gurobi_model.addVar(lb=0, name=f'total_second_stage_{seed}')
#     gurobi_model.addConstr(total_second_stage == grb.quicksum([y* scaler_y.scale_[0] + scaler_y.mean_[0] for y in y_approx_lst]))
#     combined_obj = existing_obj + alpha * (total_second_stage)
#     gurobi_model.setObjective(combined_obj, grb.GRB.MINIMIZE)
        
#     gurobi_model.update()
    
#     return gurobi_model, pred_constr, xi_values





# work by each period
# def ML_embedding(instance, solver, gurobi_model, regression_model, seed, scaler_y, feature_names, n_v_features, n_xi_features):
#     num_periods = len(instance.PeriodActive)
#     logging.info(f"Number of periods: {num_periods}")

#     # Load the CSV file containing 'i' and 'xi_i' columns
#     xi_values_df = pd.read_csv('results/training_scenario.csv')
#     # Load saved PCA model
#     scaler = joblib.load('v_i_results_scaler.pkl')
#     pca = joblib.load('v_i_results_pca.pkl')

#     mean = scaler.mean_  
#     scale = scaler.scale_
#     components = pca.components_

#     total_second_stage = gurobi_model.addVar(lb=0, name=f'total_second_stage_{seed}')
#     total_second_stage_terms = []

#     pred_constr_list = []
#     xi_values_dict = {}

#     for period in range(1, num_periods + 1):
#         # Load xi_values for this period
#         xi_vals = xi_values_df[(xi_values_df['seed'] == seed) & 
#                                (xi_values_df['period'] == period)]['xi_i']

#         xi_vector = xi_vals.apply(ast.literal_eval).sample(n=1).values[0]
#         xi_vector = xi_vector[:n_xi_features]
#         xi_values = xi_vector
#         xi_values_dict[period] = xi_values

#         # Map variables
#         indices, pyomo_var_to_gurobi_var_v = all_var_mapping(instance, solver, period)
#         v_vars = [pyomo_var_to_gurobi_var_v[name] for name in indices]

#         # Create PCA transformed variables
#         v_vars_reduced = []
#         for i in range(len(components)):
#             pca_var = gurobi_model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, 
#                                           name=f'v_pca_{period}_{i}')
#             v_vars_reduced.append(pca_var)

#         # Add PCA transformation constraints
#         for i in range(len(components)):
#             gurobi_model.addConstr(
#                 grb.quicksum((v_vars[j] - mean[j]) / scale[j] * components[i][j] 
#                              for j in range(len(v_vars))) == v_vars_reduced[i],
#                 name=f'pca_transform_{period}_{i}'
#             )

#         y_approx = gurobi_model.addVar(lb=0, name=f'y_approx_{seed}_{period}')

#         gurobi_model.update()

#         # Prepare features for ML
#         feats_dict = {}
#         for i in range(n_v_features):
#             var = v_vars_reduced[i] if i < len(v_vars_reduced) else 0
#             feats_dict[f'v_{i}'] = var
#         for i in range(n_xi_features):
#             xi_val = xi_values[i] if i < len(xi_values) else 0
#             feats_dict[f'xi_{i}'] = xi_val

#         feats = pd.DataFrame([feats_dict])

#         # Add predictor constraints
#         pred_constr = add_predictor_constr(gurobi_model, regression_model, feats, pd.DataFrame({'y_approx': [y_approx]}))
#         pred_constr_list.append(pred_constr)

#         # Adjust the objective function
#         existing_obj = gurobi_model.getObjective()
#         alpha = 1

#         # Scale back y_approx
#         scaled_y_approx = y_approx * scaler_y.scale_[0] + scaler_y.mean_[0]

#         # Accumulate total second stage cost
#         total_second_stage_terms.append(scaled_y_approx)

#     # After looping over periods, add total_second_stage constraint
#     gurobi_model.addConstr(total_second_stage == grb.quicksum(total_second_stage_terms))
#     # Set the combined objective function
#     combined_obj = existing_obj + alpha * total_second_stage
#     gurobi_model.setObjective(combined_obj, grb.GRB.MINIMIZE)

#     gurobi_model.update()
#     print(f"input values for ML: {pred_constr.input_values}")
#     return gurobi_model, pred_constr_list, xi_values_dict




def selected_var_mapping(instance, solver, i):
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
        if (n,g) in desired_data['Generation']:
            var = instance.genInstalledCap[n,g,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[instance.genInstalledCap[n,g,i]]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 3. Storage Power installed capacities
    for (n,b) in instance.StoragesOfNode:
        if (n,b) in desired_data['Storage Power']:
            var = instance.storPWInstalledCap[n,b,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 4. Storage Energy installed capacities
    for (n,b) in instance.StoragesOfNode:
        if (n,b) in desired_data['Storage Energy']:
            var = instance.storENInstalledCap[n,b,i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    # 2. Transmission installed capacities
    for (n1,n2) in instance.BidirectionalArc:
        var = instance.transmissionInstalledCap[n1,n2,i]
        gurobi_var = solver._pyomo_var_to_solver_var_map[var]
        pyomo_var_to_gurobi_var[var.name] = gurobi_var

    indices = pyomo_var_to_gurobi_var.keys()
    
    return indices, pyomo_var_to_gurobi_var


def all_var_mapping(instance, solver, period, var_type):
    """
    Maps Pyomo variables to Gurobi variables for a specific period and variable type ('v' or 'x'),
    applying the same filtering as in the PCA preprocessing.
    """
    pyomo_var_to_gurobi_var = {}
    
    if var_type == 'v':
        # 1. Generator installed capacities (excluding 'existing' and 'CCS')
        for (n, g) in instance.GeneratorsOfNode:
            if 'existing' in g or 'CCS' in g:
                continue  # Skip excluded generators
            var = instance.genInstalledCap[n, g, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 2. Transmission installed capacities
        for (n1, n2) in instance.BidirectionalArc:
            var = instance.transmissionInstalledCap[n1, n2, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 3. Storage Power installed capacities
        for (n, b) in instance.StoragesOfNode:
            var = instance.storPWInstalledCap[n, b, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 4. Storage Energy installed capacities
        for (n, b) in instance.StoragesOfNode:
            var = instance.storENInstalledCap[n, b, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    elif var_type == 'x':
        # 1. Generator investment capacities (excluding 'existing' and 'CCS')
        for (n, g) in instance.GeneratorsOfNode:
            if 'existing' in g or 'CCS' in g:
                continue  # Skip excluded generators
            var = instance.genInvCap[n, g, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 2. Transmission investment capacities
        for (n1, n2) in instance.BidirectionalArc:
            var = instance.transmissionInvCap[n1, n2, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 3. Storage Power investment capacities
        for (n, b) in instance.StoragesOfNode:
            var = instance.storPWInvCap[n, b, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 4. Storage Energy investment capacities
        for (n, b) in instance.StoragesOfNode:
            var = instance.storENInvCap[n, b, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var
    else:
        raise ValueError("Invalid var_type. Expected 'v' or 'x'.")

    indices = list(pyomo_var_to_gurobi_var.keys())
    
    return indices, pyomo_var_to_gurobi_var



def main(SEED):
    
    UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

    USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
    temp_dir = UserRunTimeConfig["temp_dir"]
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
    lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
    discountrate = UserRunTimeConfig["discountrate"]
    WACC = UserRunTimeConfig["WACC"]
    solver = UserRunTimeConfig["solver"]
    scenariogeneration = UserRunTimeConfig["scenariogeneration"]
    fix_sample = UserRunTimeConfig["fix_sample"]
    LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
    filter_make = UserRunTimeConfig["filter_make"] 
    filter_use = UserRunTimeConfig["filter_use"]
    n_cluster = UserRunTimeConfig["n_cluster"]
    moment_matching = UserRunTimeConfig["moment_matching"]
    n_tree_compare = UserRunTimeConfig["n_tree_compare"]
    EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]
    IAMC_PRINT = UserRunTimeConfig["IAMC_PRINT"]
    WRITE_LP = UserRunTimeConfig["WRITE_LP"]
    PICKLE_INSTANCE = UserRunTimeConfig["PICKLE_INSTANCE"] 


    #############################
    ##Non configurable settings##
    #############################

    NoOfRegSeason = 4
    regular_seasons = ["winter", "spring", "summer", "fall"]
    NoOfPeakSeason = 2
    lengthPeakSeason = 7
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    if version in ["europe_v50"]:
        north_sea = False
    elif version in ["reduced"]:
        north_sea = False
    else:
        north_sea = True


    SEED_range = [SEED+i for i in range(NoOfScenarios)] 
    print(f'Seed Sets {SEED_range}') 
    #######
    ##RUN##
    #######


    name = version + '_reg' + str(lengthRegSeason) + \
        '_peak' + str(lengthPeakSeason) + \
        '_sce' + str(NoOfScenarios)
    if scenariogeneration and not fix_sample:
            name = name + "_randomSGR"
    else:
        name = name + "_noSGR"
    if filter_use:
        name = name + "_filter" + str(n_cluster)
    if moment_matching:
        name = name + "_moment" + str(n_tree_compare)
    name = name + str(datetime.now().strftime("_%Y%m%d%H%M"))
    workbook_path = 'Data handler/' + version
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name + f'_{SEED}'
    scenario_data_path = 'Data handler/' + version + '/ScenarioData'
    result_file_path = 'Results/' + name
    FirstHoursOfRegSeason = [lengthRegSeason*i + 1 for i in range(NoOfRegSeason)]
    FirstHoursOfPeakSeason = [lengthRegSeason*NoOfRegSeason + lengthPeakSeason*i + 1 for i in range(NoOfPeakSeason)]
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
    Scenario = ["scenario"+str(i + 1) for i in range(NoOfScenarios)]
    peak_seasons = ['peak'+str(i + 1) for i in range(NoOfPeakSeason)]
    Season = regular_seasons + peak_seasons
    Operationalhour = [i + 1 for i in range(FirstHoursOfPeakSeason[-1] + lengthPeakSeason - 1)]
    HoursOfRegSeason = [(s,h) for s in regular_seasons for h in Operationalhour \
                    if h in list(range(regular_seasons.index(s)*lengthRegSeason+1,
                                regular_seasons.index(s)*lengthRegSeason+lengthRegSeason+1))]
    HoursOfPeakSeason = [(s,h) for s in peak_seasons for h in Operationalhour \
                        if h in list(range(lengthRegSeason*len(regular_seasons)+ \
                                            peak_seasons.index(s)*lengthPeakSeason+1,
                                            lengthRegSeason*len(regular_seasons)+ \
                                                peak_seasons.index(s)*lengthPeakSeason+ \
                                                    lengthPeakSeason+1))]
    HoursOfSeason = HoursOfRegSeason + HoursOfPeakSeason
    dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}
        

    if scenariogeneration:
        generate_random_scenario(filepath = scenario_data_path,
                                tab_file_path = tab_file_path,
                                scenarios = NoOfScenarios,
                                seasons = regular_seasons,
                                Periods = len(Period),
                                regularSeasonHours = lengthRegSeason,
                                peakSeasonHours = lengthPeakSeason,
                                dict_countries = dict_countries,
                                time_format = time_format,
                                filter_make = filter_make,
                                filter_use = filter_use,
                                n_cluster = n_cluster,
                                moment_matching = moment_matching,
                                n_tree_compare = n_tree_compare,
                                fix_sample = fix_sample,
                                north_sea = False,
                                LOADCHANGEMODULE = LOADCHANGEMODULE,
                                seed = SEED_range)

    generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)


    objective_value, expected_second_stage_value = run_empire(name = name, 
            tab_file_path = tab_file_path,
            result_file_path = result_file_path, 
            scenariogeneration = scenariogeneration,
            scenario_data_path = scenario_data_path,
            solver = solver,
            temp_dir = temp_dir, 
            FirstHoursOfRegSeason = FirstHoursOfRegSeason, 
            FirstHoursOfPeakSeason = FirstHoursOfPeakSeason, 
            lengthRegSeason = lengthRegSeason,
            lengthPeakSeason = lengthPeakSeason,
            Period = Period, 
            Operationalhour = Operationalhour,
            Scenario = Scenario,
            Season = Season,
            HoursOfSeason = HoursOfSeason,
            discountrate = discountrate, 
            WACC = WACC, 
            LeapYearsInvestment = LeapYearsInvestment,
            IAMC_PRINT = IAMC_PRINT, 
            WRITE_LP = WRITE_LP, 
            PICKLE_INSTANCE = PICKLE_INSTANCE, 
            EMISSION_CAP = EMISSION_CAP,
            USE_TEMP_DIR = USE_TEMP_DIR,
            LOADCHANGEMODULE = LOADCHANGEMODULE,
            seed = SEED)
    print("Objective Value :", objective_value)
    print("Expected Second Stage Value :", expected_second_stage_value)

    model,data = run_first_stage(name = name, 
            tab_file_path = tab_file_path,
            result_file_path = result_file_path, 
            scenariogeneration = scenariogeneration,
            scenario_data_path = scenario_data_path,
            solver = solver,
            temp_dir = temp_dir, 
            FirstHoursOfRegSeason = FirstHoursOfRegSeason, 
            FirstHoursOfPeakSeason = FirstHoursOfPeakSeason, 
            lengthRegSeason = lengthRegSeason,
            lengthPeakSeason = lengthPeakSeason,
            Period = Period, 
            Operationalhour = Operationalhour,
            Scenario = Scenario,
            Season = Season,
            HoursOfSeason = HoursOfSeason,
            discountrate = discountrate, 
            WACC = WACC, 
            LeapYearsInvestment = LeapYearsInvestment,
            IAMC_PRINT = IAMC_PRINT, 
            WRITE_LP = WRITE_LP, 
            PICKLE_INSTANCE = PICKLE_INSTANCE, 
            EMISSION_CAP = EMISSION_CAP,
            USE_TEMP_DIR = USE_TEMP_DIR,
            LOADCHANGEMODULE = LOADCHANGEMODULE)
    
    instance = model.create_instance(data)


    # Add at the beginning of your main function
    threshold = 5  # Your specified threshold
    ratio_sol = 100  # Initialize to a value greater than the threshold
    max_iterations = 10  # To prevent infinite loops
    iteration = 0

    file_path = 'results/training_data3.csv'    
    df = load_data(file_path)
    X_train, y_train, X_test, y_test, scaler_y = preprocessing_data(df)
    # Extract feature names
    feature_names = X_train.columns.tolist()
    n_v_features = len([col for col in feature_names if col.startswith('v_')])
    n_xi_features = len([col for col in feature_names if col.startswith('xi_')])


    while ratio_sol > threshold and iteration < max_iterations:
        # Preprocess data

        # Train ML model
        print(f'\nTraining Model - Iteration {iteration+1}')
        trained_lr, trained_dt, trained_rf, trained_gb = ML_training(
            X_train, y_train, X_test, y_test
        )

        # Solve optimization problem with embedded ML model
        solver = SolverFactory('gurobi_persistent')
        solver.set_instance(instance) 
        gurobi_model = solver._solver_model
        gurobi_model.update()
        
        # Modify ML_embedding to return xi_values
        embedded_model = ML_embedding_by_period(
            instance, solver, gurobi_model, trained_gb, SEED, feature_names, n_v_features, n_xi_features
        )

        # Set Gurobi parameters
        embedded_model.Params.NonConvex = 2
        embedded_model.Params.TimeLimit = 100

        # Optimize the Gurobi model
        embedded_model.optimize()

        if embedded_model.Status == GRB.OPTIMAL:
            # Compute ratio_sol
            objective_value_embed = embedded_model.objVal
            ratio_sol_simple = abs(objective_value - objective_value_embed) / objective_value * 100
            print("Gap (%) :", ratio_sol_simple)

            # Save the solution to csv
            results_df = save_results_to_csv(embedded_model, solver, f"MLsols/ML_Embed_solution_{SEED_range}.csv")

            # Run validation to get actual second-stage cost
            fsd_file_path = f"MLsols/ML_Embed_solution_{SEED_range}.csv"
            FSD = read_fsd_from_csv(fsd_file_path)

            objective_value_embedding_sol, expected_second_stage_value_embedding_sol, v_i, Q_i = empire_validation(
                    name = name, 
                    tab_file_path = tab_file_path,
                    result_file_path = result_file_path, 
                    scenariogeneration = scenariogeneration,
                    scenario_data_path = scenario_data_path,
                    solver = "Gurobi",
                    temp_dir = temp_dir, 
                    FirstHoursOfRegSeason = FirstHoursOfRegSeason, 
                    FirstHoursOfPeakSeason = FirstHoursOfPeakSeason, 
                    lengthRegSeason = lengthRegSeason,
                    lengthPeakSeason = lengthPeakSeason,
                    Period = Period, 
                    Operationalhour = Operationalhour,
                    Scenario = Scenario,
                    Season = Season,
                    HoursOfSeason = HoursOfSeason,
                    discountrate = discountrate, 
                    WACC = WACC, 
                    LeapYearsInvestment = LeapYearsInvestment,
                    IAMC_PRINT = IAMC_PRINT, 
                    FSD = FSD,
                    WRITE_LP = WRITE_LP, 
                    PICKLE_INSTANCE = PICKLE_INSTANCE, 
                    EMISSION_CAP = EMISSION_CAP,
                    USE_TEMP_DIR = USE_TEMP_DIR,
                    LOADCHANGEMODULE = LOADCHANGEMODULE,
                    seed = SEED)

            print("objective_value_embedding_sol: ", objective_value_embedding_sol)
            print("expected_second_stage_value_embedding_sol: ", expected_second_stage_value_embedding_sol)
            print("ratio_sol (%):", ratio_sol)

            ratio_sol = abs(objective_value - objective_value_embedding_sol) / objective_value * 100
            print("Gap (%) :", ratio_sol)

            if ratio_sol > threshold:
                # Extract features and target from the solution
                new_data_rows = []
                num_periods = len(instance.PeriodActive)
                # n_xi_features = len([col for col in trained_dt.feature_names_in_ if col.startswith('xi_')])

                new_X = []
                new_y = []

                scaler = joblib.load('v_i_results_scaler.pkl')
                pca = joblib.load('v_i_results_pca.pkl')
                
                for period in range(1, num_periods + 1):
                    # Extract v_i and xi_i
                    v_i_period = v_i.get(period, {})
                    # selected_v_values = extract_selected_variables(v_i_period)
                    selected_v_values = []
        
                    # Process genInstalledCap
                    if 'genInstalledCap' in v_i_period:
                        gen_data = v_i_period['genInstalledCap']
                        sorted_keys = sorted(gen_data.keys())  # Sort keys for consistent ordering
                        selected_v_values.extend([gen_data[k] for k in sorted_keys])
                        
                    # Process transmissionInstalledCap
                    if 'transmissionInstalledCap' in v_i_period:
                        trans_data = v_i_period['transmissionInstalledCap']
                        sorted_keys = sorted(trans_data.keys())
                        selected_v_values.extend([trans_data[k] for k in sorted_keys])
                        
                    # Process storPWInstalledCap
                    if 'storPWInstalledCap' in v_i_period:
                        stor_pw_data = v_i_period['storPWInstalledCap']
                        sorted_keys = sorted(stor_pw_data.keys())
                        selected_v_values.extend([stor_pw_data[k] for k in sorted_keys])
                        
                    # Process storENInstalledCap
                    if 'storENInstalledCap' in v_i_period:
                        stor_en_data = v_i_period['storENInstalledCap']
                        sorted_keys = sorted(stor_en_data.keys())
                        selected_v_values.extend([stor_en_data[k] for k in sorted_keys])
                    
                    # Convert to numpy array and reshape
                    v_vars_matrix = np.array([selected_v_values])
                    
                    # Apply transformations
                    v_vars_normalized = scaler.transform(v_vars_matrix)
                    new_v_reduced = pca.transform(v_vars_normalized)[0]

                    xi_vals = xi_values.get(period, [])
                    # feature_vector = selected_v_values + xi_vals
                    feature_vector = list(new_v_reduced) + list(xi_vals)
                    new_X.append(feature_vector)
                    y_i = Q_i[period][0]
                    new_y.append(y_i)

                new_X = np.array(new_X)
                new_y = scaler_y.fit_transform(np.array(new_y).reshape(-1, 1)).flatten()
                X_train = np.vstack([X_train, new_X])
                y_train = np.concatenate([y_train, new_y])

                # Increase iteration count
                iteration += 1
            else:
                break
        else:
            print("Optimization did not converge to optimality.")
            break


        
    return None 



# Mapping Pyomo variables to Gurobi variables
def print_pyomo_to_gurobi_mapping(solver,gurobi_model):
    print("\nMapping of Pyomo Variables to Gurobi Variables:")
    pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map  # Existing mapping

    for pyomo_var, gurobi_var in pyomo_var_to_gurobi_var.items():
        print(f"Pyomo Variable: {pyomo_var.name}, Index: {pyomo_var.index()}, Corresponding Gurobi Variable: {gurobi_var.VarName}, Value: {gurobi_var.x}")

    for v in gurobi_model.getVars():
            if v.VarName.startswith('y'):
                print(v.VarName, "=", v.x)

def save_results_to_csv(gurobi_model, solver, output_filename):
    output_path = Path(output_filename)
    output_dir = output_path.parent
    
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {output_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return None
    # Create empty lists to store the data
    results_data = []
    
    # Get the mapping between Pyomo and Gurobi variables
    pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map
    
    # Iterate through the mapping to extract results
    for pyomo_var, gurobi_var in pyomo_var_to_gurobi_var.items():
        # Get the index from Pyomo variable
        index = pyomo_var.index()
        value = gurobi_var.x
        
        # Determine the type and create appropriate entry based on variable name
        var_name = pyomo_var.name
        
        if 'genInvCap' in var_name:
            # Generation type
            node, energy_type, period = index
            entry_type = 'Generation'
            results_data.append({
                'Node': node,
                'Energy_Type': energy_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
            
        elif 'storPWInvCap' in var_name:
            # Storage Power type
            node, storage_type, period = index
            entry_type = 'Storage Power'
            results_data.append({
                'Node': node,
                'Energy_Type': storage_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
            
        elif 'storENInvCap' in var_name:
            # Storage Energy type
            node, storage_type, period = index
            entry_type = 'Storage Energy'
            results_data.append({
                'Node': node,
                'Energy_Type': storage_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
            
        elif 'transmisionInvCap' in var_name:
            # Transmission type
            node_from, node_to, period = index
            entry_type = 'Transmission'
            results_data.append({
                'Node': node_from,
                'Energy_Type': node_to,  # Using Energy_Type column for the destination node
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    # Sort the DataFrame to match input format
    df = df.sort_values(['Node', 'Energy_Type', 'Period', 'Type'])
    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"Results have been saved to {output_filename}")
    # Return DataFrame for potential further analysis
    return df



def embedding_solution_parsing():
    # Configuration for desired data
    desired_data_v = {
        'Generation': {
            'Germany': ['Solar', 'GasCCGT', 'Bio', 'Bio10cofiring'],
            'France': ['Windonshore', 'Solar', 'GasCCGT', 'Bio'],
            'Denmark': ['Solar', 'GasCCGT', 'Windonshore']
        },
        'Storage Power': {
            'Germany': ['Li-Ion_BESS'],
            'France': ['Li-Ion_BESS'],
            'Denmark': ['Li-Ion_BESS']
        },
        'Storage Energy': {
            'Germany': ['Li-Ion_BESS'],
            'France': ['Li-Ion_BESS'],
            'Denmark': ['Li-Ion_BESS']
        }
    }

    filtered_data = []
    v_i_data = data.get('v_i', {})
    
    for category, country_tech in desired_data_v.items():
        for country, technologies in country_tech.items():
            for tech in technologies:
                key = (country, tech)
                if category == 'Generation':
                    value = v_i_data.get('genInstalledCap', {}).get(str(key), None)
                elif category == 'Storage Power':
                    value = v_i_data.get('storPWInstalledCap', {}).get(str(key), None)
                elif category == 'Storage Energy':
                    value = v_i_data.get('storENInstalledCap', {}).get(str(key), None)
                
                if value is not None:
                    filtered_data.append(value)

    transmission_data = v_i_data.get('transmissionInstalledCap', {})
    for value in transmission_data.values():
        if value is not None:
            filtered_data.append(value)
    

    return filtered_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Specific seed')
    args = parser.parse_args()
    specific_seed = args.seed
    ratio = main(specific_seed)