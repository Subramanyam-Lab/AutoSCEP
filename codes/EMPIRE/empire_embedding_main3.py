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
import onnx
import onnxruntime as ort
from onnx2torch import convert
import random
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
# from model_training import (
#     load_data,
#     preprocessing_data,
#     ML_training,
#     save_models,
#     load_models
# )


def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data


# Load the ONNX model and convert it back to PyTorch
def load_and_convert_model(model_path):
    onnx_model = onnx.load(f'{model_path}')
    pytorch_model = convert(onnx_model).float()
    print("ONNX model converted back to PyTorch model.")
    # Extract layers and create a sequential model
    pytorch_model.eval()
    layers = []
    for name, layer in pytorch_model.named_children():
      layers.append(layer)
    trained_load_model = nn.Sequential(*layers)

    return trained_load_model


##############################
######### REAL ###############
##############################


def ML_embedding(instance, solver, gurobi_model, regression_model_E_Q, regression_model_LL_AMT):

    # Load scalers
    scaler_v = joblib.load(f'scaler_pca/scaler2.joblib')
    mean_v_input = scaler_v.mean_
    scale_v_input = scaler_v.scale_

    # Map Pyomo variables to Gurobi variables
    indices_v, pyomo_var_to_gurobi_var_v = v_var_mapping(instance, solver)
    v_vars = [pyomo_var_to_gurobi_var_v[name] for name in indices_v]

    # Scale v variables
    scaled_v_vars = []
    for i, var in enumerate(v_vars):
        scaled_v_var = gurobi_model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                                        name=f'scaled_v_{i}')
        gurobi_model.addConstr(
            scaled_v_var == (var - float(mean_v_input[i])) / float(scale_v_input[i]),
            name=f'scaled_v_constr'
        )
        scaled_v_vars.append(scaled_v_var)
    
    # # Load period-specific PCA models
    # pca_v = joblib.load(f'scaler_pca/pca2.pkl')
    # components_v = pca_v.components_

    # n_components_v = components_v.shape[0]
    # pca_v_vars = []
    # for j in range(n_components_v):
    #     pca_v_var = gurobi_model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
    #                                     name=f'pca_v_component')
    #     gurobi_model.addConstr(
    #         pca_v_var == grb.quicksum(components_v[j, i] * scaled_v_vars[i] for i in range(len(scaled_v_vars))),
    #         name=f'pca_v_constr'
    #     )
    #     pca_v_vars.append(pca_v_var)

    # Combine all features: PCA-reduced x, PCA-reduced v, and scaled xi
    final_input_vars = scaled_v_vars
    gurobi_model.update()

    # Create a variable for y_approx
    # y_approx = gurobi_model.addVar(lb=-GRB.INFINITY, name=f'y_approx')
    y_approx_E_Q = gurobi_model.addVar(lb=-GRB.INFINITY, name=f'y_approx_E_Q')
    y_approx_LL_AMT = gurobi_model.addVar(lb=-GRB.INFINITY, name=f'y_approx_LL_AMT')

    pred_constr_E_Q = add_predictor_constr(gurobi_model, regression_model_E_Q, scaled_v_vars, y_approx_E_Q)
    gurobi_model.update()

    pred_constr_LL_AMT = add_predictor_constr(gurobi_model, regression_model_LL_AMT, scaled_v_vars, y_approx_LL_AMT)
    gurobi_model.update()
    
    # Load global scalers
    scaler_y_E_Q = joblib.load('scaler_pca/scaler_E_Q_ELSE.joblib')
    
    mean_output = scaler_y_E_Q.mean_
    scale_output = scaler_y_E_Q.scale_
    scaled_y_E_Q = y_approx_E_Q * scale_output + mean_output

    # min_output = scaler_y_E_Q.data_min_
    # max_output = scaler_y_E_Q.data_max_
    # scaled_y_E_Q = y_approx_E_Q * (max_output - min_output) + min_output

    scaler_y_LL_AMT = joblib.load('scaler_pca/scaler_LL_AMT.joblib')
    
    mean_output = scaler_y_LL_AMT.mean_
    scale_output = scaler_y_LL_AMT.scale_
    scaled_y_LL_AMT = y_approx_LL_AMT * scale_output + mean_output

    # min_output = scaler_y_LL_AMT.data_min_
    # max_output = scaler_y_LL_AMT.data_max_
    # scaled_y_LL_AMT = y_approx_LL_AMT * (max_output - min_output) + min_output


    VOLL = 22000 
    
    # min_output = scaler_y.data_min_
    # max_output = scaler_y.data_max_
    # print(f"min_output : {min_output}, max_output : {max_output}")

    # scaled_y_approx = y_approx * (max_output - min_output) + min_output

    gurobi_model.update()

    # Update the objective
    existing_obj = gurobi_model.getObjective()
    combined_obj = existing_obj + scaled_y_E_Q + VOLL*scaled_y_LL_AMT
    gurobi_model.setObjective(combined_obj, grb.GRB.MINIMIZE)
    gurobi_model.update()

    return gurobi_model,pred_constr_E_Q, pred_constr_LL_AMT



def v_var_mapping(instance, solver):

    pyomo_var_to_gurobi_var = {}
    
    for period in instance.PeriodActive:
        # 1. Generator installed capacities (excluding 'existing' and 'CCS')
        for (n, g) in instance.GeneratorsOfNode:
            # if 'existing' in g or 'CCS' in g:
            #     continue  # Skip excluded generators
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
    lengthPeakSeason = 24
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    if version in ["europe_v50"]:
        north_sea = False
    elif version in ["reduced"]:
        north_sea = False
    else:
        north_sea = True


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
        

    start_time = time.time()
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
                                north_sea = north_sea,
                                LOADCHANGEMODULE = LOADCHANGEMODULE,
                                seed = SEED)

    generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)


    objective_value, expected_second_stage_value, second_stage_variance = run_empire(name = name, 
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
            seed = SEED,
            north_sea = north_sea)
    end_time = time.time()
    print("Objective Value :", objective_value)
    print("Expected Second Stage Value :", expected_second_stage_value)
    print("Second Stage Variance : ", second_stage_variance)
    print("Total Solving Time :", end_time - start_time)

    
    start_time = time.time()

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
            LOADCHANGEMODULE = LOADCHANGEMODULE,
            north_sea = north_sea)
    
    instance = model.create_instance(data)

    # model load 
    model_path = "scaler_pca/pytorch_regression_E_Q_ELSE.onnx"
    trained_model_E_Q = load_and_convert_model(model_path)

    model_path = "scaler_pca/pytorch_regression_LL_AMT.onnx"
    trained_model_LL_AMT = load_and_convert_model(model_path)

    # pyomo model load 
    solver = SolverFactory('gurobi_persistent')
    solver.set_instance(instance) 
    gurobi_model = solver._solver_model
    gurobi_model.update()
    
    # Modify ML_embedding to return xi_values
    embedded_model,pred_constr_E_Q,pred_constr_LL_AMT  = ML_embedding(instance, solver, gurobi_model, trained_model_E_Q,trained_model_LL_AMT)

    # Set Gurobi parameters
    embedded_model.setParam('MIPFocus', 1)
    embedded_model.setParam("NumericFocus", 3)
    embedded_model.setParam('TimeLimit', 3600)
    embedded_model.setParam('NonConvex', 2)
    embedded_model.printStats()

    # Optimize the Gurobi model
    embedded_model.optimize()
    status = embedded_model.Status


    print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr_E_Q.get_error())
    ))

    print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr_LL_AMT.get_error())
    ))

    end_time = time.time()
    print("ML embedded problem Solving time : ", end_time - start_time)


    if status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        embedded_model.computeIIS()
        # Write the IIS to a file
        embedded_model.write("infeasible.ilp")

        # Print the IIS members (constraints and variables)
        print("\nIrreducible Inconsistent Subsystem (IIS):")
        for c in embedded_model.getConstrs():
            if c.IISConstr:
                print(f"Infeasible Constraint: {c.ConstrName}")
        for v in embedded_model.getVars():
            if v.IISLB > 0 or v.IISUB > 0:
                print(f"Infeasible Variable Bound: {v.VarName}, IISLB: {v.IISLB}, IISUB: {v.IISUB}")


    if embedded_model.Status == GRB.OPTIMAL:
        # Compute ratio_sol
        objective_value_embed = embedded_model.objVal
        ratio_sol_simple = abs(objective_value - objective_value_embed) / objective_value * 100
        print("Gap (%) :", ratio_sol_simple)

        # Save the solution to csv
        results_df = save_results_to_csv(embedded_model, solver, f"MLsols/ML_Embed_solution_{SEED}_sce{NoOfScenarios}.csv")
        results_df_v = save_results_to_csv_v(embedded_model, solver, f"MLsols/ML_Embed_installed_solution_{SEED}_sce{NoOfScenarios}.csv")

        # Run validation to get actual second-stage cost
        fsd_file_path = f"MLsols/ML_Embed_solution_{SEED}_sce{NoOfScenarios}.csv"
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
                seed = SEED,
                north_sea = north_sea)

        print("objective_value_embedding_sol: ", objective_value_embedding_sol)
        print("expected_second_stage_value_embedding_sol: ", expected_second_stage_value_embedding_sol)
        ratio_sol = abs(objective_value - objective_value_embedding_sol) / objective_value * 100
        print("Gap (%) :", ratio_sol)


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




def save_results_to_csv_v(gurobi_model, solver, output_filename):
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
        
        if 'genInstalledCap' in var_name:
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
            
        elif 'storPWInstalledCap' in var_name:
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
            
        elif 'storENInstalledCap' in var_name:
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
            
        elif 'transmissionInstalledCap' in var_name:
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





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Specific seed')
    args = parser.parse_args()
    specific_seed = args.seed
    ratio = main(specific_seed)
