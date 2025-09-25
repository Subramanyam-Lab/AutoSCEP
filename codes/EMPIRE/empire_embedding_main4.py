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
from scenario_random import generate_random_scenario
from Expected_Second_Stage_data2 import run_second_stage


def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data

def log_neural_net_results(instance, logfile="nn_log.txt"):
    with open(logfile, "w") as f:
        # Log each layer’s variables
        num_layers = 0
        for block_component_name in instance.NeuralNet.component_objects(Var, active=True):
            if "x_" in block_component_name.name or "z_" in block_component_name.name:
                # e.g., block_component_name might be x_1, x_2, z_1, etc.
                var_block = getattr(instance.NeuralNet, block_component_name.name)
                f.write(f"=== {block_component_name.name} ===\n")
                for idx in var_block:
                    val = var_block[idx].value
                    f.write(f"   {block_component_name.name}[{idx}] = {val}\n")
                f.write("\n")
                num_layers += 1

        # Log the final output
        f.write(f"nn_output = {instance.nn_output.value}\n\n")
        
        # If needed, log the scaled inputs
        f.write("=== nn_inputs ===\n")
        for i in range(1, len(instance.nn_inputs) + 1):
            val = instance.nn_inputs[i].value
            f.write(f" nn_inputs[{i}] = {val}\n")
        f.write("\n")

        # Log objective value
        f.write(f"Objective (Minimized): {instance.Obj.expr()}\n")  # or instance.Obj()

    print(f"Neural net results logged to {logfile}.")


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
    
    generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)
    start_time = time.time()

    instance = run_first_stage(name = name, 
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
    
    
    if solver == "Gurobi":
        opt = SolverFactory('gurobi', Verbose=True)
        opt.options["Crossover"]=0
        opt.options["Method"]=2
    

    results = opt.solve(instance, tee=True, logfile="gurobi_output.log")# , logfile=result_file_path + '\logfile_' + name + '.log' , keepfiles=True, symbolic_solver_labels=True)
    end_time = time.time()
    print("Solver Status:", results.solver.status)
    print("Solver Termination Condition:", results.solver.termination_condition)
    print("ML embedded problem Solving time : ", end_time - start_time)

    print(value(instance.Obj))
    print(value(instance.ml_output))
    instance.solutions.load_from(results)
    objective_value = value(instance.Obj)
    save_results(instance, NoOfScenarios)
    save_results_v(instance, NoOfScenarios)

# Mapping Pyomo variables to Gurobi variables
def print_pyomo_to_gurobi_mapping(solver,gurobi_model):
    print("\nMapping of Pyomo Variables to Gurobi Variables:")
    pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map  # Existing mapping

    for pyomo_var, gurobi_var in pyomo_var_to_gurobi_var.items():
        print(f"Pyomo Variable: {pyomo_var.name}, Index: {pyomo_var.index()}, Corresponding Gurobi Variable: {gurobi_var.VarName}, Value: {gurobi_var.x}")

    for v in gurobi_model.getVars():
            if v.VarName.startswith('y'):
                print(v.VarName, "=", v.x)



def save_results(instance, NoSce):
    # Retrieve relevant data from the instance
    gen_inv_cap = instance.genInvCap.get_values()
    transmision_inv_cap = instance.transmisionInvCap.get_values()
    stor_pw_inv_cap = instance.storPWInvCap.get_values()
    stor_en_inv_cap = instance.storENInvCap.get_values()

    total_fsd_length = len(gen_inv_cap) + len(transmision_inv_cap) + len(stor_pw_inv_cap) + len(stor_en_inv_cap)

    # Add a generator type label to each entry
    gen_inv_cap = {(k[0], k[1], k[2], 'Generation'): v for k, v in gen_inv_cap.items()}
    transmision_inv_cap = {(k[0], k[1], k[2], 'Transmission'): v for k, v in transmision_inv_cap.items()}
    stor_pw_inv_cap = {(k[0], k[1], k[2], 'Storage Power'): v for k, v in stor_pw_inv_cap.items()}
    stor_en_inv_cap = {(k[0], k[1], k[2], 'Storage Energy'): v for k, v in stor_en_inv_cap.items()}

    # Combine all investment capacities and costs into dictionaries
    inv_cap_data = {**gen_inv_cap, **transmision_inv_cap, **stor_pw_inv_cap, **stor_en_inv_cap}

    # Convert the capacity data into a DataFrame
    data = [(k[0], k[1], k[2], k[3], v) for k, v in inv_cap_data.items()]
    df = pd.DataFrame(data, columns=['Node', 'Energy_Type', 'Period', 'Type', 'Value'])

    # Create output directories if they don't exist
    output_dir = "MLSOLS"
    os.makedirs(output_dir, exist_ok=True)

    # Save capacity and cost data to CSV
    # output_file_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d%H%M')}_{total_fsd_length}_seed_{seed}_inv_cap.csv")
    output_file_path = os.path.join(output_dir, f"ML_Embed_solution_ad_LS_10000.csv")
    df.to_csv(output_file_path, index=False)

    print("DataFrames created and saved successfully.")


def save_results_v(instance, NoSce):
    # Retrieve relevant data from the instance
    gen_installed_cap = instance.genInstalledCap.get_values()
    transmision_installed_cap = instance.transmissionInstalledCap.get_values()
    stor_pw_installed_cap = instance.storPWInstalledCap.get_values()
    stor_en_installed_cap = instance.storENInstalledCap.get_values()

    total_fsd_length = len(gen_installed_cap) + len(transmision_installed_cap) + len(stor_pw_installed_cap) + len(stor_en_installed_cap)

    # Add a generator type label to each entry
    gen_installed_cap = {(k[0], k[1], k[2], 'Generation'): v for k, v in gen_installed_cap.items()}
    transmision_installed_cap = {(k[0], k[1], k[2], 'Transmission'): v for k, v in transmision_installed_cap.items()}
    stor_pw_installed_cap = {(k[0], k[1], k[2], 'Storage Power'): v for k, v in stor_pw_installed_cap.items()}
    stor_en_installed_cap = {(k[0], k[1], k[2], 'Storage Energy'): v for k, v in stor_en_installed_cap.items()}

    # Combine all investment capacities and costs into dictionaries
    installed_cap_data = {**gen_installed_cap, **transmision_installed_cap, **stor_pw_installed_cap, **stor_en_installed_cap}

    # Convert the capacity data into a DataFrame
    data = [(k[0], k[1], k[2], k[3], v) for k, v in installed_cap_data.items()]
    df = pd.DataFrame(data, columns=['Node', 'Energy_Type', 'Period', 'Type', 'Value'])

    # Create output directories if they don't exist
    output_dir = "MLSOLS"
    os.makedirs(output_dir, exist_ok=True)

    # Save capacity and cost data to CSV
    # output_file_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d%H%M')}_{total_fsd_length}_seed_{seed}_inv_cap.csv")
    output_file_path = os.path.join(output_dir, f"ML_Embed_installed_solution_ad_LS_10000.csv")
    df.to_csv(output_file_path, index=False)


    print("DataFrames created and saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Specific seed')
    args = parser.parse_args()
    specific_seed = args.seed
    ratio = main(specific_seed)
