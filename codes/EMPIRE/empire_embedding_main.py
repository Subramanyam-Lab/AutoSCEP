#!/usr/bin/env python
from reader import generate_tab_files
from first_stage_empire import run_first_stage
from NEUREMPIRE import run_empire
from ml_preprocessing import load_data, ML_training, ML_embedding,selected_var_mapping, preprocessing_data,input_var_mapping
from scenario_random import generate_random_scenario
from datetime import datetime
from yaml import safe_load
import time
import pandas as pd
import csv
from gurobipy import GRB, quicksum
from gurobipy import Model as GurobiModel
from gurobi_ml import add_predictor_constr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
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
import logging
import pickle
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager
import csv
import sys
import cloudpickle
import time
from datetime import datetime
import os
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import multiprocessing
import json
from gurobi_ml import add_predictor_constr
from gurobi_ml.sklearn import add_decision_tree_regressor_constr,add_linear_regression_constr,add_mlp_regressor_constr
from gurobi_ml.sklearn import add_standard_scaler_constr
import argparse
import csv
import os
from pathlib import Path
from Embed_Model_validation import run_validation


def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data

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

    ##################################
    ######### Model Train ############
    file_path = 'training_data5.csv'
    v_i,xi_i,y = load_data(file_path)
    X_train, y_train, X_test, y_test = preprocessing_data(v_i,xi_i,y)

    print('\nTraining Model')
    # Initialize scalers for final evaluation
    trained_lr, trained_dt, trained_mlp = ML_training(
        X_train, y_train, X_test, y_test
    )
    ######### Model Train ############
    ##################################

    solver = SolverFactory('gurobi_persistent')
    instance = model.create_instance(data)
    solver.set_instance(instance) 
    gurobi_model = solver._solver_model
    gurobi_model.update()

    # Create a mapping from Pyomo variables to Gurobi variables
    pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map
    # gurobi_model = get_gurobi_installed_cap_vars(instance, gurobi_model,pyomo_var_to_gurobi_var)
    
    for seed in SEED_range:
            # gurobi_inv_cap_vars = get_gurobi_inv_cap_vars(instance, gurobi_model, period)
        # indices, pyomo_var_to_gurobi_var_ml = selected_var_mapping(instance, solver)
        # gurobi_model = ML_embedding(instance, gurobi_model, trained_lr,trained_dt, trained_mlp, indices, 
        #             pyomo_var_to_gurobi_var_ml, seed,SEED_range)
        indices, pyomo_var_to_gurobi_var_ml = input_var_mapping(instance, solver)
        gurobi_model = ML_embedding(instance, gurobi_model, trained_lr,trained_dt, trained_mlp,indices,
                    pyomo_var_to_gurobi_var_ml, seed,SEED_range)
                    
    # Set Gurobi parameters
    gurobi_model.Params.NonConvex = 2
    gurobi_model.Params.NumericFocus = 3
    gurobi_model.Params.TimeLimit = 300

    # Optimize the Gurobi model
    gurobi_model.optimize()


    if gurobi_model.Status == GRB.OPTIMAL:
        objective_value_embed = gurobi_model.objVal
        ratio = abs(objective_value-objective_value_embed) / objective_value
        print("Gap (%) :", ratio*100)

        results_df = save_results_to_csv(gurobi_model, solver, f"MLsols/ML_Embed_solution_{SEED_range}.csv")

        fsd_file_path = f"MLsols/ML_Embed_solution_{SEED_range}.csv"
        # fsd_file_path = f"FSD/202411042138_616_seed_58_inv_cap.csv"
        FSD = read_fsd_from_csv(fsd_file_path)

        objective_value_embedding_sol, expected_second_stage_value_embedding_sol = run_validation(name = name, 
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

        ratio_sol = (abs(objective_value_embedding_sol-objective_value)/objective_value)*100
        print("objective_value_embedding_sol: ",objective_value_embedding_sol)
        print("expected_second_stage_value_embedding_sol: ",expected_second_stage_value_embedding_sol)
        print("ratio_sol (%):",ratio_sol)
        print_pyomo_to_gurobi_mapping(solver)
        for v in gurobi_model.getVars():
            if v.VarName.startswith('y') or v.VarName.startswith('x'):
            # if v.VarName.startswith('y'):
                print(v.VarName, "=", v.x)


        return ratio 

def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data



# Ensure this code is placed after you have set up and solved your model

# Mapping Pyomo variables to Gurobi variables
def print_pyomo_to_gurobi_mapping(solver):
    print("\nMapping of Pyomo Variables to Gurobi Variables:")
    pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map  # Existing mapping

    for pyomo_var, gurobi_var in pyomo_var_to_gurobi_var.items():
        print(f"Pyomo Variable: {pyomo_var.name}, Index: {pyomo_var.index()}, Corresponding Gurobi Variable: {gurobi_var.VarName}")



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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Specific seed')
    args = parser.parse_args()
    specific_seed = args.seed
    ratio = main(specific_seed)

    # ##### SAVE RESULTS #########
    # csv_file_path = 'results.csv'

    # with open(csv_file_path, 'r', newline='') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     exst_data = list(reader)

    # new_row = {
    #     'seed': json.dumps(specific_seed),
    #     'Gap (%)': json.dumps(ratio)
    # }
    # exst_data.append(new_row)

    # fieldnames = ['seed', 'Gap (%)']
    # with open(csv_file_path, 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(exst_data)    