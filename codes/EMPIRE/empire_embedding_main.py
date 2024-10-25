#!/usr/bin/env python
from reader import generate_tab_files
from first_stage_empire import run_empire
from ml_preprocessing import load_data, ML_trainig, ML_embedding, var_mapping, load_pca_data, get_gurobi_inv_cap_vars
from scenario_random import generate_random_scenario
from datetime import datetime
from yaml import safe_load
import time
import pandas as pd
import csv
from gurobipy import GRB, quicksum
import gurobipy
from gurobi_ml import add_predictor_constr
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



def main():


    # Set random seeds for reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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

    second_stage = True
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
    
    specific_period = 2
   
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
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name
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
                                LOADCHANGEMODULE = LOADCHANGEMODULE)

    generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)


    instance = run_empire(name = name, 
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
            specific_period = specific_period)
    
    solver = SolverFactory('gurobi_persistent')
    solver.set_instance(instance) 

    file_path = 'cleaned_unique_combination_data.csv'
    X, y = load_data(file_path)

    # Split data into training and test sets (80-20 split)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    print('\nTraining Model')
    # Initialize scalers for final evaluation
    scaler_X_final = StandardScaler()
    scaler_y_final = StandardScaler()
    trained_lr, trained_dt = ML_trainig(
        X_train_full, y_train_full, X_test, y_test,
        scaler_X_final, scaler_y_final
    )

    
    # Access the Gurobi model
    gurobi_model = solver._solver_model
    
    # Create a mapping from Pyomo variables to Gurobi variables
    # pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map
    
    for period in range(1,2):
        gurobi_inv_cap_vars = get_gurobi_inv_cap_vars(instance, gurobi_model, period)
        sorted_indices, pyomo_var_to_gurobi_var_ml = var_mapping(instance, solver, period)
        gurobi_model = ML_embedding(gurobi_model,gurobi_inv_cap_vars, pyomo_var_to_gurobi_var_ml,trained_dt,sorted_indices, instance, period)
    
    # Set Gurobi parameters
    gurobi_model.Params.NonConvex = 2
    gurobi_model.Params.TimeLimit = 100

    # Optimize the Gurobi model
    gurobi_model.optimize()

    if gurobi_model.Status == GRB.OPTIMAL or gurobi_model.Status == GRB.TIME_LIMIT:
        # Update Pyomo variables with values from Gurobi variables
        for pyomo_var, gurobi_var in pyomo_var_to_gurobi_var.items():
            pyomo_var.set_value(gurobi_var.X)
        
        # Retrieve the value of 'y_approx'
        y_approx_value = y_approx.X

        # Display the results
        print(f"Approximated second-stage cost (y_approx): {y_approx_value}")

        # Now you can access and display the Pyomo variable values as needed
        # For example:
        for (n, g) in instance.GeneratorsOfNode:
            for i in instance.PeriodActive:
                gen_installed_cap = value(instance.genInstalledCap[n, g, i])
                print(f"Node {n}, Generator {g}, Period {i}: Installed Capacity = {gen_installed_cap}")
    else:
        print("Optimization did not converge to a solution.")


if __name__ == '__main__':
    main()