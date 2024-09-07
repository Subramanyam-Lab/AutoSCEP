#!/usr/bin/env python
from reader import generate_tab_files
from NEUREMPIRE import run_empire
from scenario_random import generate_random_scenario
from datetime import datetime
from yaml import safe_load
import time
import pandas as pd
import csv
import os
import numpy as np
import multiprocessing
from functools import partial
import shutil

start = time.time()

__author__ = "Stian Backe"
__license__ = "MIT"
__maintainer__ = "Stian Backe"
__email__ = "stian.backe@ntnu.no"

UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
temp_dir = UserRunTimeConfig["temp_dir"]
version = UserRunTimeConfig["version"]
Horizon = UserRunTimeConfig["Horizon"]
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
second_stage = False
NoOfRegSeason = 4
regular_seasons = ["winter", "spring", "summer", "fall"]
NoOfPeakSeason = 2
lengthPeakSeason = 7
LeapYearsInvestment = 5
time_format = "%d/%m/%Y %H:%M"
if version in ["europe_v50"]:
    north_sea = False
else:
    north_sea = True


def generate_scenarios_and_tabs(base_seed, num_scenarios, scenario_params, workbook_path, tab_file_path):
    generate_random_scenario(**scenario_params, scenarios=num_scenarios, seed=base_seed)
    generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)
    # 생성된 시나리오 데이터 반환
    return load_scenario_data(tab_file_path)

def load_scenario_data(tab_file_path):
    data = {}
    for file_name in ['Stochastic_StochasticAvailability.tab', 'Stochastic_ElectricLoadRaw.tab', 'Stochastic_HydroGenMaxSeasonalProduction.tab']:
        data[file_name] = pd.read_csv(os.path.join(tab_file_path, file_name), sep='\t')
    return data

def caculations(empire_params, tab_file_path,base_seed):
    empire_params["tab_file_path"] = tab_file_path
    caculated_value,variance = run_empire(**empire_params, seed=base_seed)
    return caculated_value

def process_single_scenario(i, run):
    workbook_path, scenario_params, tab_file_path, empire_params = params(i, 1)
    generate_scenarios_and_tabs(i+1, 1, scenario_params, workbook_path, tab_file_path)
    value = caculations(empire_params, tab_file_path, i+1)
    return value, tab_file_path  # tab_file_path 반환

def cal_perfectinfo(run, num_scenarios):
    process_scenario = partial(process_single_scenario, run=run)
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_scenario, range(num_scenarios))
    
    perfect_info_values = [r[0] for r in results]
    tab_file_paths = [r[1] for r in results]  # 각 시나리오의 tab_file_path 저장

    return np.mean(perfect_info_values), tab_file_paths


def calculate_pf_value(run, num_scenarios):
    # Perfect info value calculation
    perfect_info_value_dict = []
    for i in range(run):
        perfect_info_value, tab_file_paths = cal_perfectinfo(i, num_scenarios)
        # 임시 디렉토리 정리
        perfect_info_value_dict.append(perfect_info_value)
        for path in tab_file_paths:
            if os.path.exists(path):
                shutil.rmtree(path)

    return np.mean(perfect_info_value_dict)

def params(run,num_scenarios):
    name = version + '_reg' + str(lengthRegSeason) + \
            '_peak' + str(lengthPeakSeason) + \
            '_sce' + str(num_scenarios) + \
            f"_run{run+1}" + str(datetime.now().strftime("_%Y%m%d%H%M"))
        
    workbook_path = 'Data handler/' + version
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name
    scenario_data_path = 'Data handler/' + version + '/ScenarioData'
    result_file_path = 'Results/' + name
    
    FirstHoursOfRegSeason = [lengthRegSeason*i + 1 for i in range(NoOfRegSeason)]
    FirstHoursOfPeakSeason = [lengthRegSeason*NoOfRegSeason + lengthPeakSeason*i + 1 for i in range(NoOfPeakSeason)]
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
    Scenario = ["scenario"+str(i + 1) for i in range(num_scenarios)]
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
        
    print('++++++++')
    print('+EMPIRE+')
    print('++++++++')
    print(f'Running experiment {run+1} for {num_scenarios} scenarios')
    print('LOADCHANGEMODULE: ' + str(LOADCHANGEMODULE))
    print('Solver: ' + solver)
    print('Scenario Generation: ' + str(scenariogeneration))
    print('++++++++')
    print('ID: ' + name)
    print('++++++++')

    scenario_params = {
        'filepath': scenario_data_path,
        'tab_file_path': tab_file_path,
        'seasons': regular_seasons,
        'Periods': len(Period),
        'regularSeasonHours': lengthRegSeason,
        'peakSeasonHours': lengthPeakSeason,
        'dict_countries': dict_countries,
        'time_format': time_format,
        'filter_make': filter_make,
        'filter_use': filter_use,
        'n_cluster': n_cluster,
        'moment_matching': moment_matching,
        'n_tree_compare': n_tree_compare,
        'fix_sample': fix_sample,
        'north_sea': False,
        'LOADCHANGEMODULE': LOADCHANGEMODULE
    }

    empire_params = {
        'name': name,
        'tab_file_path': tab_file_path,
        'result_file_path': result_file_path,
        'scenariogeneration': scenariogeneration,
        'scenario_data_path': scenario_data_path,
        'solver': solver,
        'temp_dir': temp_dir,
        'FirstHoursOfRegSeason': FirstHoursOfRegSeason,
        'FirstHoursOfPeakSeason': FirstHoursOfPeakSeason,
        'lengthRegSeason': lengthRegSeason,
        'lengthPeakSeason': lengthPeakSeason,
        'Period': Period,
        'Operationalhour': Operationalhour,
        'Scenario': Scenario,
        'Season': Season,
        'HoursOfSeason': HoursOfSeason,
        'discountrate': discountrate,
        'WACC': WACC,
        'LeapYearsInvestment': LeapYearsInvestment,
        'IAMC_PRINT': IAMC_PRINT,
        'WRITE_LP': WRITE_LP,
        'PICKLE_INSTANCE': PICKLE_INSTANCE,
        'EMISSION_CAP': EMISSION_CAP,
        'USE_TEMP_DIR': USE_TEMP_DIR,
        'LOADCHANGEMODULE': LOADCHANGEMODULE
    }


    return workbook_path,scenario_params, tab_file_path,empire_params

def main():
    scenario_numbers = [1,3,9,27,81]
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    result_file_path = f'Results/perfect_info_value_{timestamp}.csv'

    with open(result_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Scenarios", "Perfect Info Value"])

    for num_scenarios in scenario_numbers:
        print(f"Starting experiments for {num_scenarios} scenarios")
        run = 10

        perfect_info_value = calculate_pf_value(run, num_scenarios)
            
        with open(result_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_scenarios, perfect_info_value])
        
        print(f"Completed run for {num_scenarios} scenarios")
        print(f"Perfect Info Value: {perfect_info_value}")
        print("-------------------")

    print(f"Completed all runs for {num_scenarios} scenarios")
    print("===================")
    print("All experiments completed")
    print(f"Results saved to: {result_file_path}")

if __name__ == "__main__":
    main()