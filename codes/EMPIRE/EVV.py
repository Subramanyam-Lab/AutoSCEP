import os
import csv
from datetime import datetime
from NEUREMPIRE_EVV import run_empire
from scenario_random import generate_random_scenario
from reader import generate_tab_files
from datetime import datetime
from yaml import safe_load
import time
import pandas as pd
import csv
import os
import numpy as np
import multiprocessing as mp
from functools import partial
import shutil


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

def generate_scenarios_and_tabs(base_seed, num_scenarios, scenario_params, workbook_path, tab_file_path):
    generate_random_scenario(**scenario_params, scenarios=num_scenarios, seed=base_seed)
    generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)

def caculations(empire_params, tab_file_path,base_seed):
    empire_params["tab_file_path"] = tab_file_path
    caculated_value = run_empire(**empire_params, seed=base_seed)
    return caculated_value

# def calculate_eev_value(run, num_scenarios):
#     # Perfect info value calculation
#     eev_value_dict = []
#     for i in range(run):
#         base_seed = i+1
#         workbook_path,scenario_params, tab_file_path,empire_params = params(i,num_scenarios)
#         generate_scenarios_and_tabs(base_seed, num_scenarios, scenario_params, workbook_path, tab_file_path)
#         eev_value = caculations(empire_params, tab_file_path,base_seed)
#         eev_value_dict.append(eev_value)
        
#     return np.mean(eev_value_dict)

def worker_function(run_id, num_scenarios):
    base_seed = run_id + 1
    workbook_path, scenario_params, tab_file_path, empire_params = params(run_id, num_scenarios)
    generate_scenarios_and_tabs(base_seed, num_scenarios, scenario_params, workbook_path, tab_file_path)
    eev_value = caculations(empire_params, tab_file_path, base_seed)
    return eev_value

def calculate_eev_value(run, num_scenarios):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        eev_value_list = pool.starmap(worker_function, [(i, num_scenarios) for i in range(run)])
    
    return np.mean(eev_value_list)

def main():
    scenario_numbers = [1,3,9,27,81]
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    result_file_path = f'Results/evv_results_{timestamp}.csv'

    # 결과 파일 헤더 작성
    with open(result_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Scenarios", "EEV Value"])

    for num_scenarios in scenario_numbers:
        print(f"Starting experiments for {num_scenarios} scenarios")
        run = 10
        # EV 및 EEV 계산
        eev_value = calculate_eev_value(run, num_scenarios)

        # 결과 저장
        with open(result_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_scenarios,eev_value])

        print(f"Completed run for {num_scenarios} scenarios")
        print(f"EEV Value: {eev_value}")
        print("-------------------")

    print("All experiments completed")
    print(f"Results saved to: {result_file_path}")

if __name__ == "__main__":
    main()