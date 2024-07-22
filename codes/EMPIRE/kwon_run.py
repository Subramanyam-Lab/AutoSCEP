#!/usr/bin/env python
import os
import csv
import logging
import random
import numpy as np
import pandas as pd
import multiprocessing
import psutil
from datetime import datetime
from yaml import safe_load
from reader import generate_tab_files
from NEUREMPIRE import run_empire
from scenario_random import generate_random_scenario

__author__ = "Stian Backe"
__license__ = "MIT"
__maintainer__ = "Stian Backe"
__email__ = "stian.backe@ntnu.no"

# Load configuration
UserRunTimeConfig = safe_load(open("config_run.yaml"))
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
Iterations = UserRunTimeConfig["Iterations"]

# Non-configurable settings
NoOfRegSeason = 4
regular_seasons = ["winter", "spring", "summer", "fall"]
NoOfPeakSeason = 2
lengthPeakSeason = 24
LeapYearsInvestment = 5
time_format = "%d/%m/%Y %H:%M"
north_sea = version not in ["europe_v50"]

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths and naming
name = f"{version}_reg{lengthRegSeason}_peak{lengthPeakSeason}_sce{NoOfScenarios}"
if scenariogeneration and not fix_sample:
    name += "_randomSGR"
else:
    name += "_noSGR"
if filter_use:
    name += f"_filter{n_cluster}"
if moment_matching:
    name += f"_moment{n_tree_compare}"
name += datetime.now().strftime("_%Y%m%d%H%M")

workbook_path = f'Data handler/{version}'
tab_file_path = f'Data handler/{version}/Tab_Files_{name}'
scenario_data_path = f'Data handler/{version}/ScenarioData'
result_file_path = f'Results/{name}'

# Prepare directories
os.makedirs('NN_training_data', exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FirstHoursOfRegSeason = [lengthRegSeason * i + 1 for i in range(NoOfRegSeason)]
FirstHoursOfPeakSeason = [lengthRegSeason * NoOfRegSeason + lengthPeakSeason * i + 1 for i in range(NoOfPeakSeason)]
Period = [i + 1 for i in range(int((Horizon - 2020) / LeapYearsInvestment))]
Scenario = ["scenario" + str(i + 1) for i in range(NoOfScenarios)]
peak_seasons = ['peak' + str(i + 1) for i in range(NoOfPeakSeason)]
Season = regular_seasons + peak_seasons
Operationalhour = [i + 1 for i in range(FirstHoursOfPeakSeason[-1] + lengthPeakSeason - 1)]
HoursOfRegSeason = [(s, h) for s in regular_seasons for h in Operationalhour if h in list(range(regular_seasons.index(s) * lengthRegSeason + 1, regular_seasons.index(s) * lengthRegSeason + lengthRegSeason + 1))]
HoursOfPeakSeason = [(s, h) for s in peak_seasons for h in Operationalhour if h in list(range(lengthRegSeason * len(regular_seasons) + peak_seasons.index(s) * lengthPeakSeason + 1, lengthRegSeason * len(regular_seasons) + peak_seasons.index(s) * lengthPeakSeason + lengthPeakSeason + 1))]
HoursOfSeason = HoursOfRegSeason + HoursOfPeakSeason

dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR", "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia", "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "MF": "MorayFirth", "FF": "FirthofForth", "DB": "DoggerBank", "HS": "Hornsea", "OD": "OuterDowsing", "NF": "Norfolk", "EA": "EastAnglia", "BS": "Borssele", "HK": "HollandseeKust", "HB": "HelgolanderBucht", "NS": "Nordsoen", "UN": "UtsiraNord", "SN1": "SorligeNordsjoI", "SN2": "SorligeNordsjoII"}

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_single_iteration(iteration):
    logging.info(f"Starting iteration {iteration}")
    random.seed(iteration)
    np.random.seed(iteration)
    
    iteration_name = f"{name}_{iteration}"
    iteration_tab_file_path = f"{tab_file_path}_{iteration}"
    iteration_result_file_path = f"{result_file_path}_{iteration}"
    iteration_scenario_path = scenario_data_path

    try:
        ensure_directory_exists(iteration_tab_file_path)
        ensure_directory_exists(iteration_result_file_path)
        ensure_directory_exists(temp_dir)

        if scenariogeneration:
            generate_random_scenario(filepath=iteration_scenario_path,
                                     tab_file_path=iteration_tab_file_path,
                                     scenarios=NoOfScenarios,
                                     seasons=regular_seasons,
                                     Periods=len(Period),
                                     regularSeasonHours=lengthRegSeason,
                                     peakSeasonHours=lengthPeakSeason,
                                     dict_countries=dict_countries,
                                     time_format=time_format,
                                     filter_make=filter_make,
                                     filter_use=filter_use,
                                     n_cluster=n_cluster,
                                     moment_matching=moment_matching,
                                     n_tree_compare=n_tree_compare,
                                     fix_sample=fix_sample,
                                     north_sea=north_sea,
                                     LOADCHANGEMODULE=LOADCHANGEMODULE)

        generate_tab_files(filepath=workbook_path, tab_file_path=iteration_tab_file_path)

        input_vector, expected_second_stage_value = run_empire(name=iteration_name, 
                   tab_file_path=iteration_tab_file_path,
                   result_file_path=iteration_result_file_path, 
                   scenariogeneration=scenariogeneration,
                   scenario_data_path=iteration_scenario_path,
                   solver=solver,
                   temp_dir=f"{temp_dir}_{iteration}", 
                   FirstHoursOfRegSeason=FirstHoursOfRegSeason, 
                   FirstHoursOfPeakSeason=FirstHoursOfPeakSeason, 
                   lengthRegSeason=lengthRegSeason,
                   lengthPeakSeason=lengthPeakSeason,
                   Period=Period, 
                   Operationalhour=Operationalhour,
                   Scenario=Scenario,
                   Season=Season,
                   HoursOfSeason=HoursOfSeason,
                   discountrate=discountrate, 
                   WACC=WACC, 
                   LeapYearsInvestment=LeapYearsInvestment,
                   IAMC_PRINT=IAMC_PRINT, 
                   WRITE_LP=WRITE_LP, 
                   PICKLE_INSTANCE=PICKLE_INSTANCE, 
                   EMISSION_CAP=EMISSION_CAP,
                   USE_TEMP_DIR=USE_TEMP_DIR,
                   LOADCHANGEMODULE=LOADCHANGEMODULE)

        logging.info(f"Iteration {iteration} completed successfully")
        return iteration, input_vector, expected_second_stage_value

    except Exception as e:
        logging.error(f"Error in iteration {iteration}: {str(e)}")
        return iteration, None, None

def monitor_resources():
    logging.info(f"CPU Usage: {psutil.cpu_percent()}%")
    logging.info(f"Memory Usage: {psutil.virtual_memory().percent}%")
    logging.info(f"Disk Usage: {psutil.disk_usage('/').percent}%")

if __name__ == '__main__':
    logging.info('++++++++')
    logging.info('+EMPIRE+')
    logging.info('++++++++')
    logging.info('LOADCHANGEMODULE: ' + str(LOADCHANGEMODULE))
    logging.info('Solver: ' + solver)
    logging.info('Scenario Generation: ' + str(scenariogeneration))
    logging.info('++++++++')
    logging.info(f'ID: {name}')
    logging.info('++++++++')
    
    num_processes = min(Iterations, psutil.cpu_count())
    logging.info(f"Running with {num_processes} processes")

    with multiprocessing.Pool(num_processes) as pool:
        results = []
        for result in pool.imap_unordered(run_single_iteration, range(1, Iterations + 1)):
            results.append(result)
            monitor_resources()

    valid_results = [r for r in results if r[1] is not None]

    csv_file_path = os.path.join('NN_training_data', 'nn_training_data.csv')
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'X', 'Expected_Second_Stage_Value'])
        for result in valid_results:
            writer.writerow(result)

    logging.info(f"All iterations completed. Results saved to {csv_file_path}")
    logging.info(f"Successful iterations: {len(valid_results)} out of {Iterations}")