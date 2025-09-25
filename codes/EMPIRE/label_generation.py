
#!/usr/bin/env python
from sympy import sec
from second_stage_label import run_second_stage
from scenario_random import generate_random_scenario
from datetime import datetime
from yaml import safe_load
import time
import pandas as pd
import csv
import os
import multiprocessing
import psutil
import time
import logging
import argparse
import random
import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from scipy.stats import norm  
import matplotlib.pyplot as plt


# os.environ["GRB_LICENSE_FILE"] = "gurobi.lic"

# Global Settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
temp_dir = UserRunTimeConfig["temp_dir"]
version = UserRunTimeConfig["version"]
Horizon = UserRunTimeConfig["Horizon"]
discountrate = UserRunTimeConfig["discountrate"]
WACC = UserRunTimeConfig["WACC"]
solver = UserRunTimeConfig["solver"]
scenariogeneration = UserRunTimeConfig["scenariogeneration"]
fix_sample = UserRunTimeConfig["fix_sample"]
LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
filter_use = UserRunTimeConfig["filter_use"]
n_cluster = UserRunTimeConfig["n_cluster"]
moment_matching = UserRunTimeConfig["moment_matching"]
n_tree_compare = UserRunTimeConfig["n_tree_compare"]
EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]
WRITE_LP = UserRunTimeConfig["WRITE_LP"]
PICKLE_INSTANCE = UserRunTimeConfig["PICKLE_INSTANCE"]

# Non-configurable settings
lengthPeakSeason = 24
NoOfRegSeason = 4
regular_seasons = ["winter", "spring", "summer", "fall"]
NoOfPeakSeason = 2
LeapYearsInvestment = 5
north_sea = True if version in ["europe_v51", "europe_reduced_v51"] else False

# seeds_pool = list(range(100, 1000))



def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data


def scenario_generation(lengthRegSeason, seed):      
    
    # Extract configuration variables
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    NoOfScenarios = 1  # single scenario per simulation
    fix_sample = UserRunTimeConfig["fix_sample"]
    LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
    filter_make = UserRunTimeConfig["filter_make"]
    filter_use = UserRunTimeConfig["filter_use"]
    n_cluster = UserRunTimeConfig["n_cluster"]
    moment_matching = UserRunTimeConfig["moment_matching"]
    n_tree_compare = UserRunTimeConfig["n_tree_compare"]
    
    # Non-configurable settings
    regular_seasons = ["winter", "spring", "summer", "fall"]
    LeapYearsInvestment = 5
    lengthPeakSeason = 24
    time_format = "%d/%m/%Y %H:%M"
    north_sea = True if version in ["europe_v51", "europe_reduced_v51"] else False
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]

    # Define country dictionary for scenario generation
    if version in ["europe_v51", "europe_reduced_v51"]:
        dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
                          "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
                          "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
                          "ES": "Spain", "FI": "Finland", "FR": "France",
                          "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
                          "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
                          "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
                          "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
                          "PL": "Poland", "PT": "Portugal", "RO": "Romania",
                          "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
                          "SK": "Slovakia", "MF": "MorayFirth", "FF": "FirthofForth",
                          "DB": "DoggerBank", "HS": "Hornsea", "OD": "OuterDowsing",
                          "NF": "Norfolk", "EA": "EastAnglia", "BS": "Borssele",
                          "HK": "HollandseeKust", "HB": "HelgolanderBucht", "NS": "Nordsoen",
                          "UN": "UtsiraNord", "SN1": "SorligeNordsjoI", "SN2": "SorligeNordsjoII"}

    elif version in ["reduced"]:
        dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}

    else:
        dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
                          "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
                          "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
                          "ES": "Spain", "FI": "Finland", "FR": "France",
                          "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
                          "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
                          "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
                          "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
                          "PL": "Poland", "PT": "Portugal", "RO": "Romania",
                          "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
                          "SK": "Slovakia"}

    scenario_data_path = f"Data handler/{version}/ScenarioData"
    tab_file_path = f"Data handler/{version}/Tab_Files/scenario_{seed}"
    scenario_folder = os.path.join(tab_file_path, f"len_{lengthRegSeason}")
    os.makedirs(scenario_folder, exist_ok=True)


    generate_random_scenario(
        filepath=scenario_data_path,
        tab_file_path=scenario_folder,
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
        LOADCHANGEMODULE=LOADCHANGEMODULE,
        seed=seed
    )

    return scenario_folder



def run_single_seed(seed, version, lengthRegSeason, lengthPeakSeason, NoOfScenarios, specific_period, file_num,
                NoOfRegSeason, NoOfPeakSeason, regular_seasons, Horizon, LeapYearsInvestment, 
                temp_dir, discountrate, WACC, FSD, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE,north_sea,tab_file_path,hour_decision):


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

    first_stage_obj, second_obj_val, v_dict = run_second_stage(
        tab_file_path=tab_file_path,
        temp_dir=temp_dir,
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
        FSD=FSD,
        EMISSION_CAP=EMISSION_CAP,
        USE_TEMP_DIR=USE_TEMP_DIR,
        LOADCHANGEMODULE=LOADCHANGEMODULE,
        seed=seed,
        specific_period=specific_period,
        file_num=file_num,
        north_sea = north_sea,
        hour_decision = hour_decision,
        version = version
    )
    

    return first_stage_obj, second_obj_val, v_dict



def batch_estimator(file_num, specific_period, S, lengthRegSeason,FSD):
    global seeds_pool

    NoOfScenarios = 1
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    if len(seeds_pool) < S:
        raise ValueError(f"Not enough seeds in pool: requested {S}, have {len(seeds_pool)}")
    seeds = random.sample(seeds_pool, S)
    for s in seeds:
        seeds_pool.remove(s)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for seed in seeds:
            scenario_folder = scenario_generation(lengthRegSeason, seed)
            futures.append(
                executor.submit(
                    run_single_seed,
                    seed, version, lengthRegSeason, lengthPeakSeason, NoOfScenarios,specific_period, 
                    file_num, NoOfRegSeason, NoOfPeakSeason,regular_seasons, Horizon, LeapYearsInvestment,
                    temp_dir, discountrate, WACC, FSD, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, north_sea, scenario_folder, True
                )
            )
        for future in as_completed(futures):
            f_obj, s_obj, v_dict = future.result()
            results.append(f_obj + s_obj)

    return results, f_obj, v_dict



def coefficient_of_variation(costs):
    mean = np.mean(costs)
    std  = np.std(costs, ddof=1)
    return std / mean if mean else np.inf



def label_generation(file_num, specific_period, lengthRegSeason, FSD):
    global seeds_pool
    seeds_pool = list(range(100, 5000))
    
    start = time.time()
    threshold_r, threshold_h = 0.1, 0.1
    h_increment = 6
    confidence_level = 0.95
    initial_num_sce = 5
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    MAX_L = 72
    MAX_N = 500
    
    results, first_obj, v_dict = batch_estimator(file_num, specific_period, initial_num_sce, lengthRegSeason, FSD)
    
    h_is_fixed = False

    while True:
        N = len(results)
        if N == 0:
            logging.error("Batch estimator returned no results. Terminating.")
            break
            
        if N >= MAX_N:
            logging.warning(f"Scenario count N ({N}) exceeded or met MAX_N ({MAX_N}). Terminating.")
            break
        
        mean = np.mean(results)
        std = np.std(results, ddof=1) if N > 1 else 0
        
        half_w = z * std / np.sqrt(N) if N > 1 else float('inf')
        r_error = half_w / mean if mean else float('inf')
        rel_half = coefficient_of_variation(results)
        
        logging.info(f"N={N}, L={lengthRegSeason}, mean={mean:.4f}, label={mean-first_obj:.4f}, r_error={r_error:.4f}, CV={rel_half:.4f}")


        if not h_is_fixed and rel_half > threshold_h:
            lengthRegSeason += h_increment
            
            if lengthRegSeason > MAX_L:
                logging.warning(f"Next lengthRegSeason ({lengthRegSeason}) will exceed MAX_L ({MAX_L}). Terminating.")
                break
            
            logging.info(f"Relative variation is too high (CV={rel_half:.4f}). Increasing fidelity to lengthRegSeason={lengthRegSeason}.")
            results, _, _ = batch_estimator(file_num, specific_period, N, lengthRegSeason, FSD)
            continue
        
        h_is_fixed = True
        
        if r_error > threshold_r:
            needed = int(np.ceil((z * std / (threshold_r * mean))**2)) - N if mean > 0 else initial_num_sce
            needed = max(1, needed)
            
            logging.info(f"Relative error is too high (r_error={r_error:.4f}). Requiring {needed} more scenarios.")
            more_results, _, _ = batch_estimator(file_num, specific_period, needed, lengthRegSeason, FSD)
            results = np.concatenate([results, more_results])
            continue

        else:
            logging.info("Convergence criteria met. Finalizing label.")
            break


    final_mean = np.mean(results)
    final_std = np.std(results, ddof=1)
    final_N = len(results)
    final_half_w = z * final_std / np.sqrt(final_N) if final_N > 1 else float('inf')
    final_r_error = final_half_w / final_mean if final_mean else float('inf')
    final_rel_half = coefficient_of_variation(results)

    status = "Terminated_Unknown"
    if final_r_error <= threshold_r and final_rel_half <= threshold_h:
        status = "Converged"
    elif lengthRegSeason > MAX_L:
        status = "Terminated_Max_L"
    elif final_N >= MAX_N:
        status = "Terminated_Max_N"
    else:
        status = "Terminated_Not_Converged"
    
    exec_time = time.time() - start
    label = np.mean(results) - first_obj
    return {
        'file_num': file_num,
        'period': specific_period,
        'v_i': v_dict,
        'c_i': first_obj,
        'E_Q_i': label,
        'N_i': len(results),
        'lengthRegSeason': lengthRegSeason,
        'execution_time': exec_time,
        'status': status 
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--start_index', type=int, required=True)
    parser.add_argument('--end_index', type=int, required=True)
    args = parser.parse_args()


    ops = [((i-1)//8, (i-1)%8+1) for i in range(args.start_index, args.end_index+1)]
    results = []
    
    logging.info(f"Task {args.task_id}: Starting")
    FSD = read_fsd_from_csv(f"DataSamples_EMPIRE5/sample_{args.task_id}.csv")
    
    initial_lengthRegSeason = 6

    for file_num, period in ops:
        try:
            logging.info(f"File {file_num}, period={period} is running")
            
            results.append(label_generation(file_num, period, initial_lengthRegSeason,FSD))
        except Exception as e:
            logging.error(f"Error file={file_num}, period={period}: {e}")

    if results:
        os.makedirs('training_data6', exist_ok=True)
        # training_data2
        out = f"training_data6/experiment_results_task_{args.task_id}.csv"
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

if __name__ == "__main__":
    main()
