
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
import time
import logging
import argparse
import random
import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
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


def scenario_folder_generation(lengthRegSeason, seed):      
    
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
    
    if version in ["reduced"]:
        dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}
    else: # full version
        dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
                        "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
                        "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
                        "ES": "Spain", "FI": "Finland", "FR": "France",
                        "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
                        "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
                        "LT": "Lithuania", "LU": "Luxemb."}

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
                temp_dir, discountrate, WACC, FSD, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE,north_sea,hour_decision):
    
    tab_file_path = scenario_folder_generation(lengthRegSeason, seed)

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



def batch_estimator(file_num, specific_period, S, lengthRegSeason, FSD, seeds_to_run, num_workers):
    
    NoOfScenarios = 1
    results = []
    feature_vectors = []
    dummy_f_obj = None
    dummy_v_dict = None
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for seed in seeds_to_run:
            
            future = executor.submit(
                    run_single_seed,
                    seed, version, lengthRegSeason, lengthPeakSeason, NoOfScenarios,specific_period, 
                    file_num, NoOfRegSeason, NoOfPeakSeason,regular_seasons, Horizon, LeapYearsInvestment,
                    temp_dir, discountrate, WACC, FSD, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, north_sea, True
            )
            futures[future] = seed
        
        for future in as_completed(futures):
            seed = futures[future]
            try:
                f_obj, s_obj, v_dict = future.result()
                dummy_f_obj = f_obj
                dummy_v_dict = v_dict
                results.append(f_obj + s_obj)
                
            except Exception as e:
                logging.error(f"Seed {seed} failed in batch_estimator: {e}", exc_info=True)
                
    if not results:
        return [], 0, {}
    
    
    return results, dummy_f_obj, dummy_v_dict


def coefficient_of_variation(costs):
    mean = np.mean(costs)
    std  = np.std(costs, ddof=1)
    return std / mean if mean else np.inf



def label_generation(file_num, specific_period, lengthRegSeason, num_scenarios, FSD, num_workers,master_seed):
    seeds_pool = list(range(100, 5000))
    random.seed(master_seed)
    random.shuffle(seeds_pool)
    
    start = time.time()
    seeds_to_run = [seeds_pool.pop() for _ in range(num_scenarios)]
    
    logging.info(f"running: L={lengthRegSeason}, N={num_scenarios}.")

    results, first_obj, v_dict = batch_estimator(
        file_num, specific_period, num_scenarios, lengthRegSeason, FSD, seeds_to_run, num_workers
    )

    if not results:
        logging.error("batch_estimator returned no results.")
        return {}

    mean = np.mean(results)

    exec_time = time.time() - start
    label = mean - first_obj

    logging.info(f"job ({file_num}, {specific_period}): label generation completed.")
    logging.info(f"N={len(results)}, L={lengthRegSeason}, mean={mean:.4f}, 1st stage cost={first_obj:.4f}, label (E[Q])={label:.4f}")
    
    return {
        'file_num': file_num,
        'period': specific_period,
        'v_i': v_dict,
        'c_i': first_obj,
        'E_Q_i': label,
        'N_i': len(results),
        'lengthRegSeason': lengthRegSeason,
        'execution_time': exec_time,
        'status': 'Fixed_L_N_Run', 
        'master_seed': master_seed,
        'used_seeds': seeds_to_run
    }
    
    




def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, required=True, help='Length of regular season')
    parser.add_argument('--N', type=int, required=True, help='Number of scenarios')
    parser.add_argument('--period', type=int, required=True, help='Period number')
    parser.add_argument('--master_seed', type=int, required=True, help='Master seed for reproducibility')
    parser.add_argument('--num_cpus', type=int, default=5, help='Number of CPUs')
    parser.add_argument('--file_num', type=int, default=1, help='File number')
    args = parser.parse_args()
    
    start_time = time.time()
    
    FSD = read_fsd_from_csv(f"100_seed_5_inv_cap.csv")
    
    logging.info(f"Starting: L={args.L}, N={args.N}, Period={args.period}, MasterSeed={args.master_seed}")
    
    result_data = label_generation(
        args.file_num, args.period, args.L, args.N, FSD, args.num_cpus, args.master_seed
    )
    
    if result_data:
        output_dir = "sampling_convergence_controlled"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(
            output_dir,
            f"fixed_{args.L}_{args.N}_{args.period}_mseed{args.master_seed}.csv"
        )
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_data.keys())
            writer.writeheader()
            writer.writerow(result_data)
        
        logging.info(f"SUCCESS: Saved to {output_path} in {time.time() - start_time:.2f}s")
    else:
        logging.error("FAILED: No result data")


if __name__ == "__main__":
    main()
    