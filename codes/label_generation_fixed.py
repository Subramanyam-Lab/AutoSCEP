
#!/usr/bin/env python
from second_stage_label import run_second_stage
from scenario_random import generate_random_scenario
from yaml import safe_load
import time
import csv
import os
import time
import logging
import argparse
import random
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Global Settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UserRunTimeConfig = safe_load(open("config_run.yaml"))

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
north_sea = False

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
    north_sea = False
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]

    if version in ["reduced"]:
        dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"} # reduced version
    else: # full version
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
                f_obj, s_obj, v_dict= future.result()
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



def label_generation(file_num, specific_period, lengthRegSeason, num_scenarios, FSD, num_workers):
    seeds_pool = list(range(100, 5000))
    random.shuffle(seeds_pool)
    
    start = time.time()
    seeds_to_run = [seeds_pool.pop() for _ in range(num_scenarios)]
    
    logging.info(f"Fixed values simulation run: L={lengthRegSeason}, N={num_scenarios}.")

    results, first_obj, v_dict = batch_estimator(
        file_num, specific_period, num_scenarios, lengthRegSeason, FSD, seeds_to_run, num_workers
    )


    mean = np.mean(results)
    exec_time = time.time() - start
    label = mean - first_obj

    logging.info(f"Job ({file_num}, {specific_period}): Label generation completed.")
    logging.info(f"N={len(results)}, L={lengthRegSeason}, Mean total cost={mean:.4f}, First stage cost={first_obj:.4f}, Label (E[Q])={label:.4f}")
    
    return {
        'file_num': file_num,
        'period': specific_period,
        'v_i': v_dict,
        'c_i': first_obj,
        'E_Q_i': label,
        'N_i': len(results),
        'lengthRegSeason': lengthRegSeason,
        'execution_time': exec_time,
        'used_seeds': seeds_to_run
    }
    
    




def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_num', type=int, required=True)
    parser.add_argument('--period', type=int, required=True)
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of CPUs for this specific job.')
    parser.add_argument('--numsam', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    file_num = args.file_num
    period = args.period  
    num_cpus = args.num_cpus 
    seed = args.seed
    numsam = args.numsam

    fixed_lengthRegSeason = 48  
    fixed_num_scenarios = 20 
    
    logging.info(f"====== Starting Job for (file_num={file_num}, period={period}) with {num_cpus} CPUs ======")
    
    FSD = read_fsd_from_csv(f"DataSamples_{numsam}_{seed}/sample_{file_num}.csv")
    
    try:
        result_data = label_generation(file_num, period, fixed_lengthRegSeason, fixed_num_scenarios, FSD, num_cpus)
        
        output_dir = f"training_data_fixed_{numsam}_{seed}/file_{file_num}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"period_{period}.csv")

        file_exists = os.path.isfile(output_path)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_data)

        total_duration = time.time() - start_time
        logging.info(f"SUCCESS: Job ({file_num}, {period}) finished in {total_duration:.2f}s.")

        
    except Exception as e:
        logging.error(f"FAILURE: Job ({file_num}, {period}) failed.", exc_info=True)


if __name__ == "__main__":
    main()
    