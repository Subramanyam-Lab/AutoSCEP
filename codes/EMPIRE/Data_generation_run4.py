
#!/usr/bin/env python
from reader import generate_tab_files
from Expected_Second_Stage_data2 import run_second_stage
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
import re
import glob
import ast
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from scipy.stats import norm
import fcntl  
from multiprocessing import Pool


def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data


def run_single_seed(seed, version, lengthRegSeason, lengthPeakSeason, NoOfScenarios, scenariogeneration,
                fix_sample, filter_use, n_cluster, moment_matching, n_tree_compare, specific_period, file_num,
                NoOfRegSeason, NoOfPeakSeason, regular_seasons, Horizon, LeapYearsInvestment, time_format, filter_make, 
                temp_dir, solver, discountrate, WACC, FSD, WRITE_LP, PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE):

    try:
        timestamp = datetime.now().strftime("_%Y%m%d%H%M%S%f")
        name = f"{version}_reg{lengthRegSeason}_peak{lengthPeakSeason}_sce{NoOfScenarios}"
        if scenariogeneration and not fix_sample:
            name += "_randomSGR"
        else:
            name += "_noSGR"
        if filter_use:
            name += f"_filter{n_cluster}"
        if moment_matching:
            name += f"_moment{n_tree_compare}"
        name += f"_seed{seed}_period{specific_period}_filenum{file_num}_{timestamp}"

        workbook_path = 'Data handler/' + version
        tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name
        scenario_data_path = 'Data handler/' + version + '/ScenarioData' 
        result_file_path = 'Results/' + name

        # os.makedirs(tab_file_path, exist_ok=True)
        # os.makedirs(result_file_path, exist_ok=True)

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
        # dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
        #             "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
        #             "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
        #             "ES": "Spain", "FI": "Finland", "FR": "France",
        #             "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
        #             "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
        #             "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
        #             "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
        #             "PL": "Poland", "PT": "Portugal", "RO": "Romania",
        #             "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
        #             "SK": "Slovakia", "MF": "MorayFirth", "FF": "FirthofForth",
        #             "DB": "DoggerBank", "HS": "Hornsea", "OD": "OuterDowsing",
        #             "NF": "Norfolk", "EA": "EastAnglia", "BS": "Borssele",
        #             "HK": "HollandseeKust", "HB": "HelgolanderBucht", "NS": "Nordsoen",
        #             "UN": "UtsiraNord", "SN1": "SorligeNordsjoI", "SN2": "SorligeNordsjoII"}

        dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}
        

        first_stage_obj, second_obj_val, second_stage_else, v_i, ll_amt = run_second_stage(
            name=name,
            tab_file_path=tab_file_path,
            result_file_path=result_file_path,
            scenariogeneration=scenariogeneration,
            scenario_data_path=scenario_data_path,
            solver=solver,
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
            WRITE_LP=WRITE_LP,
            PICKLE_INSTANCE=PICKLE_INSTANCE,
            EMISSION_CAP=EMISSION_CAP,
            USE_TEMP_DIR=USE_TEMP_DIR,
            LOADCHANGEMODULE=LOADCHANGEMODULE,
            seed=seed,
            specific_period=specific_period,
            file_num=file_num
        )
        

        return first_stage_obj, second_obj_val, second_stage_else, v_i, ll_amt

    except Exception as e:
        print(f"Error in run_single_seed with seed={seed}: {str(e)}")
        raise



def run_experiment(file_num,specific_period):
    start_time = time.time()
    UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

    # Extract all the configuration variables as in your original code
    USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
    temp_dir = UserRunTimeConfig["temp_dir"]
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    # NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
    NoOfScenarios = 1
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

    # Non-configurable settings
    NoOfRegSeason = 4
    regular_seasons = ["winter", "spring", "summer", "fall"]
    NoOfPeakSeason = 2
    lengthPeakSeason = 24 # original run 24
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    


    FSD = read_fsd_from_csv(f"DataSamples_EMPIRE/sample_{file_num}.csv")
    confidence_level = 0.95
    threshold = 0.05
    # threshold = 1e-5 # for original
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    r_error = float('inf')
    label = float('inf')
    avg_ll_amt= float('inf')
    avg_q_i_elst = float('inf')
    num_seed = 5
    seed_increment = 10
    q_i = []
    q_i_else = []
    first_stage_value = float('inf')
    v_i = {} 
    ll_amt_lst = []


    used_index = 0
    
    while r_error > threshold:

        if used_index >= 1000:  
            print("No more samples available. Terminating process.")
            label = np.mean(np.hstack(q_i))
            avg_q_i_elst = np.mean(np.hstack(q_i_else))
            avg_ll_amt = np.mean(np.hstack(ll_amt_lst))
            break

        if r_error == float('inf'):
            seeds = list(range(used_index + 1, min(used_index + num_seed + 1, 1001)))
            used_index += num_seed
        else:
            seeds = list(range(used_index + 1, min(used_index + seed_increment + 1, 1001)))
            used_index += seed_increment

        # Multiprocessing
        with Pool(processes=seed_increment) as pool:
            results = pool.starmap(
                run_single_seed,
                [(seed, version, lengthRegSeason, lengthPeakSeason, NoOfScenarios, scenariogeneration,
                  fix_sample, filter_use, n_cluster, moment_matching, n_tree_compare, specific_period, file_num,
                  NoOfRegSeason, NoOfPeakSeason, regular_seasons, Horizon, LeapYearsInvestment, time_format,
                  filter_make, temp_dir, solver, discountrate, WACC, FSD, WRITE_LP, PICKLE_INSTANCE,
                  EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE)
                 for seed in seeds]
            )

        for result in results:
            if result is not None:
                first_stage_obj, second_stage_obj, second_stage_obj_else, rsl_v_i, ll_amt = result
                first_stage_value = first_stage_obj
                q_i.append(second_stage_obj)
                q_i_else.append(second_stage_obj_else)
                v_i = rsl_v_i
                ll_amt_lst.append(ll_amt)



        flattened_q_i = np.hstack(q_i)
        mean_q_i = np.mean(flattened_q_i)
        sample_var = np.var(flattened_q_i, ddof=1)
        N = len(flattened_q_i)
        var_of_mean = sample_var / N
        std_err_of_mean = np.sqrt(var_of_mean)
        half_width = z * std_err_of_mean
        r_error = (2 * half_width) / (first_stage_value+ mean_q_i) if mean_q_i != 0 else float('inf')

        q_i_else_avg = np.mean(np.hstack(q_i_else))
        ll_amt_avg = np.mean(np.hstack(ll_amt_lst))

        print(f"num seed: {num_seed}, mean q_i: {mean_q_i}, var q_i: {sample_var}, relative error: {r_error}, ll_amt: {ll_amt_avg}") 
        if r_error <= threshold:
            label = mean_q_i
            avg_q_i_elst = q_i_else_avg
            avg_ll_amt = ll_amt_avg
            break
        else:
            num_seed += seed_increment

    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"all q_i : {q_i}")
    print(f"final label {label} ,Execution time: {execution_time:.6f} seconds")
    

    result_data = {
        'file_num': file_num,
        'period': specific_period,
        'v_i': v_i,
        'E_Q_i': label,
        'E_Q_i_else' : avg_q_i_elst,
        'LL_amt': avg_ll_amt,
        'execution_time': execution_time
    }

    return result_data



def run_experiment_wrapper(args):
    """Wrapper function to unpack arguments for run_experiment."""
    file_num, period = args
    return run_experiment(file_num, period)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True, help='SLURM array task ID')
    parser.add_argument('--start_index', type=int, required=True, help='Start index for operations')
    parser.add_argument('--end_index', type=int, required=True, help='End index for operations')
    args = parser.parse_args()

    task_id = args.task_id
    start_index = args.start_index
    end_index = args.end_index

    # Prepare list of (file_num, period) tuples
    operations = []
    for operation_index in range(start_index, end_index + 1):
        file_num = (operation_index - 1) // 8
        period = (operation_index - 1) % 8 + 1
        operations.append((file_num, period))

    results = []

    print(f"Task {task_id}: Starting")

    for op in operations:
        file_num, period = op
        try:
            result_data = run_experiment(file_num, period)
            results.append(result_data)
        except Exception as e:
            print(f"Error processing file_num={file_num}, period={period}: {str(e)}")
            continue

    # 각 SLURM 작업의 결과를 개별 CSV 파일로 저장
    if results:
        os.makedirs('results_empire2', exist_ok=True)  # 결과 저장 디렉토리 생성
        result_file = f"results_empire2/experiment_results_task_{task_id}.csv"
        with open(result_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Task {task_id} completed. Results saved to {result_file}.")
    else:
        print(f"Task {task_id} completed with no results.")

if __name__ == "__main__":
    main()