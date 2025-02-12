#!/usr/bin/env python
from reader import generate_tab_files
from NEUREMPIRE import run_empire
from Expected_Second_Stage import run_second_stage
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
from concurrent.futures import ProcessPoolExecutor



def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data


def run_experiment(seed,specific_period,file_num, fsd_file_path):
# def run_experiment(seed,file_num,fsd_file_path):
    UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

    # Extract all the configuration variables as in your original code
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

    # Non-configurable settings
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

    # Build a unique name including seed and timestamp
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
    # name += f"_seed{seed}_filenum{file_num}_{timestamp}"

    workbook_path = 'Data handler/' + version
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name
    scenario_data_path = 'Data handler/' + version + '/ScenarioData' 
    result_file_path = 'Results/' + name

    os.makedirs(tab_file_path, exist_ok=True)
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
        

    print(f'Running scenario with SEED={seed} and PERIOD={specific_period}')
    # print(f'Running scenario with SEED={seed}')

    if scenariogeneration:
        generate_random_scenario(
            filepath=scenario_data_path,
            tab_file_path=tab_file_path,
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

    FSD = read_fsd_from_csv(fsd_file_path)

    # Run the model
    expected_second_stage_value = run_second_stage(name = name, 
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
        FSD = FSD, 
        WRITE_LP = WRITE_LP, 
        PICKLE_INSTANCE = PICKLE_INSTANCE, 
        EMISSION_CAP = EMISSION_CAP,
        USE_TEMP_DIR = USE_TEMP_DIR,
        LOADCHANGEMODULE = LOADCHANGEMODULE,
        seed=seed,
        specific_period = specific_period,
        file_num = file_num)


def extract_file_info(filename):
    """Extracts the file number from the filename."""
    pattern = r'DataSamples_zero_prob/sample_(\d+)'
    match = re.search(pattern, filename)
    if match:
        filenum = int(match.group(1))  # Extract the numeric part as an integer
        return filenum
    return None


def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run experiment with specific parameters')
    parser.add_argument('--file_num', type=int, help='File number to process')
    parser.add_argument('--period', type=int, help='Period to process')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    # Construct file path
    file_path = f"DataSamples_EMPIRE/sample_{args.file_num}.csv"
    
    # Log the parameters for debugging
    print(f"Running with parameters: File={args.file_num}, Period={args.period}, Seed={args.seed}")
    
    try:
        # Run the experiment with provided parameters
        run_experiment(
            seed=args.seed,
            specific_period=args.period,
            file_num=args.file_num,
            fsd_file_path=file_path
        )
        print(f"Successfully completed experiment: File={args.file_num}, Period={args.period}")
        
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        raise  # Re-raise the exception to ensure the job fails properly

if __name__ == "__main__":
    main()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num', type=int, required=True, help='File num')
#     args = parser.parse_args()

#     i = args.num
#     fsd_file_path = f"DataSamples/sample_{i+1}.csv"
    
#     seed = np.random.randint(1,10000)
    
#     print(f"Running for seed: {seed} and for period: {j} about {i+1}-th fsd file")
#     run_experiment(seed, j ,i+1,fsd_file_path)


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--period', type=int, required=True, help='Specific period')
    # args = parser.parse_args()

    # specific_period = args.period

    # for i in range(30):
    #     fsd_file_path = f"FSDsamples/sampled_data_{i+1}.csv"

    #     N = 10   # Number of seeds to generate
    #     M = 10   # maximum number of parallel processes

    #     # Generate N random seeds
    #     seeds = [random.randint(1, 1000) for _ in range(N)]

    #     print(f"Running for PERIOD={specific_period} with seeds: {seeds} about {i+1}-th fsd file")

    #     # Limit the number of processes if needed
    #     num_processes = min(N, M)  # Adjust '10' to the desired maximum number of parallel processes

    #     # Use multiprocessing to run experiments in parallel
    #     with multiprocessing.Pool(processes=num_processes) as pool:
    #         pool.starmap(run_experiment, [(seed, specific_period,fsd_file_path) for seed in seeds])



# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num', type=int, required=True, help='File num')
#     args = parser.parse_args()

#     i = args.num

#     fsd_file_path = f"DataSamples/sample_{i+1}.csv"
    
#     # Generate N random seeds
#     seed = np.random.randint(1,100)

#     print(f"Running for seed: {seed} about {i+1}-th fsd file")
#     run_experiment(seed, i+1 ,fsd_file_path)

# if __name__ == "__main__":
#     main()

