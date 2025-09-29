#!/usr/bin/env python
# File & System imports
import os
import csv
import logging
from pathlib import Path
import argparse
from datetime import datetime
from yaml import safe_load
import warnings
warnings.filterwarnings('ignore')  # To suppress any warnings for cleaner output

from pyomo.environ import *
from second_stage import run_second_stage

from filelock import FileLock
import csv
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data

def main(solution_numbers,numsce,seednum,model,scenario_set_num):
    UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

    USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
    temp_dir = UserRunTimeConfig["temp_dir"]
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
    NoOfScenarios = 10
    # lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
    lengthRegSeason = 72
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
    lengthPeakSeason = 24
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    if version in ["europe_v50"]:
        north_sea = False
    elif version in ["reduced"]:
        north_sea = False
    else:
        north_sea = True


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
    scenario_data_path = 'Data handler/' + version + '/ScenarioData'
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
    
    # fsd_file_path = f"sol_sets/100_seed_5_inv_cap.csv" # near-optimal
    # fsd_file_path = f"sol_sets/ef_solution_{numsce}_{solution_numbers}.csv" # EF-validation
    # fsd_file_path = f"sol_sets/ph_solution_{numsce}_{solution_numbers}_{soltime}.csv" # PH-validation
    fsd_file_path = f"MLEMBEDSOLS_fixed/ML_Embed_solution_{model}_{numsce}_{solution_numbers}.csv"
    FSD = read_fsd_from_csv(fsd_file_path)
    
    logging.info(f"{solution_numbers} seed start!")
    sol_num = seednum + 1000
    output_dir = f"Data handler/scenarios_output/{sol_num}"
    scenario_folder = os.path.join(output_dir, f"scenario_set_{scenario_set_num}")
    
    # os.makedirs(scenario_folder, exist_ok=True)
    
    # generate_random_scenario(
    #     filepath=scenario_data_path,
    #     tab_file_path=scenario_folder,
    #     scenarios=NoOfScenarios,
    #     seasons=regular_seasons,
    #     Periods=len(Period),
    #     regularSeasonHours=lengthRegSeason,
    #     peakSeasonHours=lengthPeakSeason,
    #     dict_countries=dict_countries,
    #     time_format=time_format,
    #     filter_make=filter_make,
    #     filter_use=filter_use,
    #     n_cluster=n_cluster,
    #     moment_matching=moment_matching,
    #     n_tree_compare=n_tree_compare,
    #     fix_sample=fix_sample,
    #     north_sea=north_sea,
    #     LOADCHANGEMODULE=LOADCHANGEMODULE,
    #     seed=sol_num
    # )
    
    
    total_obj = run_second_stage(
        tab_file_path=scenario_folder,
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
        seed=sol_num,
        file_num=sol_num,
        north_sea=north_sea,
        version=version
    )
    
    logging.info(f"objective_value_ML: {total_obj}")
    
    csv_path = Path(f"validation_fixed_log.csv")
    lock = FileLock(str(csv_path) + ".lock")

    row = [
        model,
        solution_numbers,
        sol_num,
        scenario_set_num,
        numsce,
        total_obj
    ]

    with lock:
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "model",
                    "solution_number",
                    "seed",
                    "scenario_set_num",
                    "number of scenarios",
                    "total_objective_value"
                ])
            writer.writerow(row)
    logging.info(f"Saved results to {csv_path}")

    return result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_number', type=int, required=True, help='Solution Number')
    parser.add_argument('--numsce', type=int, required=True, help='Number of scenarios')
    parser.add_argument('--seednum', type=int, required=True, help='Seed Number')
    parser.add_argument('--method', type=str, required=True, help='Method used (e.g., MLP, LR)')
    parser.add_argument('--setnum', type=int, required=True, help='Scenario set number')
    # parser.add_argument('--soltime', type=int, required=True, help='Solution time')
    
    args = parser.parse_args()

    solution_numbers = args.solution_number
    numsce = args.numsce
    seednum = args.seednum
    model_name = args.method
    setnum = args.setnum
    # soltime = args.soltime
    
    result = main(solution_numbers, numsce, seednum, model_name, setnum)
