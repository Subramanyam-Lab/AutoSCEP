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
import multiprocessing
import numpy as np

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

#######
##RUN##
#######

scenario_numbers = [9]

for NoOfScenarios in scenario_numbers:
    for run in range(10):
        seed = run + 1
        
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
        name = name + f"_run{run+1}" + str(datetime.now().strftime("_%Y%m%d%H%M"))
        
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
            
        print('++++++++')
        print('+EMPIRE+')
        print('++++++++')
        print(f'Running scenario {NoOfScenarios}, run {run+1}')
        print('LOADCHANGEMODULE: ' + str(LOADCHANGEMODULE))
        print('Solver: ' + solver)
        print('Scenario Generation: ' + str(scenariogeneration))
        print('++++++++')
        print('ID: ' + name)
        print('++++++++')

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
                                    LOADCHANGEMODULE = LOADCHANGEMODULE,
                                    seed=seed)

        generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)

        run_start = time.time()
        expected_second_stage_value = run_empire(name = name, 
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
                seed = seed)
        run_end = time.time()
        run_time = run_end - run_start

        # Save results
        os.makedirs(result_file_path, exist_ok=True)
        with open(f"{result_file_path}/results_summary.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Scenarios", "Run", "Expected Second Stage Value", "Runtime"])
            writer.writerow([NoOfScenarios, run+1, expected_second_stage_value, run_time])

        print(f"Completed run {run+1} for {NoOfScenarios} scenarios")
        print(f"Runtime: {run_time} seconds")
        print(f"Expected Second Stage Value: {expected_second_stage_value}")
        print("-------------------")

    print(f"Completed all runs for {NoOfScenarios} scenarios")
    print("===================")

end = time.time()
print("All experiments completed")
print(f"Total runtime: {end - start} seconds")