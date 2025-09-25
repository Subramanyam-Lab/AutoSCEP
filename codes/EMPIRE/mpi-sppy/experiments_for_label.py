#!/usr/bin/env python
from sympy import sec
from second_stage import run_second_stage
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


# 설정 로드
import yaml
UserRunTimeConfig = yaml.safe_load(open("config_reducedrun.yaml"))

# Global Settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 전역 설정
seeds = list(range(1000, 1030))  # 10개 시드
length_list = [6, 12, 18, 24]    # 예시: 정규시즌 길이 변화
scenario_counts = [5, 10, 20, 30]  # 시나리오 개수 변화


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



def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data


def scenario_generation(lengthRegSeason, NoOfScenarios, seed):      
    
    # Extract configuration variables
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
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
    tab_file_path = f"Data handler/{version}/Tab_Files"
    scenario_folder = os.path.join(tab_file_path, f"scenario_{seed}_length_{lengthRegSeason}")
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



def run_single_seed(seed,version, lengthRegSeason, lengthPeakSeason, NoOfScenarios, file_num,
                NoOfRegSeason, NoOfPeakSeason, regular_seasons, Horizon, LeapYearsInvestment, 
                temp_dir, discountrate, WACC, FSD, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE,north_sea):

    scenario_folder = scenario_generation(lengthRegSeason,NoOfScenarios, seed)

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

    logging.info(f"Running seed {seed} with length {lengthRegSeason} and S {NoOfScenarios}")

    first_stage_obj, second_obj_val, _, _, _ = run_second_stage(
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
        seed=file_num,
        file_num=file_num,
        north_sea=north_sea,
        version=version
    )
    

    return second_obj_val



def main_experiment():
    FSD = read_fsd_from_csv("sol_sets/solution_11_5.csv")
    base_dir = 'experiment_results'
    dist_dir = os.path.join(base_dir, 'distributions')
    os.makedirs(dist_dir, exist_ok=True)
    summary = []
    for length in length_list:
        for S in scenario_counts:
            # 각 조합에 대해 10 시드 실행
            costs = []
            with ProcessPoolExecutor() as exe:
                futures = [exe.submit(run_single_seed, seed, version, length, lengthPeakSeason, S, 0,
                NoOfRegSeason, NoOfPeakSeason, regular_seasons, Horizon, LeapYearsInvestment, 
                temp_dir, discountrate, WACC, FSD, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE,north_sea) for seed in seeds]
                for fut in as_completed(futures):
                    costs.append(fut.result())
            # Raw costs 저장
            df = pd.DataFrame({'cost': costs})
            dist_csv = os.path.join(dist_dir, f'dist_L{length}_S{S}.csv')
            df.to_csv(dist_csv, index=False)


            plt.figure(figsize=(8, 6))
            plt.hist(costs, bins='auto', alpha=0.7, density=False)
            
            mean_val   = np.mean(costs)
            median_val = np.median(costs)
            
            plt.axvline(mean_val,   linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2e}')
            plt.axvline(median_val, linestyle=':',  linewidth=1.5, label=f'Median: {median_val:.2e}')
            
            plt.title(f'Distribution (L={length}, S={S})', fontsize=14)
            plt.xlabel('Second Stage Cost',   fontsize=12)
            plt.ylabel('Frequency',            fontsize=12)
            plt.grid(True, alpha=0.3)  # 연한 회색 격자
            plt.legend(frameon=False, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f'hist_L{length}_S{S}.png'))
            plt.close()

            # 통계 요약 추가
            summary.append({
                'lengthRegSeason': length,
                'numScenarios': S,
                'mean_cost': np.mean(costs),
                'std_cost': np.std(costs, ddof=1),
                'q1': np.quantile(costs, 0.25),
                'median': np.quantile(costs, 0.5),
                'q3': np.quantile(costs, 0.75)
            })

    # 요약 CSV 저장
    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(base_dir, 'summary_L_S_results.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"저장 완료: summary -> {summary_csv}")

if __name__ == '__main__':
    main_experiment()
