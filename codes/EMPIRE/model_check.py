#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import pickle
import cloudpickle
import joblib
import logging
from pathlib import Path
import argparse
from datetime import datetime
from yaml import safe_load
import time
import ast
import onnx
import onnxruntime as ort
from onnx2torch import convert
import random
import warnings
warnings.filterwarnings('ignore')  # To suppress any warnings for cleaner output

# Data manipulation and analysis
import numpy as np
import pandas as pd
import yaml
from ast import literal_eval

# Machine Learning imports
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from skopt import BayesSearchCV

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

# Optimization imports
import gurobipy as grb
from gurobipy import GRB, quicksum, Model as GurobiModel
from gurobi_ml import add_predictor_constr
from gurobi_ml.sklearn import (
    add_decision_tree_regressor_constr,
    add_linear_regression_constr,
    add_mlp_regressor_constr,
    add_standard_scaler_constr,
    add_pipeline_constr
)
import gurobipy_pandas as gppd
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager

# Visualization
import matplotlib.pyplot as plt

# Custom modules (사용자 환경에 맞게 import 경로 조정)
from reader import generate_tab_files
from first_stage_empire import run_first_stage
from NEUREMPIRE import run_empire
from scenario_random import generate_random_scenario
from Embed_Model_validation import empire_validation


# -------------------------------------------------------
# 추가 1) 체크 함수: Pyomo 변수 순서 vs 학습 시 사용된 피처 순서 일치 여부 확인
# -------------------------------------------------------
def check_embedding_order(instance, solver, scaler_v, verbose=True):
    """
    Pyomo 모델에서 v_var_mapping 순으로 가져온 변수(=학습 모델 입력)와
    실제 학습할 때 사용했던 피처(=scaler_v.mean_, scaler_v.scale_) 순서가
    일치하는지 간단히 확인하는 함수입니다.

    Args:
        instance      : Pyomo 모델 인스턴스
        solver        : SolverFactory('gurobi_persistent') 등의 solver 객체
        scaler_v      : 학습 시 사용한 StandardScaler (또는 Pipeline 내 scaler)
        verbose       : True면 상세 출력을 합니다.
    """
    # 1. Pyomo에서 추출한 변수를 순서대로 얻는다.
    indices, pyomo_var_to_gurobi_var = v_var_mapping(instance, solver)
    
    # 2. scaler_v.mean_, scale_v_input 길이와 Pyomo 변수 개수가 같은지 먼저 확인
    n_pyomo_vars = len(indices)
    n_scaler_vars = len(scaler_v.mean_)
    
    if verbose:
        print("=== [check_embedding_order] V 변수 매핑 정보 ===")
        print(f"Pyomo에서 추출된 v 변수 수: {n_pyomo_vars}")
        print(f"scaler_v 스케일러가 바라보는 피처 수: {n_scaler_vars}")

    if n_pyomo_vars != n_scaler_vars:
        print("[주의] Pyomo 변수 개수와 스케일러 피처 개수가 다릅니다!")
        print("      학습 시 변수 순서 또는 개수가 달라졌을 가능성이 큽니다.")
        return
    
    # 3. 변수명(또는 index) 확인을 위해, Pyomo -> (node, gen, period) 등으로 추출
    #    여기서는 단순히 순서만 출력해주고, 사용자가 맞는지 직접 확인하도록 합니다.
    if verbose:
        print("\nPyomo v 변수 순서 (index, var.name):")
        for i, idx in enumerate(indices):
            print(f"{i} : {idx}")  # idx는 ('Germany','Lignite', 1) 같은 튜플
    
    # 4. 사용자가 실제 ML 학습 시에 어떤 순서로 피처를 만들었는지(예: CSV 열 순서, dict 순서 등)를
    #    별도의 리스트로 관리하고 있었다면, 여기서 그 리스트와 비교하셔야 합니다.
    #    예: train_features_columns = [...]
    #    현재는 scaler_v 객체 내에 직접 열 정보를 넣지 않았다면, 매핑 확인을 위해
    #    '학습 당시에 사용한 변수명 리스트'를 별도 로딩해야 합니다.
    
    if verbose:
        print("\nscaler_v.mean_ (피처 순서) 길이: ", len(scaler_v.mean_))
        print("예시로 첫 몇 개 mean_, scale_ 값을 출력해봅니다.")
        for i in range(min(5, len(scaler_v.mean_))):
            print(f"  Feature {i}, mean = {scaler_v.mean_[i]:.4f}, scale = {scaler_v.scale_[i]:.4f}")
    
    print("\n[알림] check_embedding_order를 마쳤습니다.")
    print("      만약 Pyomo 변수 목록과 학습에 사용된 피처 목록이 순서/개수/이름 면에서 다르다면,")
    print("      모델 임베딩 시 잘못된 입력이 들어갈 가능성이 높으니 수정이 필요합니다.\n")


def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data


# -------------------------------------------------------
# 기존 함수: ML 모델 임베딩
# -------------------------------------------------------
def ML_embedding(instance, solver, gurobi_model, regression_model, scaler_v, scaler_y):
    """
    주어진 Pyomo 모델의 v 변수를 잡아서, 학습된 regression_model(두단계 비용 예측)을
    Gurobi에 제약으로 부착(add_predictor_constr)하는 함수.
    """
    # Load scalers
    mean_v_input = scaler_v.mean_
    scale_v_input = scaler_v.scale_

    # Map Pyomo variables to Gurobi variables
    indices_v, pyomo_var_to_gurobi_var_v = v_var_mapping(instance, solver)
    v_vars = [pyomo_var_to_gurobi_var_v[name] for name in indices_v]

    # Scale v variables
    scaled_v_vars = []
    for i, var in enumerate(v_vars):
        scaled_v_var = gurobi_model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                                           name=f'scaled_v_{i}')
        gurobi_model.addConstr(
            scaled_v_var == (var - float(mean_v_input[i])) / float(scale_v_input[i]),
            name=f'scaled_v_constr_{i}'
        )
        scaled_v_vars.append(scaled_v_var)

    # 만약 PCA 등을 적용했다면 이 자리에서 PCA 결과를 적용하는 로직을 삽입

    # 최종 feature로 scaled_v_vars 사용
    final_input_vars = scaled_v_vars
    gurobi_model.update()

    # 예측값 y_approx
    y_approx = gurobi_model.addVar(lb=-GRB.INFINITY, name=f'y_approx')

    # gurobi_ml의 add_predictor_constr 이용
    pred_constr = add_predictor_constr(gurobi_model, regression_model, final_input_vars, y_approx)
    gurobi_model.update()

    # 최종 예측값 스케일링 복원
    mean_output = scaler_y.mean_
    scale_output = scaler_y.scale_
    scaled_y = y_approx * scale_output + mean_output

    gurobi_model.update()

    # 1단계 모델의 기존 목적식에, 예측된 2단계 비용 scaled_y를 추가
    existing_obj = gurobi_model.getObjective()
    combined_obj = existing_obj + scaled_y
    gurobi_model.setObjective(combined_obj, grb.GRB.MINIMIZE)
    gurobi_model.update()

    return gurobi_model, pred_constr


def v_var_mapping(instance, solver):
    """
    Pyomo 변수 중에서 1단계에 해당하는 v 변수를 순서대로 추출.
    반환:
        indices: Pyomo 변수의 이름(또는 index) 리스트
        pyomo_var_to_gurobi_var: Pyomo -> Gurobi 변수 매핑 딕셔너리
    """
    pyomo_var_to_gurobi_var = {}
    
    for period in instance.PeriodActive:
        # 1. Generator installed capacities
        for (n, g) in instance.GeneratorsOfNode:
            var = instance.genInstalledCap[n, g, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 2. Transmission installed capacities
        for (n1, n2) in instance.BidirectionalArc:
            var = instance.transmissionInstalledCap[n1, n2, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 3. Storage Power installed capacities
        for (n, b) in instance.StoragesOfNode:
            var = instance.storPWInstalledCap[n, b, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

        # 4. Storage Energy installed capacities
        for (n, b) in instance.StoragesOfNode:
            var = instance.storENInstalledCap[n, b, period]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[var.name] = gurobi_var

    indices = list(pyomo_var_to_gurobi_var.keys())
    
    return indices, pyomo_var_to_gurobi_var


# -------------------------------------------------------
# 추가 2) 디버깅/로그 출력 함수
# -------------------------------------------------------
def print_debug_logs(gurobi_model, pred_constr):
    """
    Gurobi 모델 및 예측 제약과 관련된 디버그 정보를 간단히 출력합니다.
    예: 제약 오차, 변수 이름/값 등
    """
    print("\n===== [Debug Logs] =====")
    # 예측 오차 (gu robi_ml)
    max_err = np.max(pred_constr.get_error())
    print(f"- Regression approximation max error: {max_err:.6f}")

    # 모델 통계
    gurobi_model.printStats()

    print("========================\n")


def print_pyomo_to_gurobi_mapping(solver, gurobi_model):
    print("\nMapping of Pyomo Variables to Gurobi Variables:")
    pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map  # 기존 매핑 딕셔너리

    for pyomo_var, gurobi_var in pyomo_var_to_gurobi_var.items():
        val = gurobi_var.x if gurobi_var.x is not None else None
        print(f"Pyomo Var: {pyomo_var.name}, Index: {pyomo_var.index()}, "
              f"Gurobi Var: {gurobi_var.VarName}, Value: {val}")

    for v in gurobi_model.getVars():
        if v.VarName.startswith('y'):
            print(v.VarName, "=", v.x)


# -------------------------------------------------------
# 결과 CSV 저장
# -------------------------------------------------------
def save_results_to_csv(gurobi_model, solver, output_filename):
    output_path = Path(output_filename)
    output_dir = output_path.parent
    
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {output_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return None

    results_data = []
    pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map

    for pyomo_var, gurobi_var in pyomo_var_to_gurobi_var.items():
        index = pyomo_var.index()
        value = gurobi_var.x
        var_name = pyomo_var.name
        
        if 'genInvCap' in var_name:
            node, energy_type, period = index
            entry_type = 'Generation'
            results_data.append({
                'Node': node,
                'Energy_Type': energy_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
        elif 'storPWInvCap' in var_name:
            node, storage_type, period = index
            entry_type = 'Storage Power'
            results_data.append({
                'Node': node,
                'Energy_Type': storage_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
        elif 'storENInvCap' in var_name:
            node, storage_type, period = index
            entry_type = 'Storage Energy'
            results_data.append({
                'Node': node,
                'Energy_Type': storage_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
        elif 'transmisionInvCap' in var_name:  # 오타 'transmision'? 실제 코드 확인 필요
            node_from, node_to, period = index
            entry_type = 'Transmission'
            results_data.append({
                'Node': node_from,
                'Energy_Type': node_to,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
    
    df = pd.DataFrame(results_data)
    df = df.sort_values(['Node', 'Energy_Type', 'Period', 'Type'])
    df.to_csv(output_filename, index=False)
    print(f"Results have been saved to {output_filename}")
    return df


def save_results_to_csv_v(gurobi_model, solver, output_filename):
    output_path = Path(output_filename)
    output_dir = output_path.parent
    
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {output_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return None

    results_data = []
    pyomo_var_to_gurobi_var = solver._pyomo_var_to_solver_var_map

    for pyomo_var, gurobi_var in pyomo_var_to_gurobi_var.items():
        index = pyomo_var.index()
        value = gurobi_var.x
        var_name = pyomo_var.name

        if 'genInstalledCap' in var_name:
            node, energy_type, period = index
            entry_type = 'Generation'
            results_data.append({
                'Node': node,
                'Energy_Type': energy_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
        elif 'storPWInstalledCap' in var_name:
            node, storage_type, period = index
            entry_type = 'Storage Power'
            results_data.append({
                'Node': node,
                'Energy_Type': storage_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
        elif 'storENInstalledCap' in var_name:
            node, storage_type, period = index
            entry_type = 'Storage Energy'
            results_data.append({
                'Node': node,
                'Energy_Type': storage_type,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
        elif 'transmissionInstalledCap' in var_name:
            node_from, node_to, period = index
            entry_type = 'Transmission'
            results_data.append({
                'Node': node_from,
                'Energy_Type': node_to,
                'Period': period,
                'Type': entry_type,
                'Value': value
            })
    
    df = pd.DataFrame(results_data)
    df = df.sort_values(['Node', 'Energy_Type', 'Period', 'Type'])
    df.to_csv(output_filename, index=False)
    print(f"Results have been saved to {output_filename}")
    return df


# -------------------------------------------------------
# 메인 함수
# -------------------------------------------------------
def main(SEED):
    
    UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

    USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
    temp_dir = UserRunTimeConfig["temp_dir"]
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
    lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
    discountrate = UserRunTimeConfig["discountrate"]
    WACC = UserRunTimeConfig["WACC"]
    solver_name = UserRunTimeConfig["solver"]
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

    name = (version + '_reg' + str(lengthRegSeason) +
            '_peak' + str(lengthPeakSeason) +
            '_sce' + str(NoOfScenarios))
    if scenariogeneration and not fix_sample:
        name += "_randomSGR"
    else:
        name += "_noSGR"
    if filter_use:
        name += "_filter" + str(n_cluster)
    if moment_matching:
        name += "_moment" + str(n_tree_compare)
    name += str(datetime.now().strftime("_%Y%m%d%H%M"))

    workbook_path = 'Data handler/' + version
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name + f'_{SEED}'
    scenario_data_path = 'Data handler/' + version + '/ScenarioData'
    result_file_path = 'Results/' + name
    FirstHoursOfRegSeason = [lengthRegSeason*i + 1 for i in range(NoOfRegSeason)]
    FirstHoursOfPeakSeason = [lengthRegSeason*NoOfRegSeason + lengthPeakSeason*i + 1 for i in range(NoOfPeakSeason)]
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
    Scenario = ["scenario"+str(i + 1) for i in range(NoOfScenarios)]
    peak_seasons = ['peak'+str(i + 1) for i in range(NoOfPeakSeason)]
    Season = regular_seasons + peak_seasons
    Operationalhour = [i + 1 for i in range(FirstHoursOfPeakSeason[-1] + lengthPeakSeason - 1)]
    HoursOfRegSeason = [(s,h) for s in regular_seasons for h in Operationalhour 
                        if h in range(regular_seasons.index(s)*lengthRegSeason+1,
                                      regular_seasons.index(s)*lengthRegSeason+lengthRegSeason+1)]
    HoursOfPeakSeason = [(s,h) for s in peak_seasons for h in Operationalhour 
                         if h in range(lengthRegSeason*len(regular_seasons)+ 
                                       peak_seasons.index(s)*lengthPeakSeason+1,
                                       lengthRegSeason*len(regular_seasons)+ 
                                       peak_seasons.index(s)*lengthPeakSeason+ 
                                       lengthPeakSeason+1)]
    HoursOfSeason = HoursOfRegSeason + HoursOfPeakSeason
    dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}
    
    generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)
    start_time = time.time()

    model,data = run_first_stage(name = name, 
                                 tab_file_path = tab_file_path,
                                 result_file_path = result_file_path, 
                                 scenariogeneration = scenariogeneration,
                                 scenario_data_path = scenario_data_path,
                                 solver = solver_name,
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
                                 north_sea = north_sea)
    
    instance = model.create_instance(data)
    
    num_sam = 2500
    # model load 
    model_path = f'scaler_pca2/best_model_{num_sam}.pkl'
    with open(model_path, 'rb') as f:
        trained_model = pickle.load(f)

    scaler_v = joblib.load(f'scaler_pca2/scaler_{num_sam}.joblib')
    scaler_y = joblib.load(f'scaler_pca2/scaler_y_{num_sam}.joblib')

    # Pyomo model load
    solver = SolverFactory('gurobi_persistent')
    solver.set_instance(instance) 
    gurobi_model = solver._solver_model
    gurobi_model.update()
    
    # (A) 임베딩 순서가 맞는지 확인
    check_embedding_order(instance, solver, scaler_v, verbose=True)

    # (B) ML 임베딩
    embedded_model, pred_constr = ML_embedding(instance, solver, gurobi_model,
                                               trained_model, scaler_v, scaler_y)

    # Set Gurobi parameters
    embedded_model.setParam('MIPFocus', 1)
    embedded_model.setParam("NumericFocus", 3)
    embedded_model.setParam('TimeLimit', 3600)
    embedded_model.setParam('NonConvex', 2)

    # (C) 디버그 로그 출력
    print_debug_logs(embedded_model, pred_constr)

    # Optimize
    embedded_model.optimize()
    status = embedded_model.Status

    end_time = time.time()
    print("ML embedded problem Solving time : ", end_time - start_time)

    if status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        embedded_model.computeIIS()
        embedded_model.write("infeasible.ilp")

        print("\nIrreducible Inconsistent Subsystem (IIS):")
        for c in embedded_model.getConstrs():
            if c.IISConstr:
                print(f"Infeasible Constraint: {c.ConstrName}")
        for v in embedded_model.getVars():
            if v.IISLB > 0 or v.IISUB > 0:
                print(f"Infeasible Variable Bound: {v.VarName}, IISLB: {v.IISLB}, IISUB: {v.IISUB}")

    if embedded_model.Status == GRB.OPTIMAL:
        # Save the solution to csv
        results_df = save_results_to_csv(embedded_model, solver,
                                         f"MLsols/ML_Embed_solution_{NoOfScenarios}.csv")
        results_df_v = save_results_to_csv_v(embedded_model, solver,
                                             f"MLsols/ML_Embed_installed_solution_{NoOfScenarios}.csv")

        # Run validation to get actual second-stage cost
        fsd_file_path = f"MLsols/ML_Embed_solution_{NoOfScenarios}.csv"
        FSD = read_fsd_from_csv(fsd_file_path)
        
        objective_value_ML, expected_second_stage_value_ML, v_i, Q_i = empire_validation(
            name = name, 
            tab_file_path = tab_file_path,
            result_file_path = result_file_path, 
            scenariogeneration = scenariogeneration,
            scenario_data_path = scenario_data_path,
            solver = "Gurobi",
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
            FSD = FSD,
            WRITE_LP = WRITE_LP, 
            PICKLE_INSTANCE = PICKLE_INSTANCE, 
            EMISSION_CAP = EMISSION_CAP,
            USE_TEMP_DIR = USE_TEMP_DIR,
            LOADCHANGEMODULE = LOADCHANGEMODULE,
            seed = SEED,
            north_sea = north_sea)

        print("objective_value_ML: ", objective_value_ML)
        print("expected_second_stage_value_ML: ", expected_second_stage_value_ML)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Specific seed')
    args = parser.parse_args()
    specific_seed = args.seed
    ratio = main(specific_seed)
