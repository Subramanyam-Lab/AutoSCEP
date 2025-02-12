from __future__ import division
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager
import csv
import sys
import cloudpickle
import time
from datetime import datetime
import os
import joblib
import pandas as pd
import numpy as np
import multiprocessing
import json
import pandas as pd
import numpy as np
import glob
from FSD_sampling_violation import create_model, load_investment_data, inv_allo
from pyomo.environ import *
import io
import contextlib
from pyomo.core.expr.visitor import identify_variables
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
from first_stage_empire_sampling import run_first_stage
from yaml import safe_load
import random

def get_inv_cap_bounds(instance):
    bounds_dict = {}

    # Generator Investment Capacity Bounds
    gen_inv_cap_bounds = {}
    for (n, g, i) in instance.genInvCap:
        lb = 0
        # Find the technology t corresponding to generator g
        t_list = [t for (t, g1) in instance.GeneratorsOfTechnology if g1 == g]
        if not t_list:
            ub = 500000.0  # default upper bounds
        else:
            t = t_list[0]
            # Maximum built capacity in period i
            max_built_cap = value(instance.genMaxBuiltCap[n, t, i])
            # Upper bound is the minimum of max_additional_cap and max_built_cap
            ub = max(0, max_built_cap)
        gen_inv_cap_bounds[(n, g, i)] = (lb, ub)

    # Transmission Investment Capacity Bounds
    transmision_inv_cap_bounds = {}
    for (n1, n2, i) in instance.transmisionInvCap:
        lb = 0
        max_built_cap = value(instance.transmissionMaxBuiltCap[n1, n2, i])
        ub = max(0, max_built_cap)
        transmision_inv_cap_bounds[(n1, n2, i)] = (lb, ub)

    # Storage Power Investment Capacity Bounds
    stor_pw_inv_cap_bounds = {}
    for (n, b, i) in instance.storPWInvCap:
        lb = 0
        max_built_cap = value(instance.storPWMaxBuiltCap[n, b, i])
        ub = max(0, max_built_cap)
        stor_pw_inv_cap_bounds[(n, b, i)] = (lb, ub)

    # Storage Energy Investment Capacity Bounds
    stor_en_inv_cap_bounds = {}
    for (n, b, i) in instance.storENInvCap:
        lb = 0
        max_built_cap = value(instance.storENMaxBuiltCap[n, b, i])
        ub = max(0,max_built_cap)
        stor_en_inv_cap_bounds[(n, b, i)] = (lb, ub)

    bounds_dict['genInvCap'] = gen_inv_cap_bounds
    bounds_dict['transmisionInvCap'] = transmision_inv_cap_bounds
    bounds_dict['storPWInvCap'] = stor_pw_inv_cap_bounds
    bounds_dict['storENInvCap'] = stor_en_inv_cap_bounds

    return bounds_dict





def new_sampling_method(instance):
    # Step 1: Get investment capacity variable bounds
    # This function must be defined by you to return a dictionary of investment capacity bounds
    bounds_dict = get_inv_cap_bounds(instance)  
    
    # Create variables dictionary only for investment capacities
    variables = {}  # key: (var_type, var_index), value: {'ub': upper bound, 'value': None, 'constraints': []}
    for var_type, var_bounds in bounds_dict.items():
        for var_index, (lb, ub) in var_bounds.items():
            variables[(var_type, var_index)] = {
                'ub': ub,
                'value': None,
                'constraints': []
            }

    # print("\n=== Initial Variables and Bounds ===")
    # for var, info in variables.items():
    #     print(f"Variable: {var}, Upper Bound: {info['ub']}")

    # Step 2: Set up constraints
    constraints = []

    # Investment capacity constraints for generation
    for t in instance.Technology:
        for n in instance.Node:
            for i in instance.PeriodActive:
                lhs_vars = {}
                for g in instance.Generator:
                    if (n, g) in instance.GeneratorsOfNode and (t, g) in instance.GeneratorsOfTechnology:
                        var_index = ('genInvCap', (n, g, i))
                        if var_index in variables:
                            lhs_vars[var_index] = 1
                            variables[var_index]['constraints'].append(len(constraints))
                rhs = value(instance.genMaxBuiltCap[n, t, i])
                constraints.append({
                    'vars': lhs_vars,
                    'sense': '<=',
                    'rhs': rhs,
                    'type': 'investment_gen_cap'
                })

    # Investment capacity constraints for transmission
    for (n1, n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            lhs_vars = {}
            var_index = ('transmisionInvCap', (n1, n2, i))
            if var_index in variables:
                lhs_vars[var_index] = 1
                variables[var_index]['constraints'].append(len(constraints))
            rhs = value(instance.transmissionMaxBuiltCap[n1, n2, i])
            constraints.append({
                'vars': lhs_vars,
                'sense': '<=',
                'rhs': rhs,
                'type': 'investment_trans_cap'
            })

    # Investment capacity constraints for storage energy
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            lhs_vars = {}
            var_index = ('storENInvCap', (n, b, i))
            if var_index in variables:
                lhs_vars[var_index] = 1
                variables[var_index]['constraints'].append(len(constraints))
            rhs = value(instance.storENMaxBuiltCap[n, b, i])
            constraints.append({
                'vars': lhs_vars,
                'sense': '<=',
                'rhs': rhs,
                'type': 'investment_storage_energy_cap'
            })

    # Investment capacity constraints for storage power
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            lhs_vars = {}
            var_index = ('storPWInvCap', (n, b, i))
            if var_index in variables:
                lhs_vars[var_index] = 1
                variables[var_index]['constraints'].append(len(constraints))
            rhs = value(instance.storPWMaxBuiltCap[n, b, i])
            constraints.append({
                'vars': lhs_vars,
                'sense': '<=',
                'rhs': rhs,
                'type': 'investment_storage_power_cap'
            })


    # Max Installed Capacity Constraints (No installedCap variable)
    # Generation: installedCap(n,g,i) = genInitCap(n,g,i) + sum_{j<=i} genInvCap(n,g,j)
    # We enforce: installedCap(n,g,i) <= genMaxInstalledCap(n,t,i)
    # This becomes: sum_{j<=i} genInvCap(n,g,j) <= genMaxInstalledCap(n,t,i) - genInitCap(n,g,i)
    for t in instance.Technology:
        for n in instance.Node:
            for i in instance.PeriodActive:
                # For each generator of this technology/node
                # sum_{g in G_t} [ sum_{j<=i} genInvCap(n,g,j) ] + genInitCap(n,g,i) <= genMaxInstalledCap(n,t,i)
                # Move initCap to the right-hand side:
                rhs = value(instance.genMaxInstalledCap[n, t, i])
                # We'll accumulate all genInvCap terms that are relevant
                lhs_vars = {}
                # For each generator in this tech at node n
                total_init_cap = 0
                for g in instance.Generator:
                    if (n, g) in instance.GeneratorsOfNode and (t, g) in instance.GeneratorsOfTechnology:
                        init_cap = value(instance.genInitCap[n, g, i])
                        total_init_cap += init_cap
                        # Add all genInvCap(n,g,j) with j <= i
                        for j in instance.PeriodActive:
                            if j <= i:
                                var_index = ('genInvCap', (n, g, j))
                                if var_index in variables:
                                    if var_index not in lhs_vars:
                                        lhs_vars[var_index] = 0
                                    lhs_vars[var_index] += 1
                # Adjust RHS by subtracting total initial capacity
                rhs -= total_init_cap
                # Add the constraint if there are any variables
                if lhs_vars:
                    for var_index in lhs_vars:
                        variables[var_index]['constraints'].append(len(constraints))
                    constraints.append({
                        'vars': lhs_vars,
                        'sense': '<=',
                        'rhs': rhs,
                        'type': 'max_installed_gen_cap'
                    })

    # Transmission: installedCap(n1,n2,i) = transmissionInitCap(n1,n2,i) + sum_{j<=i} transmissionInvCap(n1,n2,j)
    # installedCap(n1,n2,i) <= transmissionMaxInstalledCap(n1,n2,i)
    # => sum_{j<=i} transmissionInvCap(n1,n2,j) <= transmissionMaxInstalledCap(n1,n2,i) - transmissionInitCap(n1,n2,i)
    for (n1,n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            rhs = value(instance.transmissionMaxInstalledCap[n1, n2, i])
            init_cap = value(instance.transmissionInitCap[n1, n2, i])
            rhs -= init_cap
            lhs_vars = {}
            for j in instance.PeriodActive:
                if j <= i:
                    var_index = ('transmisionInvCap', (n1, n2, j))
                    if var_index in variables:
                        lhs_vars[var_index] = 1
            if lhs_vars:
                for var_index in lhs_vars:
                    variables[var_index]['constraints'].append(len(constraints))
                constraints.append({
                    'vars': lhs_vars,
                    'sense': '<=',
                    'rhs': rhs,
                    'type': 'max_installed_trans_cap'
                })

    # Storage Energy: installedCap(n,b,i) = storENInitCap(n,b,i) + sum_{j<=i} storENInvCap(n,b,j)
    # => sum_{j<=i} storENInvCap(n,b,j) <= storENMaxInstalledCap(n,b,i) - storENInitCap(n,b,i)
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            rhs = value(instance.storENMaxInstalledCap[n, b, i])
            init_cap = value(instance.storENInitCap[n, b, i])
            rhs -= init_cap
            lhs_vars = {}
            for j in instance.PeriodActive:
                if j <= i:
                    var_index = ('storENInvCap', (n, b, j))
                    if var_index in variables:
                        lhs_vars[var_index] = 1
            if lhs_vars:
                for var_index in lhs_vars:
                    variables[var_index]['constraints'].append(len(constraints))
                constraints.append({
                    'vars': lhs_vars,
                    'sense': '<=',
                    'rhs': rhs,
                    'type': 'max_installed_storage_energy_cap'
                })

    # Storage Power: installedCap(n,b,i) = storPWInitCap(n,b,i) + sum_{j<=i} storPWInvCap(n,b,j)
    # => sum_{j<=i} storPWInvCap(n,b,j) <= storPWMaxInstalledCap(n,b,i) - storPWInitCap(n,b,i)
    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            rhs = value(instance.storPWMaxInstalledCap[n, b, i])
            init_cap = value(instance.storPWInitCap[n, b, i])
            rhs -= init_cap
            lhs_vars = {}
            for j in instance.PeriodActive:
                if j <= i:
                    var_index = ('storPWInvCap', (n, b, j))
                    if var_index in variables:
                        lhs_vars[var_index] = 1
            if lhs_vars:
                for var_index in lhs_vars:
                    variables[var_index]['constraints'].append(len(constraints))
                constraints.append({
                    'vars': lhs_vars,
                    'sense': '<=',
                    'rhs': rhs,
                    'type': 'max_installed_storage_power_cap'
                })

#    print("\n=== Constraints ===")
#    for i, constraint in enumerate(constraints):
#        print(f"Constraint {i}: Type: {constraint['type']}, Variables: {list(constraint['vars'].keys())}, RHS: {constraint['rhs']}")

    # Step 3: Initialize sets I and J
    I = set(variables.keys())
    J = set()

    # Step 4: Iteratively sample variables
    # print("\n=== Sampling Process ===")
    while J != I:
        # Choose an unassigned variable
        i = (I - J).pop()
        var_info = variables[i]

        # Sample x_i ~ U[0, ub_i]
        x_i = random.uniform(0, var_info['ub'])
        var_info['value'] = x_i

        # Log the sampled value
#        print(f"Sampled Variable: {i}, Value: {x_i:.4f}, Updated Upper Bound: {var_info['ub']:.4f}")

        # Update constraints where x_i appears
        for constraint_idx in var_info['constraints']:
            constraint = constraints[constraint_idx]

            # Update rhs of the constraint
            if 'updated_rhs' not in constraint:
                constraint['updated_rhs'] = constraint['rhs']
            constraint['updated_rhs'] -= constraint['vars'][i] * x_i

            # Update upper bounds of unassigned variables in the constraint
            for var_j, coeff in constraint['vars'].items():
                if var_j != i and var_j not in J:
                    var_j_info = variables[var_j]
                    if constraint['sense'] == '<=':
                        if coeff != 0:
                            new_ub = constraint['updated_rhs'] / coeff
                            # ensure non-negativity
                            new_ub = max(new_ub, 0)
                            var_j_info['ub'] = min(var_j_info['ub'], new_ub)
#                            print(f"Updated Variable: {var_j}, New Upper Bound: {var_j_info['ub']:.4f}")

        # Add variable to J
        J.add(i)

    # Collect the sampled values
    sampled_values = {var: info['value'] for var, info in variables.items()}

    # print("\n=== Final Sampled Values ===")
    # for var, sampled_value in sampled_values.items():
    #     print(f"Variable: {var}, Sampled Value: {sampled_value:.4f}")

    return sampled_values


def has_negative_values(fsd_data):
    for row in fsd_data:
        cap_value = float(row[4])  # Assuming 'Value' is the 5th column
        if cap_value < 0:
            return True
    return False


def check_model_feasibility(instance):

    solver = SolverFactory('glpk')  # You can change this to your preferred solver
    try:
        results = solver.solve(instance, tee=False)
    except Exception as e:
        print(f"Solver exception: {e}")
        return False

    if results.solver.termination_condition == TerminationCondition.optimal:
        return True
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # print("Model is infeasible.")
        return False
    else:
        print(f"Solver Termination Condition: {results.solver.termination_condition}")
        print("Couldn't evaluate feasibility")
        return None


def build_sample_for_checking(sampled_values):
    sampled_data = []
    for keys, samples in sampled_values.items():
        var_type = keys[0]
        if var_type == 'genInvCap':
            n, g, i = keys[1][0],keys[1][1],keys[1][2]
            sampled_data.append({
                'Node': n,
                'Energy_Type': g,
                'Period': i,
                'Type': 'Generation',
                'Value': samples
            })
        elif var_type == 'transmisionInvCap':
            n1, n2, i = keys[1][0],keys[1][1],keys[1][2]
            sampled_data.append({
                'Node': n1,
                'Energy_Type': n2,
                'Period': i,
                'Type': 'Transmission',
                'Value': samples
            })
        elif var_type == 'storPWInvCap':
            n, b, i = keys[1][0],keys[1][1],keys[1][2]
            sampled_data.append({
                'Node': n,
                'Energy_Type': b,
                'Period': i,
                'Type': 'Storage Power',
                'Value': samples
            })
        elif var_type == 'storENInvCap':
            n, b, i = keys[1][0],keys[1][1],keys[1][2]
            sampled_data.append({
                'Node': n,
                'Energy_Type': b,
                'Period': i,
                'Type': 'Storage Energy',
                'Value': samples
            })

    return pd.DataFrame(sampled_data)


def sampling(instance,start_sample_num, north_sea, max_attempts=1000):
    data_folder = 'Data handler/sampling/reduced'
    base_dir = 'DataSamples_EMPIRE3'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    feasible_samples = 0
    total_attempts = 0
    
    while total_attempts < max_attempts:
        total_attempts += 1
        sampled_values = new_sampling_method(instance)
        fsd = build_sample_for_checking(sampled_values)
        fsd_data = fsd.values.tolist()

        if has_negative_values(fsd_data):
            print(f"Attempt {total_attempts} has negative capacities. Discarding.")
            continue

        try:
            gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(fsd_data)
            instance = create_model(data_folder, gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap,north_sea)
            fsd_instance = inv_allo(instance, gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)
            
            if check_model_feasibility(fsd_instance):
                output_file = os.path.join(base_dir, f'sample_{int(start_sample_num+ feasible_samples)}.csv')
                fsd.to_csv(output_file, index=False)
                print(f"Sample {start_sample_num+ feasible_samples} is feasible. Saved as {output_file}")
                feasible_samples += 1
            else:
                print(f"Attempt {total_attempts} is infeasible. Discarding.")
        
        except Exception as e:
            print(f"Error in attempt {total_attempts}: {e}")
            continue
        
        if total_attempts % 50 == 0:
            current_ratio = feasible_samples / total_attempts
            print(f"Current accept-rejection ratio: {current_ratio:.6f}")


    final_ratio = feasible_samples / total_attempts if total_attempts > 0 else 0

    print(f"Generated {feasible_samples} feasible samples.")
    print(f"Final accept-rejection ratio: {final_ratio:.6f}")
    print(f"Total attempts: {total_attempts}")
    
    print("\nExperiment complete.")

if __name__ == '__main__':
    UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

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


    #############################
    ##Non configurable settings##
    #############################

    NoOfRegSeason = 4
    regular_seasons = ["winter", "spring", "summer", "fall"]
    NoOfPeakSeason = 2
    lengthPeakSeason = 24 # reduced 7
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    if version in ["europe_v50"]:
        north_sea = False
    elif version in ["reduced"]:
        north_sea = False
    else:
        north_sea = True


    SEED = 42


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

    # dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
    #                   "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
    #                   "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    #                   "ES": "Spain", "FI": "Finland", "FR": "France",
    #                   "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
    #                   "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    #                   "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
    #                   "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
    #                   "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    #                   "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
    #                   "SK": "Slovakia", "MF": "MorayFirth", "FF": "FirthofForth",
    #                   "DB": "DoggerBank", "HS": "Hornsea", "OD": "OuterDowsing",
    #                   "NF": "Norfolk", "EA": "EastAnglia", "BS": "Borssele",
    #                   "HK": "HollandseeKust", "HB": "HelgolanderBucht", "NS": "Nordsoen",
    #                   "UN": "UtsiraNord", "SN1": "SorligeNordsjoI", "SN2": "SorligeNordsjoII"}

    model,data = run_first_stage(name = name, 
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
            north_sea = north_sea)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prob', type=int, required=True, help='zero_prob')
    args = parser.parse_args()
    start_sample_num = (args.prob)*1000
    instance = model.create_instance(data)
    sampling(instance,start_sample_num,north_sea)
