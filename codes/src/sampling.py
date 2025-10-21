from __future__ import division
import re
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager
import csv
from datetime import datetime
import os
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from first_stage import run_first_stage
from pyomo.environ import *
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
from yaml import safe_load
import random
import logging
import time


logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%d/%m/%Y %H:%M')


def get_inv_cap_bounds(instance):
    bounds_dict = {}

    gen_inv_cap_bounds = {}
    for (n, g, i) in instance.genInvCap:
        lb = 0
        t_list = [t for (t, g1) in instance.GeneratorsOfTechnology if g1 == g]
        if not t_list:
            ub = 500000.0  # default upper bounds
        else:
            t = t_list[0]
            max_built_cap = value(instance.genMaxBuiltCap[n, t, i])
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


def build_preprocessed_data(instance):
    bounds_dict = get_inv_cap_bounds(instance)
    variables = {}
    for var_type, var_bounds in bounds_dict.items():
        for var_index, (lb, ub) in var_bounds.items():
            if ub == 0:
                variables[(var_type, var_index)] = {
                    'lb': lb,
                    'ub': ub,
                    'value': 0.0,
                    'constraints': []
                }
            else:
                variables[(var_type, var_index)] = {
                    'lb': lb,
                    'ub': ub,
                    'value': None,
                    'constraints': []
                }

    constraints = []

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

    for t in instance.Technology:
        for n in instance.Node:
            for i in instance.PeriodActive:
                lhs_vars = {}
                total_init_cap = 0  
                for g in instance.Generator:
                    if (n, g) in instance.GeneratorsOfNode and (t, g) in instance.GeneratorsOfTechnology:
                        startPeriod_int = 1
                        calcStart = value(1 + i - (instance.genLifetime[g]/instance.LeapYearsInvestment))
                        if calcStart > startPeriod_int:
                            startPeriod_int = calcStart
                        init_cap = value(instance.genInitCap[n, g, i])
                        total_init_cap += init_cap
                        for j in instance.PeriodActive:
                            if j <= i and j >= startPeriod_int:
                                var_index = ('genInvCap', (n, g, j))
                                if var_index in variables:
                                    lhs_vars[var_index] = lhs_vars.get(var_index, 0) + 1
                rhs_val = value(instance.genMaxInstalledCap[n, t, i]) - total_init_cap
                if lhs_vars:
                    for var_index in lhs_vars:
                        variables[var_index]['constraints'].append(len(constraints))
                    constraints.append({
                        'vars': lhs_vars,
                        'sense': '<=',
                        'rhs': rhs_val,
                        'type': 'max_installed_gen_cap_with_lifetime'
                    })

    for (n1,n2) in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            startPeriod = 1
            calcStart = value(1 + i - instance.transmissionLifetime[n1, n2]/instance.LeapYearsInvestment)
            if calcStart > startPeriod:
                startPeriod = calcStart      
            rhs = value(instance.transmissionMaxInstalledCap[n1, n2, i])
            init_cap = value(instance.transmissionInitCap[n1, n2, i])
            rhs -= init_cap
            lhs_vars = {}
            for j in instance.PeriodActive:
                if j <= i and j >= startPeriod:
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

    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            startPeriod = 1
            calcStart = value(1 + i - instance.storageLifetime[b]/instance.LeapYearsInvestment)
            if calcStart > startPeriod:
                startPeriod = calcStart

            rhs = value(instance.storENMaxInstalledCap[n, b, i])
            init_cap = value(instance.storENInitCap[n, b, i])
            rhs -= init_cap
            lhs_vars = {}
            for j in instance.PeriodActive:
                if j <= i and j >= startPeriod:
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

    for (n,b) in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            startPeriod = 1
            calcStart = value(1 + i - instance.storageLifetime[b]/instance.LeapYearsInvestment)
            if calcStart > startPeriod:
                startPeriod = calcStart

            rhs = value(instance.storPWMaxInstalledCap[n, b, i])
            init_cap = value(instance.storPWInitCap[n, b, i])
            rhs -= init_cap
            lhs_vars = {}
            for j in instance.PeriodActive:
                if j <= i and j >= startPeriod:
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
                    'type': 'max_installed_power_energy_cap'
                })

    logging.info("Constraint Setting Done!")

    var_keys = list(variables.keys())  # ex) [('genInvCap',(n,g,i)), ('transmisionInvCap',(...)), ...]
    var2idx   = {v: idx for idx, v in enumerate(var_keys)}
    idx2var   = {idx: v for v, idx in var2idx.items()}
    n_vars = len(var_keys)

    lb_array = np.zeros(n_vars, dtype=np.float64)
    ub_array = np.zeros(n_vars, dtype=np.float64)
    val_array = np.full(n_vars, np.nan, dtype=np.float64)

    for v_key, info in variables.items():
        i = var2idx[v_key]
        lb_array[i] = info.get('lb', 0.0)
        ub_array[i] = info['ub']
        if info['value'] is not None:
            val_array[i] = info['value']

    constraints_of_var = [[] for _ in range(n_vars)]
    for c_idx, cdict in enumerate(constraints):
        for v_key in cdict['vars'].keys():
            vidx = var2idx[v_key]
            constraints_of_var[vidx].append(c_idx)

    array_constraints = []
    for c_idx, cdict in enumerate(constraints):
        var_idxs = []
        coefs    = []
        for v_key, coeff in cdict['vars'].items():
            var_idxs.append(var2idx[v_key])
            coefs.append(coeff)
        var_idxs = np.array(var_idxs, dtype=np.int32)
        coefs    = np.array(coefs,   dtype=np.float64)

        sense = cdict['sense']
        rhs   = cdict['rhs']

        array_constraints.append({
            'var_idxs': var_idxs,
            'coefs': coefs,
            'sense': sense,
            'rhs': rhs,
            'updated_rhs': rhs,
            'unassigned_mask': np.full(len(var_idxs), True, dtype=bool),
            'type': cdict['type']
        })

    for v_key in var_keys:
        if v_key[0] == 'genInvCap':
            node, generator, period = v_key[1]
            if ('existing' in str(generator)) or ('CCS' in str(generator)):
                i = var2idx[v_key]
                val_array[i] = 0.0

    preprocessed_data = {
        'var_keys': var_keys,
        'var2idx': var2idx,
        'idx2var': idx2var,
        'lb_array': lb_array,
        'ub_array': ub_array,
        'val_array': val_array,
        'constraints_of_var': constraints_of_var,
        'array_constraints': array_constraints
    }

    return preprocessed_data


def get_key_from_var(var_key):
    var_type, index_tuple = var_key
    if var_type == 'genInvCap':
        node, generator, period = index_tuple
        return (node, generator, period, "Generation")
    elif var_type == 'transmisionInvCap':
        node1, node2, period = index_tuple
        return (node1, node2, period, "Transmission")
    elif var_type == 'storPWInvCap':
        node, storage, period = index_tuple
        return (node, storage, period, "Storage Power")
    elif var_type == 'storENInvCap':
        node, storage, period = index_tuple
        return (node, storage, period, "Storage Energy")
    return None

def data_preprocessing(preprocessed_data, optimal_values):
    val_array = preprocessed_data['val_array'].copy()
    var_keys = preprocessed_data['var_keys']
    var2idx = preprocessed_data['var2idx']

    for i, var_key in enumerate(var_keys):
        if np.isnan(val_array[i]): 
            opt_key = get_key_from_var(var_key)
            val_array[i] = optimal_values.get(opt_key, preprocessed_data['lb_array'][i])

    return val_array

def resample_single_variable(idx, val_array, p_data):
    var_key = p_data['idx2var'][idx]
    lb = p_data['lb_array'][idx]
    dynamic_ub = p_data['ub_array'][idx]

    for c_idx in p_data['constraints_of_var'][idx]:
        c = p_data['array_constraints'][c_idx]
        if c['sense'] == '<=':
            var_pos = np.where(c['var_idxs'] == idx)[0]
            if var_pos.size == 0: continue
            
            coeff = c['coefs'][var_pos[0]]
            if coeff > 1e-9: 
                other_vars_sum = np.dot(c['coefs'], val_array[c['var_idxs']]) - coeff * val_array[idx]
                potential_ub = (c['rhs'] - other_vars_sum) / coeff
                dynamic_ub = min(dynamic_ub, potential_ub)

    lo = lb
    hi = max(lb, dynamic_ub)
    if hi < lo: hi = lo 
    
    sampled_val = random.uniform(lo, hi)
    
    return max(lb, min(p_data['ub_array'][idx], sampled_val))



def simple_sampler(preprocessed_data):
    val_array = data_preprocessing(preprocessed_data, {})
    indices_to_sample = [i for i, ub in enumerate(preprocessed_data['ub_array']) if ub > 1e-9]

    var2idx = preprocessed_data['var2idx']
    random.shuffle(indices_to_sample)
    
    for idx in indices_to_sample:
        if np.isnan(val_array[idx]): continue 

        new_val = resample_single_variable(idx, val_array, preprocessed_data)
        val_array[idx] = new_val

        var_type, idx_tuple = preprocessed_data['idx2var'][idx]
        if var_type in ('storPWInvCap', 'storENInvCap') and idx_tuple[1] == "Li-Ion_BESS":
            if var_type == 'storPWInvCap':
                paired_key = ('storENInvCap', (idx_tuple[0], idx_tuple[1], idx_tuple[2]))
                factor = 2.0
            else: # storENInvCap
                paired_key = ('storPWInvCap', (idx_tuple[0], idx_tuple[1], idx_tuple[2]))
                factor = 0.5
            
            paired_idx = var2idx.get(paired_key)
            if paired_idx and paired_idx in indices_to_sample:
                val_array[paired_idx] = new_val * factor
    
    return {preprocessed_data['idx2var'][i]: val for i, val in enumerate(val_array)}





def check_sample_feasibility(sampled_values, preprocessed_data, tol=1e-6):
    var_keys = preprocessed_data['var_keys']
    array_constraints = preprocessed_data['array_constraints']
    feasible = True

    for c in array_constraints:
        var_idxs = c['var_idxs']
        coefs = c['coefs']
        sense = c['sense']
        rhs = c['rhs']
        lhs_value = 0.0

        for i, var_idx in enumerate(var_idxs):
            var_key = var_keys[var_idx]
            if var_key not in sampled_values:
                print(f"Variable {var_key} not found in the sampled values.")
                feasible = False
                continue
            lhs_value += coefs[i] * sampled_values[var_key]
        
        if sense == '<=':
            if lhs_value > rhs + tol:
                print(f"Constraint violated (<=): LHS = {lhs_value} > RHS = {rhs} (Constraint type: {c['type']})")
                feasible = False
        elif sense == '>=':
            if lhs_value < rhs - tol:
                print(f"Constraint violated (>=): LHS = {lhs_value} < RHS = {rhs} (Constraint type: {c['type']})")
                feasible = False
        else:
            print(f"Unsupported constraint sense '{sense}' encountered.")
            feasible = False

    return feasible

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


def sampling(instance, start_sample_num, base_dir, max_attempts):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    logging.info("Preprocessing constraints & variables just once...")
    preprocessed = build_preprocessed_data(instance)
    logging.info("Done Preprocessing!")

    feasible_samples = 0
    total_attempts = 0

    while total_attempts < max_attempts:
        total_attempts += 1
        logging.info(f"Attempt {total_attempts}/{max_attempts} | Generating a sample with Simple Sampler...")
        
        sampled_values = simple_sampler(preprocessed_data=preprocessed)
        
        if sampled_values is None:
            logging.warning("Sampling failed for this attempt.")    
            continue

        is_feasible = check_sample_feasibility(sampled_values, preprocessed)
        
        if is_feasible:
            feasible_samples += 1
            logging.info(f"Sample {start_sample_num + feasible_samples} is feasible!")
            fsd = build_sample_for_checking(sampled_values)
            output_file = os.path.join(base_dir, f'sample_{int(start_sample_num + feasible_samples)}.csv')
            fsd.to_csv(output_file, index=False)
            logging.info(f"Feasible sample saved to {output_file}")
        else:
            logging.error("A sample from Simple sampler was found to be infeasible. Check implementation.")

    final_ratio = feasible_samples / total_attempts if total_attempts > 0 else 0
    logging.info("\n==================== Experiment Complete ====================")
    logging.info(f"Generated {feasible_samples} feasible samples out of {total_attempts} attempts.")
    logging.info(f"Final acceptance ratio: {final_ratio:.4f}")
    logging.info("===========================================================")



if __name__ == '__main__':
    UserRunTimeConfig = safe_load(open("config_run.yaml"))

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
    north_sea = False


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
    workbook_path = '../Data handler/' + version
    tab_file_path = '../Data handler/' + version + '/Tab_Files_' + name + f'_{SEED}'
    scenario_data_path = '../Data handler/' + version + '/ScenarioData'
    result_file_path = '../Results/' + name
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
    
    
    logging.info("Now creating instance")
    
    
    instance = run_first_stage(version, tab_file_path, result_file_path, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
            lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
            discountrate, WACC, LeapYearsInvestment, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, north_sea)

    logging.info("Now instance created")

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, required=True, help='num_samples') # it should be integer (e.g. 1000)
    parser.add_argument('--seed', type=int, required=True, help='seed') # it should be integer (e.g. 42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    base_dir = f'DataSamples_{args.num_samples}_{args.seed}'
    logging.info("The created samples will be stored at %s directory", base_dir)
    start_sample_num = 0
    
    logging.info(f"Starting sampling with Seed: {args.seed}, Num Samples: {args.num_samples}")
    start_time = time.time()

    sampling(instance, start_sample_num, base_dir, args.num_samples)

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Sampling process finished in {duration:.2f} seconds.")
    
    
    csv_filename = 'dataset_generation_log_adaptive.csv'
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(['seed', 'num_samples', 'sampling(s)', 'labeling(s)'])
            
        writer.writerow([args.seed, args.num_samples, f"{duration:.4f}", 0.0])

    logging.info(f"Results for seed {args.seed} appended to {csv_filename}")