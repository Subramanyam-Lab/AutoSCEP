from __future__ import division
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager
import csv
from datetime import datetime
import os
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from FSD_sampling_violation import run_first_stage
from pyomo.environ import *
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
from yaml import safe_load
import random
import logging
import math


# Configure logging at the beginning (adjust level and format as needed)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%d/%m/%Y %H:%M')


def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data


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


def build_preprocessed_data(instance):
    bounds_dict = get_inv_cap_bounds(instance)

    # Create variables dictionary only for investment capacities
    # 각 변수마다 lb, ub, value, 그리고 연결된 constraints 인덱스 리스트를 저장
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

    # max_installed_gen_cap_with_lifetime
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
                        total_init_cap += value(instance.genInitCap[n, g, i])
                        for j in instance.PeriodActive:
                            if j >= startPeriod_int and j <= i:
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
    
    # max_installed_trans_cap
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

    # max_installed_storage_energy_cap
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

    # max_installed_power_energy_cap
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

        # Sum over each variable in the constraint using its coefficient and sampled value
        for i, var_idx in enumerate(var_idxs):
            var_key = var_keys[var_idx]
            if var_key not in sampled_values:
                print(f"Variable {var_key} not found in the sampled values.")
                feasible = False
                continue
            lhs_value += coefs[i] * sampled_values[var_key]
        
        # Check the constraint based on its sense
        if sense == '<=':
            if lhs_value > rhs + tol:
                print(f"Constraint violated (<=): LHS = {lhs_value} > RHS = {rhs} (Constraint type: {c['type']})")
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

    
    
def simplified_sampler(preprocessed_data):
    var_keys = preprocessed_data['var_keys']
    var2idx = preprocessed_data['var2idx']
    idx2var = {i: v for v, i in var2idx.items()}
    
    # Create local copies of arrays to modify during sampling
    lb_array = preprocessed_data['lb_array'].copy()
    ub_array = preprocessed_data['ub_array'].copy()
    val_array = preprocessed_data['val_array'].copy()

    # Deep copy constraints to modify them locally
    local_constraints = [{
        'var_idxs': c['var_idxs'].copy(),
        'coefs': c['coefs'].copy(),
        'sense': c['sense'],
        'rhs': c['rhs'],
        'updated_rhs': c['rhs'],
        'type': c.get('type', 'N/A')  # 'type' 키를 복사하고, 없을 경우 'N/A'로 처리
    } for c in preprocessed_data['array_constraints']]

    # Group unassigned variables
    indices_to_sample = [i for i, v in enumerate(val_array) if np.isnan(v)]
    
    transmission_indices = [i for i in indices_to_sample if var_keys[i][0] == 'transmisionInvCap']
    storage_indices = [i for i in indices_to_sample if var_keys[i][0] in ('storPWInvCap', 'storENInvCap')]
    gen_indices = [i for i in indices_to_sample if var_keys[i][0] == 'genInvCap']

    def sample_group(indices_group):
        local_indices_group = list(indices_group)

        while local_indices_group:
            # Pick the next variable to sample (RANDOM CHOICE)
            chosen_idx = random.choice(local_indices_group)
            local_indices_group.remove(chosen_idx) # Prevent re-sampling

            # Dynamically calculate the current tightest upper bound from all constraints
            current_ub = ub_array[chosen_idx]
            for c in local_constraints:
                # Find if the chosen variable is in this constraint
                pos = np.where(c['var_idxs'] == chosen_idx)[0]
                if pos.size > 0:
                    coeff = c['coefs'][pos[0]]
                    if c['sense'] == '<=' and coeff > 1e-9:
                        potential_ub = c['updated_rhs'] / coeff
                        current_ub = min(current_ub, potential_ub)
            
            # Update the upper bound if we found a tighter one
            ub_array[chosen_idx] = current_ub
            
            # Check for feasibility before sampling
            lo, hi = lb_array[chosen_idx], ub_array[chosen_idx]
            if lo > hi + 1e-9:
                # This path is infeasible
                logging.warning(f"  [Infeasible] Lower bound {lo:.2f} > Upper bound {hi:.2f}. Sampling failed for this path.")
                return False

            # Sample a value from the valid range
            x_val = random.uniform(lo, hi)
            val_array[chosen_idx] = x_val
            
            # Update the RHS of all constraints involving this newly sampled variable
            for c in local_constraints:
                pos = np.where(c['var_idxs'] == chosen_idx)[0]
                if pos.size > 0:
                    c['updated_rhs'] -= c['coefs'][pos[0]] * x_val
            
            # Handle special logic for Li-Ion battery pairing
            var_type, idx_tuple = var_keys[chosen_idx]
            if var_type in ('storPWInvCap', 'storENInvCap') and idx_tuple[1] == "Li-Ion_BESS":
                if var_type == 'storPWInvCap':
                    paired_key = ('storENInvCap', (idx_tuple[0], idx_tuple[1], idx_tuple[2]))
                    factor = 2.0
                else: # storENInvCap
                    paired_key = ('storPWInvCap', (idx_tuple[0], idx_tuple[1], idx_tuple[2]))
                    factor = 0.5
                
                paired_idx = var2idx.get(paired_key)
                if paired_idx and np.isnan(val_array[paired_idx]):
                    paired_val = x_val * factor
                    val_array[paired_idx] = paired_val
                    if paired_idx in local_indices_group:
                        local_indices_group.remove(paired_idx)
                    # Update constraints for the paired variable
                    for c in local_constraints:
                        pos = np.where(c['var_idxs'] == paired_idx)[0]
                        if pos.size > 0:
                            c['updated_rhs'] -= c['coefs'][pos[0]] * paired_val
        return True

    # Create a list of groups to be processed
    groups_to_process = [transmission_indices, storage_indices, gen_indices]
    
    # Shuffle the order of the groups (RANDOMIZED ORDER)
    random.shuffle(groups_to_process)
    
    # Sample each group in the new random order
    for group in groups_to_process:
        if not sample_group(group):
            return None # Sampling failed for this attempt
    
    # If successful, return the dictionary of sampled values
    return {var_keys[i]: val for i, val in enumerate(val_array)}



def sampling(instance,start_sample_num, base_dir, max_attempts):
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    logging.info("Preprocessing constraints & variables just once...")
    preprocessed = build_preprocessed_data(instance)
    logging.info("Done Preprocessing!")
    n_vars      = len(preprocessed['var_keys'])
    lb_array    = preprocessed['lb_array']
    ub_array    = preprocessed['ub_array']
    history     = {i: [] for i in range(n_vars)}


    feasible_samples = 0
    total_attempts = 0


    while total_attempts < max_attempts:
        total_attempts += 1
        logging.info("Attempt %d go!", total_attempts)
        
        
        sampled_values = simplified_sampler(preprocessed)
        
        for (var_type, idx_tuple), x_val in sampled_values.items():
            idx = preprocessed['var2idx'][(var_type, idx_tuple)]
            history[idx].append(x_val)
        # sampled_values = sample_with_preprocessed(preprocessed,distribution_result)
        is_feasible = check_sample_feasibility(sampled_values, preprocessed)
        if is_feasible:
            logging.info("The sample is feasible!")
            fsd = build_sample_for_checking(sampled_values)
        else:
            logging.info("The sample violates some constraints.")

        output_file = os.path.join(base_dir, f'sample_{int(start_sample_num + feasible_samples)}.csv')
        fsd.to_csv(output_file, index=False)
        logging.info("Sample %d is feasible. Saved as %s", start_sample_num + feasible_samples, output_file)
        feasible_samples += 1
        if total_attempts % 100 == 0:
            current_ratio = feasible_samples / total_attempts
            logging.info("Current accept-rejection ratio: %.6f", current_ratio)        

    final_ratio = feasible_samples / total_attempts if total_attempts > 0 else 0

    logging.info("Generated %d feasible samples.", feasible_samples)
    logging.info("Final accept-rejection ratio: %.6f", final_ratio)
    logging.info("Total attempts: %d", total_attempts)
    logging.info("\nExperiment complete.")







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
    if version in ["europe_v51","europe_reduced_v51"]:
        north_sea = True
    else:
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

    if version in ["europe_v51","europe_reduced_v51"]:
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
        
    else :
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
    
    
    data_folder = f'Data handler/base/{version}'
    base_dir = 'DataSamples_EMPIRE3'

    logging.info("The created samples will be stored at %s directory", base_dir)


    model,data = run_first_stage(name = name, 
            tab_file_path = data_folder,
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
            north_sea = north_sea,
            scenariopath = tab_file_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prob', type=int, required=True, help='zero_prob')
    args = parser.parse_args()
    start_sample_num = (args.prob)*1000
    logging.info("Now creating instance")
    instance = model.create_instance(data)
    logging.info("Now instance created")
    max_attempts = 1000
    sampling(instance,start_sample_num,base_dir,max_attempts)
