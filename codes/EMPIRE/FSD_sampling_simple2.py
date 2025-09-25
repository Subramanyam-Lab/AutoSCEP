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


# def load_optimal_values_from_fsd(fsd_data):
#     opt_dict = {}
#     for row in fsd_data:
#         if len(row) < 5:
#             continue  # 혹시 컬럼이 부족한 행 무시
#         node, energy_type, period_str, var_type, val_str = row
#         period = int(period_str)
#         val = float(val_str)
#         # key = (node, energy_type, period, var_type)
#         opt_dict[(node, energy_type, period, var_type)] = val
#     return opt_dict

def load_optimal_values_from_fsd(fsd_data):
    opt_dict = {}
    for row in fsd_data:
        if len(row) < 5:
            continue
        
        node, energy_type, period_str, var_type_str, val_str = row
        period = int(period_str)
        val = float(val_str)

        # CSV의 var_type 문자열을 내부 key 형식으로 변환
        if var_type_str == "Generation":
            key = ('genInvCap', (node, energy_type, period))
        elif var_type_str == "Transmission":
            # energy_type에 두 번째 노드 이름이 들어감
            key = ('transmisionInvCap', (node, energy_type, period))
        elif var_type_str == "Storage Power":
            key = ('storPWInvCap', (node, energy_type, period))
        elif var_type_str == "Storage Energy":
            key = ('storENInvCap', (node, energy_type, period))
        else:
            continue # 해당 없는 타입은 무시

        opt_dict[key] = val
    return opt_dict



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




def get_opt_key_from_var(var_key):
    """내부 변수 키를 최적해 CSV 키 형식으로 변환"""
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

def initialize_sample_with_optimum(preprocessed_data, optimal_values):
    """최적해를 사용하여 샘플 배열을 초기화"""
    val_array = preprocessed_data['val_array'].copy()
    var_keys = preprocessed_data['var_keys']
    var2idx = preprocessed_data['var2idx']

    for i, var_key in enumerate(var_keys):
        if np.isnan(val_array[i]): # 아직 값이 할당되지 않은 변수만 처리
            opt_key = get_opt_key_from_var(var_key)
            # 최적해 딕셔너리에서 값을 찾고, 없으면 하한(주로 0)으로 설정
            val_array[i] = optimal_values.get(opt_key, preprocessed_data['lb_array'][i])

    return val_array

# def resample_single_variable(idx, val_array, p_data):
#     var_key = p_data['idx2var'][idx]
#     lb = p_data['lb_array'][idx]
#     dynamic_ub = p_data['ub_array'][idx]

#     # 이 변수가 포함된 제약조건들을 순회하며 동적 상한 계산
#     for c_idx in p_data['constraints_of_var'][idx]:
#         c = p_data['array_constraints'][c_idx]
#         if c['sense'] == '<=':
#             var_pos = np.where(c['var_idxs'] == idx)[0]
#             if var_pos.size == 0: continue
            
#             coeff = c['coefs'][var_pos[0]]
#             if coeff > 1e-9: # 양수 계수일 때만 상한에 영향을 줌
#                 # 현재 변수를 제외한 나머지 변수들의 값 합산
#                 other_vars_sum = np.dot(c['coefs'], val_array[c['var_idxs']]) - coeff * val_array[idx]
                
#                 # 새로운 상한 계산
#                 potential_ub = (c['rhs'] - other_vars_sum) / coeff
#                 dynamic_ub = min(dynamic_ub, potential_ub)

#     lo = lb
#     hi = max(lb, dynamic_ub)
#     if hi < lo: hi = lo # 부동소수점 오류 등으로 상한이 하한보다 작아지는 경우 방지

#     # 단순 균등 샘플링
#     # sampled_val = random.uniform(lo, hi)
    
#     alpha = random.random()
#     if alpha >= 0.5:
#         f = random.betavariate(1-alpha, alpha)   # 평균 = 1/(1+3)=0.25, 잔여의 25%만 쓰는 경향
#     else:
#         f = random.betavariate(alpha, 1-alpha)

#     sampled_val = lo + f * (hi - lo)
    
#     # 최종적으로 변수 자체의 절대 상한을 넘지 않도록 보정
#     return max(lb, min(p_data['ub_array'][idx], sampled_val))



def resample_single_variable(idx, val_array, p_data, bias=None):
    var_key = p_data['idx2var'][idx]
    lb = p_data['lb_array'][idx]
    dynamic_ub = p_data['ub_array'][idx]

    # 이 변수가 포함된 제약조건들을 순회하며 동적 상한 계산
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

    sampled_val = 0.0
    if lo >= hi:
        sampled_val = lo
    elif bias == 'low':
        f = random.betavariate(1, 5)
        sampled_val = lo + f * (hi - lo)
    elif bias == 'extreme_low':
        ### 방법 2: 95% 확률로 0을 선택, 5% 확률로 작은 값 선택 ###
        if random.random() < 0.95:
            sampled_val = lo  # 대부분의 경우 하한(0)을 선택
        else:
            f = random.betavariate(1, 10) # 0에 매우 가깝게 편향된 분포
            sampled_val = lo + f * (hi - lo)
    else: # bias가 None이면 균등 샘플링
        sampled_val = random.uniform(lo, hi)

    return max(lb, min(p_data['ub_array'][idx], sampled_val))



def generate_convex_sample(known_x_star, preprocessed_data, alpha=0.99):
    logging.info("... (convex) generating a feasible random point using simple_sampler...")
    x_rand_dict = simple_sampler(preprocessed_data, iterations=100)
    
    x_new = {}
    var_keys = preprocessed_data['var_keys']
    var2idx = preprocessed_data['var2idx']
    lb_array = preprocessed_data['lb_array']
    ub_array = preprocessed_data['ub_array']

    for var_key in var_keys:
        idx = var2idx[var_key]
        
        # known_x_star에 값이 있는지 확인, 없으면 하한값 사용
        val_star = known_x_star.get(var_key, lb_array[idx])
        
        # simple_sampler로 생성한 랜덤 샘플의 값
        val_rand = x_rand_dict.get(var_key, lb_array[idx])
        
        # Convex Combination 수식 적용
        new_val = alpha * val_star + (1 - alpha) * val_rand
        
        # 최종 값은 변수의 원래 경계를 넘지 않도록 보정
        x_new[var_key] = max(lb_array[idx], min(ub_array[idx], new_val))

    return x_new



# def simple_sampler(preprocessed_data, iterations=100):
#     # 1. 최적해로 샘플 배열 초기화
#     val_array = initialize_sample_with_optimum(preprocessed_data, {})
    
#     # 샘플링할 변수들의 인덱스 목록 (상한이 0보다 큰 변수만)
#     indices_to_sample = [i for i, ub in enumerate(preprocessed_data['ub_array']) if ub > 1e-9]

#     var2idx = preprocessed_data['var2idx']

#     # 2. Gibbs 샘플링 반복 (Mixing/Burn-in)
#     for i in range(iterations):
#         # 매 반복마다 순서를 섞어주어 시스템적 편향을 방지
#         random.shuffle(indices_to_sample)
        
#         for idx in indices_to_sample:
#             # 변수가 이미 다른 변수(예: Li-Ion)에 의해 설정되었다면 건너뛰기
#             if np.isnan(val_array[idx]): continue 

#             # 단일 변수 리샘플링
#             new_val = resample_single_variable(idx, val_array, preprocessed_data)
#             val_array[idx] = new_val

#             # Li-Ion 배터리 특별 처리 로직
#             var_type, idx_tuple = preprocessed_data['idx2var'][idx]
#             if var_type in ('storPWInvCap', 'storENInvCap') and idx_tuple[1] == "Li-Ion_BESS":
#                 if var_type == 'storPWInvCap':
#                     paired_key = ('storENInvCap', (idx_tuple[0], idx_tuple[1], idx_tuple[2]))
#                     factor = 2.0
#                 else: # storENInvCap
#                     paired_key = ('storPWInvCap', (idx_tuple[0], idx_tuple[1], idx_tuple[2]))
#                     factor = 0.5
                
#                 paired_idx = var2idx.get(paired_key)
#                 if paired_idx and paired_idx in indices_to_sample:
#                     # 쌍이 되는 변수 값을 비율에 맞춰 강제 업데이트
#                     val_array[paired_idx] = new_val * factor
    
#     # 최종적으로 생성된 샘플을 딕셔너리 형태로 변환하여 반환
#     return {preprocessed_data['idx2var'][i]: val for i, val in enumerate(val_array)}



def simple_sampler(preprocessed_data, iterations=100, bias=None):
    val_array = initialize_sample_with_optimum(preprocessed_data, {})
    
    indices_to_sample = [i for i, ub in enumerate(preprocessed_data['ub_array']) if ub > 1e-9]

    var2idx = preprocessed_data['var2idx']

    for i in range(iterations):
        random.shuffle(indices_to_sample)
        
        for idx in indices_to_sample:
            if not np.isnan(val_array[idx]):
                # bias 인자를 전달
                new_val = resample_single_variable(idx, val_array, preprocessed_data, bias=bias)
                val_array[idx] = new_val

            var_type, idx_tuple = preprocessed_data['idx2var'][idx]
            if var_type in ('storPWInvCap', 'storENInvCap') and idx_tuple[1] == "Li-Ion_BESS":
                if var_type == 'storPWInvCap':
                    paired_key = ('storENInvCap', (idx_tuple[0], idx_tuple[1], idx_tuple[2]))
                    factor = 2.0
                else:
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


# def sampling(instance, start_sample_num, base_dir, max_attempts):
    
#     if not os.path.exists(base_dir):
#         os.makedirs(base_dir)

#     logging.info("Preprocessing constraints & variables just once...")
#     preprocessed = build_preprocessed_data(instance)
#     logging.info("Done Preprocessing!")
    
    
#     ###################### new #####################
    
#     optimal_csv_path = '100_seed_5_inv_cap.csv' # x* 데이터가 있는 파일 경로
#     logging.info(f"Loading optimal solution from {optimal_csv_path}...")
#     fsd_data = read_fsd_from_csv(optimal_csv_path)
#     known_x_star = load_optimal_values_from_fsd(fsd_data)
#     logging.info(f"Loaded {len(known_x_star)} known variable values for convex sampling.")

#     convex_sample_ratio = 0.3
#     logging.info(f"Mixing convex samples and simple samples at a ratio of {convex_sample_ratio * 100:.0f}:{100 - convex_sample_ratio * 100:.0f}.")
    
#     ################################################
#     feasible_samples = 0
#     total_attempts = 0

#     while total_attempts < max_attempts:
#         total_attempts += 1
#         logging.info(f"Attempt {total_attempts}/{max_attempts}")
        
#         # sampled_values = simple_sampler(
#         #     preprocessed_data=preprocessed,
#         #     iterations=100
#         # )
        
        
#         ######################### new ########################
#         if random.random() < convex_sample_ratio:
#             logging.info("...generating sample using Convex Combination.")
#             sampled_values = generate_convex_sample(
#                 known_x_star=known_x_star,
#                 preprocessed_data=preprocessed,
#                 alpha=0.7
#             )
#         else:
#             logging.info("...generating sample using simple sampler.")
#             sampled_values = simple_sampler(
#                 preprocessed_data=preprocessed,
#                 iterations=100
#             )
            
        
#         #######################################################
        
#         if sampled_values is None:
#             logging.warning("Sampling failed for this attempt.")
#             continue

#         # 생성된 샘플의 타당성 검증 (이론적으로는 필요 없지만, 최종 확인용)
#         is_feasible = check_sample_feasibility(sampled_values, preprocessed)
        
#         if is_feasible:
#             feasible_samples += 1
#             logging.info(f"Sample {start_sample_num + feasible_samples} is feasible!")
#             fsd = build_sample_for_checking(sampled_values)
#             output_file = os.path.join(base_dir, f'sample_{int(start_sample_num + feasible_samples)}.csv')
#             fsd.to_csv(output_file, index=False)
#             logging.info(f"Feasible sample saved to {output_file}")
#         else:
#             logging.error("A sample was found to be infeasible. Check implementation.")

#     final_ratio = feasible_samples / total_attempts if total_attempts > 0 else 0
#     logging.info("\n==================== Experiment Complete ====================")
#     logging.info(f"Generated {feasible_samples} feasible samples out of {total_attempts} attempts.")
#     logging.info(f"Final acceptance ratio: {final_ratio:.4f}")
#     logging.info("===========================================================")


def sampling(instance, start_sample_num, base_dir, max_attempts):
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    logging.info("Preprocessing constraints & variables just once...")
    preprocessed = build_preprocessed_data(instance)
    logging.info("Done Preprocessing!")
    
    optimal_csv_path = '100_seed_5_inv_cap.csv'
    logging.info(f"Loading optimal solution from {optimal_csv_path}...")
    fsd_data = read_fsd_from_csv(optimal_csv_path)
    known_x_star = load_optimal_values_from_fsd(fsd_data)
    logging.info(f"Loaded {len(known_x_star)} known variable values for convex sampling.")

    # ==================== 샘플링 전략 비율 설정 ====================
    # 10%: 최적해 근방 탐색, 40%: 과소투자 탐색, 50%: 일반 탐색
    near_optimal_ratio = 0.1
    underinvestment_ratio = 0.3
    broad_ratio = 0.5  
    extreme_sparse_ratio = 0.1 
    # =============================================================
    
    logging.info(f"Sampling Profile Ratios -> Near-Optimal: {near_optimal_ratio*100:.0f}%, Under-investment: {underinvestment_ratio*100:.0f}%, Broad: {(1-near_optimal_ratio-underinvestment_ratio)*100:.0f}%")
    
    feasible_samples = 0
    total_attempts = 0

    while total_attempts < max_attempts:
        total_attempts += 1
        logging.info(f"Attempt {total_attempts}/{max_attempts}")
        
        # ==================== 확률적으로 샘플링 전략 선택 ====================
        strategy_choice = random.random()
        
        if strategy_choice < near_optimal_ratio:
            # 1. 프로필 1: 최적해 근방 탐색
            logging.info("...generating sample using Convex Combination (Near-Optimal).")
            sampled_values = generate_convex_sample(
                known_x_star=known_x_star,
                preprocessed_data=preprocessed,
                alpha=0.8  # alpha 값을 조금 낮춰 탐색 반경을 넓힐 수도 있음
            )
        elif strategy_choice < near_optimal_ratio + underinvestment_ratio:
            # 2. 프로필 2: 과소투자 집중 탐색
            logging.info("...generating sample using biased sampler (Under-investment Focused).")
            sampled_values = simple_sampler(
                preprocessed_data=preprocessed,
                iterations=100,
                bias='low' # 낮은 값으로 편향
            )
        elif strategy_choice < near_optimal_ratio + underinvestment_ratio + broad_ratio:
            # 3. 프로필 3: 넓은 영역 탐색 (편향 없음)
            logging.info("...generating sample using simple sampler (Broad Exploration).")
            sampled_values = simple_sampler(
                preprocessed_data=preprocessed,
                iterations=100,
                bias=None # 균등 분포 샘플링
            )
        
        elif strategy_choice < near_optimal_ratio + underinvestment_ratio + broad_ratio + extreme_sparse_ratio:
            ### 방법 2: '거의 0' 샘플 생성 (아래 설명) ###
            logging.info("...generating a 'nearly zero' sparse sample.")
            sampled_values = simple_sampler(
                preprocessed_data=preprocessed,
                iterations=100,
                bias='extreme_low' # 새로운 bias 모드
            )
        else:
            ### 방법 1: '순수 0' 샘플 직접 주입 ###
            logging.info("...injecting a deterministic 'pure zero' sample.")
            # 모든 변수 키에 대해 값을 0.0으로 설정하는 딕셔너리를 생성
            sampled_values = {key: 0.0 for key in preprocessed['var_keys']}
            
            
            
        # =================================================================
        
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
            logging.error("A sample was found to be infeasible. Check implementation.")

    final_ratio = feasible_samples / total_attempts if total_attempts > 0 else 0
    logging.info("\n==================== Experiment Complete ====================")
    logging.info(f"Generated {feasible_samples} feasible samples out of {total_attempts} attempts.")
    logging.info(f"Final acceptance ratio: {final_ratio:.4f}")
    logging.info("===========================================================")




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
    base_dir = 'DataSamples_EMPIRE6'

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
