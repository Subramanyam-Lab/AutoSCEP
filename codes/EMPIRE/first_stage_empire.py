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
import tensorflow as tf
import pandas as pd
import numpy as np
import multiprocessing
import json
import torch
from mymodel import SimpleMLP

__author__ = "Stian Backe"
__license__ = "MIT"
__maintainer__ = "Stian Backe"
__email__ = "stian.backe@ntnu.no"



########################################### LR #########################################################

def embed_linear_regression_in_pyomo(abstract_model, lr_model, input_vars, output_var_name):

    model = abstract_model
    output_var = Var(domain=Reals, name=output_var_name)
    model.add_component(output_var_name, output_var)
    
    intercept = lr_model.intercept_
    coef = lr_model.coef_  
    def linear_pred_constraint_rule(m):
        return m.component(output_var_name) == intercept + sum(coef[j]*input_vars[j] for j in range(len(coef)))
    
    constraint_name = f"linear_regression_constraint"
    model.add_component(constraint_name, Constraint(expr=linear_pred_constraint_rule(model)))
    return output_var


def integrate_linear_regression_with_pyomo_model(lr_model_path, instance, output_var_name):
    lr_model = joblib.load(lr_model_path)
    n_features = len(list(instance.nn_inputs.keys()))
    input_vars = [instance.nn_inputs[i] for i in range(1, n_features+1)]
    output_var = embed_linear_regression_in_pyomo(instance, lr_model, input_vars, output_var_name)
    return output_var

########################################### LR #########################################################

########################################### DT #########################################################
from collections import defaultdict

def embed_decision_tree_in_pyomo(model, dt_model, input_vars, output_var_name,
                                 bigM=1e5, epsilon=1e-6):

    # Extract tree components
    tree_ = dt_model.tree_
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    features = tree_.feature
    thresholds = tree_.threshold

    print(n_nodes)
    print(children_left)
    print(children_right)
    print(features)
    print(thresholds)
    
    # Extract leaf node information
    node_values = tree_.value.squeeze(axis=1) if tree_.value.shape[2] == 1 else tree_.value.squeeze(axis=2)

    print(node_values)
    
    # Identify leaf nodes
    is_leaf = np.zeros(n_nodes, dtype=bool)
    for i in range(n_nodes):
        if children_left[i] == -1 and children_right[i] == -1:
            is_leaf[i] = True
    leaf_indices = np.where(is_leaf)[0]
    
    # Extract leaf values (handle both single and multi-output trees)
    if len(node_values.shape) > 1 and node_values.shape[1] > 1:
        # Multi-output case - take first output for now (could be extended)
        leaf_values = [node_values[idx, 0] for idx in leaf_indices]
    else:
        # Single output case
        leaf_values = [node_values[idx] for idx in leaf_indices]
    
    # 1) Create output variable
    output_var = Var(domain=Reals, name=output_var_name)
    model.add_component(output_var_name, output_var)

    # 2) Create binary variables for leaf selection
    leaf_var = {}
    for lf_idx in leaf_indices:
        var_name = f"leaf_var_{lf_idx}"
        leaf_var[lf_idx] = Var(domain=Binary, name=var_name)
        model.add_component(var_name, leaf_var[lf_idx])
    
    # 3) Ensure exactly one leaf is selected
    def single_leaf_rule(m):
        return sum(leaf_var[lf] for lf in leaf_indices) == 1
    model.single_leaf_constraint = Constraint(rule=single_leaf_rule)
    
    # 4) Link leaf selection to output value using Big-M constraints
    for lf_idx, lf_val in zip(leaf_indices, leaf_values):
        # Upper bound constraint
        c_ub = f"output_ub_leaf_{lf_idx}"
        model.add_component(
            c_ub,
            Constraint(expr=output_var <= lf_val + bigM*(1 - leaf_var[lf_idx]))
        )
        # Lower bound constraint
        c_lb = f"output_lb_leaf_{lf_idx}"
        model.add_component(
            c_lb,
            Constraint(expr=output_var >= lf_val - bigM*(1 - leaf_var[lf_idx]))
        )

    # 5) Add path constraints for each leaf
    # Build subtree cache for efficient path finding
    subtree_cache = defaultdict(set)
    
    def get_subtree_nodes(node):
        """Recursively get all nodes in the subtree rooted at 'node'"""
        if node == -1:
            return set()
        if node in subtree_cache:
            return subtree_cache[node]
        st = {node}
        st |= get_subtree_nodes(children_left[node])
        st |= get_subtree_nodes(children_right[node])
        subtree_cache[node] = st
        return st

    def path_to_leaf(leaf_id):
        """Find the path from root to leaf_id as a list of (node_id, direction) tuples"""
        path = []
        node_current = 0  # root node
        while node_current != leaf_id:
            if children_left[node_current] != -1 and leaf_id in get_subtree_nodes(children_left[node_current]):
                path.append((node_current, 'L'))
                node_current = children_left[node_current]
            else:
                path.append((node_current, 'R'))
                node_current = children_right[node_current]
        return path

    # Add constraints for each leaf node's path
    for lf_idx in leaf_indices:
        path = path_to_leaf(lf_idx)
        for (node_id, direction) in path:
            thr_val = thresholds[node_id]
            f_idx = features[node_id]
            
            # Skip if threshold is None or NaN
            if thr_val is None or (isinstance(thr_val, float) and np.isnan(thr_val)):
                continue
            
            try:
                # Verify feature index is valid
                _ = input_vars[f_idx]
            except (IndexError, KeyError):
                print(f"Warning: Feature index {f_idx} is out of bounds. Skipping constraint.")
                continue
                
            if direction == 'L':
                # Left branch: x[f_idx] <= thr_val
                cname = f"node_{node_id}_leaf_{lf_idx}_left"
                model.add_component(
                    cname,
                    Constraint(expr=input_vars[f_idx] <= thr_val + bigM*(1 - leaf_var[lf_idx]))
                )
            else:
                # Right branch: x[f_idx] > thr_val, approximated as x[f_idx] >= thr_val + epsilon
                cname = f"node_{node_id}_leaf_{lf_idx}_right"
                model.add_component(
                    cname,
                    Constraint(expr=input_vars[f_idx] >= thr_val + epsilon - bigM*(1 - leaf_var[lf_idx]))
                )
    
    return output_var


def integrate_decision_tree_with_pyomo_model(dt_model_path, instance, output_var_name,
                                             bigM=1e8, epsilon=1e-2):
    import joblib
    dt_model = joblib.load(dt_model_path)
    
    n_features = len(list(instance.nn_inputs.keys()))
    input_vars = [instance.nn_inputs[i] for i in range(1, n_features+1)]
    
    output_var = embed_decision_tree_in_pyomo(
        instance, dt_model, input_vars, output_var_name, bigM=bigM, epsilon=epsilon
    )
    return output_var



########################################### DT #########################################################


########################################### NN #########################################################
def extract_neural_network_params(model):
    weights = []
    biases = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.numpy())
        elif 'bias' in name:
            biases.append(param.data.numpy())
    
    return weights, biases


def embed_relu_network_in_pyomo(abstract_model, weights, biases, input_vars, output_var_name, M=1000):
    model = abstract_model
    num_layers = len(weights)
    
    # 은닉층 노드 변수 및 이진 변수 선언
    hidden_vars = []
    binary_vars = []
    pre_activation_vars = []
    
    # 각 층마다 변수 생성
    for layer_idx in range(num_layers):
        if layer_idx == 0:
            # 첫 번째 층의 입력 차원
            input_dim = weights[layer_idx].shape[1]
        else:
            # 이전 층의 출력 차원
            input_dim = weights[layer_idx-1].shape[0]
            
        # 현재 층의 출력 차원
        output_dim = weights[layer_idx].shape[0]
        
        # 활성화 이전 값을 위한 변수 (z)
        pre_vars = []
        for j in range(output_dim):
            var_name = f'z_{layer_idx}_{j}'
            var = Var(domain=Reals, name=var_name)
            model.add_component(var_name, var)
            pre_vars.append(var)
        pre_activation_vars.append(pre_vars)
        
        # 마지막 층이 아닌 경우 ReLU 활성화 함수를 위한 변수 추가
        if layer_idx < num_layers - 1:
            # 활성화 이후 값을 위한 변수 (a)
            h_vars = []
            b_vars = []
            
            for j in range(output_dim):
                # 활성화 이후 값
                a_name = f'a_{layer_idx}_{j}'
                a_var = Var(domain=NonNegativeReals, name=a_name)
                model.add_component(a_name, a_var)
                h_vars.append(a_var)
                
                # 이진 변수 (ReLU on/off 결정)
                b_name = f'b_{layer_idx}_{j}'
                b_var = Var(domain=Binary, name=b_name)
                model.add_component(b_name, b_var)
                b_vars.append(b_var)
            
            hidden_vars.append(h_vars)
            binary_vars.append(b_vars)
    
    # 제약조건 추가
    # 첫 번째 층 입력과 선형 결합 제약조건
    for j in range(len(pre_activation_vars[0])):
        expr = sum(weights[0][j, i] * input_vars[i] for i in range(len(input_vars)))
        expr += biases[0][j]
        
        constraint_name = f'first_layer_constraint_{j}'
        constraint = pre_activation_vars[0][j] == expr
        model.add_component(constraint_name, Constraint(expr=constraint))
    
    # 은닉층 사이의 연결과 ReLU 활성화 함수를 위한 제약조건
    for layer_idx in range(num_layers - 1):
        output_dim = weights[layer_idx].shape[0]
        
        for j in range(output_dim):
            # ReLU 제약조건 (Big-M 방법)
            # 1. a >= z
            constraint_name = f'relu_lb_{layer_idx}_{j}'
            constraint = hidden_vars[layer_idx][j] >= pre_activation_vars[layer_idx][j]
            model.add_component(constraint_name, Constraint(expr=constraint))
            
            # 2. a <= z + M(1-b)
            constraint_name = f'relu_ub1_{layer_idx}_{j}'
            constraint = hidden_vars[layer_idx][j] <= pre_activation_vars[layer_idx][j] + M * (1 - binary_vars[layer_idx][j])
            model.add_component(constraint_name, Constraint(expr=constraint))
            
            # 3. a <= Mb
            constraint_name = f'relu_ub2_{layer_idx}_{j}'
            constraint = hidden_vars[layer_idx][j] <= M * binary_vars[layer_idx][j]
            model.add_component(constraint_name, Constraint(expr=constraint))
            
        # 다음 층이 있는 경우, 입력 연결
        if layer_idx < num_layers - 2:
            next_dim = weights[layer_idx+1].shape[0]
            
            for j in range(next_dim):
                expr = sum(weights[layer_idx+1][j, i] * hidden_vars[layer_idx][i] for i in range(output_dim))
                expr += biases[layer_idx+1][j]
                
                constraint_name = f'hidden_layer_constraint_{layer_idx+1}_{j}'
                constraint = pre_activation_vars[layer_idx+1][j] == expr
                model.add_component(constraint_name, Constraint(expr=constraint))
    
    # 출력층 처리
    last_layer_idx = num_layers - 1
    
    if last_layer_idx > 0:
        # 이전 은닉층의 출력과 연결
        prev_dim = weights[last_layer_idx-1].shape[0]
        output_dim = weights[last_layer_idx].shape[0]
        
        for j in range(output_dim):
            expr = sum(weights[last_layer_idx][j, i] * hidden_vars[last_layer_idx-1][i] for i in range(prev_dim))
            expr += biases[last_layer_idx][j]
            
            constraint_name = f'output_layer_constraint_{j}'
            constraint = pre_activation_vars[last_layer_idx][j] == expr
            model.add_component(constraint_name, Constraint(expr=constraint))
    
    # 출력 변수 생성 (단일 출력 가정)
    output_var = Var(domain=Reals, name=output_var_name)
    model.add_component(output_var_name, output_var)
    
    # 출력 변수와 마지막 층 연결
    if len(pre_activation_vars[last_layer_idx]) == 1:
        # 단일 출력
        constraint_name = f'final_output_constraint'
        constraint = output_var == pre_activation_vars[last_layer_idx][0]
        model.add_component(constraint_name, Constraint(expr=constraint))
    else:
        # 다중 출력 (예: 첫 번째 출력만 사용)
        constraint_name = f'final_output_constraint'
        constraint = output_var == pre_activation_vars[last_layer_idx][0]
        model.add_component(constraint_name, Constraint(expr=constraint))
        
    return output_var


def integrate_nn_with_pyomo_model(pytorch_model, instance, output_var_name, M=1000):
    # 1. PyTorch 모델에서 파라미터 추출
    weights, biases = extract_neural_network_params(pytorch_model)
    
    # 2. 입력 변수 정의 (이미 instance.nn_inputs에 정의되어 있음)
    n_features = len(list(instance.nn_inputs.keys()))
    input_vars = [instance.nn_inputs[i] for i in range(1, n_features+1)]
    
    # 3. 신경망 임베딩
    output_var = embed_relu_network_in_pyomo(
        instance, weights, biases, input_vars, output_var_name, M
    )
    
    return output_var



########################################### NN #########################################################


def run_first_stage(name, tab_file_path, result_file_path, scenariogeneration, scenario_data_path,
               solver, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
               lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
               discountrate, WACC, LeapYearsInvestment, IAMC_PRINT, WRITE_LP,
               PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, north_sea):

    if USE_TEMP_DIR:
        TempfileManager.tempdir = temp_dir

    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    model = AbstractModel()

    
    ########
    ##SETS##
    ########


    #Define the sets


    #Supply technology sets
    model.Generator = Set(ordered=True) #g
    model.Technology = Set(ordered=True) #t
    model.Storage =  Set() #b

    #Temporal sets
    model.Period = Set(ordered=True) #max period,|I|,it becomes 8
    # model.PeriodActive = Set(initialize=period_filter)
    model.PeriodActive = Set(ordered=True, initialize=Period) #i
    model.Operationalhour = Set(ordered=True, initialize=Operationalhour) #h
    model.Season = Set(ordered=True, initialize=Season) #s

    #Spatial sets
    model.Node = Set(ordered=True) #n
    if north_sea:
        model.OffshoreNode = Set(ordered=True, within=model.Node) #n
    model.DirectionalLink = Set(dimen=2, within=model.Node*model.Node, ordered=True) #a
    model.TransmissionType = Set(ordered=True)

    #Stochastic sets
    model.Scenario = Set(ordered=True, initialize=Scenario) #w

    #Subsets
    model.GeneratorsOfTechnology=Set(dimen=2) #(t,g) for all t in T, g in G_t
    model.GeneratorsOfNode = Set(dimen=2) #(n,g) for all n in N, g in G_n
    model.TransmissionTypeOfDirectionalLink = Set(dimen=3) #(n1,n2,t) for all (n1,n2) in L, t in T
    model.ThermalGenerators = Set(within=model.Generator) #g_ramp
    model.RegHydroGenerator = Set(within=model.Generator) #g_reghyd
    model.HydroGenerator = Set(within=model.Generator) #g_hyd
    model.StoragesOfNode = Set(dimen=2) #(n,b) for all n in N, b in B_n
    model.DependentStorage = Set() #b_dagger
    model.HoursOfSeason = Set(dimen=2, ordered=True, initialize=HoursOfSeason) #(s,h) for all s in S, h in H_s
    model.FirstHoursOfRegSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfRegSeason) # begining hour of the Regular Season
    model.FirstHoursOfPeakSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfPeakSeason) # begining hour of the Peak Season

    #Load the data

    data = DataPortal()
    data.load(filename=tab_file_path + "/" + 'Sets_Generator.tab',format="set", set=model.Generator)
    data.load(filename=tab_file_path + "/" + 'Sets_ThermalGenerators.tab',format="set", set=model.ThermalGenerators)
    data.load(filename=tab_file_path + "/" + 'Sets_HydroGenerator.tab',format="set", set=model.HydroGenerator)
    data.load(filename=tab_file_path + "/" + 'Sets_HydroGeneratorWithReservoir.tab',format="set", set=model.RegHydroGenerator)
    data.load(filename=tab_file_path + "/" + 'Sets_Storage.tab',format="set", set=model.Storage)
    data.load(filename=tab_file_path + "/" + 'Sets_DependentStorage.tab',format="set", set=model.DependentStorage)
    data.load(filename=tab_file_path + "/" + 'Sets_Technology.tab',format="set", set=model.Technology)
    data.load(filename=tab_file_path + "/" + 'Sets_Node.tab',format="set", set=model.Node)
    if north_sea:
        data.load(filename=tab_file_path + "/" + 'Sets_OffshoreNode.tab',format="set", set=model.OffshoreNode)
    data.load(filename=tab_file_path + "/" + 'Sets_Horizon.tab',format="set", set=model.Period)
    data.load(filename=tab_file_path + "/" + 'Sets_DirectionalLines.tab',format="set", set=model.DirectionalLink)
    data.load(filename=tab_file_path + "/" + 'Sets_LineType.tab',format="set", set=model.TransmissionType)
    data.load(filename=tab_file_path + "/" + 'Sets_LineTypeOfDirectionalLines.tab',format="set", set=model.TransmissionTypeOfDirectionalLink)
    data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfTechnology.tab',format="set", set=model.GeneratorsOfTechnology)
    data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfNode.tab',format="set", set=model.GeneratorsOfNode)
    data.load(filename=tab_file_path + "/" + 'Sets_StorageOfNodes.tab',format="set", set=model.StoragesOfNode)


    #Build arc subsets

    def NodesLinked_init(model, node):
        retval = []
        for (i,j) in model.DirectionalLink:
            if j == node:
                retval.append(i)
        return retval
    model.NodesLinked = Set(model.Node, initialize=NodesLinked_init)

    def BidirectionalArc_init(model):
        retval = []
        for (i,j) in model.DirectionalLink:
            if i != j and (not (j,i) in retval):
                retval.append((i,j))
        return retval
    model.BidirectionalArc = Set(dimen=2, initialize=BidirectionalArc_init, ordered=True) #l

    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    #Scaling

    model.discountrate = Param(initialize=discountrate)
    model.WACC = Param(initialize=WACC)
    model.LeapYearsInvestment = Param(initialize=LeapYearsInvestment) # 5 year
    model.operationalDiscountrate = Param(mutable=True)
    model.sceProbab = Param(model.Scenario, mutable=True) # pi_w, \sum pi_w =1
    model.seasScale = Param(model.Season, initialize=1.0, mutable=True) # \alpha_s
    model.lengthRegSeason = Param(initialize=lengthRegSeason) # 168
    model.lengthPeakSeason = Param(initialize=lengthPeakSeason) # 48

    #Cost

    model.genCapitalCost = Param(model.Generator, model.Period, default=0, mutable=True)
    model.transmissionTypeCapitalCost = Param(model.TransmissionType, model.Period, default=0, mutable=True)
    model.storPWCapitalCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.storENCapitalCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.genFixedOMCost = Param(model.Generator, model.Period, default=0, mutable=True)
    model.transmissionTypeFixedOMCost = Param(model.TransmissionType, model.Period, default=0, mutable=True)
    model.storPWFixedOMCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.storENFixedOMCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.genInvCost = Param(model.Generator, model.Period, default=9000000, mutable=True)
    model.transmissionInvCost = Param(model.BidirectionalArc, model.Period, default=3000000, mutable=True)
    model.storPWInvCost = Param(model.Storage, model.Period, default=1000000, mutable=True)
    model.storENInvCost = Param(model.Storage, model.Period, default=800000, mutable=True)
    model.transmissionLength = Param(model.BidirectionalArc, default=0, mutable=True)
    model.genVariableOMCost = Param(model.Generator, default=0.0, mutable=True)
    model.genFuelCost = Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genMargCost = Param(model.Generator, model.Period, default=600, mutable=True)
    model.genCO2TypeFactor = Param(model.Generator, default=0.0, mutable=True)
    # model.nodeLostLoadCost = Param(model.Node, model.Period, default=22000.0)
    model.nodeLostLoadCost = Param(model.Node, model.Period, default=1e+8, mutable=False)
    model.CO2price = Param(model.Period, default=0.0, mutable=True)
    model.CCSCostTSFix = Param(initialize=1149873.72) #NB! Hard-coded
    model.CCSCostTSVariable = Param(model.Period, default=0.0, mutable=True)
    model.CCSRemFrac = Param(initialize=0.9)

    #Node dependent technology limitations

    model.genRefInitCap = Param(model.GeneratorsOfNode, default=0.0, mutable=True)
    model.genScaleInitCap = Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genInitCap = Param(model.GeneratorsOfNode, model.Period, default=0.0, mutable=True)
    model.transmissionInitCap = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWInitCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENInitCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.genMaxBuiltCap = Param(model.Node, model.Technology, model.Period, default=500000.0, mutable=True)
    model.transmissionMaxBuiltCap = Param(model.BidirectionalArc, model.Period, default=20000.0, mutable=True)
    model.storPWMaxBuiltCap = Param(model.StoragesOfNode, model.Period, default=500000.0, mutable=True)
    model.storENMaxBuiltCap = Param(model.StoragesOfNode, model.Period, default=500000.0, mutable=True)
    model.genMaxInstalledCapRaw = Param(model.Node, model.Technology, default=0.0, mutable=True)
    model.genMaxInstalledCap = Param(model.Node, model.Technology, model.Period, default=0.0, mutable=True)
    model.transmissionMaxInstalledCapRaw = Param(model.BidirectionalArc, model.Period, default=0.0)
    model.transmissionMaxInstalledCap = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWMaxInstalledCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storPWMaxInstalledCapRaw = Param(model.StoragesOfNode, default=0.0, mutable=True)
    model.storENMaxInstalledCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENMaxInstalledCapRaw = Param(model.StoragesOfNode, default=0.0, mutable=True)

    #Type dependent technology limitations

    model.genLifetime = Param(model.Generator, default=0.0, mutable=True)
    model.transmissionLifetime = Param(model.BidirectionalArc, default=40.0, mutable=True)
    model.storageLifetime = Param(model.Storage, default=0.0, mutable=True)
    model.genEfficiency = Param(model.Generator, model.Period, default=1.0, mutable=True)
    model.lineEfficiency = Param(model.DirectionalLink, default=0.97, mutable=True)
    model.storageChargeEff = Param(model.Storage, default=1.0, mutable=True)
    model.storageDischargeEff = Param(model.Storage, default=1.0, mutable=True)
    model.storageBleedEff = Param(model.Storage, default=1.0, mutable=True)
    model.genRampUpCap = Param(model.ThermalGenerators, default=0.0, mutable=True)
    model.storageDiscToCharRatio = Param(model.Storage, default=1.0, mutable=True) #NB! Hard-coded
    model.storagePowToEnergy = Param(model.DependentStorage, default=1.0, mutable=True)

    #Stochastic input

    model.sloadRaw = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    model.sloadAnnualDemand = Param(model.Node, model.Period, default=0.0, mutable=True)
    model.sload = Param(model.Node, model.Operationalhour, model.Period, model.Scenario, default=0.0, mutable=True)
    model.genCapAvailTypeRaw = Param(model.Generator, default=1.0, mutable=True)
    model.genCapAvailStochRaw = Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    model.genCapAvail = Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    model.maxRegHydroGenRaw = Param(model.Node, model.Period, model.HoursOfSeason, model.Scenario, default=0.0, mutable=True)
    model.maxRegHydroGen = Param(model.Node, model.Period, model.Season, model.Scenario, default=0.0, mutable=True)
    model.maxHydroNode = Param(model.Node, default=0.0, mutable=True)
    model.storOperationalInit = Param(model.Storage, default=0.0, mutable=True) #Percentage of installed energy capacity initially

    if EMISSION_CAP:
        model.CO2cap = Param(model.Period, default=5000.0, mutable=True)

    if LOADCHANGEMODULE:
        model.sloadMod = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)



    #Load the parameters

    data.load(filename=tab_file_path + "/" + 'Generator_CapitalCosts.tab', param=model.genCapitalCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_FixedOMCosts.tab', param=model.genFixedOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_VariableOMCosts.tab', param=model.genVariableOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_FuelCosts.tab', param=model.genFuelCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_CCSCostTSVariable.tab', param=model.CCSCostTSVariable, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_Efficiency.tab', param=model.genEfficiency, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_RefInitialCap.tab', param=model.genRefInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_ScaleFactorInitialCap.tab', param=model.genScaleInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_InitialCapacity.tab', param=model.genInitCap, format="table") #node_generator_intial_capacity.xlsx
    data.load(filename=tab_file_path + "/" + 'Generator_MaxBuiltCapacity.tab', param=model.genMaxBuiltCap, format="table")#?
    data.load(filename=tab_file_path + "/" + 'Generator_MaxInstalledCapacity.tab', param=model.genMaxInstalledCapRaw, format="table")#maximum_capacity_constraint_040317_high
    data.load(filename=tab_file_path + "/" + 'Generator_CO2Content.tab', param=model.genCO2TypeFactor, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_RampRate.tab', param=model.genRampUpCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_GeneratorTypeAvailability.tab', param=model.genCapAvailTypeRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_Lifetime.tab', param=model.genLifetime, format="table")

    data.load(filename=tab_file_path + "/" + 'Transmission_InitialCapacity.tab', param=model.transmissionInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_MaxBuiltCapacity.tab', param=model.transmissionMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_MaxInstallCapacityRaw.tab', param=model.transmissionMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_Length.tab', param=model.transmissionLength, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_TypeCapitalCost.tab', param=model.transmissionTypeCapitalCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_TypeFixedOMCost.tab', param=model.transmissionTypeFixedOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_lineEfficiency.tab', param=model.lineEfficiency, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_Lifetime.tab', param=model.transmissionLifetime, format="table")

    data.load(filename=tab_file_path + "/" + 'Storage_StorageBleedEfficiency.tab', param=model.storageBleedEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageChargeEff.tab', param=model.storageChargeEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageDischargeEff.tab', param=model.storageDischargeEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StoragePowToEnergy.tab', param=model.storagePowToEnergy, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyCapitalCost.tab', param=model.storENCapitalCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyFixedOMCost.tab', param=model.storENFixedOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyInitialCapacity.tab', param=model.storENInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyMaxBuiltCapacity.tab', param=model.storENMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyMaxInstalledCapacity.tab', param=model.storENMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageInitialEnergyLevel.tab', param=model.storOperationalInit, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_PowerCapitalCost.tab', param=model.storPWCapitalCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_PowerFixedOMCost.tab', param=model.storPWFixedOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_InitialPowerCapacity.tab', param=model.storPWInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_PowerMaxBuiltCapacity.tab', param=model.storPWMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_PowerMaxInstalledCapacity.tab', param=model.storPWMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_Lifetime.tab', param=model.storageLifetime, format="table")

    # data.load(filename=tab_file_path + "/" + 'Node_NodeLostLoadCost.tab', param=model.nodeLostLoadCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Node_ElectricAnnualDemand.tab', param=model.sloadAnnualDemand, format="table")
    data.load(filename=tab_file_path + "/" + 'Node_HydroGenMaxAnnualProduction.tab', param=model.maxHydroNode, format="table")
   

    model.exp_sload = Param(model.Node, model.Period, model.Operationalhour, default=0.0, mutable = True)
    model.avg_cap_avail = Param(model.GeneratorsOfNode, model.Operationalhour, model.Period, default=0.0, mutable=True)
    data.load(filename=f'Data handler/base/reduced/Average_sload_{str(lengthRegSeason)}_2.tab', param=model.exp_sload, format="table")
    data.load(filename=f'Data handler/base/reduced/Average_Cap_Avail_{str(lengthRegSeason)}.tab', param=model.avg_cap_avail, format="table")


    # scenariopath = f'Data handler/scenarios_PH/5/10'

    # data.load(filename=scenariopath + "/" + 'Stochastic_HydroGenMaxSeasonalProduction.tab', param=model.maxRegHydroGenRaw, format="table")
    # data.load(filename=scenariopath + "/" + 'Stochastic_StochasticAvailability.tab', param=model.genCapAvailStochRaw, format="table")
    # data.load(filename=scenariopath + "/" + 'Stochastic_ElectricLoadRaw.tab', param=model.sloadRaw, format="table")

    data.load(filename=tab_file_path + "/" + 'General_seasonScale.tab', param=model.seasScale, format="table")

    if EMISSION_CAP:
        data.load(filename=tab_file_path + "/" + 'General_CO2Cap.tab', param=model.CO2cap, format="table")
    else:
        data.load(filename=tab_file_path + "/" + 'General_CO2Price.tab', param=model.CO2price, format="table")

    # if LOADCHANGEMODULE:
    #     data.load(filename=scenariopath + "/" + 'LoadchangeModule/Stochastic_ElectricLoadMod.tab', param=model.sloadMod, format="table")


    # It means that the probability of scenarios is equally same regardless of scenario, and the expected second stage value is just average of second stage value.
    def prepSceProbab_rule(model):
        #Build an equiprobable probability distribution for scenarios

        for sce in model.Scenario:
            model.sceProbab[sce] = value(1/len(model.Scenario))

    model.build_SceProbab = BuildAction(rule=prepSceProbab_rule)


    # This function consists the costs per period for each generator, storage, transmission
    def prepInvCost_rule(model):
        #Build investment cost for generators, storages and transmission. Annual cost is calculated for the lifetime of the generator and discounted for a year.
        #Then cost is discounted for the investment period (or the remaining lifetime). CCS generators has additional fixed costs depending on emissions.

        #Generator
        for g in model.Generator:
            for i in model.PeriodActive:
                costperyear=(model.WACC/(1-((1+model.WACC)**(-model.genLifetime[g]))))*model.genCapitalCost[g,i]+model.genFixedOMCost[g,i]
                costperperiod=costperyear*1000*(1-(1+model.discountrate)**-(min(value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), value(model.genLifetime[g]))))/(1-(1/(1+model.discountrate)))
                if ('CCS',g) in model.GeneratorsOfTechnology:
                    costperperiod+=model.CCSCostTSFix*model.CCSRemFrac*model.genCO2TypeFactor[g]*(3.6/model.genEfficiency[g,i])
                model.genInvCost[g,i]=costperperiod

        #Storage
        for b in model.Storage:
            for i in model.PeriodActive:
                costperyearPW=(model.WACC/(1-((1+model.WACC)**(-model.storageLifetime[b]))))*model.storPWCapitalCost[b,i]+model.storPWFixedOMCost[b,i]
                costperperiodPW=costperyearPW*1000*(1-(1+model.discountrate)**-(min(value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), value(model.storageLifetime[b]))))/(1-(1/(1+model.discountrate)))
                model.storPWInvCost[b,i]=costperperiodPW
                costperyearEN=(model.WACC/(1-((1+model.WACC)**(-model.storageLifetime[b]))))*model.storENCapitalCost[b,i]+model.storENFixedOMCost[b,i]
                costperperiodEN=costperyearEN*1000*(1-(1+model.discountrate)**-(min(value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), value(model.storageLifetime[b]))))/(1-(1/(1+model.discountrate)))
                model.storENInvCost[b,i]=costperperiodEN

        #Transmission
        for (n1,n2) in model.BidirectionalArc:
            for i in model.PeriodActive:
                for t in model.TransmissionType:
                    if (n1,n2,t) in model.TransmissionTypeOfDirectionalLink:
                        costperyear=(model.WACC/(1-((1+model.WACC)**(-model.transmissionLifetime[n1,n2]))))*model.transmissionLength[n1,n2]*model.transmissionTypeCapitalCost[t,i]+model.transmissionTypeFixedOMCost[t,i]
                        costperperiod=costperyear*(1-(1+model.discountrate)**-(min(value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), value(model.transmissionLifetime[n1,n2]))))/(1-(1/(1+model.discountrate)))
                        model.transmissionInvCost[n1,n2,i]=costperperiod

    model.build_InvCost = BuildAction(rule=prepInvCost_rule) # This is the cost vector of first stage.

    # This function consists of cost vector for generation, such as q^{gen}
    def prepOperationalCostGen_rule(model):
        #Build generator short term marginal costs

        for g in model.Generator:
            for i in model.PeriodActive:
                if ('CCS',g) in model.GeneratorsOfTechnology:
                    costperenergyunit=(3.6/model.genEfficiency[g,i])*(model.genFuelCost[g,i]+(1-model.CCSRemFrac)*model.genCO2TypeFactor[g]*model.CO2price[i])+ \
                    (3.6/model.genEfficiency[g,i])*(model.CCSRemFrac*model.genCO2TypeFactor[g]*model.CCSCostTSVariable[i])+ \
                    model.genVariableOMCost[g]
                else:
                    costperenergyunit=(3.6/model.genEfficiency[g,i])*(model.genFuelCost[g,i]+model.genCO2TypeFactor[g]*model.CO2price[i])+ \
                    model.genVariableOMCost[g]
                model.genMargCost[g,i]=costperenergyunit

    model.build_OperationalCostGen = BuildAction(rule=prepOperationalCostGen_rule)

    # This is \bar_{x}
    def prepInitialCapacityNodeGen_rule(model):
        #Build initial capacity for generator type in node

        for (n,g) in model.GeneratorsOfNode:
            for i in model.PeriodActive:
                if value(model.genInitCap[n,g,i]) == 0:
                    model.genInitCap[n,g,i] = model.genRefInitCap[n,g]*(1-model.genScaleInitCap[g,i])

    model.build_InitialCapacityNodeGen = BuildAction(rule=prepInitialCapacityNodeGen_rule)

    # This is \bar_{x}
    def prepInitialCapacityTransmission_rule(model):
        #Build initial capacity for transmission lines to ensure initial capacity is the upper installation bound if infeasible

        for (n1,n2) in model.BidirectionalArc:
            for i in model.PeriodActive:
                if value(model.transmissionMaxInstalledCapRaw[n1,n2,i]) <= value(model.transmissionInitCap[n1,n2,i]):
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionInitCap[n1,n2,i]
                else:
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionMaxInstalledCapRaw[n1,n2,i]

    model.build_InitialCapacityTransmission = BuildAction(rule=prepInitialCapacityTransmission_rule)

    # This is V on the mathematical obejctive function
    def prepOperationalDiscountrate_rule(model):
        #Build operational discount rate

        model.operationalDiscountrate = sum((1+model.discountrate)**(-j) for j in list(range(0,value(model.LeapYearsInvestment))))

    model.build_operationalDiscountrate = BuildAction(rule=prepOperationalDiscountrate_rule)


    # Following functions are represent \bar_{V}
    def prepGenMaxInstalledCap_rule(model):
        #Build resource limit (installed limit) for all periods. Avoid infeasibility if installed limit lower than initially installed cap.

        for t in model.Technology:
            for n in model.Node:
                for i in model.PeriodActive:
                    if value(model.genMaxInstalledCapRaw[n,t] <= sum(model.genInitCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology)):
                        model.genMaxInstalledCap[n,t,i]=sum(model.genInitCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology)
                    else:
                        model.genMaxInstalledCap[n,t,i]=model.genMaxInstalledCapRaw[n,t]

    model.build_genMaxInstalledCap = BuildAction(rule=prepGenMaxInstalledCap_rule)

    def storENMaxInstalledCap_rule(model):
        #Build installed limit (resource limit) for storEN

        for (n,b) in model.StoragesOfNode:
            for i in model.PeriodActive:
                model.storENMaxInstalledCap[n,b,i]=model.storENMaxInstalledCapRaw[n,b]

    model.build_storENMaxInstalledCap = BuildAction(rule=storENMaxInstalledCap_rule)

    def storPWMaxInstalledCap_rule(model):
        #Build installed limit (resource limit) for storPW

        for (n,b) in model.StoragesOfNode:
            for i in model.PeriodActive:
                model.storPWMaxInstalledCap[n,b,i]=model.storPWMaxInstalledCapRaw[n,b]

    model.build_storPWMaxInstalledCap = BuildAction(rule=storPWMaxInstalledCap_rule)


    #############
    ##VARIABLES##
    #############

    # First Stage Decisions (x)
    model.genInvCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmisionInvCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)

    # First Stage Decisions (v)
    model.genInstalledCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmissionInstalledCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInstalledCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInstalledCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)

    ###############
    ##EXPRESSIONS##
    ###############

    def multiplier_rule(model,period):
        coeff=1
        if period>1:
            coeff=pow(1.0+model.discountrate,(-LeapYearsInvestment*(int(period)-1)))
        return coeff
    model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)

    #############
    ##OBJECTIVE##
    #############

    # def Obj_rule(model):
    #     first_stage_value = sum(model.discount_multiplier[i]*(sum(model.genInvCost[g,i]* model.genInvCap[n,g,i] for (n,g) in model.GeneratorsOfNode ) + \
    #         sum(model.transmissionInvCost[n1,n2,i]*model.transmisionInvCap[n1,n2,i] for (n1,n2) in model.BidirectionalArc ) + \
    #         sum((model.storPWInvCost[b,i]*model.storPWInvCap[n,b,i]+model.storENInvCost[b,i]*model.storENInvCap[n,b,i]) for (n,b) in model.StoragesOfNode )
    #         ) for i in model.PeriodActive)
    #     return first_stage_value
    # model.Obj = Objective(rule=Obj_rule, sense=minimize)

    ###############
    ##CONSTRAINTS##
    ###############

    #### First Stage Constraints ####

    def lifetime_rule_gen(model, n, g, i):
        startPeriod=1
        if value(1+i-(model.genLifetime[g]/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.genLifetime[g]/model.LeapYearsInvestment)
        return sum(model.genInvCap[n,g,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.genInstalledCap[n,g,i] + model.genInitCap[n,g,i]== 0   #
    model.installedCapDefinitionGen = Constraint(model.GeneratorsOfNode, model.PeriodActive, rule=lifetime_rule_gen)

    #################################################################

    def lifetime_rule_storEN(model, n, b, i):
        startPeriod=1
        if value(1+i-model.storageLifetime[b]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.storageLifetime[b]/model.LeapYearsInvestment)
        return sum(model.storENInvCap[n,b,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.storENInstalledCap[n,b,i] + model.storENInitCap[n,b,i]== 0   #
    model.installedCapDefinitionStorEN = Constraint(model.StoragesOfNode, model.PeriodActive, rule=lifetime_rule_storEN)

    #################################################################

    def lifetime_rule_storPOW(model, n, b, i):
        startPeriod=1
        if value(1+i-model.storageLifetime[b]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.storageLifetime[b]/model.LeapYearsInvestment)
        return sum(model.storPWInvCap[n,b,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.storPWInstalledCap[n,b,i] + model.storPWInitCap[n,b,i]== 0   #
    model.installedCapDefinitionStorPOW = Constraint(model.StoragesOfNode, model.PeriodActive, rule=lifetime_rule_storPOW)

    #################################################################

    def lifetime_rule_trans(model, n1, n2, i):
        startPeriod=1
        if value(1+i-model.transmissionLifetime[n1,n2]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.transmissionLifetime[n1,n2]/model.LeapYearsInvestment)
        return sum(model.transmisionInvCap[n1,n2,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.transmissionInstalledCap[n1,n2,i] + model.transmissionInitCap[n1,n2,i] == 0   #
    model.installedCapDefinitionTrans = Constraint(model.BidirectionalArc, model.PeriodActive, rule=lifetime_rule_trans)

    #################################################################

    def investment_gen_cap_rule(model, t, n, i):
        return sum(model.genInvCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology) - model.genMaxBuiltCap[n,t,i] <= 0
    model.investment_gen_cap = Constraint(model.Technology, model.Node, model.PeriodActive, rule=investment_gen_cap_rule)

    #################################################################

    def investment_trans_cap_rule(model, n1, n2, i):
        return model.transmisionInvCap[n1,n2,i] - model.transmissionMaxBuiltCap[n1,n2,i] <= 0
    model.investment_trans_cap = Constraint(model.BidirectionalArc, model.PeriodActive, rule=investment_trans_cap_rule)

    #################################################################

    def investment_storage_power_cap_rule(model, n, b, i):
        return model.storPWInvCap[n,b,i] - model.storPWMaxBuiltCap[n,b,i] <= 0
    model.investment_storage_power_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=investment_storage_power_cap_rule)

    #################################################################

    def investment_storage_energy_cap_rule(model, n, b, i):
        return model.storENInvCap[n,b,i] - model.storENMaxBuiltCap[n,b,i] <= 0
    model.investment_storage_energy_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=investment_storage_energy_cap_rule)

    ################################################################

    def installed_gen_cap_rule(model, t, n, i):
        return sum(model.genInstalledCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology) - model.genMaxInstalledCap[n,t,i] <= 0
    model.installed_gen_cap = Constraint(model.Technology, model.Node, model.PeriodActive, rule=installed_gen_cap_rule)

    #################################################################

    def installed_trans_cap_rule(model, n1, n2, i):
        return model.transmissionInstalledCap[n1,n2,i] - model.transmissionMaxInstalledCap[n1,n2,i] <= 0
    model.installed_trans_cap = Constraint(model.BidirectionalArc, model.PeriodActive, rule=installed_trans_cap_rule)

    #################################################################

    def installed_storage_power_cap_rule(model, n, b, i):
        return model.storPWInstalledCap[n,b,i] - model.storPWMaxInstalledCap[n,b,i] <= 0
    model.installed_storage_power_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=installed_storage_power_cap_rule)

    #################################################################

    def installed_storage_energy_cap_rule(model, n, b, i):
        return model.storENInstalledCap[n,b,i] - model.storENMaxInstalledCap[n,b,i] <= 0
    model.installed_storage_energy_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=installed_storage_energy_cap_rule)

    #################################################################
    if north_sea:
        def wind_farm_tranmission_cap_rule(model, n1, n2, i):
            if n1 in model.OffshoreNode or n2 in model.OffshoreNode:
                if (n1,n2) in model.BidirectionalArc:
                    if n1 in model.OffshoreNode:
                        return model.transmissionInstalledCap[(n1,n2),i] <= sum(model.genInstalledCap[n1,g,i] for g in model.Generator if (n1,g) in model.GeneratorsOfNode)
                    else:
                        return model.transmissionInstalledCap[(n1,n2),i] <= sum(model.genInstalledCap[n2,g,i] for g in model.Generator if (n2,g) in model.GeneratorsOfNode)
                elif (n2,n1) in model.BidirectionalArc:
                    if n1 in model.OffshoreNode:
                        return model.transmissionInstalledCap[(n2,n1),i] <= sum(model.genInstalledCap[n1,g,i] for g in model.Generator if (n1,g) in model.GeneratorsOfNode)
                    else:
                        return model.transmissionInstalledCap[(n2,n1),i] <= sum(model.genInstalledCap[n2,g,i] for g in model.Generator if (n2,g) in model.GeneratorsOfNode)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        model.wind_farm_transmission_cap = Constraint(model.Node, model.Node, model.PeriodActive, rule=wind_farm_tranmission_cap_rule)

    #################################################################

    def power_energy_relate_rule(model, n, b, i):
        if b in model.DependentStorage:
            return model.storPWInstalledCap[n,b,i] - model.storagePowToEnergy[b]*model.storENInstalledCap[n,b,i] == 0   #
        else:
            return Constraint.Skip
    model.power_energy_relate = Constraint(model.StoragesOfNode, model.PeriodActive, rule=power_energy_relate_rule)

    #################################################################

    def version1_rule(model, n, h, i):
        gen_avail_capacity = sum(model.genInstalledCap[n, g, i] * model.avg_cap_avail[n,g,h,i] for g in model.Generator if (n, g) in model.GeneratorsOfNode)
        return model.exp_sload[n,i,h]*1.3-gen_avail_capacity <= 0 
    model.version1 = Constraint(model.Node, model.Operationalhour, model.PeriodActive,rule=version1_rule)

    ######################## ML Embedding #####################################
    
    # New constraints
    low_cost_tech = ['Solar', 'Windonshore', 'Windoffshore', 'Hydroregulated', 'Hydrorun-of-the-river', 'Geo', 'Wave', 'Nuclear', 'Lignite']
    def low_cost_init(model):
        return [(n,g) for (n,g) in model.GeneratorsOfNode if g in low_cost_tech]
    model.GeneratorsOfLowCost = Set(within=model.GeneratorsOfNode, initialize=low_cost_init)
    

    alpha_dict = {1: 0.5, 2: 0.55, 3: 0.6, 4: 0.65, 5: 0.78, 6: 0.78, 7: 0.80, 8: 0.85}
    model.alpha = Param(model.PeriodActive, initialize=alpha_dict)

    def low_cost_share_rule(model, n, i):
        total_cap = sum(model.genInstalledCap[n, g, i] 
                        for (n_tmp, g) in model.GeneratorsOfNode if n_tmp == n)
        low_cost_cap = sum(model.genInstalledCap[n, g, i] 
                        for (n_tmp, g) in model.GeneratorsOfLowCost if n_tmp == n)
        return (model.alpha[i] * total_cap - low_cost_cap <= 0)

    model.low_cost_share_constraint = Constraint(model.Node, model.PeriodActive, rule=low_cost_share_rule)

    def storage_enforce(model, n, i):
        if not any((n, b) in model.StoragesOfNode and b in model.DependentStorage for b in model.DependentStorage):
            return Constraint.Skip
        return (0.03 * sum(model.genInstalledCap[n, g, i] for g in model.Generator if (n, g) in model.GeneratorsOfNode)- sum(model.storPWInstalledCap[n, b, i] for (n2, b) in model.StoragesOfNode if n2 == n and b in model.DependentStorage)) <= 0
    model.storage_enforce = Constraint(model.Node, model.PeriodActive, rule=storage_enforce)

    n_features = 616
    num_sce = 10
    num_sam = 10000
    scaler_v = joblib.load(f'scaler_pca4/scaler_ad_{num_sam}.joblib')
    data_min = scaler_v.data_min_
    data_max = scaler_v.data_max_
    data_range = data_max - data_min

    scaler_y = joblib.load(f'scaler_pca_ad/scaler_y_ad_{num_sam}.joblib')
    mean_output = scaler_y.mean_
    scale_output = scaler_y.scale_

    # pca_obj = joblib.load('scaler_pca2/pca_ad.pkl')
    # components = pca_obj.components_  # Shape: (n_pca, n_features)
    # pca_mean = pca_obj.mean_          # Shape: (n_features,)
    # # Compute the offset for each PCA component: dot product of component and PCA mean
    # offset = np.dot(components, pca_mean)  # Array of shape (n_pca,)
    # n_pca = components.shape[0]
    
    print(f"mean_output:{mean_output}, scale_output: {scale_output}")
    print(f"n_features : {n_features}")

    # --- PCA 변환을 위해 pca_ad_1000.pkl 로드 후, PCA 성분을 적용 --- #
    # (4) PCA 모델 불러오기
    pca_obj = joblib.load(f'scaler_pca4/pca_ad_{num_sam}.pkl')
    components = pca_obj.components_  # (n_pca, n_features)
    pca_mean = pca_obj.mean_          # (n_features,)
    offset = np.dot(components, pca_mean)  # (n_pca,)
    n_pca = components.shape[0]

    model.nn_inputs = Var(RangeSet(1, n_pca), domain=Reals)
    model.v_scaled = Var(RangeSet(1, n_features), domain=Reals)
    
    instance = model.create_instance(data)
    instance.dual = Suffix(direction=Suffix.IMPORT) 


    # Compute the total number of features
    n_periods = len(list(instance.PeriodActive))
    n_gen = len(list(instance.GeneratorsOfNode))
    n_trans = len(list(instance.BidirectionalArc))
    n_storage = len(list(instance.StoragesOfNode))
    n_features = n_periods * (n_gen + n_trans + 2 * n_storage)

    # Build an ordered list of ML input variables (Python list)
    ordered_ml_inputs = []
    for i in instance.PeriodActive:
        # 1. Generator installed capacities
        for (n, g) in instance.GeneratorsOfNode:
            ordered_ml_inputs.append(instance.genInstalledCap[n, g, i])
        # 2. Transmission installed capacities
        for (n1, n2) in instance.BidirectionalArc:
            ordered_ml_inputs.append(instance.transmissionInstalledCap[n1, n2, i])
        # 3. Storage Power installed capacities
        for (n, b) in instance.StoragesOfNode:
            ordered_ml_inputs.append(instance.storPWInstalledCap[n, b, i])
        # 4. Storage Energy installed capacities
        for (n, b) in instance.StoragesOfNode:
            ordered_ml_inputs.append(instance.storENInstalledCap[n, b, i])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 저장된 PyTorch 모델 로드 (torch.load로 로드)
    # 전체 모델 객체를 불러옵니다.

    
    def v_scailing(instance, i):
        j = i - 1  # 0-based index
        if data_range[j] == 0:
            return instance.v_scaled[i] == 0
        else:
            return instance.v_scaled[i] == ((ordered_ml_inputs[j] - data_min[j]) / data_range[j])
    instance.ml_input_scaled_constraints = Constraint(RangeSet(1, n_features),rule=v_scailing)


    # (6) pca_vars[k] = Σ_j [components[k-1, j-1] * nn_inputs[j]] - offset[k-1]
    def pca_constraint_rule(m, k):
        # k는 1-based, components[k-1]는 0-based
        return m.nn_inputs[k] == sum(components[k-1, j-1]*m.v_scaled[j] 
                                    for j in range(1, n_features+1)) - offset[k-1]
    instance.pca_constraint = Constraint(RangeSet(1, n_pca), rule=pca_constraint_rule)


    output_var_name = 'ml_output'

    pytorch_model = SimpleMLP(n_pca)
    pytorch_model.load_state_dict(torch.load(f'scaler_pca_ad/best_model_ad_{num_sam}.pth', weights_only=True))
    pytorch_model.eval()
    
    # ml_output = integrate_nn_with_pyomo_model(pytorch_model, instance, output_var_name, M=1000) # NN
    ml_output = integrate_linear_regression_with_pyomo_model(f"scaler_pca_ad/linear_regression_model_{num_sam}.joblib", instance, output_var_name) # LR
    # ml_output = integrate_decision_tree_with_pyomo_model(f"scaler_pca_ad/decision_tree_model_{num_sam}.joblib", instance, output_var_name) # DT


    def Obj_rule(instance):
        first_stage_value = sum(
            instance.discount_multiplier[i] * (
                sum(instance.genInvCost[g,i] * instance.genInvCap[n,g,i] for (n, g) in instance.GeneratorsOfNode) +
                sum(instance.transmissionInvCost[n1,n2,i] * instance.transmisionInvCap[n1,n2,i] for (n1, n2) in instance.BidirectionalArc) +
                sum((instance.storPWInvCost[b,i] * instance.storPWInvCap[n,b,i] + instance.storENInvCost[b,i] * instance.storENInvCap[n,b,i])
                    for (n, b) in instance.StoragesOfNode)
            )
            for i in instance.PeriodActive
        )
        # second_stage_value = ml_output * scale_output + mean_output
        second_stage_value = ml_output* scale_output + mean_output
        Total_cost = first_stage_value+second_stage_value
        return Total_cost

    instance.Obj = Objective(rule=Obj_rule, sense=minimize)


    return instance
