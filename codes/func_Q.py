# from pyomo.environ import *
# import numpy as np
# import glob
# import os
# import re


# def Q(model, y_fixed, capcity, trans_cost, size):
#     M = np.sum(trans_cost)
#     clients, facilities = size
#     data_dir = f"data/CPLP_{clients}_{facilities}"
#     data_files = glob.glob(os.path.join(data_dir, "*.dat"))
#     data_files_sorted = sorted(data_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
#     second_stage_value_lst = []
    
#     y_fixed_value = {f'P{i}': y_fixed [i] for i in range(len(y_fixed))}
#     capcity_value = {f'P{i}': capcity [i] for i in range(len(capcity))}
#     trans_dict = {f'C{i}': {f'P{j}': trans_cost[i * facilities + j] for j in range(facilities)} for i in range(clients)}

#     def sub_objective_rule(model):
#         return sum(trans_dict[c][p] * model.x[c, p] for c in model.C for p in model.P) + M*sum(model.s[p] for p in model.P)
#     model.obj = Objective(rule=sub_objective_rule, sense=minimize)
    
#     def sub_demand_constraint_rule(model, c):
#         return sum(model.x[c, p] for p in model.P) == 1
#     model.demand_constraint = Constraint(model.C, rule=sub_demand_constraint_rule)
        
#     def sub_capacity_constraint_rule(model, p):
#         return sum(model.d[c] * model.x[c,p] for c in model.C) <= capcity_value[p] * y_fixed_value[p] + model.s[p]
#     model.capacity_constraint = Constraint(model.P, rule=sub_capacity_constraint_rule)

#     # For each scenario
#     for data_file in data_files_sorted:
#             # Create a model instance and load data.
#             instance = model.create_instance(data_file)          
#             solver = SolverFactory('glpk')                 
#             solver.solve(instance, tee=False)
#             second_stage_value_lst.append(value(instance.obj))

#     return np.mean(second_stage_value_lst)


# Version 2

from pyomo.environ import *
import numpy as np

def load_demand_from_dat(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    demands = {}
    mode = None
    headers = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "param d :" in line:
            mode = "d"
            continue
        elif line.startswith(";"):
            if mode == "d":
                break  # 세미콜론을 발견하면 'd' 모드를 종료합니다.

        if mode == "d" and "demand_" in line:
            headers = line.split()
            for header in headers:
                if header != ":=":  # 이 부분을 추가하여 ":="를 제외합니다.
                    demands[header] = []
            continue

        if mode == "d":
            parts = line.split()
            values = [float(value) for value in parts[1:]]

            for header, value in zip(headers, values):
                if header in demands:  # 추가된 조건
                    demands[header].append(value)

    return demands





def Q(model, y_fixed, capcity, trans_cost, size, data_file):
    M = np.sum(trans_cost)
    clients, facilities = size

    y_fixed_value = {f'P{i}': y_fixed[i] for i in range(len(y_fixed))}
    capcity_value = {f'P{i}': capcity[i] for i in range(len(capcity))}
    trans_dict = {f'C{i}': {f'P{j}': trans_cost[i * facilities + j] for j in range(facilities)} for i in range(clients)}
    
    
    def sub_objective_rule(model):
        return sum(trans_dict[c][p] * model.x[c, p] for c in model.C for p in model.P) + M * sum(model.s[p] for p in model.P)

    model.obj = Objective(rule=sub_objective_rule, sense=minimize)

    def sub_demand_constraint_rule(model, c):
        return sum(model.x[c, p] for p in model.P) == 1

    model.demand_constraint = Constraint(model.C, rule=sub_demand_constraint_rule)

    def sub_capacity_constraint_rule(model, p):
        return sum(model.demands[c] * model.x[c, p] for c in model.C) <= capcity_value[p] * y_fixed_value[p] + model.s[p]

    model.capacity_constraint = Constraint(model.P, rule=sub_capacity_constraint_rule)

    
    demands = load_demand_from_dat(data_file)
    print(demands)
    exit(3)
    
    # Create a model instance
    instance = model.create_instance(data_file)
    
    # List to store results for each demand scenario
    second_stage_value_lst = []

    # Demand scenario headers
    demand_headers = [f'demand_{i}' for i in range(clients)]

    # Solve for each demand scenario
    for header in demand_headers:
        # Set the demand for each client from the demand matrix
        for client in model.C:
            instance.demands[client] = getattr(instance, 'demands')[client, header]

        # Solve the model
        solver = SolverFactory('glpk')
        solver.solve(instance, tee=False)

        second_stage_value_lst.append(value(instance.obj))

    return np.mean(second_stage_value_lst)
