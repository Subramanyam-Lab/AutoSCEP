import time
start_time = time.time()
from pyomo.environ import *
import pandas as pd
import os
import glob
from func_Q2 import *
import numpy as np
from model_definition import define_model
import re
import random
import multiprocessing

# Define the model
model = AbstractModel()

# Define sets
model.P = Set()  # Set of potential plant locations
model.C = Set()  # Set of customers

# Define parameters
model.f = Param(model.P, within=NonNegativeReals)  # Cost of setting up a plant at each location
model.c = Param(model.P, within=NonNegativeReals)  # Capacity of each plant
model.demands = Param(model.C, mutable=True, within=NonNegativeReals)  # Demand of each customer
model.t = Param(model.C, model.P, within=NonNegativeReals)  # Transport cost from plant to customer


# Define variables
model.x = Var(model.C, model.P, within=NonNegativeReals, bounds=(0, 1))  # Supply ratio from plant to customer
model.y = Var(model.P, within=Binary)  # Binary variable indicating whether a plant is built


# Define objective function: minimize setup and transport costs
def objective_rule(model):
    return sum(model.f[p]*model.y[p] for p in model.P) + \
           sum(model.t[c,p]*model.x[c,p] for c in model.C for p in model.P) 
model.obj = Objective(rule=objective_rule, sense=minimize)

# Define constraints
def demand_constraint_rule(model, c):
    return sum(model.x[c,p] for p in model.P) == 1  
model.demand_constraint = Constraint(model.C, rule=demand_constraint_rule)

def capacity_constraint_rule(model, p):
    return sum(model.demands[c] * model.x[c,p] for c in model.C) <= model.c[p] * model.y[p]
model.capacity_constraint = Constraint(model.P, rule=capacity_constraint_rule)


# Post-processing: display variable values and save to CSV
def pyomo_postprocess(options=None, instance=None, filename='optimization_results.csv',expected_second_stage_value=None,first_stage_decisions=None, feasibility= None, size=None):
    
    clients, facilities = size
    directory = f"results_V1/CPLP_{clients}_{facilities}/"
    full_filename = os.path.join(directory, filename)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if len(first_stage_decisions) != len(expected_second_stage_value):
        raise ValueError("first_stage_decisions and expected_second_stage_value should have the same length")
    
    # If the file exists, remove it to start fresh
    if os.path.exists(full_filename):
        os.remove(full_filename)

    df = pd.DataFrame({
        'first stage decision': [str(decision) for decision in first_stage_decisions],
        'expected second stage value': expected_second_stage_value,
        'feasibility': feasibility_lst
    })

    file_exists = os.path.isfile(full_filename)

    df.to_csv(full_filename, mode='a', header=not file_exists, index=False)
    

def solve_for_file(data_file, size, model, num_iteration):
    clients, facilities = size
    first_stage_decisions_lst = []
    second_stage_value_lst = []
    feasibility_lst = []

    problem_size = f"{clients}_{facilities}_{data_file.split('_')[-1].split('.')[0]}"  
    result_filename = f"results_{problem_size}.csv"

    seen_decisions = {}

    for i in range(num_iteration):
        print(f"data_file: {data_file} | {i} trial")
        instance = model.create_instance(data_file)
        demand_values = {c: random.uniform(5, 35) for c in instance.C}
        for c in instance.C:
            instance.demands[c] = demand_values[c]

        solver = SolverFactory('gurobi')
        results = solver.solve(instance, tee=False)
        
        first_stage_decisions = tuple(int(value(instance.y[p])) for p in instance.P)
        decision_str = str(first_stage_decisions)

        if decision_str in seen_decisions:
            continue

        seen_decisions[decision_str] = True
        capacity = [value(instance.c[p]) for p in instance.P]
        transportation_cost = [value(instance.t[c, p]) for c in instance.C for p in instance.P]

        m = define_model()
        expected_second_stage_value = Q(m, first_stage_decisions, capacity, transportation_cost, size, data_file)
        if expected_second_stage_value ==0:
            feasibility_lst.append(0)
        else:
            feasibility_lst.append(1)
        
        first_stage_decisions_lst.append(first_stage_decisions)
        second_stage_value_lst.append(expected_second_stage_value)

    # Post-process results.
    pyomo_postprocess(options=None, instance=instance, filename=result_filename,
                      expected_second_stage_value=second_stage_value_lst,
                      first_stage_decisions=first_stage_decisions_lst, 
                      feasibility = feasibility_lst, size=size)

if __name__ == '__main__':
    problem_sizes = [(10, 10), (25, 25), (50, 50)]
    # problem_sizes = [(10, 10)]
    
    for size in problem_sizes:
        clients, facilities = size
        if clients ==10:
            num_iteration = 1000
        elif clients ==25:
            num_iteration = 2000
        else:
            num_iteration = 2000
        
        ### for real ####
        data_dir = f"data/CPLP_{clients}_{facilities}"
        data_files = glob.glob(os.path.join(data_dir, "*.dat"))
        data_files_sorted = sorted(data_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

        pool = multiprocessing.Pool(processes=30)  
        results = [pool.apply_async(solve_for_file, args=(data_file, size, model, num_iteration)) for data_file in data_files_sorted]
        
        
        # data_dir = f"data/CPLP_{clients}_{facilities}"
        # data_files = glob.glob(os.path.join(data_dir, "*.dat"))
        # data_files_sorted = sorted(data_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
        # data_files_to_load = data_files_sorted[:num_files_to_load]

        # pool = multiprocessing.Pool(processes=3)  
        # results = [pool.apply_async(solve_for_file, args=(data_file, size, model, num_iteration)) for data_file in data_files_to_load]
        
        pool.close()
        pool.join()
    
    end_time = time.time()
    execution_time = end_time - start_time

    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"The code ran for {int(hours)}h {int(minutes)}m {seconds}s")