from pyomo.environ import *
import pandas as pd
import os
import glob
from func_Q import *
import numpy as np
from model_definition import define_model
from func_Q import Q
import re


M = 10000

# Define the model
model = AbstractModel()

# Define sets
model.P = Set()  # Set of potential plant locations
model.C = Set()  # Set of customers

# Define parameters
model.f = Param(model.P, within=NonNegativeReals)  # Cost of setting up a plant at each location
model.c = Param(model.P, within=NonNegativeReals)  # Capacity of each plant
model.d = Param(model.C, within=NonNegativeReals)  # Demand of each customer
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
    return sum(model.d[c] * model.x[c,p] for c in model.C) <= model.c[p] * model.y[p]
model.capacity_constraint = Constraint(model.P, rule=capacity_constraint_rule)


# Post-processing: display variable values and save to CSV
def pyomo_postprocess(options=None, instance=None, filename='optimization_results.csv',expected_second_stage_value=None,first_stage_decisions=None):
    # instance.x.display()
    # instance.y.display()

    # Prepare data for CSV
    first_stage = first_stage_decisions
    expected_second_stage_values = expected_second_stage_value
    
    # Combine all data into a single row
    data = [first_stage, expected_second_stage_values]
    
    # Create DataFrame
    df = pd.DataFrame([data], columns=['first stage decision', 'expected second stage value'])
    
    # Check if the file exists to avoid writing headers multiple times
    file_exists = os.path.isfile(filename)
    
    # Save to CSV (append if file exists, write header only if file does not exist)
    df.to_csv(filename, mode='a', header=not file_exists, index=False)


if __name__ == '__main__':
    # Retrieve data files from different directories based on problem size.
    # problem_sizes = [(10, 10), (25, 25), (50, 50)]
    problem_sizes = [(10, 10)]
    
    for size in problem_sizes:
        clients, facilities = size
        data_dir = f"data/CPLP_{clients}_{facilities}"
        
        # Extract all file paths
        data_files = glob.glob(os.path.join(data_dir, "*.dat"))
        
        # Sort files by the number in their name
        data_files_sorted = sorted(data_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
        
        # For each scenario
        for data_file in data_files_sorted:
            print("data_file: ",data_file)
            # Create a model instance and load data.
            instance = model.create_instance(data_file)
            
            # Solve the problem.
            solver = SolverFactory("gurobi")
            results = solver.solve(instance, tee=False)
            
            first_stage_decisions = [int(value(instance.y[p])) for p in instance.P]
            print(first_stage_decisions)
            capacity = [value(instance.c[p]) for p in instance.P]
            transportation_cost = [value(instance.t[c,p]) for c in instance.C for p in instance.P]
            
            m = define_model()
            expected_second_stage_value = Q(m, first_stage_decisions,capacity, transportation_cost, size)
            print("expected_second_stage_value:",expected_second_stage_value)
            
            # Display or save the results.
            # results.write()
            # print("\nDisplaying Solution\n" + '-'*60)
            
            problem_size = f"{clients}_{facilities}"  # Extracting the problem size from the folder name
            result_filename = f"results_{problem_size}.csv"
            
            # Post-process results.
            pyomo_postprocess(options=None, instance=instance, filename=result_filename,expected_second_stage_value= expected_second_stage_value, first_stage_decisions = first_stage_decisions)