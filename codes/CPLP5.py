import time
import pandas as pd
import os
import glob
import random
import multiprocessing
from pyomo.environ import *
import numpy as np
import re

# Define the model creation function
def create_model(num_scenarios):
    model = AbstractModel()

    # Sets
    model.P = Set()  # Potential plant locations
    model.C = Set()  # Customers
    model.Scenarios = RangeSet(num_scenarios)  # Scenarios for SAA

    # Parameters
    model.f = Param(model.P, within=NonNegativeReals)  # Setup costs
    model.c = Param(model.P, within=NonNegativeReals)  # Plant capacities
    model.demands = Param(model.C, mutable=True, within=NonNegativeReals)  # Customer demands
    model.t = Param(model.C, model.P, within=NonNegativeReals)  # Transport costs
    model.DemandScenarios = Param(model.C, model.Scenarios, mutable=True)  # Demand in each scenario

    # Variables
    model.x = Var(model.C, model.P, within=NonNegativeReals, bounds=(0, 1))  # Fraction supplied
    model.y = Var(model.P, within=Binary)  # Plant setup decisions
    model.s = Var(model.P, within=NonNegativeReals)  # Slack for over-capacity

    # Objective: Minimize setup and transport costs
    def objective_rule(m):
        return sum(m.f[p] * m.y[p] for p in m.P) + \
               sum(m.t[c, p] * m.x[c, p] for c in m.C for p in m.P) + \
               sum(m.s[p] for p in m.P)  # Adjusted to remove redundant sum

    model.obj = Objective(rule=objective_rule, sense=minimize)

    # Constraints
    def demand_constraint_rule(m, c):
        return sum(m.x[c, p] for p in m.P) == 1

    model.demand_constraint = Constraint(model.C, rule=demand_constraint_rule)

    def capacity_constraint_rule(m, p, s):
        return sum(m.DemandScenarios[c, s] * m.x[c, p] for c in m.C) <= m.c[p] * m.y[p] + m.s[p]

    model.capacity_constraint = Constraint(model.P, model.Scenarios, rule=capacity_constraint_rule)

    return model


# Function to solve a single scenario
def solve_scenario(data_file, model, scenario_index, num_scenarios):
    # Load data and create a model instance
    instance = model.create_instance(data_file)
    solver = SolverFactory('gurobi')

    # Generate random demands for the scenario
    for c in instance.C:
        instance.demands[c] = random.uniform(5, 35)  # Random demand

    # Solve the model
    solver.solve(instance, tee=False)

    # Extract and return results
    first_stage_decision = tuple(value(instance.y[p]) for p in instance.P)
    second_stage_value = calculate_second_stage_value(instance, first_stage_decision)
    return first_stage_decision, second_stage_value

# Function to post-process and save results
def post_process_results(results, size):
    directory = f"results4/CPLP_{size[0]}_{size[1]}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_filename = os.path.join(directory, "optimization_results.csv")

    # Convert results to DataFrame and save
    df = pd.DataFrame(results, columns=['First Stage Decision', 'Expected Second Stage Value'])
    df.to_csv(full_filename, index=False)

# Main execution function
def execute_saa(problem_sizes, num_files_to_load, num_scenarios):
    start_time = time.time()

    for size in problem_sizes:
        model = create_model(num_scenarios)
        data_dir = f"data/CPLP_{size[0]}_{size[1]}"
        data_files = glob.glob(os.path.join(data_dir, "*.dat"))[:num_files_to_load]
        results = []

        with multiprocessing.Pool(min(len(data_files), multiprocessing.cpu_count())) as pool:
            for data_file in data_files:
                for scenario_index in range(num_scenarios):
                    results.append(pool.apply_async(solve_scenario, (data_file, model, scenario_index, num_scenarios)))

        # Collect results
        collected_results = [res.get() for res in results]
        post_process_results(collected_results, size)

    end_time = time.time()
    print_execution_time(start_time, end_time)

# Helper function to print execution time
def print_execution_time(start, end):
    hours, remainder = divmod(end - start, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"The code ran for {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == '__main__':
    problem_sizes = [(10, 10), (25, 25), (50, 50)]
    num_files_to_load = 3
    num_scenarios = 1000
    execute_saa(problem_sizes, num_files_to_load, num_scenarios)
    
    
    
    

def Q(model, y_fixed, capcity, trans_cost, size, data_file):
    M = np.sum(trans_cost)
    clients, facilities = size
    second_stage_value_lst = []

    y_fixed_value = {f'P{i}': y_fixed[i] for i in range(len(y_fixed))}
    capacity_value = {f'P{i}': capcity[i] for i in range(len(capcity))}
    trans_dict = {f'C{i}': {f'P{j}': trans_cost[i * facilities + j] for j in range(facilities)} for i in range(clients)}

    def sub_objective_rule(model):
        return sum(trans_dict[c][p] * model.x[c, p] for c in model.C for p in model.P) + M * sum(model.s[p] for p in model.P)

    model.obj = Objective(rule=sub_objective_rule, sense=minimize)

    def sub_demand_constraint_rule(model, c):
        return sum(model.x[c, p] for p in model.P) == 1

    model.demand_constraint = Constraint(model.C, rule=sub_demand_constraint_rule)

    def sub_capacity_constraint_rule(model, p):
        return sum(model.demands[c] * model.x[c, p] for c in model.C) <= capacity_value[p] * y_fixed_value[p] + model.s[p]

    model.capacity_constraint = Constraint(model.P, rule=sub_capacity_constraint_rule)

    load_demands = load_demand_from_dat(data_file)

    # Demand scenario headers
    demand_headers = [f'demand_{i}' for i in range(len(load_demands))]
    # solver = SolverFactory("cplex", excutable= "/storage/icds/RISE/sw8/cplex/22.1.1/cplex/bin/x86-64_linux/cplex")
    solver = SolverFactory('gurobi')
    
    results_accumulated = []
    previous_checkpoint = 0
    sub_checkpoint = [10*2**i for i in range(10)]
    check_points = sub_checkpoint + [len(demand_headers)] 
    
    for check_point in check_points:
        headers_subset = demand_headers[previous_checkpoint:check_point]
        for header in headers_subset:
            result = solve_instance(header, load_demands, model.create_instance(data_file), solver)
            results_accumulated.append(result)
        
        if len(results_accumulated) > 1 and np.std(results_accumulated) / np.mean(results_accumulated) <= 0.05:
            print(f"In {data_file}, At checkpoint {check_point}, the ratio of results is below 0.05. Stopping further calculations.")
            print(f"Average result up to this checkpoint: {np.mean(results_accumulated)}")
            break
        
        previous_checkpoint = check_point
        
    return np.mean(results_accumulated)

# Post-processing: display variable values and save to CSV
def pyomo_postprocess(options=None, instance=None, filename='optimization_results.csv',expected_second_stage_value=None,first_stage_decisions=None,size=None):
    
    clients, facilities = size
    directory = f"results4/CPLP_{clients}_{facilities}/"
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
        'expected second stage value': expected_second_stage_value
    })

    file_exists = os.path.isfile(full_filename)

    df.to_csv(full_filename, mode='a', header=not file_exists, index=False)
    

def solve_for_file(data_file, size, base_model, num_scenarios):
    clients, facilities = size
    first_stage_decisions_lst = []
    second_stage_value_lst = []

    # Load data file and create a base instance
    instance = base_model.create_instance(data_file)

    # Set up the solver
    solver = SolverFactory('gurobi')

    for scenario_index in range(num_scenarios):
        print(f"data_file: {data_file} | Scenario {scenario_index + 1}")

        # Generate a random demand for each customer for the current scenario
        demand_values = {c: random.uniform(5, 35) for c in instance.C}
        for c in instance.C:
            instance.demands[c] = demand_values[c]  # Assign to the mutable param

        # Solve the instance with the current demand scenario
        solver.solve(instance, tee=False)

        # Extract first-stage decisions and calculate expected second-stage value
        first_stage_decisions = tuple(int(value(instance.y[p])) for p in instance.P)
        capacity = [value(instance.c[p]) for p in instance.P]
        transportation_cost = [value(instance.t[c, p]) for c in instance.C for p in instance.P]

        # Define a model for the second stage and calculate the expected value
        m = define_model()
        expected_second_stage_value = Q(m, first_stage_decisions, capacity, transportation_cost, size, data_file)

        # Store the results
        first_stage_decisions_lst.append(first_stage_decisions)
        second_stage_value_lst.append(expected_second_stage_value)

    # Post-process results after all scenarios are solved
    pyomo_postprocess(options=None, instance=instance, filename=f"results_{size}.csv",
                      expected_second_stage_value=second_stage_value_lst,
                      first_stage_decisions=first_stage_decisions_lst, size=size)
    
if __name__ == '__main__':
    start_time = time.time()
    problem_sizes = [(10, 10), (25, 25), (50, 50)]
    num_files_to_load = 3
    num_scenarios = 1000  # Number of scenarios for SAA
    
    # Create results directory if it doesn't exist
    if not os.path.exists("results4"):
        os.makedirs("results4")
    
    for size in problem_sizes:
        clients, facilities = size
        num_iteration = 1000 if clients == 10 else 2000  # Adjusted as per size
        
        data_dir = f"data/CPLP_{clients}_{facilities}"
        data_files = glob.glob(os.path.join(data_dir, "*.dat"))
        data_files_sorted = sorted(data_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
        data_files_to_load = data_files_sorted[:num_files_to_load]
        

        # Determine the number of processors to use
        num_processors = min(3, multiprocessing.cpu_count())
        pool = multiprocessing.Pool(processes=num_processors)  
        results = [pool.apply_async(solve_for_file, args=(data_file, size, define_model(), num_iteration)) for data_file in data_files_to_load]
        
        pool.close()
        pool.join()
    
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"The code ran for {int(hours)}h {int(minutes)}m {seconds}s")
    




#### for real ####
# data_dir = f"data/CPLP_{clients}_{facilities}"
# data_files = glob.glob(os.path.join(data_dir, "*.dat"))
# data_files_sorted = sorted(data_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

# pool = multiprocessing.Pool(processes=30)  
# results = [pool.apply_async(solve_for_file, args=(data_file, size, model, num_iteration)) for data_file in data_files_sorted]