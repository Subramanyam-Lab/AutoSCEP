from pyomo.environ import *
import numpy as np
import multiprocessing
import random

def demand_generate(num_vectors, vector_length):
    demand_vectors = {}
    for vector_num in range(num_vectors):
        demand_key = f'demand_{vector_num}'
        demand_vectors[demand_key] = [random.uniform(5, 35) for _ in range(vector_length)]

    return demand_vectors

def solve_instance(demand_header, demands,instance, solver):
    for idx, demand_value in enumerate(demands[demand_header]):
        client = 'C{}'.format(idx)  # Convert 0, 1, ... to 'C0', 'C1', ...
        instance.demands[client] = demand_value
    result = solver.solve(instance, tee=False)
    if result.solver.termination_condition == TerminationCondition.infeasible:
        return 0
    return value(instance.obj)

def Q(model, y_fixed, capcity, trans_cost, size, data_file, sample_size):
    M = np.sum(trans_cost)
    clients, facilities = size
    second_stage_value_lst = []

    y_fixed_value = {f'P{i}': y_fixed[i] for i in range(len(y_fixed))}
    capacity_value = {f'P{i}': capcity[i] for i in range(len(capcity))}
    trans_dict = {f'C{i}': {f'P{j}': trans_cost[i * facilities + j] for j in range(facilities)} for i in range(clients)}

    def sub_objective_rule(model):
        return sum(trans_dict[c][p] * model.x[c, p] for c in model.C for p in model.P) + M*sum(model.s[p] for p in model.P)

    model.obj = Objective(rule=sub_objective_rule, sense=minimize)

    def sub_demand_constraint_rule(model, c):
        return sum(model.x[c, p] for p in model.P) == 1

    model.demand_constraint = Constraint(model.C, rule=sub_demand_constraint_rule)

    def sub_capacity_constraint_rule(model, p):
        return sum(model.demands[c] * model.x[c, p] for c in model.C) <= capacity_value[p] * y_fixed_value[p]+ model.s[p]

    model.capacity_constraint = Constraint(model.P, rule=sub_capacity_constraint_rule)

    
    solver = SolverFactory('gurobi')
    load_demands = demand_generate(sample_size, clients)
    demand_headers = [f'demand_{i}' for i in range(len(load_demands))]
    
    results_accumulated = []
    previous_checkpoint = 0
    sub_checkpoint = [10*2**i for i in range(10)]
    check_points = sub_checkpoint + [len(demand_headers)] 
    
    for check_point in check_points:
        headers_subset = demand_headers[previous_checkpoint:check_point]
        for header in headers_subset:
            result = solve_instance(header, load_demands, model.create_instance(data_file), solver)
            if np.mean(results_accumulated) > 100:
                break
            results_accumulated.append(result)
        
        if len(results_accumulated) > 1 and np.std(results_accumulated) / np.mean(results_accumulated) <= 0.05:
            print(f"In {data_file}, At checkpoint {check_point}, the ratio of results is below 0.05. Stopping further calculations.")
            print(f"Average result up to this checkpoint: {np.mean(results_accumulated)}")
            break
        
        previous_checkpoint = check_point
                
    return np.mean(results_accumulated)