from pyomo.environ import *
import numpy as np
import multiprocessing

def solve_instance(instance, solver):
    demand_values = {c: random.uniform(5, 35) for c in instance.C}
    for c in instance.C:
        instance.demands[c] = demand_values[c]
    solver.solve(instance, tee=False)
    return value(instance.obj)

def Q(model, y_fixed, capcity, trans_cost, size, data_file, sample_size):
    print("Q entered")
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

    solver = SolverFactory('glpk')
    
    results_accumulated = []
    previous_checkpoint = 0
    sub_checkpoint = [10*2**i for i in range(10)]
    check_points = sub_checkpoint + [len(demand_headers)] 
    
    for check_point in check_points:
        for i in range(sample_size):
            print(i)
            result = solve_instance(model.create_instance(data_file), solver)
            print(result)
            results_accumulated.append(result)
        
        if len(results_accumulated) > 1 and np.std(results_accumulated) / np.mean(results_accumulated) <= 0.05:
            print(f"In {data_file}, At checkpoint {check_point}, the ratio of results is below 0.05. Stopping further calculations.")
            print(f"Average result up to this checkpoint: {np.mean(results_accumulated)}")
            break
        
        previous_checkpoint = check_point
        
    
                
    return np.mean(results_accumulated)