from pyomo.environ import *
import numpy as np
import multiprocessing

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
                break 

        if mode == "d" and "demand_" in line:
            headers = line.split()
            for header in headers:
                if header != ":=":  
                    demands[header] = []
            continue

        if mode == "d":
            parts = line.split()
            values = [float(value) for value in parts[1:]]

            for header, value in zip(headers, values):
                if header in demands:
                    demands[header].append(value)

    return demands

def solve_instance(demand_header, demands, instance, solver):
    for idx, demand_value in enumerate(demands[demand_header]):
        client = 'C{}'.format(idx)  # Convert 0, 1, ... to 'C0', 'C1', ...
        instance.demands[client] = demand_value

    solver.solve(instance, tee=False)
    return value(instance.obj)

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
    solver = SolverFactory('glpk')
    
    
    results_accumulated = []
    previous_checkpoint = 0
    sub_checkpoint = [10*2**i for i in range(10)]
    check_points = sub_checkpoint + [len(demand_headers)] 
    
    for check_point in check_points:
        headers_subset = demand_headers[previous_checkpoint:check_point]  # 현재 체크 포인트까지의 헤더만 선택
        with multiprocessing.Pool(processes=10) as pool:
            results = pool.starmap(solve_instance, [(header, load_demands, model.create_instance(data_file), solver) for header in headers_subset])
        
        results_accumulated.extend(results)
        
        if len(results_accumulated) > 1 and np.std(results_accumulated) / np.mean(results_accumulated) <= 0.05:
            print(f"At checkpoint {check_point}, the ratio (standard deviation / mean) of results is below 0.05. Stopping further calculations.")
            print(f"Average result up to this checkpoint: {np.mean(results_accumulated)}")
            break
        
        previous_checkpoint = check_point
        
    return np.mean(results_accumulated)

    # # Using multiprocessing to solve the instances in parallel "multiprocessing.cpu_count()"
    # with multiprocessing.Pool(processes=100) as pool:
    #     results = pool.starmap(solve_instance, [(header, load_demands, model.create_instance(data_file), solver) for header in demand_headers])
    # return np.mean(results)