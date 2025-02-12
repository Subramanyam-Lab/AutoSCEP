import random
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from gurobi_ml import add_predictor_constr
from gurobipy import GRB
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from datetime import datetime
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')



def generate_data(n_I, n_J, n_K, n_L):
    cost_coefficients = {i: np.random.uniform(1, 3) for i in range(1, n_I+1)}
    tech_matrix = {(j, i): np.random.uniform(0.5, 2.0) for j in range(1, n_J+1) for i in range(1, n_I+1)}
    rhs_values = {j: np.random.uniform(8, 15) for j in range(1, n_J+1)}

    d = {k: np.random.uniform(3, 5) for k in range(1, n_K+1)}
    p_data = {(l, k): np.random.uniform(1.0, 2.0) for l in range(1, n_L+1) for k in range(1, n_K+1)}
    q_data = {(l, i): np.random.uniform(0.1, 0.5) for l in range(1, n_L+1) for i in range(1, n_I+1)}
    r_data = {l: np.random.uniform(20, 30) for l in range(1, n_L+1)}

    return cost_coefficients, tech_matrix, rhs_values, d, p_data, q_data, r_data

def solve_full_model(x_fixed=None, data=None, n_I=5, n_J=3, n_K=3, n_L=2):
    cost_coefficients, tech_matrix, rhs_values, d, p_data, q_data, r_data = data

    model = pyo.ConcreteModel()

    model.I = pyo.RangeSet(1, n_I)
    model.J = pyo.RangeSet(1, n_J)
    model.K = pyo.RangeSet(1, n_K)
    model.L = pyo.RangeSet(1, n_L)

    model.costs = pyo.Param(model.I, initialize=cost_coefficients)
    model.tech_coef = pyo.Param(model.J, model.I, initialize=tech_matrix)
    model.rhs = pyo.Param(model.J, initialize=rhs_values)

    model.d = pyo.Param(model.K, initialize=d)
    model.p = pyo.Param(model.L, model.K, initialize=p_data)
    model.q = pyo.Param(model.L, model.I, initialize=q_data)
    model.r = pyo.Param(model.L, initialize=r_data)

    model.x = pyo.Var(model.I, bounds=(0, 20))
    model.y = pyo.Var(model.K, bounds=(0, 30))

    if x_fixed is not None:
        for i in model.I:
            model.x[i].fix(x_fixed[i-1])

    def obj_rule(m):
        first_stage = sum(m.costs[i] * m.x[i] for i in m.I)
        second_stage = sum(m.d[k] * m.y[k] for k in m.K)
        return first_stage + second_stage
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def first_stage_constr_rule(m, j):
        return sum(m.tech_coef[j, i] * m.x[i] for i in m.I) >= m.rhs[j]
    model.first_stage_constr = pyo.Constraint(model.J, rule=first_stage_constr_rule)

    def second_stage_constr_rule(m, l):
        return (sum(m.p[l, k] * m.y[k] for k in m.K) +
                sum(m.q[l, i] * m.x[i] for i in m.I) >= m.r[l])
    model.second_stage_constr = pyo.Constraint(model.L, rule=second_stage_constr_rule)

    solver = SolverFactory('gurobi')
    results = solver.solve(model)

    if results.solver.status == pyo.SolverStatus.ok:
        return {
            'objective': pyo.value(model.obj),
            'x_values': [pyo.value(model.x[i]) for i in model.I],
            'y_values': [pyo.value(model.y[k]) for k in model.K],
            'first_stage_cost': sum(model.costs[i] * pyo.value(model.x[i]) for i in model.I),
            'second_stage_cost': sum(d[k] * pyo.value(model.y[k]) for k in model.K)
        }
    else:
        return None

def solve_second_stage(x_values, data, n_I=5, n_K=3, n_L=2):
    cost_coefficients, tech_matrix, rhs_values, d, p_data, q_data, r_data = data

    model = pyo.ConcreteModel()

    model.K = pyo.RangeSet(1, n_K)
    model.L = pyo.RangeSet(1, n_L)

    model.d = pyo.Param(model.K, initialize=d)
    model.p = pyo.Param(model.L, model.K, initialize=p_data)
    model.q = pyo.Param(model.L, range(1, n_I+1), initialize=q_data)
    model.r = pyo.Param(model.L, initialize=r_data)

    model.y = pyo.Var(model.K, bounds=(0, 30))

    def obj_rule(m):
        return sum(m.d[k] * m.y[k] for k in m.K)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def const_rule(m, l):
        lhs = sum(m.p[l, k] * m.y[k] for k in m.K)
        rhs = m.r[l] - sum(m.q[l, i] * x_values[i-1] for i in range(1, n_I+1))
        return lhs >= rhs
    model.constraints = pyo.Constraint(model.L, rule=const_rule)

    solver = SolverFactory('gurobi')
    results = solver.solve(model)

    if results.solver.status == pyo.SolverStatus.ok:
        return pyo.value(model.obj)
    else:
        return np.inf

def ML_model_training(X_train, y_train):
    print("ML model training...")
    ml_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            alpha=0.2,
            max_iter=500,
            random_state=42
        ))
    ])

    ml_model.fit(X_train, y_train)
    return ml_model

def first_stage_model(data, n_I=5, n_J=3):
    cost_coefficients, tech_matrix, rhs_values, _, _, _, _ = data

    model = pyo.ConcreteModel()

    model.I = pyo.RangeSet(1, n_I)
    model.J = pyo.RangeSet(1, n_J)

    model.costs = pyo.Param(model.I, initialize=cost_coefficients)
    model.tech_coef = pyo.Param(model.J, model.I, initialize=tech_matrix)
    model.rhs = pyo.Param(model.J, initialize=rhs_values)

    model.x = pyo.Var(model.I, within=pyo.Reals, bounds=(0, 20))

    def first_stage_obj_rule(m):
        return sum(m.costs[i] * m.x[i] for i in m.I)
    model.first_stage_obj = pyo.Objective(rule=first_stage_obj_rule, sense=pyo.minimize)

    def first_stage_constr_rule(m, j):
        return sum(m.tech_coef[j, i] * m.x[i] for i in m.I) >= m.rhs[j]
    model.first_stage_constr = pyo.Constraint(model.J, rule=first_stage_constr_rule)

    return model

def generate_convex_combinations(x, n, bounds):
    x = np.array(x)
    samples = []
    for _ in range(n):
        random_point = np.array([np.random.uniform(low, high) for low, high in bounds])
        alpha = np.random.uniform(0, 1)
        new_sample = alpha * x + (1 - alpha) * random_point
        samples.append(new_sample)
    return samples

def dataset_split(initial_n_samples, data, n_I, n_K, n_L):
    print("Initial data generation...")
    X = np.random.uniform(0, 20, size=(initial_n_samples, n_I))
    y = [solve_second_stage(x, data, n_I, n_K, n_L) for x in X]

    return X.tolist(),y



def main(n_I, n_J, n_K, n_L, initial_n_samples, threshold, max_iterations=100, seed=42):

    np.random.seed(seed)
    random.seed(seed)
    data = generate_data(n_I, n_J, n_K, n_L)

    iteration = 0
    error_obj_gap = float('inf')
    error_second_gap = float('inf')

    X_lst, y_lst= dataset_split(initial_n_samples, data, n_I, n_K, n_L)

    while (error_obj_gap > threshold) and iteration < max_iterations:
        print(f"\nIteration {iteration+1}:")

        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X_lst), np.array(y_lst), test_size=0.2, random_state=42
        )


        ml_model = ML_model_training(X_train, y_train)

        model = first_stage_model(data, n_I, n_J)

        solver = SolverFactory('gurobi_persistent')
        solver.set_instance(model)
        gurobi_model = solver._solver_model
        gurobi_model.update()

        pyomo_var_to_gurobi_var = {}
        for i in model.I:
            var = model.x[i]
            gurobi_var = solver._pyomo_var_to_solver_var_map[var]
            pyomo_var_to_gurobi_var[i] = gurobi_var

        y_approx = gurobi_model.addVar(lb=-GRB.INFINITY, name="y_approx")
        gurobi_x_vars = [pyomo_var_to_gurobi_var[i] for i in model.I]
        pred_constr = add_predictor_constr(gurobi_model, ml_model, gurobi_x_vars, y_approx)

        gurobi_model.update()
        existing_obj = gurobi_model.getObjective()
        gurobi_model.setObjective(existing_obj + y_approx, GRB.MINIMIZE)

        gurobi_model.Params.NonConvex = 2
        gurobi_model.Params.TimeLimit = 400
        gurobi_model.setParam("OutputFlag", 0)

        print("Optimizing embedded model...")
        gurobi_model.optimize()

        if gurobi_model.Status == GRB.OPTIMAL or gurobi_model.Status == GRB.TIME_LIMIT:
            solution = {}
            for i in model.I:
                var = model.x[i]
                gurobi_var = pyomo_var_to_gurobi_var[i]
                var.set_value(gurobi_var.X)
                solution[i] = gurobi_var.X

            surrogate_solution = {
                'first_stage_cost': pyo.value(model.first_stage_obj),
                'second_stage_approx': y_approx.X
            }

            true_solution = solve_full_model(data=data, n_I=n_I, n_J=n_J, n_K=n_K, n_L=n_L)
            true_obj_embed_sol = solve_full_model(x_fixed=[solution[i] for i in model.I], data=data, n_I=n_I, n_J=n_J, n_K=n_K, n_L=n_L)

            first_stage_cost = pyo.value(model.first_stage_obj)
            approx_second_stage = y_approx.X
            true_second_stage = solve_second_stage([solution[i] for i in model.I], data, n_I, n_K, n_L)

            error_obj = abs(true_solution['objective'] - true_obj_embed_sol['objective'])
            error_obj_gap = (error_obj / true_solution['objective']) * 100
            print(f"Objective Gap: {error_obj_gap:.4f}%")

            error_second = abs(true_solution['second_stage_cost'] - surrogate_solution['second_stage_approx'])
            error_second_gap = (error_second / true_solution['second_stage_cost']) * 100
            print(f"Prediction Gap: {error_second_gap:.4f}%")

            if error_obj_gap <= threshold:
                print("Desired accuracy achieved.")
                break

            print("Generating new samples...")
            bounds = [(0, 20) for _ in range(n_I)]
            new_samples = generate_convex_combinations([solution[i] for i in model.I], n=2**(iteration), bounds=bounds)

            for x_new in new_samples:
                y_new = solve_second_stage(x_new, data, n_I, n_K, n_L)
                X_lst.append(x_new)
                y_lst.append(y_new)

            iteration += 1
        else:
            print("Fail to optimize")
            print(f"Status: {gurobi_model.Status}")
            break

    print("\nFinal Optimization Results:")
    print("-" * 40)
    print("\nFirst-Stage Decisions:")
    for i in model.I:
        print(f"x[{i}] = {solution[i]:.4f}")

    if true_solution:
        print("\nDecision variables (y):")
        for k in range(n_K):
            print(f"y[{k+1}] = {true_solution['y_values'][k]:.4f}")

        print("\nCost Analysis:")
        print(f"First-Stage Cost: {surrogate_solution['first_stage_cost']:.4f}")
        print(f"Approximated Second-Stage Cost: {surrogate_solution['second_stage_approx']:.4f}")
        print(f"Actual Second-Stage Cost: {true_solution['second_stage_cost']:.4f}")
        print(f"Original model cost: {true_solution['objective']:.4f}")
        print(f"Embedded model cost: {surrogate_solution['first_stage_cost'] + surrogate_solution['second_stage_approx']:.4f}")

        print("\nSolution Error Analysis:")
        print(f"Relative prediction error: {error_second_gap:.8f}%")

    else:
        print("\nWarning: Fail")

    return error_obj_gap, error_second_gap, iteration

def get_problem_threshold(n_I):
    if n_I <= 5:
        return 3.0
    elif n_I <= 10:
        return 3.0
    elif n_I <= 20:
        return 3.0
    elif n_I <= 40:
        return 3.0
    else:
        return 3.0


def run_experiment_with_seed(seed, problem_sizes):
    results = []
    print(f"\nRunning experiments with seed {seed}")
    
    for n_I in problem_sizes:
        n_J = max(3, int(n_I * 0.6))  
        n_K = max(3, int(n_I * 1.2))  
        n_L = max(2, int(n_I * 0.8))  
        
        initial_n_samples = 2  
        threshold = get_problem_threshold(n_I)  
        
        print(f"\nSeed {seed} - Running with n_I={n_I}, n_J={n_J}, n_K={n_K}, n_L={n_L}")
        print(f"Initial samples: {initial_n_samples}, Threshold: {threshold}%")
        
        optimality_gap, prediction_gap, iterations = main(
            n_I, n_J, n_K, n_L, 
            initial_n_samples, threshold, seed=seed
        )
        
        results.append({
            'seed': seed,
            'n_I': n_I,
            'n_J': n_J,
            'n_K': n_K,
            'n_L': n_L,
            'final_samples': initial_n_samples + sum(2**i for i in range(iterations)),
            'iterations': iterations,
            'optimality_gap': optimality_gap,
            'prediction_gap': prediction_gap
        })
        
    return results


if __name__ == "__main__":
    problem_sizes = [5, 10, 20, 40, 80]
    random_seeds = [int(seed) for seed in np.random.choice(range(1, 101), size=30, replace=False)]
    
    num_cores = max(1, int(mp.cpu_count() * 0.75))
    print(f"Using {num_cores} CPU cores")
    
    with mp.Pool(num_cores) as pool:
        partial_run = partial(run_experiment_with_seed, problem_sizes=problem_sizes)
        all_results = pool.map(partial_run, random_seeds)
    
    flattened_results = [item for sublist in all_results for item in sublist]
    df = pd.DataFrame(flattened_results)
    
    summary_stats = df.groupby('n_I').agg({
        'optimality_gap': ['mean', 'std'],
        'prediction_gap': ['mean', 'std'],
        'iterations': ['mean', 'std'],
        'final_samples': ['mean', 'std']
    }).round(4)

    print("\nSummary Statistics:")
    print(summary_stats)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'detailed_results_{current_time}_toy1.csv', index=False)
    summary_stats.to_csv(f'summary_results_{current_time}_toy1.csv')
    print("\nDetailed results saved to:", f'detailed_results_{current_time}_toy1.csv')
    print("Summary results saved to:", f'summary_results_{current_time}_toy1.csv')
