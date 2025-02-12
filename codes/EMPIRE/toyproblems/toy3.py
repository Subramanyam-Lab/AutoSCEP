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
import time
from datetime import datetime
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def get_problem_threshold(n_I):
    """Define convergence threshold based on problem size"""
    if n_I <= 5:
        return 5.0
    elif n_I <= 10:
        return 5.0
    elif n_I <= 20:
        return 5.0
    elif n_I <= 40:
        return 5.0
    else:
        return 5.0

def generate_data(n_I, n_J, n_K, n_L, n_scenarios):
    """Generate problem data including uncertainty scenarios"""
    # First stage parameters
    c = {i: np.random.uniform(1, 3) for i in range(1, n_I+1)}
    A = {(j, i): np.random.uniform(0.5, 2.0) for j in range(1, n_J+1) for i in range(1, n_I+1)}
    b = {j: np.random.uniform(8, 15) for j in range(1, n_J+1)}
    
    # Investment impact matrices (D_ij)
    D = {(i, j): np.diag([np.random.uniform(0.8, 1.0) for _ in range(n_K)]) 
         for i in range(1, n_I+1) for j in range(1, i+1)}
    
    # Initial state
    x0 = {i: np.random.uniform(0, 2) for i in range(1, n_I+1)}
    
    # Second stage parameters for each scenario
    scenarios = {}
    for s in range(n_scenarios):
        W = {(l, k): np.random.uniform(0.5, 1.5) for l in range(1, n_L+1) for k in range(1, n_K+1)}
        T = {(l, k): np.random.uniform(0.1, 0.3) for l in range(1, n_L+1) for k in range(1, n_K+1)}
        h = {l: np.random.uniform(20, 30) for l in range(1, n_L+1)}
        q = {k: np.random.uniform(1, 2) for k in range(1, n_K+1)}
        
        scenarios[s] = {'W': W, 'T': T, 'h': h, 'q': q}
    
    return {
        'c': c, 'A': A, 'b': b, 'D': D, 'x0': x0,
        'scenarios': scenarios
    }

def solve_second_stage(v_values, scenario_data, n_K, n_L):
    """Solve second stage problem Q(v, ξ) for given v and scenario"""
    model = pyo.ConcreteModel()
    
    model.K = pyo.RangeSet(1, n_K)
    model.L = pyo.RangeSet(1, n_L)
    
    # Second stage decision variables
    model.y = pyo.Var(model.K, bounds=(0, None))
    
    # Objective: minimize q'y
    def obj_rule(m):
        return sum(scenario_data['q'][k] * m.y[k] for k in m.K)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Constraints: Wy ≥ h - Tv
    def constr_rule(m, l):
        lhs = sum(scenario_data['W'][(l,k)] * m.y[k] for k in m.K)
        rhs = scenario_data['h'][l] - sum(scenario_data['T'][(l,k)] * v_values[k-1] for k in m.K)
        return lhs >= rhs
    model.constraints = pyo.Constraint(model.L, rule=constr_rule)
    
    solver = SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    results = solver.solve(model)
    
    if results.solver.status == pyo.SolverStatus.ok:
        return pyo.value(model.obj), [pyo.value(model.y[k]) for k in model.K]
    else:
        return np.inf, None

def solve_full_model(x_fixed=None, data=None, n_I=5, n_J=3, n_K=3, n_L=2):
    """Solve the complete two-stage model"""
    model = pyo.ConcreteModel()
    
    model.I = pyo.RangeSet(1, n_I)
    model.J = pyo.RangeSet(1, n_J)
    model.K = pyo.RangeSet(1, n_K)
    model.L = pyo.RangeSet(1, n_L)
    
    # Decision variables
    model.x = pyo.Var(model.I, bounds=(0, None))
    model.v = pyo.Var(model.I, model.K, bounds=(0, None))
    model.y = pyo.Var(model.K, bounds=(0, None))
    
    if x_fixed is not None:
        for i in model.I:
            model.x[i].fix(x_fixed[i-1])
    
    # Objective function
    def obj_rule(m):
        first_stage = sum(data['c'][i] * m.x[i] for i in m.I)
        second_stage = sum(data['scenarios'][0]['q'][k] * m.y[k] for k in m.K)
        return first_stage + second_stage
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # First stage constraints
    def resource_constr_rule(m, j):
        return sum(data['A'][(j,i)] * m.x[i] for i in m.I) >= data['b'][j]
    model.resource_constr = pyo.Constraint(model.J, rule=resource_constr_rule)
    
    # State equation
    def state_constr_rule(m, i, k):
        return (m.v[i,k] == data['x0'][i] + 
                sum(data['D'][(i,j)][k-1,k-1] * m.x[j] for j in range(1, i+1)))
    model.state_constr = pyo.Constraint(model.I, model.K, rule=state_constr_rule)


    # Second stage constraints
    def second_stage_constr_rule(m, l):
        scenario = data['scenarios'][0]
        lhs = sum(scenario['W'][(l,k)] * m.y[k] for k in m.K)
        rhs = scenario['h'][l] - sum(scenario['T'][(l,k)] * m.v[n_I,k] for k in m.K)
        return lhs >= rhs
    model.second_stage_constr = pyo.Constraint(model.L, rule=second_stage_constr_rule)
    
    solver = SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    results = solver.solve(model)
    
    if results.solver.status == pyo.SolverStatus.ok:
        return {
            'objective': pyo.value(model.obj),
            'x_values': [pyo.value(model.x[i]) for i in model.I],
            'v_values': [[pyo.value(model.v[i,k]) for k in model.K] for i in model.I],
            'y_values': [pyo.value(model.y[k]) for k in model.K],
            'first_stage_cost': sum(data['c'][i] * pyo.value(model.x[i]) for i in model.I),
            'second_stage_cost': sum(data['scenarios'][0]['q'][k] * pyo.value(model.y[k]) for k in model.K)
        }
    else:
        return None

def ML_model_training(X_train, y_train):
    """Train ML model to approximate second stage cost with scenario features"""
    print("ML model training...")
    ml_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(100, 50),  
            activation='relu',
            alpha=0.2,
            max_iter=1000,
            random_state=42
        ))
    ])
    
    ml_model.fit(X_train, y_train)
    return ml_model

def first_stage_model(data, n_I, n_J, n_K):
    """Create first stage optimization model"""
    model = pyo.ConcreteModel()
    
    model.I = pyo.RangeSet(1, n_I)
    model.J = pyo.RangeSet(1, n_J)
    model.K = pyo.RangeSet(1, n_K)
    
    # First stage decision variables
    model.x = pyo.Var(model.I, bounds=(0, None))
    model.v = pyo.Var(model.I, model.K, bounds=(0, None))
    
    # First stage objective
    def obj_rule(m):
        return sum(data['c'][i] * m.x[i] for i in m.I)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Resource constraints
    def resource_constr_rule(m, j):
        return sum(data['A'][(j,i)] * m.x[i] for i in m.I) >= data['b'][j]
    model.resource_constr = pyo.Constraint(model.J, rule=resource_constr_rule)
    
    # State equation
    def state_constr_rule(m, i, k):
        return (m.v[i,k] == data['x0'][i] + 
                sum(data['D'][(i,j)][k-1,k-1] * m.x[j] for j in range(1, i+1)))
    model.state_constr = pyo.Constraint(model.I, model.K, rule=state_constr_rule)
    
    return model

def generate_convex_combinations(x, n, bounds):
    """Generate convex combinations of points for sampling"""
    x = np.array(x)
    samples = []
    for _ in range(n):
        random_point = np.array([np.random.uniform(low, high) for low, high in bounds])
        alpha = np.random.uniform(0, 1)
        new_sample = alpha * x + (1 - alpha) * random_point
        samples.append(new_sample)
    return samples

def scenario_to_features(scenario):
    """Convert scenario data to flat feature vector"""
    features = []
    # W 행렬 요소들
    for l, k in scenario['W'].keys():
        features.append(scenario['W'][(l,k)])
    # T 행렬 요소들
    for l, k in scenario['T'].keys():
        features.append(scenario['T'][(l,k)])
    # h 벡터
    for l in scenario['h'].keys():
        features.append(scenario['h'][l])
    # q 벡터
    for k in scenario['q'].keys():
        features.append(scenario['q'][k])
    return features


def dataset_split(initial_n_samples, data, n_I, n_K, n_L):
    """Generate initial dataset for ML training with scenario features"""
    print("Initial data generation...")
    X = np.random.uniform(0, 20, size=(initial_n_samples, n_I))
    
    # Calculate total feature dimension
    scenario_dim = n_L * n_K + n_L * n_K + n_L + n_K  # W, T, h, q dimensions
    total_v_dim = n_I * n_K  # Total dimension for v values
    
    # Initialize arrays with correct shapes
    V = np.zeros((initial_n_samples, total_v_dim))
    scenario_features = np.zeros((initial_n_samples, scenario_dim))
    y = np.zeros(initial_n_samples)
    
    for i in range(initial_n_samples):
        # Calculate v values
        v_values = []
        for j in range(n_I):
            for k in range(n_K):
                cumsum = data['x0'][j+1]
                for t in range(j+1):
                    cumsum += data['D'][(j+1,t+1)][k,k] * X[i,t]
                v_values.append(cumsum)
        V[i] = np.array(v_values)
        
        # Get scenario features
        scenario_features[i] = scenario_to_features(data['scenarios'][0])
        
        # Solve second stage
        second_stage_cost, _ = solve_second_stage(v_values, data['scenarios'][0], n_K, n_L)
        y[i] = second_stage_cost
    
    # Combine v_values and scenario features
    X_combined = np.column_stack([V, scenario_features])
    
    print(f"Training data shapes: X_combined: {X_combined.shape}, y: {y.shape}")
    return X_combined, y



def main(n_I, n_J, n_K, n_L, initial_n_samples, threshold, max_iterations=100, seed=42):
    """Main optimization loop"""
    np.random.seed(seed)
    random.seed(seed)
    
    start_time = time.time()
    n_scenarios = 1  # Using single scenario for simplicity
    data = generate_data(n_I, n_J, n_K, n_L, n_scenarios)
    
    iteration = 0
    error_obj_gap = float('inf')
    error_second_gap = float('inf')
    
    # Initialize data storage with correct shapes
    total_v_dim = n_I * n_K
    scenario_dim = n_L * n_K + n_L * n_K + n_L + n_K
    X_data, y_data = dataset_split(initial_n_samples, data, n_I, n_K, n_L)
    
    while (error_obj_gap > threshold) and iteration < max_iterations:
        print(f"\nIteration {iteration+1}:")
        
        if time.time() - start_time > 3600:
            print("Time limit reached")
            break
        
        # Train ML model with properly shaped arrays
        V_train, V_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
        
        ml_model = ML_model_training(V_train, y_train)
        
        # Solve first stage problem with ML approximation
        model = first_stage_model(data, n_I, n_J, n_K)
        solver = SolverFactory('gurobi_persistent')
        solver.set_instance(model)
        gurobi_model = solver._solver_model
        gurobi_model.update()
        
        # Map Pyomo variables to Gurobi variables
        pyomo_var_to_gurobi_var = {}
        for i in model.I:
            var_x = model.x[i]
            gurobi_var_x = solver._pyomo_var_to_solver_var_map[var_x]
            pyomo_var_to_gurobi_var[('x', i)] = gurobi_var_x
            for k in model.K:
                var_v = model.v[i,k]
                gurobi_var_v = solver._pyomo_var_to_solver_var_map[var_v]
                pyomo_var_to_gurobi_var[('v', i, k)] = gurobi_var_v
        
        # Add ML prediction to objective
        y_approx = gurobi_model.addVar(lb=-GRB.INFINITY, name="y_approx")
        gurobi_v_vars = []
        for i in model.I:
            for k in model.K:
                gurobi_v_vars.append(pyomo_var_to_gurobi_var[('v', i, k)])
        
        # Add scenario features as constants
        scenario_features = scenario_to_features(data['scenarios'][0])
        scenario_vars = [gurobi_model.addVar(lb=val, ub=val, name=f"scenario_{i}")
                        for i, val in enumerate(scenario_features)]
        
        # Combine v_vars and scenario_vars for prediction
        all_vars = gurobi_v_vars + scenario_vars
        pred_constr = add_predictor_constr(gurobi_model, ml_model, all_vars, y_approx)
        
        gurobi_model.update()
        existing_obj = gurobi_model.getObjective()
        gurobi_model.setObjective(existing_obj + y_approx, GRB.MINIMIZE)
        
        # Set solver parameters
        gurobi_model.Params.NonConvex = 2
        gurobi_model.Params.TimeLimit = 800
        gurobi_model.setParam("OutputFlag", 0)
        
        # Solve the model
        print("Optimizing embedded model...")
        gurobi_model.optimize()
        
        if gurobi_model.Status == GRB.OPTIMAL or gurobi_model.Status == GRB.TIME_LIMIT:
            # Extract solution
            x_solution = {}
            v_solution = {}
            for i in model.I:
                var_x = model.x[i]
                gurobi_var_x = pyomo_var_to_gurobi_var[('x', i)]
                var_x.set_value(gurobi_var_x.X)
                x_solution[i] = gurobi_var_x.X
                for k in model.K:
                    var_v = model.v[i,k]
                    gurobi_var_v = pyomo_var_to_gurobi_var[('v', i, k)]
                    var_v.set_value(gurobi_var_v.X)
                    v_solution[(i,k)] = gurobi_var_v.X

            # Get solution values
            surrogate_solution = {
                'first_stage_cost': pyo.value(model.obj),
                'second_stage_approx': y_approx.X,
                'v_values': [[v_solution[(i,k)] for k in range(1, n_K+1)] for i in range(1, n_I+1)]
            }

            # Evaluate true solutions
            true_solution = solve_full_model(data=data, n_I=n_I, n_J=n_J, n_K=n_K, n_L=n_L)
            true_obj_embed_sol = solve_full_model(x_fixed=[x_solution[i] for i in model.I], 
                                                data=data, n_I=n_I, n_J=n_J, n_K=n_K, n_L=n_L)

            first_stage_cost = pyo.value(model.obj)
            approx_second_stage = y_approx.X
            true_second_stage_cost, _ = solve_second_stage(
                [v_solution[(i,k)] for i in range(1, n_I+1) for k in range(1, n_K+1)],
                data['scenarios'][0], n_K, n_L
            )


            # Calculate errors
            error_obj = abs(true_solution['objective'] - true_obj_embed_sol['objective'])
            error_obj_gap = (error_obj / true_solution['objective']) * 100
            print(f"Objective Gap: {error_obj_gap:.4f}%")

            error_second = abs(true_second_stage_cost - surrogate_solution['second_stage_approx'])
            error_second_gap = (error_second / true_second_stage_cost) * 100
            print(f"Prediction Gap: {error_second_gap:.4f}%")

            if error_obj_gap <= threshold:
                print("Desired accuracy achieved.")
                break

            # Generate new samples
            print("Generating new samples...")
            bounds = [(0, 20) for _ in range(n_I)]
            new_samples = generate_convex_combinations(
                [x_solution[i] for i in model.I], 
                n=2**(iteration), 
                bounds=bounds
            )

            # Add new samples to training data
            for x_new in new_samples:
                v_new = []
                for i in range(n_I):
                    for k in range(n_K):
                        cumsum = data['x0'][i+1]
                        for j in range(i+1):
                            cumsum += data['D'][(i+1,j+1)][k,k] * x_new[j]
                        v_new.append(cumsum)
                
                # Get scenario features for new sample
                scenario_features = scenario_to_features(data['scenarios'][0])
                
                # Combine v_new and scenario features
                X_new = np.concatenate([v_new, scenario_features])
                
                second_stage_cost, _ = solve_second_stage(v_new, data['scenarios'][0], n_K, n_L)
                
                # Append new data with correct shapes
                X_data = np.vstack([X_data, X_new])
                y_data = np.append(y_data, second_stage_cost)

            iteration += 1
        else:
            print("Failed to optimize")
            print(f"Status: {gurobi_model.Status}")
            break

    # Print final results
    print("\nFinal Optimization Results:")
    print("-" * 40)
    print("\nFirst-Stage Decisions:")
    for i in model.I:
        print(f"x[{i}] = {x_solution[i]:.4f}")
        for k in model.K:
            print(f"v[{i},{k}] = {v_solution[(i,k)]:.4f}")

    if true_solution:
        print("\nSecond-Stage Decisions:")
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
        print(f"Total iterations: {iteration}")
        print(f"Total samples: {len(V_train)}")
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    else:
        print("\nWarning: Failed to find optimal solution")

    return error_obj_gap, error_second_gap, iteration

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
    problem_sizes = [5, 10, 40]
    random_seeds = [int(seed) for seed in np.random.choice(range(1, 101), size=10, replace=False)]
    
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
    df.to_csv(f'detailed_results_{current_time}.csv', index=False)
    # summary_stats.to_csv(f'summary_results_{current_time}.csv')
    print("\nDetailed results saved to:", f'detailed_results_{current_time}.csv')
    # print("Summary results saved to:", f'summary_results_{current_time}.csv')