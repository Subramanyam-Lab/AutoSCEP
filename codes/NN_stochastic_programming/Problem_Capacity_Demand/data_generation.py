# data_generation.py

import numpy as np
import pandas as pd
from pyomo.environ import *

def generate_demand_scenarios(num_scenarios):
    """
    Generate random demand scenarios.
    """
    np.random.seed(42)
    demands = np.random.normal(loc=100, scale=20, size=num_scenarios)
    demands = np.clip(demands, a_min=0, a_max=None)  # Demand cannot be negative
    return demands

def solve_second_stage(capacity, demand, production_cost=2):
    """
    Solve the second-stage problem for a given capacity and demand.
    """
    model = ConcreteModel()
    model.y = Var(domain=NonNegativeReals)

    # Objective: Minimize production cost
    model.obj = Objective(expr=production_cost * model.y, sense=minimize)

    # Constraints
    model.capacity_constraint = Constraint(expr=model.y <= capacity)
    model.demand_constraint = Constraint(expr=model.y <= demand)

    # Solve the model
    solver = SolverFactory('glpk')
    solver.solve(model, tee=False)

    production = model.y.value
    cost = production_cost * production
    return cost

def generate_training_data(num_samples, num_scenarios):
    """
    Generate training data consisting of sampled capacities and expected second-stage costs.
    """
    capacities = np.linspace(50, 150, num_samples)  # Sample capacities
    demands = generate_demand_scenarios(num_scenarios)
    training_data = []

    for capacity in capacities:
        costs = []
        for demand in demands:
            cost = solve_second_stage(capacity, demand)
            costs.append(cost)
        expected_cost = np.mean(costs)
        training_data.append({'capacity': capacity, 'expected_cost': expected_cost})
        print(f"Capacity: {capacity}, Expected Cost: {expected_cost}")

    df = pd.DataFrame(training_data)
    return df

if __name__ == "__main__":
    # Parameters
    num_samples = 50       # Number of capacity samples
    num_scenarios = 1000   # Number of demand scenarios per capacity

    # Generate training data
    training_data = generate_training_data(num_samples, num_scenarios)

    # Save the training data to a CSV file
    training_data.to_csv('training_data.csv', index=False)
