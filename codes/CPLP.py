from pyomo.environ import *
import pandas as pd
import os
import random

NUM_ITERATIONS = 1000

# Define the model
model = AbstractModel()

# Define sets
model.plants = Set()  # Set of potential plant locations
model.customers = Set()  # Set of customers

# Define parameters
model.setup_cost = Param(model.plants)  # Cost of setting up a plant at each location
model.capacity = Param(model.plants)  # Capacity of each plant
model.demand = Param(model.customers,mutable=True)  # Demand of each customer
model.transport_cost = Param(model.plants, model.customers)  # Transport cost from plant to customer

# Define variables
model.x = Var(model.plants, model.customers, within=NonNegativeReals)  # Supply ratio from plant to customer
model.y = Var(model.plants, within=Binary)  # Binary variable indicating whether a plant is built

# Define objective function: minimize setup and transport costs
def objective_rule(model):
    return sum(model.setup_cost[p]*model.y[p] for p in model.plants) + \
           sum(model.transport_cost[p, c]*model.demand[c]*model.x[p, c] for p in model.plants for c in model.customers)

model.obj = Objective(rule=objective_rule, sense=minimize)

# Define constraints
def demand_constraint_rule(model, c):
    return sum(model.x[p, c] for p in model.plants) == 1  # Ensure the sum of supply ratios to a customer is 1

model.demand_constraint = Constraint(model.customers, rule=demand_constraint_rule)

def capacity_constraint_rule(model, p):
    # Ensure the total demand served by a plant does not exceed its capacity
    return sum(model.demand[c]*model.x[p, c] for c in model.customers) <= model.capacity[p]*model.y[p]  

model.capacity_constraint = Constraint(model.plants, rule=capacity_constraint_rule)


# Post-processing: display variable values and save to CSV
def pyomo_postprocess(options=None, instance=None, results=None):
    model.x.display()
    
    # Prepare data for CSV
    first_stage_decisions = [int(value(instance.y[p])) for p in instance.plants]
    scenarios = [value(instance.demand[c]) for c in instance.customers]
    
    # Expected second stage value: Sum of transport_cost[p, c] * demand[c] * x[p, c] for all p, c
    expected_second_stage_value = sum(
        value(instance.transport_cost[p, c] * instance.demand[c] * instance.x[p, c])
        for p in instance.plants for c in instance.customers
    )
    
    # Combine all data into a single row
    data = [first_stage_decisions, scenarios, expected_second_stage_value]
    
    # Create DataFrame
    df = pd.DataFrame([data], columns=['first stage decision', 'scenario', 'expected second stage value'])
    
    # Check if the file exists to avoid writing headers multiple times
    file_exists = os.path.isfile('optimization_results.csv')
    
    # Save to CSV (append if file exists, write header only if file does not exist)
    df.to_csv('optimization_results.csv', mode='a', header=not file_exists, index=False)



if __name__ == '__main__':
    for _ in range(NUM_ITERATIONS):
        # Instantiate the model and solve the problem
        instance = model.create_instance('data.dat')
        
        # Modify demand with random values
        for c in instance.customers:
            instance.demand[c] = random.randint(50, 100)  # Random demand between 50 and 100
        
        opt = SolverFactory("glpk")
        results = opt.solve(instance, tee=True)  # tee=True prints the solver log

        # Display results and process output
        results.write()
        print("\nDisplaying Solution\n" + '-'*60)

        # Display optimal cost and optimal values of variables
        print("Optimal Cost:", value(instance.obj))

        print("Optimal Solution:")
        for p in instance.plants:
            print(f"Install plant at location {p}: {int(value(instance.y[p]))}")
            for c in instance.customers:
                print(f"  Supply ratio to customer {c}: {value(instance.x[p, c])}")

        # Post-process results
        pyomo_postprocess(options=None, instance=instance, results=results)