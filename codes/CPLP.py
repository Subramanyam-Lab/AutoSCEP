from pyomo.environ import *

# Define the model
model = AbstractModel()

# Define sets
model.plants = Set()  # Set of potential plant locations
model.customers = Set()  # Set of customers

# Define parameters
model.setup_cost = Param(model.plants)  # Cost of setting up a plant at each location
model.capacity = Param(model.plants)  # Capacity of each plant
model.demand = Param(model.customers)  # Demand of each customer
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

# Post-processing: display variable values
def pyomo_postprocess(options=None, instance=None, results=None):
  model.x.display()
  
if __name__ == '__main__':
    # Instantiate the model and solve the problem
    instance = model.create_instance('data.dat')
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
            
# aaa
