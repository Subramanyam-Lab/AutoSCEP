# optimization.py

from pyomo.environ import *
from pyomo.opt import SolverFactory
from model_embedding import embed_neural_network

def define_optimization_model():
    """
    Define the first-stage optimization model with the embedded neural network.
    """
    model = ConcreteModel()

    # First-stage decision variable: capacity
    model.x = Var(domain=NonNegativeReals)

    # Embed the neural network into the model
    model = embed_neural_network(
        pyomo_model=model,
        input_var=model.x,
        model_file='trained_model.h5',
        scaler_X_file='scaler_X.pkl',
        scaler_y_file='scaler_y.pkl'
    )

    # First-stage cost parameters
    capacity_cost = 5  # Cost per unit of capacity

    # Objective: Minimize total cost = capacity cost + expected second-stage cost
    model.expected_cost = Var(domain=Reals)
    model.total_cost = Objective(expr=capacity_cost * model.x + model.expected_cost, sense=minimize)

    # Solve the model using a MILP solver
    solver = SolverFactory('glpk')  
    result = solver.solve(model, tee=True)

    print(f"Optimal Capacity: {model.x.value}")
    print(f"Expected Second-Stage Cost: {model.expected_cost.value}")
    print(f"Total Cost: {model.total_cost.expr()}")

    return model

if __name__ == "__main__":
    define_optimization_model()
