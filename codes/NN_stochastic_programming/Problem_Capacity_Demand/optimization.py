# optimization.py

from pyomo.environ import *
from model_embedding import load_trained_model, embed_neural_network

def define_optimization_model():
    """
    Define the first-stage optimization model with embedded neural network.
    """
    model = ConcreteModel()

    # First-stage decision variable: capacity
    model.x = Var(domain=NonNegativeReals)

    # Load the trained neural network
    nn_model = load_trained_model('trained_model.h5')

    # Embed the neural network into the Pyomo model
    model = embed_neural_network(nn_model, model, model.x)

    # First-stage cost parameters
    capacity_cost = 5  # Cost per unit of capacity

    # Objective: Minimize total cost = capacity cost + expected second-stage cost
    model.obj = Objective(expr=capacity_cost * model.x + model.expected_cost, sense=minimize)

    # Solve the model
    solver = SolverFactory('ipopt')
    result = solver.solve(model, tee=True)

    print(f"Optimal Capacity: {model.x.value}")
    print(f"Expected Second-Stage Cost: {model.expected_cost.expr()}")
    print(f"Total Cost: {model.obj.expr()}")

    return model

if __name__ == "__main__":
    define_optimization_model()
