# model_embedding.py

import numpy as np
from pyomo.environ import *
from tensorflow.keras.models import load_model
import joblib

def embed_neural_network(pyomo_model, input_var, model_file, scaler_X_file, scaler_y_file):
    """
    Embed the trained neural network into the Pyomo model.
    """
    # Load the trained neural network model and scalers
    model = load_model(model_file)
    scaler_X = joblib.load(scaler_X_file)
    scaler_y = joblib.load(scaler_y_file)

    # Extract weights and biases
    weights = []
    biases = []
    activation_functions = []

    for layer in model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:
            weights.append(layer_weights[0])
            biases.append(layer_weights[1])
            activation_functions.append(layer.activation.__name__)

    num_layers = len(weights)

    # Define variables for each layer
    layer_sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]
    pyomo_model.z = Var(range(num_layers), range(max(layer_sizes)), domain=Reals)
    pyomo_model.a = Var(range(num_layers), range(max(layer_sizes)), domain=Reals)
    pyomo_model.delta = Var(range(num_layers), range(max(layer_sizes)), domain=Binary)

    big_M = 1e4  # Big-M value, adjust based on variable ranges

    # Input scaling parameters
    X_mean = scaler_X.data_min_[0]
    X_range = scaler_X.data_range_[0]

    # Output scaling parameters
    y_min = scaler_y.data_min_[0]
    y_range = scaler_y.data_range_[0]

    # Layer 0 computations (input layer)
    def input_layer_rule(m, j):
        if j < layer_sizes[0]:
            scaled_input = (input_var - X_mean) / X_range
            return m.z[0, j] == scaled_input * weights[0][0][j] + biases[0][j]
        else:
            return Constraint.Skip
    pyomo_model.input_layer_constraint = Constraint(range(max(layer_sizes)), rule=input_layer_rule)

    # Hidden layers and output layer computations
    def layer_rule(m, l, j):
        if l == 0:
            return Constraint.Skip  # Already handled input layer
        if j < layer_sizes[l]:
            expr = sum(m.a[l - 1, k] * weights[l][k][j] for k in range(layer_sizes[l - 1])) + biases[l][j]
            return m.z[l, j] == expr
        else:
            return Constraint.Skip
    pyomo_model.layer_constraint = Constraint(range(1, num_layers), range(max(layer_sizes)), rule=layer_rule)

    # Activation functions (ReLU)
    def activation_rule(m, l, j):
        if j < layer_sizes[l]:
            # ReLU activation
            # a = max(0, z)
            # Introduce big-M constraints
            return [
                m.a[l, j] >= 0,
                m.a[l, j] >= m.z[l, j],
                m.a[l, j] <= m.z[l, j] + big_M * (1 - m.delta[l, j]),
                m.a[l, j] <= big_M * m.delta[l, j]
            ]
        else:
            return Constraint.Skip
    pyomo_model.activation_constraint = ConstraintList()
    for l in range(num_layers):
        for j in range(layer_sizes[l]):
            constraints = activation_rule(pyomo_model, l, j)
            for c in constraints:
                pyomo_model.activation_constraint.add(c)

    # Output scaling
    def output_rule(m):
        scaled_output = m.a[num_layers - 1, 0]
        output = scaled_output * y_range + y_min
        return m.expected_cost == output
    pyomo_model.output_constraint = Constraint(rule=output_rule)

    return pyomo_model
