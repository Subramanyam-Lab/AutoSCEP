# model_embedding.py

import numpy as np
from pyomo.environ import *
from tensorflow import keras

def load_trained_model(model_file):
    """
    Load the trained neural network model.
    """
    model = keras.models.load_model(model_file)
    return model

def embed_neural_network(model, pyomo_model, input_var, output_var_name='expected_cost'):
    """
    Embed the neural network into the Pyomo model.
    """
    # Extract weights and biases
    weights = []
    biases = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:
            weights.append(layer_weights[0])
            biases.append(layer_weights[1])

    num_layers = len(weights)

    # Define variables for each layer
    pyomo_model.layer_vars = Var(range(num_layers), domain=Reals)
    pyomo_model.layer_output = Var(range(num_layers), domain=Reals)

    # Input layer computation
    def input_layer_rule(m):
        return m.layer_vars[0] == weights[0][0][0] * input_var + biases[0][0]
    pyomo_model.input_layer = Constraint(rule=input_layer_rule)

    # Hidden layers computation
    def hidden_layer_rule(m, l):
        if l == 0:
            return Constraint.Skip
        prev_output = m.layer_output[l - 1]
        return m.layer_vars[l] == weights[l][0][0] * prev_output + biases[l][0]
    pyomo_model.hidden_layers = Constraint(range(num_layers), rule=hidden_layer_rule)

    # Activation function (ReLU)
    def activation_rule(m, l):
        return m.layer_output[l] == max(0, m.layer_vars[l])
    pyomo_model.activation = Constraint(range(num_layers), rule=activation_rule)

    # Output variable
    setattr(pyomo_model, output_var_name, Expression(expr=m.layer_output[num_layers - 1]))

    return pyomo_model
