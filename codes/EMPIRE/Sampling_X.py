from pyomo.environ import *
import pandas as pd
import numpy as np

# Define the model
model = ConcreteModel()

# Define sets
model.GeneratorsOfNode = Set(dimen=2)
model.BidirectionalArc = Set(dimen=2)
model.StoragesOfNode = Set(dimen=2)
model.PeriodActive = Set()

# Define parameters
model.genMaxBuiltCap = Param(model.GeneratorsOfNode, model.PeriodActive, default=500000.0, mutable=True)
model.transmissionMaxBuiltCap = Param(model.BidirectionalArc, model.PeriodActive, default=20000.0, mutable=True)
model.storPWMaxBuiltCap = Param(model.StoragesOfNode, model.PeriodActive, default=500000.0, mutable=True)
model.storENMaxBuiltCap = Param(model.StoragesOfNode, model.PeriodActive, default=500000.0, mutable=True)

# Load parameter data
def load_parameter_data(model, tab_file_path):
    data = DataPortal()
    data.load(filename=tab_file_path + '/' + 'genMaxBuiltCap.csv', param=model.genMaxBuiltCap)
    data.load(filename=tab_file_path + '/' + 'transmissionMaxBuiltCap.csv', param=model.transmissionMaxBuiltCap)
    data.load(filename=tab_file_path + '/' + 'storPWMaxBuiltCap.csv', param=model.storPWMaxBuiltCap)
    data.load(filename=tab_file_path + '/' + 'storENMaxBuiltCap.csv', param=model.storENMaxBuiltCap)
    model.data = data

tab_file_path = 'Data handler/reduced/Tab_Files_' + name
load_parameter_data(model, tab_file_path)

# Define variables
model.genInvCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
model.transmisionInvCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
model.storPWInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
model.storENInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)

# Define constraints
def gen_max_built_cap_rule(model, n, g, i):
    return model.genInvCap[n, g, i] <= model.genMaxBuiltCap[n, g, i]
model.genMaxBuiltCapConstraint = Constraint(model.GeneratorsOfNode, model.PeriodActive, rule=gen_max_built_cap_rule)

def transmission_max_built_cap_rule(model, n1, n2, i):
    return model.transmisionInvCap[n1, n2, i] <= model.transmissionMaxBuiltCap[n1, n2, i]
model.transmissionMaxBuiltCapConstraint = Constraint(model.BidirectionalArc, model.PeriodActive, rule=transmission_max_built_cap_rule)

def stor_pw_max_built_cap_rule(model, n, b, i):
    return model.storPWInvCap[n, b, i] <= model.storPWMaxBuiltCap[n, b, i]
model.storPWMaxBuiltCapConstraint = Constraint(model.StoragesOfNode, model.PeriodActive, rule=stor_pw_max_built_cap_rule)

def stor_en_max_built_cap_rule(model, n, b, i):
    return model.storENInvCap[n, b, i] <= model.storENMaxBuiltCap[n, b, i]
model.storENMaxBuiltCapConstraint = Constraint(model.StoragesOfNode, model.PeriodActive, rule=stor_en_max_built_cap_rule)

# Function to generate a feasible set of samples for a given variable
def generate_feasible_samples(model, variable, num_samples, bounds):
    samples = []
    for _ in range(num_samples):
        sample = {}
        for index in variable:
            sample_value = np.random.uniform(0, bounds[index])
            sample[index] = sample_value
        samples.append(sample)
    return samples

# Function to save samples to CSV
def save_samples_to_csv(samples, filename):
    data = []
    for sample in samples:
        for key, value in sample.items():
            data.append([key[0], key[1], key[2], value])
    df = pd.DataFrame(data, columns=['Node', 'Component', 'Period', 'Value'])
    df.to_csv(filename, index=False)

# Example usage
num_samples = 100  # Number of samples to generate

# Generate and save samples for genInvCap
genInvCap_bounds = {(n, g, i): value(model.genMaxBuiltCap[n, g, i]) for (n, g), i in model.genMaxBuiltCap}
genInvCap_samples = generate_feasible_samples(model, model.genInvCap, num_samples, genInvCap_bounds)
save_samples_to_csv(genInvCap_samples, 'genInvCap_samples.csv')

# Generate and save samples for transmisionInvCap
transmisionInvCap_bounds = {(n1, n2, i): value(model.transmissionMaxBuiltCap[n1, n2, i]) for (n1, n2), i in model.transmissionMaxBuiltCap}
transmisionInvCap_samples = generate_feasible_samples(model, model.transmisionInvCap, num_samples, transmisionInvCap_bounds)
save_samples_to_csv(transmisionInvCap_samples, 'transmisionInvCap_samples.csv')

# Generate and save samples for storPWInvCap
storPWInvCap_bounds = {(n, b, i): value(model.storPWMaxBuiltCap[n, b, i]) for (n, b), i in model.storPWMaxBuiltCap}
storPWInvCap_samples = generate_feasible_samples(model, model.storPWInvCap, num_samples, storPWInvCap_bounds)
save_samples_to_csv(storPWInvCap_samples, 'storPWInvCap_samples.csv')

# Generate and save samples for storENInvCap
storENInvCap_bounds = {(n, b, i): value(model.storENMaxBuiltCap[n, b, i]) for (n, b), i in model.storENMaxBuiltCap}
storENInvCap_samples = generate_feasible_samples(model, model.storENInvCap, num_samples, storENInvCap_bounds)
save_samples_to_csv(storENInvCap_samples, 'storENInvCap_samples.csv')