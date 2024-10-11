from pyomo.environ import *
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(filename='constraint_violations.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def create_model(tab_file_path):
    model = AbstractModel()

    # Sets
    model.Node = Set()
    model.GeneratorsOfNode = Set(dimen=2)
    model.BidirectionalArc = Set(dimen=2)
    model.StoragesOfNode = Set(dimen=2)
    model.PeriodActive = Set()
    model.Technology = Set()
    model.Generator = Set()
    model.GeneratorsOfTechnology = Set(dimen=2)

    # Parameters
    model.genInitCap = Param(model.GeneratorsOfNode, model.PeriodActive, default=0.0, mutable=True)
    model.genMaxBuiltCap = Param(model.Node, model.Technology, model.PeriodActive, default=500000.0, mutable=True)
    model.genMaxInstalledCap = Param(model.Node, model.Technology, model.PeriodActive, default=0.0, mutable=True)
    model.transmissionInitCap = Param(model.BidirectionalArc, model.PeriodActive, default=0.0, mutable=True)
    model.transmissionMaxBuiltCap = Param(model.BidirectionalArc, model.PeriodActive, default=20000.0, mutable=True)
    model.transmissionMaxInstalledCap = Param(model.BidirectionalArc, model.PeriodActive, default=0.0, mutable=True)
    model.storPWInitCap = Param(model.StoragesOfNode, model.PeriodActive, default=0.0, mutable=True)
    model.storPWMaxBuiltCap = Param(model.StoragesOfNode, model.PeriodActive, default=500000.0, mutable=True)
    model.storPWMaxInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, default=0.0, mutable=True)
    model.storENInitCap = Param(model.StoragesOfNode, model.PeriodActive, default=0.0, mutable=True)
    model.storENMaxBuiltCap = Param(model.StoragesOfNode, model.PeriodActive, default=500000.0, mutable=True)
    model.storENMaxInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, default=0.0, mutable=True)

    # Load data
    data = DataPortal()
    data.load(filename=os.path.join(tab_file_path, "Sets_Node.tab"), set=model.Node)
    data.load(filename=os.path.join(tab_file_path, "Sets_GeneratorsOfNode.tab"), set=model.GeneratorsOfNode)
    data.load(filename=os.path.join(tab_file_path, "Sets_BidirectionalArc.tab"), set=model.BidirectionalArc)
    data.load(filename=os.path.join(tab_file_path, "Sets_StoragesOfNode.tab"), set=model.StoragesOfNode)
    data.load(filename=os.path.join(tab_file_path, "Sets_GeneratorsOfTechnology.tab"), set=model.GeneratorsOfTechnology)
    data.load(filename=os.path.join(tab_file_path, "Sets_PeriodActive.tab"), set=model.PeriodActive)
    data.load(filename=os.path.join(tab_file_path, "Sets_Technology.tab"), set=model.Technology)
    data.load(filename=os.path.join(tab_file_path, "Sets_Generator.tab"), set=model.Generator)

    data.load(filename=os.path.join(tab_file_path, "Generator_InitialCapacity.tab"), param=model.genInitCap)
    data.load(filename=os.path.join(tab_file_path, "Generator_MaxBuiltCapacity.tab"), param=model.genMaxBuiltCap)
    data.load(filename=os.path.join(tab_file_path, "Generator_MaxInstalledCapacity.tab"), param=model.genMaxInstalledCap)
    data.load(filename=os.path.join(tab_file_path, "Transmission_InitialCapacity.tab"), param=model.transmissionInitCap)
    data.load(filename=os.path.join(tab_file_path, "Transmission_MaxBuiltCapacity.tab"), param=model.transmissionMaxBuiltCap)
    data.load(filename=os.path.join(tab_file_path, "Transmission_MaxInstallCapacity.tab"), param=model.transmissionMaxInstalledCap)
    data.load(filename=os.path.join(tab_file_path, "Storage_InitialPowerCapacity.tab"), param=model.storPWInitCap)
    data.load(filename=os.path.join(tab_file_path, "Storage_PowerMaxBuiltCapacity.tab"), param=model.storPWMaxBuiltCap)
    data.load(filename=os.path.join(tab_file_path, "Storage_PowerMaxInstalledCapacity.tab"), param=model.storPWMaxInstalledCap)
    data.load(filename=os.path.join(tab_file_path, "Storage_EnergyInitialCapacity.tab"), param=model.storENInitCap)
    data.load(filename=os.path.join(tab_file_path, "Storage_EnergyMaxBuiltCapacity.tab"), param=model.storENMaxBuiltCap)
    data.load(filename=os.path.join(tab_file_path, "Storage_EnergyMaxInstalledCapacity.tab"), param=model.storENMaxInstalledCap)

    return model, data

def check_constraints(instance, sample):
    violations = []

    # Check installed gen cap constraint
    for t in instance.Technology:
        for n in instance.Node:
            for i in instance.PeriodActive:
                installed_cap = sum(sample['genInstalledCap'].get((n, g, i), 0) 
                                    for g in instance.Generator 
                                    if (n, g) in instance.GeneratorsOfNode and (t, g) in instance.GeneratorsOfTechnology)
                if installed_cap > instance.genMaxInstalledCap[n, t, i]:
                    violations.append(f"Installed gen cap exceeded for tech {t}, node {n}, period {i}")


    # Check installed trans cap constraint
    for n1, n2 in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            if sample['transmissionInstalledCap'].get((n1, n2, i), 0) > instance.transmissionMaxInstalledCap[n1, n2, i]:
                violations.append(f"Installed trans cap exceeded for arc ({n1}, {n2}), period {i}")

    # Check installed storage power cap constraint
    for n, b in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            if sample['storPWInstalledCap'].get((n, b, i), 0) > instance.storPWMaxInstalledCap[n, b, i]:
                violations.append(f"Installed storage power cap exceeded for node {n}, storage {b}, period {i}")

    # Check installed storage energy cap constraint
    for n, b in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            if sample['storENInstalledCap'].get((n, b, i), 0) > instance.storENMaxInstalledCap[n, b, i]:
                violations.append(f"Installed storage energy cap exceeded for node {n}, storage {b}, period {i}")

    return violations

def sample_first_stage_decisions(instance):
    max_attempts = 1000
    for attempt in range(max_attempts):
        sampled_values = {
            'genInstalledCap': {},
            'transmissionInstalledCap': {},
            'storPWInstalledCap': {},
            'storENInstalledCap': {}
        }

        # Sample genInstalledCap
        for n, g in instance.GeneratorsOfNode:
            for i in instance.PeriodActive:
                lower_bound = instance.genInitCap[n, g, i]
                t = next(t for t in instance.Technology if (t, g) in instance.GeneratorsOfTechnology)
                upper_bound = min(
                    instance.genMaxBuiltCap[n, t, i],
                    instance.genMaxInstalledCap[n, t, i]
                )
                sampled_values['genInstalledCap'][n, g, i] = random.uniform(lower_bound, upper_bound)

        # Sample transmissionInstalledCap
        for n1, n2 in instance.BidirectionalArc:
            for i in instance.PeriodActive:
                lower_bound = instance.transmissionInitCap[n1, n2, i]
                upper_bound = min(instance.transmissionMaxBuiltCap[n1, n2, i], instance.transmissionMaxInstalledCap[n1, n2, i])
                sampled_values['transmissionInstalledCap'][n1, n2, i] = random.uniform(lower_bound, upper_bound)

        # Sample storPWInstalledCap
        for n, b in instance.StoragesOfNode:
            for i in instance.PeriodActive:
                lower_bound = instance.storPWInitCap[n, b, i]
                upper_bound = min(instance.storPWMaxBuiltCap[n, b, i], instance.storPWMaxInstalledCap[n, b, i])
                sampled_values['storPWInstalledCap'][n, b, i] = random.uniform(lower_bound, upper_bound)

        # Sample storENInstalledCap
        for n, b in instance.StoragesOfNode:
            for i in instance.PeriodActive:
                lower_bound = instance.storENInitCap[n, b, i]
                upper_bound = min(instance.storENMaxBuiltCap[n, b, i], instance.storENMaxInstalledCap[n, b, i])
                sampled_values['storENInstalledCap'][n, b, i] = random.uniform(lower_bound, upper_bound)

        # Check constraints
        violations = check_constraints(instance, sampled_values)
        if not violations:
            logging.info(f"Successfully generated sample on attempt {attempt + 1}")
            return sampled_values
        else:
            for violation in violations:
                logging.info(f"Attempt {attempt + 1}: {violation}")

    raise ValueError(f"Failed to generate a valid sample after {max_attempts} attempts")

def format_sample(sample):
    formatted_data = []
    for var_type, var_data in sample.items():
        for key, value in var_data.items():
            if var_type in ['genInstalledCap', 'storPWInstalledCap', 'storENInstalledCap']:
                n, g, i = key
                energy_type = g
                period = i
                type_ = 'Generation' if var_type == 'genInstalledCap' else 'Storage Power' if var_type == 'storPWInstalledCap' else 'Storage Energy'
            elif var_type == 'transmissionInstalledCap':
                n1, n2, i = key
                energy_type = f"{n1},{n2}"
                period = i
                type_ = 'Transmission'
            
            formatted_data.append({
                'Energy_Type': energy_type,
                'Period': period,
                'Type': type_,
                'Value': value
            })
    return pd.DataFrame(formatted_data)

def generate_samples(model, data, num_samples, output_dir):
    instance = model.create_instance(data)
    
    all_samples = []
    for sample_num in range(num_samples):
        try:
            sample = sample_first_stage_decisions(instance)
            formatted_sample = format_sample(sample)
            all_samples.append(formatted_sample)
            logging.info(f"Generated sample {sample_num + 1}/{num_samples}")
            print(f"Generated sample {sample_num + 1}/{num_samples}")
        except ValueError as e:
            error_msg = f"Failed to generate sample {sample_num + 1}: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
    
    # Combine all samples
    combined_sample = pd.concat(all_samples, ignore_index=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    filename = os.path.join(output_dir, f"samples_{num_samples}_{timestamp}.csv")
    combined_sample.to_csv(filename, index=False)
    logging.info(f"Samples saved to {filename}")
    print(f"Samples saved to {filename}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tab_file_path = os.path.join(script_dir, "Data handler", "sampling", "full")
    output_dir = os.path.join(script_dir, "SeedSamples")
    os.makedirs(output_dir, exist_ok=True)
    num_samples = 10  # Adjust as needed

    model, data = create_model(tab_file_path)
    generate_samples(model, data, num_samples, output_dir)