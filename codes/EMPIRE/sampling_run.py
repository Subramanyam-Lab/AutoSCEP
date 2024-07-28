#!/usr/bin/env python
from reader import generate_tab_files
from datetime import datetime
from yaml import safe_load
import time
import pandas as pd
import numpy as np
import os
import logging
from pyomo.environ import DataPortal

__author__ = "Stian Backe"
__license__ = "MIT"
__maintainer__ = "Stian Backe"
__email__ = "stian.backe@ntnu.no"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
temp_dir = UserRunTimeConfig["temp_dir"]
version = UserRunTimeConfig["version"]
Horizon = UserRunTimeConfig["Horizon"]
NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
discountrate = UserRunTimeConfig["discountrate"]
WACC = UserRunTimeConfig["WACC"]
scenariogeneration = UserRunTimeConfig["scenariogeneration"]
fix_sample = UserRunTimeConfig["fix_sample"]
LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
filter_make = UserRunTimeConfig["filter_make"] 
filter_use = UserRunTimeConfig["filter_use"]
n_cluster = UserRunTimeConfig["n_cluster"]
moment_matching = UserRunTimeConfig["moment_matching"]
n_tree_compare = UserRunTimeConfig["n_tree_compare"]

#############################
##Non configurable settings##
#############################
NoOfRegSeason = 4
regular_seasons = ["winter", "spring", "summer", "fall"]
NoOfPeakSeason = 2
lengthPeakSeason = 7
LeapYearsInvestment = 5
time_format = "%d/%m/%Y %H:%M"

def load_empire_data(tab_file_path):
    """
    Load relevant data from EMPIRE model files.
    """
    data = DataPortal()
    

    num_periods = int((2060 - 2020) / 5)
    period_set = list(range(1, num_periods + 1))
    data['Period'] = {None: period_set}
    logging.info(f"Generated Period set: {period_set}")

    # Load sets
    data.load(filename=os.path.join(tab_file_path, "Sets_Generator.tab"), format="set", set="Generator")
    data.load(filename=os.path.join(tab_file_path, "Sets_Node.tab"), format="set", set="Node")
    data.load(filename=os.path.join(tab_file_path, "Sets_Storage.tab"), format="set", set="Storage")
    data.load(filename=os.path.join(tab_file_path, "Sets_Technology.tab"), format="set", set="Technology")
    data.load(filename=os.path.join(tab_file_path, "Sets_DirectionalLines.tab"), format="set", set="DirectionalLink")
    data.load(filename=os.path.join(tab_file_path, "Sets_GeneratorsOfNode.tab"), format="set", set="GeneratorsOfNode")
    data.load(filename=os.path.join(tab_file_path, "Sets_StorageOfNodes.tab"), format="set", set="StoragesOfNode")
    data.load(filename=os.path.join(tab_file_path, "Sets_GeneratorsOfTechnology.tab"), format="set", set="GeneratorsOfTechnology")
    # data.load(filename=os.path.join(tab_file_path, "Sets_BidirectionalArc.tab"), format="set", set="BidirectionalArc")

    # Load parameters
    data.load(filename=os.path.join(tab_file_path, "Generator_MaxBuiltCapacity.tab"), param="genMaxBuiltCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Transmission_MaxBuiltCapacity.tab"), param="transmissionMaxBuiltCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Storage_PowerMaxBuiltCapacity.tab"), param="storPWMaxBuiltCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Storage_EnergyMaxBuiltCapacity.tab"), param="storENMaxBuiltCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Generator_MaxInstalledCapacity.tab"), param="genMaxInstalledCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Transmission_MaxInstallCapacityRaw.tab"), param="transmissionMaxInstalledCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Storage_PowerMaxInstalledCapacity.tab"), param="storPWMaxInstalledCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Storage_EnergyMaxInstalledCapacity.tab"), param="storENMaxInstalledCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Generator_InitialCapacity.tab"), param="genInitCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Transmission_InitialCapacity.tab"), param="transmissionInitCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Storage_InitialPowerCapacity.tab"), param="storPWInitCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Storage_EnergyInitialCapacity.tab"), param="storENInitCap", format="table")
    data.load(filename=os.path.join(tab_file_path, "Generator_Lifetime.tab"), param="genLifetime", format="table")
    data.load(filename=os.path.join(tab_file_path, "Storage_Lifetime.tab"), param="storageLifetime", format="table")
    data.load(filename=os.path.join(tab_file_path, "Transmission_Lifetime.tab"), param="transmissionLifetime", format="table")
    

    # Generate BidirectionalArc set
    directional_link = data['DirectionalLink']
    bidirectional_arc = set()
    for i, j in directional_link:
        if i != j and (j, i) not in bidirectional_arc:
            bidirectional_arc.add((i, j))
    data['BidirectionalArc'] = {None: list(bidirectional_arc)}
    logging.info(f"Generated BidirectionalArc set with {len(bidirectional_arc)} elements")

    return data


def generate_sample(data):
    """
    Generate a sample based on the loaded EMPIRE data.
    """
    sample = []
    
    # Sample genInvCap
    for n, g in data['GeneratorsOfNode']:
        for i in data['Period']:
            if (n, g, i) in data['genMaxBuiltCap']:
                sample.append({
                    'Country': n,
                    'Energy_Type': g,
                    'Period': i,
                    'Type': 'Generation',
                    'Value': np.random.uniform(0, data['genMaxBuiltCap'][n, g, i])
                })
    
    # Sample transmisionInvCap
    for (n1, n2) in data['DirectionalLink']:
        for i in data['Period']:
            if (n1, n2, i) in data['transmissionMaxBuiltCap']:
                sample.append({
                    'Country': f"{n1},{n2}",
                    'Energy_Type': 'Transmission',
                    'Period': i,
                    'Type': 'Transmission',
                    'Value': np.random.uniform(0, data['transmissionMaxBuiltCap'][n1, n2, i])
                })
    
    # Sample storPWInvCap and storENInvCap
    for n in data['Node']:
        for b in data['Storage']:
            for i in data['Period']:
                if (n, b, i) in data['storPWMaxBuiltCap']:
                    sample.append({
                        'Country': n,
                        'Energy_Type': b,
                        'Period': i,
                        'Type': 'Storage Power',
                        'Value': np.random.uniform(0, data['storPWMaxBuiltCap'][n, b, i])
                    })
                if (n, b, i) in data['storENMaxBuiltCap']:
                    sample.append({
                        'Country': n,
                        'Energy_Type': b,
                        'Period': i,
                        'Type': 'Storage Energy',
                        'Value': np.random.uniform(0, data['storENMaxBuiltCap'][n, b, i])
                    })
    
    return pd.DataFrame(sample)

CONSTRAINT_TOLERANCE = 1e-6  # You can adjust this value as needed

def check_constraints(sample, data):
    """
    Check if the sample satisfies the constraints.
    """
    sample_dict = {(row['Country'], row['Energy_Type'], row['Period'], row['Type']): row['Value'] 
                   for _, row in sample.iterrows()}
    
    def get_start_period(current_period, lifetime, leap_years):
        start_period = 1
        if 1 + current_period - (lifetime / leap_years) > start_period:
            start_period = 1 + current_period - lifetime / leap_years
        return max(1, int(start_period))

    def log_constraint_violation(constraint_type, details):
        logging.debug(f"Constraint violation: {constraint_type} - {details}")

    # Check generator lifetime and capacity constraints
    for n, g in data['GeneratorsOfNode']:
        for i in data['Period']:
            start_period = get_start_period(i, data['genLifetime'][g], LeapYearsInvestment)
            installed_cap = sum(sample_dict.get((n, g, j, 'Generation'), 0) for j in range(start_period, i+1))
            if abs(installed_cap - sample_dict.get((n, g, i, 'Generation'), 0) + data['genInitCap'].get((n, g, i), 0)) > CONSTRAINT_TOLERANCE:
                log_constraint_violation("Generator lifetime", f"Node: {n}, Generator: {g}, Period: {i}")
                return False
            if installed_cap > data['genMaxInstalledCap'].get((n, g, i), float('inf')):
                log_constraint_violation("Generator max capacity", f"Node: {n}, Generator: {g}, Period: {i}")
                return False

    # Check storage energy lifetime and capacity constraints
    for n, b in data['StoragesOfNode']:
        for i in data['Period']:
            start_period = get_start_period(i, data['storageLifetime'][b], LeapYearsInvestment)
            installed_cap = sum(sample_dict.get((n, b, j, 'Storage Energy'), 0) for j in range(start_period, i+1))
            if abs(installed_cap - sample_dict.get((n, b, i, 'Storage Energy'), 0) + data['storENInitCap'].get((n, b, i), 0)) > CONSTRAINT_TOLERANCE:
                log_constraint_violation("Storage energy lifetime", f"Node: {n}, Storage: {b}, Period: {i}")
                return False
            if installed_cap > data['storENMaxInstalledCap'].get((n, b, i), float('inf')):
                log_constraint_violation("Storage energy max capacity", f"Node: {n}, Storage: {b}, Period: {i}")
                return False

    # Check storage power lifetime and capacity constraints
    for n, b in data['StoragesOfNode']:
        for i in data['Period']:
            start_period = get_start_period(i, data['storageLifetime'][b], LeapYearsInvestment)
            installed_cap = sum(sample_dict.get((n, b, j, 'Storage Power'), 0) for j in range(start_period, i+1))
            if abs(installed_cap - sample_dict.get((n, b, i, 'Storage Power'), 0) + data['storPWInitCap'].get((n, b, i), 0)) > CONSTRAINT_TOLERANCE:
                log_constraint_violation("Storage power lifetime", f"Node: {n}, Storage: {b}, Period: {i}")
                return False
            if installed_cap > data['storPWMaxInstalledCap'].get((n, b, i), float('inf')):
                log_constraint_violation("Storage power max capacity", f"Node: {n}, Storage: {b}, Period: {i}")
                return False

    # Check transmission lifetime and capacity constraints
    for n1, n2 in data['BidirectionalArc']:
        for i in data['Period']:
            start_period = get_start_period(i, data['transmissionLifetime'].get((n1, n2), 40), LeapYearsInvestment)
            installed_cap = sum(sample_dict.get((f"{n1},{n2}", 'Transmission', j, 'Transmission'), 0) for j in range(start_period, i+1))
            if abs(installed_cap - sample_dict.get((f"{n1},{n2}", 'Transmission', i, 'Transmission'), 0) + data['transmissionInitCap'].get((n1, n2, i), 0)) > CONSTRAINT_TOLERANCE:
                log_constraint_violation("Transmission lifetime", f"Link: {n1}-{n2}, Period: {i}")
                return False
            if installed_cap > data['transmissionMaxInstalledCap'].get((n1, n2, i), float('inf')):
                log_constraint_violation("Transmission max capacity", f"Link: {n1}-{n2}, Period: {i}")
                return False

    # Check investment constraints
    for t in data['Technology']:
        for n in data['Node']:
            for i in data['Period']:
                gen_inv = sum(sample_dict.get((n, g, i, 'Generation'), 0) for g in data['Generator'] 
                              if (n, g) in data['GeneratorsOfNode'] and (t, g) in data['GeneratorsOfTechnology'])
                if gen_inv > data['genMaxBuiltCap'].get((n, t, i), float('inf')):
                    log_constraint_violation("Generator investment", f"Node: {n}, Technology: {t}, Period: {i}")
                    return False

    for n1, n2 in data['BidirectionalArc']:
        for i in data['Period']:
            if sample_dict.get((f"{n1},{n2}", 'Transmission', i, 'Transmission'), 0) > data['transmissionMaxBuiltCap'].get((n1, n2, i), float('inf')):
                log_constraint_violation("Transmission investment", f"Link: {n1}-{n2}, Period: {i}")
                return False

    for n, b in data['StoragesOfNode']:
        for i in data['Period']:
            if sample_dict.get((n, b, i, 'Storage Power'), 0) > data['storPWMaxBuiltCap'].get((n, b, i), float('inf')):
                log_constraint_violation("Storage power investment", f"Node: {n}, Storage: {b}, Period: {i}")
                return False
            if sample_dict.get((n, b, i, 'Storage Energy'), 0) > data['storENMaxBuiltCap'].get((n, b, i), float('inf')):
                log_constraint_violation("Storage energy investment", f"Node: {n}, Storage: {b}, Period: {i}")
                return False

    return True

def generate_and_save_samples(data, num_samples, output_file):
    """
    Generate multiple samples, check constraints, and save valid samples to a CSV file.
    """
    all_samples = []
    attempts = 0
    max_attempts = num_samples * 100  # Limit the number of attempts

    while len(all_samples) < num_samples and attempts < max_attempts:
        sample = generate_sample(data)
        if check_constraints(sample, data):
            sample['Sample_ID'] = len(all_samples) + 1
            all_samples.append(sample)
        attempts += 1

    if len(all_samples) < num_samples:
        logging.warning(f"Only {len(all_samples)} valid samples were generated out of {num_samples} requested after {attempts} attempts.")

    if not all_samples:
        logging.error("No valid samples were generated.")
        return None

    combined_samples = pd.concat(all_samples, ignore_index=True)
    combined_samples.to_csv(output_file, index=False)
    logging.info(f"Samples saved to {output_file}")
    return combined_samples


if __name__ == "__main__":
    start = time.time()

    # Generate name for this run
    name = version + '_reg' + str(lengthRegSeason) + \
        '_peak' + str(lengthPeakSeason) + \
        '_sce' + str(NoOfScenarios)
    if scenariogeneration and not fix_sample:
        name = name + "_randomSGR"
    else:
        name = name + "_noSGR"
    if filter_use:
        name = name + "_filter" + str(n_cluster)
    if moment_matching:
        name = name + "_moment" + str(n_tree_compare)
    name = name + str(datetime.now().strftime("_%Y%m%d%H%M"))

    # Set up paths
    workbook_path = 'Data handler/' + version
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name
    scenario_data_path = 'Data handler/' + version + '/ScenarioData'
    result_file_path = 'Results/' + name

    # Create necessary directories
    os.makedirs(tab_file_path, exist_ok=True)
    os.makedirs(result_file_path, exist_ok=True)

    # Set up logging for constraint violations
    logging.getLogger().setLevel(logging.DEBUG)
    constraint_log_file = os.path.join(result_file_path, f"constraint_violations_{name}.log")
    file_handler = logging.FileHandler(constraint_log_file)
    file_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)

    # Generate tab files
    generate_tab_files(filepath=workbook_path, tab_file_path=tab_file_path)

    # Load EMPIRE data
    empire_data = load_empire_data(tab_file_path)

    # Generate samples
    num_samples = 1  # You can change this or make it configurable
    output_file = os.path.join(result_file_path, f"samples_{name}.csv")
    samples = generate_and_save_samples(empire_data, num_samples, output_file)

    if samples is not None:
        logging.info(f"Generated {len(samples)} samples and saved to {output_file}")
    else:
        logging.error("Failed to generate any valid samples.")

    end = time.time()
    logging.info(f"Sampling Implementation took {end - start} seconds")
    logging.info(f"Generated {len(samples)} samples and saved to {output_file}")

    # Remove the file handler to avoid duplicate logs in future runs
    logging.getLogger().removeHandler(file_handler)
    