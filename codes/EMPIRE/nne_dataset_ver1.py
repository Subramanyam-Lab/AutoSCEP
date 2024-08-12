import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from yaml import safe_load
from sampling import new_samples, sample_model
from scenario_random import generate_random_scenario
from Expected_Second_Stage import run_second_stage
import multiprocessing
from functools import partial
import shutil
import uuid
from multiprocessing import current_process


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open("config_reducedrun.yaml", 'r') as config_file:
    UserRunTimeConfig = safe_load(config_file)


USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
temp_dir = UserRunTimeConfig["temp_dir"]
version = UserRunTimeConfig["version"]
Horizon = UserRunTimeConfig["Horizon"]
lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
discountrate = UserRunTimeConfig["discountrate"]
WACC = UserRunTimeConfig["WACC"]
solver = UserRunTimeConfig["solver"]
scenariogeneration = UserRunTimeConfig["scenariogeneration"]
fix_sample = UserRunTimeConfig["fix_sample"]
LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
filter_make = UserRunTimeConfig["filter_make"] 
filter_use = UserRunTimeConfig["filter_use"]
n_cluster = UserRunTimeConfig["n_cluster"]
moment_matching = UserRunTimeConfig["moment_matching"]
n_tree_compare = UserRunTimeConfig["n_tree_compare"]
EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]
IAMC_PRINT = UserRunTimeConfig["IAMC_PRINT"]
WRITE_LP = UserRunTimeConfig["WRITE_LP"]
PICKLE_INSTANCE = UserRunTimeConfig["PICKLE_INSTANCE"] 


#############################
##Non configurable settings##
#############################
NoOfRegSeason = 4
regular_seasons = ["winter", "spring", "summer", "fall"]
NoOfPeakSeason = 2
lengthPeakSeason = 7
LeapYearsInvestment = 5
time_format = "%d/%m/%Y %H:%M"
if version in ["europe_v50"]:
    north_sea = False
else:
    north_sea = True

#######
##RUN##
#######

FirstHoursOfRegSeason = [lengthRegSeason*i + 1 for i in range(NoOfRegSeason)]
FirstHoursOfPeakSeason = [lengthRegSeason*NoOfRegSeason + lengthPeakSeason*i + 1 for i in range(NoOfPeakSeason)]
Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
peak_seasons = ['peak'+str(i + 1) for i in range(NoOfPeakSeason)]
Season = regular_seasons + peak_seasons
Operationalhour = [i + 1 for i in range(FirstHoursOfPeakSeason[-1] + lengthPeakSeason - 1)]
HoursOfRegSeason = [(s,h) for s in regular_seasons for h in Operationalhour \
                if h in list(range(regular_seasons.index(s)*lengthRegSeason+1,
                            regular_seasons.index(s)*lengthRegSeason+lengthRegSeason+1))]
HoursOfPeakSeason = [(s,h) for s in peak_seasons for h in Operationalhour \
                    if h in list(range(lengthRegSeason*len(regular_seasons)+ \
                                        peak_seasons.index(s)*lengthPeakSeason+1,
                                        lengthRegSeason*len(regular_seasons)+ \
                                            peak_seasons.index(s)*lengthPeakSeason+ \
                                                lengthPeakSeason+1))]
HoursOfSeason = HoursOfRegSeason + HoursOfPeakSeason
dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}

# Constants
N_SAMPLES = 100  # Number of FSD samples
K_SCENARIOS = 10  # Number of scenarios for NN-E
SINGLE_SCENARIO = 1  # Number of scenario for NN-P

class NNEDataset(Dataset):
    def __init__(self, fsd_data, scenarios_data, labels):
        self.fsd_data = fsd_data
        self.scenarios_data = scenarios_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.fsd_data[idx], self.scenarios_data[idx], self.labels[idx]

def generate_scenarios(filepath, tab_file_path, num_scenarios):
    if not os.path.exists(tab_file_path):
        os.makedirs(tab_file_path)

    generate_random_scenario(filepath = filepath,
                        tab_file_path = tab_file_path,
                        scenarios = num_scenarios,
                        seasons = regular_seasons,
                        Periods = len(Period),
                        regularSeasonHours = lengthRegSeason,
                        peakSeasonHours = lengthPeakSeason,
                        dict_countries = dict_countries,
                        time_format = time_format,
                        filter_make = filter_make,
                        filter_use = filter_use,
                        n_cluster = n_cluster,
                        moment_matching = moment_matching,
                        n_tree_compare = n_tree_compare,
                        fix_sample = fix_sample,
                        north_sea = False,
                        LOADCHANGEMODULE = LOADCHANGEMODULE)

def calculate_second_stage_value(fsd, name_path, tab_file_path, Scenario, result_file_path):
    
    expected_second_stage_value = run_second_stage(name = name_path, 
        tab_file_path = tab_file_path,
        result_file_path = result_file_path, 
        scenariogeneration = scenariogeneration,
        scenario_data_path = tab_file_path,
        solver = solver,
        temp_dir = temp_dir, 
        FirstHoursOfRegSeason = FirstHoursOfRegSeason, 
        FirstHoursOfPeakSeason = FirstHoursOfPeakSeason, 
        lengthRegSeason = lengthRegSeason,
        lengthPeakSeason = lengthPeakSeason,
        Period = Period, 
        Operationalhour = Operationalhour,
        Scenario = Scenario,
        Season = Season,
        HoursOfSeason = HoursOfSeason,
        discountrate = discountrate, 
        WACC = WACC, 
        LeapYearsInvestment = LeapYearsInvestment,
        FSD = fsd, 
        WRITE_LP = WRITE_LP, 
        PICKLE_INSTANCE = PICKLE_INSTANCE, 
        EMISSION_CAP = EMISSION_CAP,
        USE_TEMP_DIR = USE_TEMP_DIR,
        LOADCHANGEMODULE = LOADCHANGEMODULE)
    return expected_second_stage_value

def process_new_samples_output(samples):
    processed_samples = []
    
    if isinstance(samples, list):
        # If samples is a list of DataFrames
        for df in samples:
            processed_samples.extend(process_dataframe(df))
    elif isinstance(samples, pd.DataFrame):
        # If samples is a single DataFrame
        processed_samples = process_dataframe(samples)
    else:
        raise ValueError("Unexpected type for samples")
    
    return processed_samples

def process_dataframe(df):
    return [
        [
            str(row['Country']),
            str(row['Energy_Type']),
            int(row['Period']),
            str(row['Type']),
            float(row['Value'])
        ]
        for _, row in df.iterrows()
    ]

def load_and_process_scenario_data(tab_file_path):
    scenario_data = []
    
    # Load and process electric load data
    electric_load_path = os.path.join(tab_file_path, 'Stochastic_ElectricLoadRaw.tab')
    scenario_data.extend(process_tab_data(electric_load_path, 'ElectricLoadRaw_in_MW'))
    
    # Load and process hydro data
    hydro_path = os.path.join(tab_file_path, 'Stochastic_HydroGenMaxSeasonalProduction.tab')
    scenario_data.extend(process_tab_data(hydro_path, 'HydroGeneratorMaxSeasonalProduction'))
    
    # Load and process availability data
    availability_path = os.path.join(tab_file_path, 'Stochastic_StochasticAvailability.tab')
    scenario_data.extend(process_tab_data(availability_path, 'GeneratorStochasticAvailabilityRaw'))
    
    return scenario_data


def process_tab_data(file_path, value_column):
    try:
        df = pd.read_csv(file_path, sep='\t')
        processed_data = df[value_column].tolist()
        # logging.info(f"Successfully processed {file_path}")
        return processed_data
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        raise

def save_nne_dataset_to_csv(nnp_dataset, filename=None):
    if filename is None:
        filename = f"nnp_dataset_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    
    data = []
    for fsd_data, scenario_data, label in nnp_dataset:
        row = {
            'fsd_data': str(fsd_data),  # Convert list to string
            'scenario_data': str(scenario_data),  # Convert list to string
            'label': label
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"NNP dataset saved to {filename}")


def get_safe_directory_name(base_path, prefix):
    process_id = current_process().pid
    unique_id = uuid.uuid4().hex[:8]
    dir_name = f"{prefix}_{process_id}_{unique_id}"

    full_path = os.path.join(base_path, dir_name)

    counter = 1
    while os.path.exists(full_path):
        dir_name = f"{prefix}_{process_id}_{unique_id}_{counter}"
        full_path = os.path.join(base_path, dir_name)
        counter += 1
    
    return full_path


# def process_scenario(scenario_index, processed_fsd, tab_file_path, scenario_data_path):
#     try:
#         logging.info(f"Processing scenarion {scenario_index + 1}")
        
#         generate_scenarios(scenario_data_path, tab_file_path, SINGLE_SCENARIO)
        
#         nne_value = calculate_second_stage_value(processed_fsd, tab_file_path, tab_file_path, ["scenario1"], 'Results/NNE')
        
#         scenario_data = load_and_process_scenario_data(tab_file_path)
        
#         return nne_value, scenario_data
#     except Exception as e:
#         logging.error(f"Error in process_scenario for {scenario_index + 1}: {str(e)}")
#         raise
#     finally:
#         if os.path.exists(tab_file_path):
#             shutil.rmtree(tab_file_path)



# Two options:
# 1) process_single_sample() with multiprocessing by scenario then calculate expeceted value,
#    prob = 1 / (len(scenarios)) * value(process_single_sample)
#    To do 1), we should make fsd_sample out of the process_single_sample()
# 2) process_multiple_sample()
#    Just calculate expeceted value to do with each processors, simply think about that

# def process_single_sample(i, sample_model_path, scenario_data_path):
#     try:
#         logging.info(f"Starting process for sample {i+1}")
        
#         model, data = sample_model(sample_model_path)
#         instance = model.create_instance(data)
#         logging.info(f"Instance created for sample {i+1}")
        
#         base_path = f'Data handler/{UserRunTimeConfig["version"]}/Tab_Files'
#         prefix = f'sce_{SINGLE_SCENARIO}'
#         tab_file_path = get_safe_directory_name(base_path, prefix)
#         logging.info(f"Directory created: {tab_file_path}")
        
#         fsd_sample = new_samples(instance, 1)
#         processed_fsd = process_new_samples_output(fsd_sample)
        
        
#         num_processes = min(K_SCENARIOS, multiprocessing.cpu_count())
#         with multiprocessing.Pool(processes=num_processes) as pool:
#             process_scenario_partial = partial(process_scenario, processed_fsd=processed_fsd, tab_file_path=tab_file_path, scenario_data_path=scenario_data_path)
#             results = pool.map(process_scenario_partial, range(K_SCENARIOS))
        
#         nne_values, scenario_data_list = zip(*results)
        
#         expected_nne_value = sum(nne_values) / K_SCENARIOS
#         logging.info(f"Expected nnp_value calculated for sample {i+1}: {expected_nne_value}")
        
#         fsd_values = [item[4] for item in processed_fsd]
        
#         # Combine scenario data from all scenarios
#         combined_scenario_data = [item for sublist in scenario_data_list for item in sublist]
        
#         return (fsd_values, combined_scenario_data, expected_nne_value)
#     except Exception as e:
#         logging.error(f"Error in process_single_sample for sample {i+1}: {str(e)}")
#         raise
#     finally:
#         if os.path.exists(tab_file_path):
#             shutil.rmtree(tab_file_path)
#             logging.info(f"Removed directory: {tab_file_path}")



def process_scenario(scenario_index, processed_fsd, tab_file_path, scenario_data_path):
    try:
        logging.info(f"Processing scenario {scenario_index + 1}")
        
        generate_scenarios(scenario_data_path, tab_file_path, SINGLE_SCENARIO)
        
        nne_value = calculate_second_stage_value(processed_fsd, os.path.basename(tab_file_path), tab_file_path, ["scenario1"], 'Results/NNE')
        
        scenario_data = load_and_process_scenario_data(tab_file_path)
        
        return nne_value, scenario_data
    except Exception as e:
        logging.error(f"Error in process_scenario for {scenario_index + 1}: {str(e)}")
        raise
    finally:
        if os.path.exists(tab_file_path):
            shutil.rmtree(tab_file_path)

def process_single_sample(i, sample_model_path, scenario_data_path):
    try:
        logging.info(f"Starting process for sample {i+1}")
        
        model, data = sample_model(sample_model_path)
        instance = model.create_instance(data)
        logging.info(f"Instance created for sample {i+1}")
        
        base_path = f'Data handler/{UserRunTimeConfig["version"]}/Tab_Files'
        prefix = f'sce_{SINGLE_SCENARIO}'
        tab_file_path = get_safe_directory_name(base_path, prefix)
        logging.info(f"Directory created: {tab_file_path}")
        
        fsd_sample = new_samples(instance, 1)
        processed_fsd = process_new_samples_output(fsd_sample)
        
        scenario_results = []
        for scenario_index in range(K_SCENARIOS):
            nne_value, scenario_data = process_scenario(scenario_index, processed_fsd, tab_file_path, scenario_data_path)
            scenario_results.append((nne_value, scenario_data))
        
        nne_values, scenario_data_list = zip(*scenario_results)
        
        expected_nne_value = sum(nne_values) / K_SCENARIOS
        logging.info(f"Expected nne_value calculated for sample {i+1}: {expected_nne_value}")
        
        fsd_values = [item[4] for item in processed_fsd]
        
        # Combine scenario data from all scenarios
        combined_scenario_data = [item for sublist in scenario_data_list for item in sublist]
        
        return (fsd_values, combined_scenario_data, expected_nne_value)
    except Exception as e:
        logging.error(f"Error in process_single_sample for sample {i+1}: {str(e)}")
        raise
    finally:
        if os.path.exists(tab_file_path):
            shutil.rmtree(tab_file_path)
            logging.info(f"Removed directory: {tab_file_path}")


# Option 2) it will take long time.
def process_multiple_sample(i, sample_model_path, scenario_data_path):
    try:
        logging.info(f"Starting process for sample {i+1}")
        
        model, data = sample_model(sample_model_path)
        instance = model.create_instance(data)
        logging.info(f"Instance created for sample {i+1}")
        
        base_path = f'Data handler/{UserRunTimeConfig["version"]}/Tab_Files'
        prefix = f'sce_{K_SCENARIOS}'
        tab_file_path = get_safe_directory_name(base_path, prefix)
        logging.info(f"Directory created: {tab_file_path}")
        
        fsd_sample = new_samples(instance, 1)
        logging.info(f"FSD sample generated for sample {i+1}")
        
        logging.info(f"Generating scenarios for sample {i+1}")
        generate_scenarios(scenario_data_path, tab_file_path, K_SCENARIOS)
        logging.info(f"Scenarios generated for sample {i+1}")
        
        processed_fsd = process_new_samples_output(fsd_sample)
        logging.info(f"FSD processed for sample {i+1}")
        Scenario = ["scenario"+str(i + 1) for i in range(K_SCENARIOS)]
        logging.info(f"Calculating second stage value for sample {i+1}")
        nnp_value = calculate_second_stage_value(processed_fsd, os.path.basename(tab_file_path), tab_file_path, Scenario, 'Results/NNE')
        logging.info(f"nnp_value calculated for sample {i+1}: {nnp_value}")
        
        fsd_values = [item[4] for item in processed_fsd]
        scenario_data = load_and_process_scenario_data(tab_file_path)
        logging.info(f"Data loaded and processed for sample {i+1}")
        
        return (fsd_values, scenario_data, nnp_value)
    except Exception as e:
        logging.error(f"Error in process_single_sample for sample {i+1}: {str(e)}")
        raise
    finally:
        if os.path.exists(tab_file_path):
            shutil.rmtree(tab_file_path)
            logging.info(f"Removed directory: {tab_file_path}")

        
# for option 1
# def create_nne_datasets():
#     sample_model_path = 'Data handler/sampling'
#     scenario_data_path = f'Data handler/reduced/ScenarioData'
    
#     # Create a partial function with fixed arguments
#     process_sample = partial(process_single_sample, sample_model_path=sample_model_path, scenario_data_path=scenario_data_path)
    
#     num_processes = max(1, multiprocessing.cpu_count())
#     print(f"Using {num_processes} processes")

#     with multiprocessing.Pool(processes=num_processes) as pool:
#         nne_dataset = pool.map(process_sample, range(N_SAMPLES))
    
#     return NNEDataset(*zip(*nne_dataset))

# for option 2
def create_nne_datasets():
    sample_model_path = 'Data handler/sampling'
    scenario_data_path = f'Data handler/reduced/ScenarioData'
    
    # Create a partial function with fixed arguments
    process_sample = partial(process_multiple_sample, sample_model_path=sample_model_path, scenario_data_path=scenario_data_path)
    
    num_processes = max(1, multiprocessing.cpu_count())
    print(f"Using {num_processes} processes")

    with multiprocessing.Pool(processes=num_processes) as pool:
        nne_dataset = pool.map(process_sample, range(N_SAMPLES))
    
    return NNEDataset(*zip(*nne_dataset))


if __name__ == "__main__":
    try:
        # nne_dataset = create_nne_datasets()
        nne_dataset = create_nne_datasets()

        # Save NNP dataset to CSV
        save_nne_dataset_to_csv(nne_dataset)

        # Create DataLoaders
        # nne_dataloader = DataLoader(nne_dataset, batch_size=32, shuffle=True, num_workers=4)
        nne_dataloader = DataLoader(nne_dataset, batch_size=32, shuffle=True, num_workers=4)

        # logging.info(f"NN-E dataset size: {len(nne_dataset)}")
        logging.info(f"NN-E dataset size: {len(nne_dataset)}")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        raise