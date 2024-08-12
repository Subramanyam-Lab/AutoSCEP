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
from functools import wraps
import time
import numpy as np
from typing import List, Tuple, Callable

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


# Constants (from original code)
N_SAMPLES = 100  # Number of FSD samples
K_SCENARIOS = 30  # Number of scenarios for NN-E
SINGLE_SCENARIO = 1  # Number of scenario for NN-P


class NNPDataset(Dataset):
    def __init__(self, fsd_data, scenario_data, labels):
        self.fsd_data = fsd_data
        self.scenario_data = scenario_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.fsd_data[idx], self.scenario_data[idx], self.labels[idx]


class FourLevelParallelization:
    def __init__(self, T: int, N: int, seasons: List[str], model, data):
        self.T = T
        self.N = N
        self.seasons = seasons
        self.model = model
        self.data = data

    def level1_sampling(self, instance) -> List[np.ndarray]:
        return new_samples(instance, 1)

    def level2_populate(self, fsd_sample: np.ndarray) -> List:
        return process_new_samples_output(fsd_sample)

    def level3_sample_scenarios(self, scenario_data_path: str, tab_file_path: str) -> None:
        generate_scenarios(scenario_data_path, tab_file_path, self.N)

    def level4_solve(self, processed_fsd: List, name_path: str, tab_file_path: str, Scenario: List[str]) -> float:
        return calculate_second_stage_value(processed_fsd, name_path, tab_file_path, Scenario, 'Results/NNE')

    def process_single_sample(self, i: int, scenario_data_path: str) -> Tuple[List, List, float]:
        try:
            logging.info(f"Starting process for sample {i+1}")
            
            instance = self.model.create_instance(self.data)
            logging.info(f"Instance created for sample {i+1}")
            
            base_path = f'Data handler/{UserRunTimeConfig["version"]}/Tab_Files'
            prefix = f'sce_{self.N}'
            tab_file_path = get_safe_directory_name(base_path, prefix)
            logging.info(f"Directory created: {tab_file_path}")
            
            fsd_sample = self.level1_sampling(instance)
            logging.info(f"FSD sample generated for sample {i+1}")
            
            Scenario = [f"scenario{j+1}" for j in range(self.N)]
            logging.info(f"Generating scenarios for sample {i+1}")
            self.level3_sample_scenarios(scenario_data_path, tab_file_path)
            logging.info(f"Scenarios generated for sample {i+1}")
            
            processed_fsd = self.level2_populate(fsd_sample)
            logging.info(f"FSD processed for sample {i+1}")
            
            logging.info(f"Calculating second stage value for sample {i+1}")
            nnp_value = self.level4_solve(processed_fsd, os.path.basename(tab_file_path), tab_file_path, Scenario)
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
                try:
                    shutil.rmtree(tab_file_path)
                    logging.info(f"Removed directory: {tab_file_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove directory {tab_file_path}: {str(e)}")

    def run(self, scenario_data_path: str) -> List[Tuple[List, List, float]]:
        num_processes = max(1, multiprocessing.cpu_count())
        logging.info(f"Using {num_processes} processes")
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            return pool.map(partial(self.process_single_sample, scenario_data_path=scenario_data_path), range(self.T))

# Helper functions (from original code)
def generate_scenarios(filepath, tab_file_path, num_scenarios):
    # ... (implementation from original code)
    return 0

def calculate_second_stage_value(fsd, name_path, tab_file_path, Scenario, result_file_path):
    # ... (implementation from original code)
    return 0

def process_new_samples_output(samples):
    # ... (implementation from original code)
    return 0

def load_and_process_scenario_data(tab_file_path):
    # ... (implementation from original code)
    return 0

def get_safe_directory_name(base_path, prefix):
    # ... (implementation from original code)
    return 0

def retry_on_exception(max_attempts=10, delay=5):
    # ... (implementation from original code)
    return 0

def save_nnp_dataset_to_csv(nnp_dataset, filename=None):
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

# Main execution
if __name__ == "__main__":
    try:
        sample_model_path = 'Data handler/sampling'
        scenario_data_path = f'Data handler/reduced/ScenarioData'
        model, data = sample_model(sample_model_path)

        parallelization = FourLevelParallelization(
            T=N_SAMPLES,
            N=SINGLE_SCENARIO,
            seasons=['winter', 'spring', 'summer', 'fall', 'peak1', 'peak2'],
            model=model,
            data=data
        )

        dataset = parallelization.run(scenario_data_path)
        nnp_dataset = NNPDataset(*zip(*dataset))

        # Save NNP dataset to CSV
        save_nnp_dataset_to_csv(nnp_dataset)

        # Create DataLoader
        nnp_dataloader = DataLoader(nnp_dataset, batch_size=32, shuffle=True, num_workers=4)

        logging.info(f"NN-P dataset size: {len(nnp_dataset)}")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        raise