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

# Constants (from original code)
N_SAMPLES = 100  # Number of FSD samples
K_SCENARIOS = 30  # Number of scenarios for NN-E
SINGLE_SCENARIO = 1  # Number of scenario for NN-P

# ... (other constants and configurations from the original code)

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

    @retry_on_exception(max_attempts=3, delay=2)
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