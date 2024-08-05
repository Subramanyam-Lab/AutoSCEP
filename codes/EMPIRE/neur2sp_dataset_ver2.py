import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from yaml import safe_load
import os
from datetime import datetime
import csv
from Expected_Second_Stage import run_second_stage
from concurrent.futures import ProcessPoolExecutor
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
try:
    with open("config_reducedrun.yaml", 'r') as config_file:
        UserRunTimeConfig = safe_load(config_file)
except FileNotFoundError:
    logging.error("Configuration file not found. Please ensure 'config_reducedrun.yaml' exists.")
    raise
except yaml.YAMLError as e:
    logging.error(f"Error parsing YAML configuration: {e}")
    raise

# Data processing functions
def process_data(file_path, process_func):
    try:
        return process_func(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Empty data file: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        raise

def process_fsd(file_path):
    df = pd.read_csv(file_path)
    return {country: df[df['Country'] == country].groupby(['Energy_Type', 'Period', 'Type'])['Value'].sum().unstack(['Energy_Type', 'Type'])
            for country in df['Country'].unique()}

def process_tab_data(file_path, pivot_columns, value_column):
    df = pd.read_csv(file_path, sep='\t')
    return {country: torch.stack([
        torch.tensor(
            df[(df['Node'] == country) & (df['Scenario'] == scenario)]
            .pivot_table(index='Period', columns=pivot_columns, values=value_column)
            .values, 
            dtype=torch.float32
        ) for scenario in df['Scenario'].unique()
    ]) for country in df['Node'].unique()}

# Dataset class
class TwoStageStochasticDataset(Dataset):
    def __init__(self, fsd_data, electric_load_data, hydro_data, availability_data, labels):
        self.fsd_data = fsd_data
        self.electric_load_data = electric_load_data
        self.hydro_data = hydro_data
        self.availability_data = availability_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.fsd_data[idx], 
                self.electric_load_data[idx], 
                self.hydro_data[idx],
                self.availability_data[idx], 
                self.labels[idx])

# Function to read FSD from CSV
def read_fsd_from_csv(file_path):
    try:
        with open(file_path, 'r') as csvfile:
            return [row for row in csv.reader(csvfile)][1:]  # Skip header
    except FileNotFoundError:
        logging.error(f"FSD file not found: {file_path}")
        raise
    except csv.Error as e:
        logging.error(f"Error reading CSV file {file_path}: {str(e)}")
        raise

# Function to run second stage optimization in parallel
def run_second_stage_parallel(args):
    filename, file_path, tab_file_path, result_file_path, config = args
    FSD = read_fsd_from_csv(file_path)
    try:
        expected_second_stage_value = run_second_stage(
            name=config['version'],
            tab_file_path=tab_file_path,
            result_file_path=result_file_path,
            FSD=FSD,
            **config
        )
        return {"Filename": filename, "ExpectedSecondStage": expected_second_stage_value}
    except Exception as e:
        logging.error(f"Error in run_second_stage for {filename}: {str(e)}")
        return {"Filename": filename, "ExpectedSecondStage": None, "Error": str(e)}

# Main function to create datasets
def create_datasets(fsd_file_path, tab_file_path, result_file_path):
    # Process data
    fsd_data = process_data(fsd_file_path, process_fsd)
    electric_load_data = process_data(
        os.path.join(tab_file_path, 'Stochastic_ElectricLoadRaw.tab'),
        lambda f: process_tab_data(f, 'Operationalhour', 'ElectricLoadRaw_in_MW')
    )
    hydro_data = process_data(
        os.path.join(tab_file_path, 'Stochastic_HydroGenMaxSeasonalProduction.tab'),
        lambda f: process_tab_data(f, ['Season', 'Operationalhour'], 'HydroGeneratorMaxSeasonalProduction')
    )
    availability_data = process_data(
        os.path.join(tab_file_path, 'Stochastic_StochasticAvailability.tab'),
        lambda f: process_tab_data(f, ['IntermitentGenerators', 'Operationalhour'], 'GeneratorStochasticAvailabilityRaw')
    )

    # Run second stage optimization in parallel
    fsd_files = [f for f in os.listdir(fsd_file_path) if f.endswith('.csv')]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_second_stage_parallel, 
            [(f, os.path.join(fsd_file_path, f), tab_file_path, result_file_path, UserRunTimeConfig) for f in fsd_files]))

    # Create labels DataFrame
    labels_df = pd.DataFrame(results)
    labels_df.to_csv(f"expected_second_stage_results_{datetime.now().strftime('%Y%m%d%H%M')}.csv", index=False)

    # Create datasets
    nne_dataset, nnp_dataset = [], []
    for country in fsd_data.keys():
        country_fsd = fsd_data[country]
        country_electric_load = electric_load_data[country]
        country_hydro = hydro_data[country]
        country_availability = availability_data[country]
        country_labels = labels_df[labels_df['Country'] == country]['ExpectedSecondStage'].values

        for i in range(len(country_labels)):
            fsd_tensor = torch.tensor(country_fsd.iloc[i].values, dtype=torch.float32)
            electric_load_tensor = country_electric_load[i]
            hydro_tensor = country_hydro[i]
            availability_tensor = country_availability[i]
            label_tensor = torch.tensor(country_labels[i], dtype=torch.float32)

            nne_dataset.append((fsd_tensor, electric_load_tensor, hydro_tensor, availability_tensor, label_tensor))
            
            nnp_dataset.extend([
                (torch.cat([fsd_tensor, 
                            electric_load_tensor[j].flatten(), 
                            hydro_tensor[j].flatten(),
                            availability_tensor[j].flatten()]), 
                 label_tensor)
                for j in range(electric_load_tensor.size(0))
            ])

    return nne_dataset, nnp_dataset

# Main execution
if __name__ == "__main__":
    try:
        fsd_file_path = 'FSDsamples'
        tab_file_path = f"Data handler/{UserRunTimeConfig['version']}/Tab_Files_{UserRunTimeConfig['version']}"
        result_file_path = f"Results/{UserRunTimeConfig['version']}"

        nne_dataset, nnp_dataset = create_datasets(fsd_file_path, tab_file_path, result_file_path)

        # Create DataLoaders
        nne_dataloader = DataLoader(nne_dataset, batch_size=32, shuffle=True, num_workers=4)
        nnp_dataloader = DataLoader(nnp_dataset, batch_size=32, shuffle=True, num_workers=4)

        logging.info(f"NN-E dataset size: {len(nne_dataset)}")
        logging.info(f"NN-P dataset size: {len(nnp_dataset)}")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        raise