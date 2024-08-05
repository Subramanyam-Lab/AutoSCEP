import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from yaml import safe_load
import os
from datetime import datetime
import csv
from Expected_Second_Stage import run_second_stage

# Load configuration
UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

# Define data processing functions (from the first code)
def process_fsd(file_path):
    df = pd.read_csv(file_path)
    countries = df['Country'].unique()
    country_data = {}
    for country in countries:
        country_df = df[df['Country'] == country]
        country_data[country] = country_df.groupby(['Energy_Type', 'Period', 'Type'])['Value'].sum().unstack(['Energy_Type', 'Type'])
    return country_data

def process_electric_load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    countries = df['Node'].unique()
    country_data = {}
    for country in countries:
        country_df = df[df['Node'] == country]
        scenarios = country_df['Scenario'].unique()
        scenario_data = []
        for scenario in scenarios:
            scenario_df = country_df[country_df['Scenario'] == scenario]
            pivot_df = scenario_df.pivot(index='Period', columns='Operationalhour', values='ElectricLoadRaw_in_MW')
            scenario_tensor = torch.tensor(pivot_df.values, dtype=torch.float32)
            scenario_data.append(scenario_tensor)
        country_data[country] = torch.stack(scenario_data)
    return country_data

def process_hydro_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    countries = df['Node'].unique()
    country_data = {}
    for country in countries:
        country_df = df[df['Node'] == country]
        scenarios = country_df['Scenario'].unique()
        scenario_data = []
        for scenario in scenarios:
            scenario_df = country_df[country_df['Scenario'] == scenario]
            pivot_df = scenario_df.pivot_table(index=['Period', 'Season'], 
                                               columns='Operationalhour', 
                                               values='HydroGeneratorMaxSeasonalProduction')
            scenario_tensor = torch.tensor(pivot_df.values, dtype=torch.float32)
            scenario_data.append(scenario_tensor)
        country_data[country] = torch.stack(scenario_data)
    return country_data

def process_availability_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    countries = df['Node'].unique()
    country_data = {}
    for country in countries:
        country_df = df[df['Node'] == country]
        scenarios = country_df['Scenario'].unique()
        scenario_data = []
        for scenario in scenarios:
            scenario_df = country_df[country_df['Scenario'] == scenario]
            pivot_df = scenario_df.pivot_table(index=['IntermitentGenerators', 'Period'], 
                                               columns='Operationalhour', 
                                               values='GeneratorStochasticAvailabilityRaw')
            scenario_tensor = torch.tensor(pivot_df.values, dtype=torch.float32)
            scenario_data.append(scenario_tensor)
        country_data[country] = torch.stack(scenario_data)
    return country_data

# Define dataset class
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

# Function to read FSD from CSV (from the second code)
def read_fsd_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  
        fsd_data = [row for row in csv_reader]
    return fsd_data

# Main function to create datasets
def create_datasets(fsd_file_path, tab_file_path, result_file_path):
    fsd_data = process_fsd(fsd_file_path)
    electric_load_data = process_electric_load_data(os.path.join(tab_file_path, 'Stochastic_ElectricLoadRaw.tab'))
    hydro_data = process_hydro_data(os.path.join(tab_file_path, 'Stochastic_HydroGenMaxSeasonalProduction.tab'))
    availability_data = process_availability_data(os.path.join(tab_file_path, 'Stochastic_StochasticAvailability.tab'))
    
    results = []
    for filename in os.listdir(fsd_file_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(fsd_file_path, filename)
            FSD = read_fsd_from_csv(file_path)
            
            # Run second stage optimization
            expected_second_stage_value = run_second_stage(
                name=UserRunTimeConfig['version'],
                tab_file_path=tab_file_path,
                result_file_path=result_file_path,
                FSD=FSD,
                **UserRunTimeConfig  # Pass other necessary parameters
            )
            
            results.append({
                "Filename": filename,
                "ExpectedSecondStage": expected_second_stage_value
            })
    
    # Create labels DataFrame
    labels_df = pd.DataFrame(results)
    labels_df.to_csv(f"expected_second_stage_results_{datetime.now().strftime('%Y%m%d%H%M')}.csv", index=False)
    
    # Create datasets
    nne_dataset = []
    nnp_dataset = []
    
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
            
            # NN-E dataset
            nne_dataset.append((fsd_tensor, electric_load_tensor, hydro_tensor, availability_tensor, label_tensor))
            
            # NN-P dataset
            for j in range(electric_load_tensor.size(0)):
                nnp_input = torch.cat([fsd_tensor, 
                                       electric_load_tensor[j].flatten(), 
                                       hydro_tensor[j].flatten(),
                                       availability_tensor[j].flatten()])
                nnp_dataset.append((nnp_input, label_tensor))
    
    return nne_dataset, nnp_dataset

# Main execution
if __name__ == "__main__":
    fsd_file_path = 'FSDsamples'
    tab_file_path = f"Data handler/{UserRunTimeConfig['version']}/Tab_Files_{UserRunTimeConfig['version']}"
    result_file_path = f"Results/{UserRunTimeConfig['version']}"
    
    nne_dataset, nnp_dataset = create_datasets(fsd_file_path, tab_file_path, result_file_path)
    
    # Create DataLoaders
    nne_dataloader = DataLoader(nne_dataset, batch_size=32, shuffle=True)
    nnp_dataloader = DataLoader(nnp_dataset, batch_size=32, shuffle=True)
    
    print(f"NN-E dataset size: {len(nne_dataset)}")
    print(f"NN-P dataset size: {len(nnp_dataset)}")