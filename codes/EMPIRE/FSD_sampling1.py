import pandas as pd
import numpy as np
from scipy import stats
import glob
from FSD_sampling_violation import create_model, check_model_feasibility, read_fsd_from_csv
from pyomo.environ import *
import csv
import io

def read_csv_files(file_pattern):
    all_data = []
    for file in glob.glob(file_pattern):
        df = pd.read_csv(file)
        all_data.append(df)
    return all_data

def build_kde_and_sample(data_list):
    combined_data = pd.concat(data_list, ignore_index=True)
    grouped = combined_data.groupby(['Node', 'Energy_Type', 'Period', 'Type'])
    sampled_data = []
    for name, group in grouped:
        values = group['Value'].values
        if len(values) >= 2:
            try:
                kde = stats.gaussian_kde(values)
                new_sample = kde.resample(1)[0][0]
                new_sample = max(min(new_sample, np.max(values)), np.min(values))
            except:
                new_sample = np.median(values)
        else:
            new_sample = np.median(values)
        sampled_data.append({
            'Node': name[0],
            'Energy_Type': name[1],
            'Period': name[2],
            'Type': name[3],
            'Value': new_sample
        })
    return pd.DataFrame(sampled_data)

def save_to_csv(df, output_file):
    df.to_csv(output_file, index=False)

def generate_feasible_samples(n, input_file_pattern, data_folder):
    data_list = read_csv_files(input_file_pattern)
    feasible_samples = 0
    sample_number = 1

    while feasible_samples < n:
        sampled_df = build_kde_and_sample(data_list)
        
        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        sampled_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        # Read CSV string as fsd_data
        csv_reader = csv.reader(csv_string.splitlines())
        next(csv_reader)  # Skip header
        fsd_data = [row for row in csv_reader]
        
        model, data = create_model(data_folder, fsd_data)
        instance = model.create_instance(data)
        
        if check_model_feasibility(instance):
            output_file = f'sampled_data_{sample_number}.csv'
            save_to_csv(sampled_df, output_file)
            print(f"Sample {sample_number} is feasible. Saved as {output_file}")
            feasible_samples += 1
        else:
            print(f"Sample {sample_number} is infeasible. Discarding.")
        
        sample_number += 1

    print(f"Generated {n} feasible samples.")

# Main execution
if __name__ == "__main__":
    input_file_pattern = "SeedSamples/fsd_seed*.csv"  # Adjust this pattern to match your input files
    data_folder = 'Data handler/sampling/full'
    n_samples = 10  # Number of feasible samples you want to generate

    generate_feasible_samples(n_samples, input_file_pattern, data_folder)