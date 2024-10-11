import pandas as pd
import numpy as np
from scipy.stats import qmc
import os
from FSD_sampling_violation import create_model, check_model_feasibility  # Assuming these are available
import csv
import io

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:  # If all columns are merged
            df = pd.read_csv(file_path, names=['Node', 'Energy_Type', 'Period', 'Type', 'Value'])
    else:
        # Attempt to read the .tab file with various delimiters
        try:
            df = pd.read_csv(file_path, sep='\t', engine='python')
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, delim_whitespace=True, engine='python')
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep=',', engine='python')
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep=';', engine='python')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    return df

def rename_columns(df):
    # Ensure that the DataFrame has at least four columns
    if len(df.columns) < 4:
        raise ValueError("DataFrame does not have enough columns to rename properly.")
    # Rename the first four columns accordingly
    df.columns = ['Node', 'Energy_Type', 'Period', 'Value'] + list(df.columns[4:])
    return df

def get_unique_keys(df):
    # Initialize the keys_by_type dictionary
    keys_by_type = {
        'Generation': [],
        'Transmission': [],
        'Storage Power': [],
        'Storage Energy': []
    }
    
    for _, row in df.iterrows():
        node, energy_type, period, type_ = row['Node'], row['Energy_Type'], row['Period'], row['Type']
        key = (node, energy_type, period)
        
        if type_.lower() == 'generation' or type_.lower() == 'generator':
            keys_by_type['Generation'].append(key)
        elif type_.lower() == 'transmission':
            keys_by_type['Transmission'].append(key)
        elif type_.lower() == 'storage power':
            keys_by_type['Storage Power'].append(key)
        elif type_.lower() == 'storage energy':
            keys_by_type['Storage Energy'].append(key)
    
    # Remove duplicate keys
    for key_type in keys_by_type:
        keys_by_type[key_type] = list(set(keys_by_type[key_type]))
    
    return keys_by_type

def prepare_bounds(keys_by_type, gen_data, trans_data, stor_pw_data, stor_en_data):
    bounds = {}
    
    def get_value_or_default(df, index_cols, value_col, default_value):
        try:
            # If the value exists, return it
            return df.loc[tuple(index_cols), value_col]
        except KeyError:
            # If the value doesn't exist, return the default value
            return default_value

    # Set bounds for each type using the keys
    for key in keys_by_type['Generation']:
        value = get_value_or_default(gen_data, key, 'Value', 500000.0)
        bounds[('Generation', *key)] = (0, value)
    
    for key in keys_by_type['Transmission']:
        value = get_value_or_default(trans_data, key, 'Value', 20000.0)
        bounds[('Transmission', *key)] = (0, value)
    
    for key in keys_by_type['Storage Power']:
        value = get_value_or_default(stor_pw_data, key, 'Value', 500000.0)
        bounds[('Storage Power', *key)] = (0, value)
    
    for key in keys_by_type['Storage Energy']:
        value = get_value_or_default(stor_en_data, key, 'Value', 500000.0)
        bounds[('Storage Energy', *key)] = (0, value)
    
    return bounds

def lhs_sampling(bounds, num_samples):
    sampler = qmc.LatinHypercube(d=len(bounds))  
    samples = sampler.random(n=num_samples)  
    
    scaled_samples = []
    for sample in samples:  
        scaled_sample = {}
        for (idx, (key, (lower, upper))) in enumerate(bounds.items()):
            scaled_value = lower + (upper - lower) * sample[idx]
            scaled_sample[key] = scaled_value
        scaled_samples.append(scaled_sample)
    
    return scaled_samples

def main():
    # Load data
    fsd_data = load_data('SeedSamples/fsd_seed1.csv')
    gen_data = load_data('Data handler/sampling/full/Generator_MaxBuiltCapacity.tab')
    trans_data = load_data('Data handler/sampling/full/Transmission_MaxBuiltCapacity.tab')
    stor_pw_data = load_data('Data handler/sampling/full/Storage_PowerMaxBuiltCapacity.tab')
    stor_en_data = load_data('Data handler/sampling/full/Storage_EnergyMaxBuiltCapacity.tab')
    
    # Check if any of the DataFrames are None
    if any(df is None for df in [fsd_data, gen_data, trans_data, stor_pw_data, stor_en_data]):
        print("One or more DataFrames could not be loaded. Please check the input files.")
        return

    # Rename columns for .tab files
    for df_name, df in [('Generator', gen_data), ('Transmission', trans_data), ('Storage Power', stor_pw_data), ('Storage Energy', stor_en_data)]:
        try:
            df = rename_columns(df)
        except ValueError as e:
            print(f"Error in renaming columns for {df_name}: {e}")
            return
        # Update the DataFrame after renaming
        if df_name == 'Generator':
            gen_data = df
        elif df_name == 'Transmission':
            trans_data = df
        elif df_name == 'Storage Power':
            stor_pw_data = df
        elif df_name == 'Storage Energy':
            stor_en_data = df

    # Set 'Value' column correctly if it's under a different name
    for df in [gen_data, trans_data, stor_pw_data, stor_en_data]:
        if 'Unnamed:_3' in df.columns:
            df['Value'] = df['Unnamed:_3']
            df.drop(columns=['Unnamed:_3'], inplace=True)
    
    # Set index
    for df in [gen_data, trans_data, stor_pw_data, stor_en_data]:
        df.set_index(['Node', 'Energy_Type', 'Period'], inplace=True)
    
    # Ensure 'Period' and 'Value' are numeric
    for df in [gen_data, trans_data, stor_pw_data, stor_en_data]:
        df.index = df.index.set_levels([df.index.levels[0],
                                        df.index.levels[1],
                                        df.index.levels[2].astype(int)], level=[0,1,2])
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Classify keys
    keys_by_type = get_unique_keys(fsd_data)
    
    # Prepare bounds
    bounds = prepare_bounds(keys_by_type, gen_data, trans_data, stor_pw_data, stor_en_data)

    # Debug: Print the bounds to ensure they are correctly prepared
    if len(bounds) == 0:
        print("No bounds found. Please check the input data.")
        return
    else:
        print(f"Number of bounds: {len(bounds)}")
        
    # Perform LHS for 5 samples
    num_samples = 5
    lhs_samples = lhs_sampling(bounds, num_samples)  # 5개의 샘플 생성
    
    num_feasible_samples = 0
    sample_number = 1

    for sample in lhs_samples:
        # Reconstruct the DataFrame with the same structure as fsd_seed1.csv
        sample_rows = []
        for (param_type, node, energy_type, period), value in sample.items():
            sample_rows.append({
                'Node': node,
                'Energy_Type': energy_type,
                'Period': int(period),
                'Type': param_type,
                'Value': value
            })
        # Create DataFrame
        sample_df = pd.DataFrame(sample_rows)
        # Ensure columns order matches fsd_seed1.csv
        sample_df = sample_df[['Node', 'Energy_Type', 'Period', 'Type', 'Value']]
        

        # Add Type_order column for sorting
        type_order = {'Generation': 1, 'Transmission': 2, 'Storage Power': 3, 'Storage Energy': 4}
        sample_df['Type_order'] = sample_df['Type'].map(type_order)
        
        # Sort the DataFrame
        sample_df.sort_values(by=['Node', 'Energy_Type', 'Type_order', 'Period'], inplace=True)
        
        # Drop the Type_order column
        sample_df.drop(columns=['Type_order'], inplace=True)

        
        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        # Read CSV string as fsd_data for model
        csv_reader = csv.reader(csv_string.splitlines())
        next(csv_reader)  # Skip header
        fsd_sample_data = [row for row in csv_reader]
        
        # Create model and check feasibility
        data_folder = 'Data handler/sampling/full'  # Adjust if needed
        try:
            model, data = create_model(data_folder, fsd_sample_data)
            instance = model.create_instance(data)
            feasible = check_model_feasibility(instance)
        except Exception as e:
            print(f"Error during model creation or feasibility check for sample {sample_number}: {e}")
            feasible = False
        
        if feasible:
            # Save to CSV with unique name
            sample_file_name = f'lhs_sample_{num_feasible_samples + 1}.csv'
            sample_df.to_csv(sample_file_name, index=False)
            print(f"Sample {num_feasible_samples + 1} is feasible. Saved as '{sample_file_name}'.")
            num_feasible_samples += 1
        else:
            print(f"Sample {sample_number} is infeasible. Discarding.")
        
        sample_number += 1

    print(f"Generated {num_feasible_samples} feasible samples.")

if __name__ == "__main__":
    main()