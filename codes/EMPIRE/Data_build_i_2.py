import numpy as np
import json
import csv
import os
import re
import glob
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import pickle
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for desired data
desired_data_v = {
    'Generation': {
        'Germany': ['Solar', 'GasCCGT', 'Bio', 'Bio10cofiring'],
        'France': ['Windonshore', 'Solar', 'GasCCGT', 'Bio'],
        'Denmark': ['Solar', 'GasCCGT', 'Windonshore']
    },
    'Storage Power': {
        'Germany': ['Li-Ion_BESS'],
        'France': ['Li-Ion_BESS'],
        'Denmark': ['Li-Ion_BESS']
    },
    'Storage Energy': {
        'Germany': ['Li-Ion_BESS'],
        'France': ['Li-Ion_BESS'],
        'Denmark': ['Li-Ion_BESS']
    }
}

desired_data_xi = {
    'Generation': [
        ('Germany', 'GasCCGT'),
        ('Denmark', 'GasCCGT'),
        ('France', 'GasCCGT'),
        ('Germany', 'Bio10cofiring'),
        ('Germany', 'Bio'),
        ('France', 'Bio'),
        ('Denmark', 'Windonshore'),
        ('France', 'Windonshore'),
        ('Germany', 'Solar'),
        ('Denmark', 'Solar'),
        ('France', 'Solar')
    ]
}

def extract_file_info(filename):
    """Extract period, seed, and filenum from filename."""
    # pattern = r'(?:v|xi_Q)_(\d+)_period_(\d+)_seed_(\d+)'
    pattern = r'(?:v|x|xi_Q)_(\d+)_period_(\d+)_seed_(\d+)'
    match = re.search(pattern, filename)
    if match:
        period, seed, filenum = map(int, match.groups())
        return period, seed, filenum
    return None, None, None

# Variable Selection

# No selection

# def filter_v_data(data):
#     """Extract all data from v_i without parsing."""
#     filtered_data = []
#     v_i_data = data.get('v_i', {})
    
#     for key, values in v_i_data.items():
#         if isinstance(values, dict):
#             # If the value is a nested dictionary, append its values
#             filtered_data.extend(values.values())
#         else:
#             # Otherwise, append the value directly
#             filtered_data.append(values)
    
#     return filtered_data

# for v_i
# def filter_v_data(data):
#     """Filter genInstalledCap data while keeping other installedCap data intact."""
#     filtered_data = []
#     v_i_data = data.get('v_i', {})

#     for key, values in v_i_data.items():
#         if key == 'genInstalledCap':
#             # Process genInstalledCap to exclude items with 'existing' or 'CCS'
#             filtered_gen_installed_cap = {
#                 sub_key: value
#                 for sub_key, value in values.items()
#                 if 'existing' not in sub_key and 'CCS' not in sub_key
#             }
#             filtered_data.extend(filtered_gen_installed_cap.values())
#         else:
#             # Keep other installedCap data intact
#             filtered_data.extend(values.values())

#     return filtered_data

def filter_v_data(data):
    """Filter genInstalledCap data while keeping other installedCap data intact."""
    filtered_data = []
    v_i_data = data.get('v_i', {})

    for key, values in v_i_data.items():
        if isinstance(values, dict):
            # If the value is a nested dictionary, append its values
            filtered_data.extend(values.values())
        else:
            # Otherwise, append the value directly
            filtered_data.append(values)

    return filtered_data



def filter_x_data(data):
    """Filter genInstalledCap data while keeping other installedCap data intact."""
    filtered_data = []
    x_i_data = data.get('x_i', {})
    for key, values in x_i_data.items():
        if key == 'genInvCap':
            # Process genInstalledCap to exclude items with 'existing' or 'CCS'
            filtered_gen_installed_cap = {
                sub_key: value
                for sub_key, value in values.items()
                if 'existing' not in sub_key and 'CCS' not in sub_key
            }
            filtered_data.extend(filtered_gen_installed_cap.values())
        else:
            # Keep other installedCap data intact
            filtered_data.extend(values.values())
    return filtered_data



def filter_xi_data(data):
    """Extract all data from v_i without parsing."""
    filtered_data = []
    xi_i_data = data.get('xi_i', {})
    
    for key, values in xi_i_data.items():
        if isinstance(values, dict):
            # If the value is a nested dictionary, append its values
            filtered_data.extend(values.values())
        else:
            # Otherwise, append the value directly
            filtered_data.append(values)
    
    return filtered_data


# def filter_xi_data(data):
#     """Extract desired variables from xi_i data."""
#     filtered_data = []
#     xi_i_data = data.get('xi_i', {})
    
#     if 'genCapAvail' in xi_i_data:
#         for k, v in xi_i_data['genCapAvail'].items():
#             try:
#                 parsed_k = ast.literal_eval(k)
#                 if len(parsed_k) >= 4:  # (country, tech, period, scenario)
#                     country, tech = parsed_k[0], parsed_k[1]
#                     for desired_country, desired_tech in desired_data_xi['Generation']:
#                         if country == desired_country and tech == desired_tech:
#                             filtered_data.append(v)
#             except (ValueError, SyntaxError) as e:
#                 logging.warning(f"Failed to parse key {k}: {str(e)}")
    
#     if 'sload' in xi_i_data:
#         for k, v in xi_i_data['sload'].items():
#             try:
#                 parsed_k = ast.literal_eval(k)
#                 if len(parsed_k) >= 1:  # (country, tech, period, scenario)
#                     country = parsed_k[0]
#                     for desired_country, desired_tech in desired_data_xi['Generation']:
#                         if country == desired_country:
#                             filtered_data.append(v)
#             except (ValueError, SyntaxError) as e:
#                 logging.warning(f"Failed to parse key {k}: {str(e)}")
    
#     if 'maxRegHydroGen' in xi_i_data:
#         for k, v in xi_i_data['maxRegHydroGen'].items():
#             try:
#                 parsed_k = ast.literal_eval(k)
#                 if len(parsed_k) >= 1:  # (country, tech, period, scenario)
#                     country = parsed_k[0]
#                     for desired_country, desired_tech in desired_data_xi['Generation']:
#                         if country == desired_country:
#                             filtered_data.append(v)
#             except (ValueError, SyntaxError) as e:
#                 logging.warning(f"Failed to parse key {k}: {str(e)}")
    
#     if not filtered_data:
#         logging.info(f"ERROR!!!")
#     # else:
#     #     logging.info(f"Extracted {len(filtered_data)} features from xi_i_data")
    
#     return filtered_data


def process_files(base_path):
    """Process all files and return organized data."""
    processed_data = {}
    
    # Process xi files first
    xi_files = glob.glob(os.path.join(base_path, 'xi_Q_*_period_*_seed_*'))
    logging.info(f"Found {len(xi_files)} xi files")
    
    for xi_file in xi_files:
        period, seed, filenum = extract_file_info(xi_file)
        if period is None:
            continue
            
        try:
            with open(xi_file, 'r') as f:
                xi_data = json.load(f)
            
            key = (period, seed, filenum)
            xi_filtered = filter_xi_data(list(xi_data.values())[0])
            
            if xi_filtered:  # Only process if we have valid xi data
                # processed_data[key] = {
                #     'period': period,
                #     'seed': seed,
                #     'filenum': filenum,
                #     'xi_i': xi_filtered,
                #     'Q_i': list(xi_data.values())[0].get('Q_i', {}).get('scenario1', '')  # Add Q_i here
                # }
                processed_data[key] = {
                    'period': period,
                    'seed': seed,
                    'filenum': filenum,
                    'xi_i': xi_filtered,
                    'operational_cost': list(xi_data.values())[0].get('Q_i', {}).get('operational_cost', {}).get('scenario1', ''),
                    'shed_cost': list(xi_data.values())[0].get('Q_i', {}).get('shed_cost', {}).get('scenario1', ''),
                    'scaled_Q_i': list(xi_data.values())[0].get('Q_i', {}).get('scaled_Q_i', {}).get('scenario1', ''),
                    'raw_Q_i': list(xi_data.values())[0].get('Q_i', {}).get('raw_Q_i', {}).get('scenario1', '')
                }
        except Exception as e:
            logging.error(f"Error processing {xi_file}: {str(e)}")
    
    # Process v files
    v_files = glob.glob(os.path.join(base_path, 'v_*_period_*_seed_*'))
    logging.info(f"Found {len(v_files)} v files")
    
    for v_file in v_files:
        period, seed, filenum = extract_file_info(v_file)
        if period is None:
            continue
            
        try:
            with open(v_file, 'r') as f:
                v_data = json.load(f)
            
            key = (period, seed, filenum)
            if key in processed_data:  # Only process if we have matching xi data
                v_filtered = filter_v_data(list(v_data.values())[0])
                if v_filtered:
                    processed_data[key]['v_i'] = v_filtered
        except Exception as e:
            logging.error(f"Error processing {v_file}: {str(e)}")



    x_files = glob.glob(os.path.join(base_path, 'x_*_period_*_seed_*'))
    logging.info(f"Found {len(x_files)} x files")
    
    for x_file in x_files:
        period, seed, filenum = extract_file_info(x_file)
        if period is None:
            continue
            
        try:
            with open(x_file, 'r') as f:
                x_data = json.load(f)
            key = (period, seed, filenum)
            if key in processed_data:  # Only process if we have matching xi data
                x_filtered = filter_x_data(list(x_data.values())[0])
                if x_filtered:
                    processed_data[key]['x_i'] = x_filtered
        except Exception as e:
            logging.error(f"Error processing {x_file}: {str(e)}")

    
    # Remove entries that don't have both v and xi data
    # processed_data = {k: v for k, v in processed_data.items() 
    #                  if 'v_i' in v and 'xi_i' in v}
    
    processed_data = {k: v for k, v in processed_data.items() 
                     if 'v_i' in v and 'xi_i' in v and 'x_i' in v}

    logging.info(f"Successfully processed {len(processed_data)} matching pairs of files")
    return processed_data

# def perform_pca(X, n_components=0.95):
#     """Perform PCA on the data."""
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     pca = PCA(n_components=n_components)
#     X_pca = pca.fit_transform(X_scaled)
#     return X_pca, pca, scaler

def ensure_directory_exists(directory_path):
    """Ensure the directory for the given path exists."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
            logging.info(f"Directory created: {directory_path}")
        except Exception as e:
            logging.error(f"Failed to create directory {directory_path}: {str(e)}")
            raise

# def save_results(processed_data, output_file='training_data.csv'):
#     """Save processed data to CSV file."""
#     try:
#         # Ensure the output directory exists
#         ensure_directory_exists(output_file)
        
#         # Create new file or overwrite existing one
#         with open(output_file, 'w', newline='') as csvfile:
#             fieldnames = ['period', 'seed', 'filenum', 'v_i', 'xi_i', 'Q_i']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
            
#             for key, data in sorted(processed_data.items()):
#                 if 'v_i' in data and 'xi_i' in data:  # Only save if both v and xi data exist
#                     row = {
#                         'period': data['period'],
#                         'seed': data['seed'],
#                         'filenum': data['filenum'],
#                         'v_i': json.dumps(data['v_i']),
#                         'xi_i': json.dumps(data['xi_i']),
#                         'Q_i': data.get('Q_i', '')  # Add Q_i if available
#                     }
#                     writer.writerow(row)
#         logging.info(f"Successfully saved results to {output_file}")
#     except Exception as e:
#         logging.error(f"Error saving results to {output_file}: {str(e)}")
#         raise

def save_pca_model(period, pca, feature_name, output_dir="period_PCA"):
    """Save PCA model for a specific feature and period."""
    try:
        # Ensure the output directory exists
        output_dir = "pca_period"
        ensure_directory_exists(output_dir)

        # File path for the specific period and feature
        file_path = os.path.join(output_dir, f"PCA_model_{feature_name}_period_{period}.pkl")
        logging.info(f"Saving PCA model to {file_path}")


        # Save the PCA model to a pickle file
        with open(file_path, 'wb') as pklfile:
            pickle.dump(pca, pklfile)

        logging.info(f"Saved PCA model for {feature_name} of period {period} to {file_path}")
    except Exception as e:
        logging.error(f"Error saving PCA model for {feature_name} of period {period}: {str(e)}")
        raise



def save_results(processed_data, output_file='training_data.csv'):
    """Save processed data to CSV file."""
    try:
        # Ensure the output directory exists
        ensure_directory_exists(output_file)
        
        # Create new file or overwrite existing one
        with open(output_file, 'w', newline='') as csvfile:
            # fieldnames = ['period', 'seed', 'filenum', 'x_i', 'v_i', 'xi_i', 'Q_i']
            fieldnames = ['period', 'seed', 'filenum', 'x_i', 'v_i', 'xi_i', 'operational_cost', 'shed_cost', 'scaled_Q_i', 'raw_Q_i']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for key, data in sorted(processed_data.items()):
                if 'v_i' in data and 'xi_i' in data and 'x_i' in data:  # Ensure all required fields exist
                    # row = {
                    #     'period': data['period'],
                    #     'seed': data['seed'],
                    #     'filenum': data['filenum'],
                    #     'x_i': json.dumps(data['x_i']),  # Add x_i
                    #     'v_i': json.dumps(data['v_i']),
                    #     'xi_i': json.dumps(data['xi_i']),
                    #     'Q_i': data.get('Q_i', '')  # Add Q_i if available
                    # }

                    row = {
                        'period': data['period'],
                        'seed': data['seed'],
                        'filenum': data['filenum'],
                        'x_i': json.dumps(data['x_i']),  # Add x_i
                        'v_i': json.dumps(data['v_i']),
                        'xi_i': json.dumps(data['xi_i']),
                        'operational_cost': data.get('operational_cost', ''),
                        'shed_cost': data.get('shed_cost', ''),
                        'scaled_Q_i': data.get('scaled_Q_i', ''),
                        'raw_Q_i': data.get('raw_Q_i', '')
                    }

                    writer.writerow(row)
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {str(e)}")
        raise


def main():
    # Configure more detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    base_path = 'results_5_scenarios'
    output_file = 'results_training_data/training_data3_5.csv'
    output_dir = 'pca_period'
    
    # Number of components for PCA
    n_components_x_i = 37  
    n_components_v_i = 41  
    n_components_xi_i = 45  
    
    # Process the files
    processed_data = process_files(base_path)
    
    if not processed_data:
        logging.error("No valid data pairs were processed. Exiting.")
        return

    # Get the list of keys in order
    keys_list = sorted(processed_data.keys())
    
    # Group keys by period
    keys_by_period = {}
    for key in keys_list:
        period = processed_data[key]['period']
        if period not in keys_by_period:
            keys_by_period[period] = []
        keys_by_period[period].append(key)
    
    # Determine min_features_v_i and min_features_xi_i
    min_features_x_i = min(len(processed_data[key]['x_i']) for key in processed_data.keys())
    min_features_v_i = min(len(processed_data[key]['v_i']) for key in processed_data.keys())
    min_features_xi_i = min(len(processed_data[key]['xi_i']) for key in processed_data.keys())
    
    # Adjust n_components to be less than or equal to min_features
    n_components_x_i = min(n_components_x_i, min_features_x_i)
    n_components_v_i = min(n_components_v_i, min_features_v_i)
    n_components_xi_i = min(n_components_xi_i, min_features_xi_i)
    
    # Log the number of components
    logging.info(f"Using n_components_x_i = {n_components_x_i} for PCA on x_i data")
    logging.info(f"Using n_components_v_i = {n_components_v_i} for PCA on v_i data")
    logging.info(f"Using n_components_xi_i = {n_components_xi_i} for PCA on xi_i data")
    

    # Perform PCA on x_i data for each period
    for period, keys in keys_by_period.items():
        x_i_data = [processed_data[key]['x_i'] for key in keys]
        X = np.array(x_i_data)
        
        if X.size == 0 or X.shape[1] == 0:
            logging.error(f"Invalid data shape for PCA on x_i data for period {period}: {X.shape}")
            continue
        
        logging.info(f"Performing PCA on x_i data for period {period} with shape: {X.shape}")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components_x_i)
        X_pca = pca.fit_transform(X_scaled)
        
        save_pca_model(period, pca, "x_i",output_dir=output_dir)

        joblib.dump(scaler, os.path.join(output_dir, f'scaler_x_{period}.joblib'))
    
        # Reconstruction
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Reconstruction error
        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        logging.info(f"Period {period}: Reconstruction error for x_i PCA: {mse}")
        
        # Update data
        for i, key in enumerate(keys):
            processed_data[key]['x_i'] = X_pca[i].tolist()


    # Perform PCA on v_i data for each period
    for period, keys in keys_by_period.items():
        v_i_data = [processed_data[key]['v_i'] for key in keys]
        X = np.array(v_i_data)
        
        if X.size == 0 or X.shape[1] == 0:
            logging.error(f"Invalid data shape for PCA on v_i data for period {period}: {X.shape}")
            continue
        
        logging.info(f"Performing PCA on v_i data for period {period} with shape: {X.shape}")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components_v_i)
        X_pca = pca.fit_transform(X_scaled)

        save_pca_model(period, pca, "v_i",output_dir=output_dir)
        joblib.dump(scaler, os.path.join(output_dir, f'scaler_v_{period}.joblib'))
        
        # Reconstruction
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Reconstruction error
        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        logging.info(f"Period {period}: Reconstruction error for v_i PCA: {mse}")
        
        # Update data
        for i, key in enumerate(keys):
            processed_data[key]['v_i'] = X_pca[i].tolist()
    
    # Perform PCA on xi_i data for each period
    for period, keys in keys_by_period.items():
        xi_i_data = [processed_data[key]['xi_i'] for key in keys]
        X = np.array(xi_i_data)
        
        if X.size == 0 or X.shape[1] == 0:
            logging.error(f"Invalid data shape for PCA on xi_i data for period {period}: {X.shape}")
            continue
        
        logging.info(f"Performing PCA on xi_i data for period {period} with shape: {X.shape}")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components_xi_i)
        X_pca = pca.fit_transform(X_scaled)
        
        save_pca_model(period, pca, "xi_i",output_dir=output_dir)
        joblib.dump(scaler, os.path.join(output_dir, f'scaler_xi_{period}.joblib'))

        # Reconstruction
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Reconstruction error
        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        logging.info(f"Period {period}: Reconstruction error for xi_i PCA: {mse}")
        
        # Update data
        for i, key in enumerate(keys):
            processed_data[key]['xi_i'] = X_pca[i].tolist()
    
    # Save results
    save_results(processed_data, output_file)
    logging.info("Data processing completed successfully.")

if __name__ == "__main__":
    main()