import numpy as np
import json
import csv
import os
import re
import glob
import ast  # Add this import
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import pickle
import joblib  # To save scaler and PCA objects

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
    pattern = r'(?:v|xi_Q)_(\d+)_period_(\d+)_seed_(\d+)'
    match = re.search(pattern, filename)
    if match:
        period, seed, filenum = map(int, match.groups())
        return period, seed, filenum
    return None, None, None

# def filter_v_data(data):
#     """Extract desired variables from v_i data."""
#     filtered_data = []
#     v_i_data = data.get('v_i', {})
    
#     for category, country_tech in desired_data_v.items():
#         for country, technologies in country_tech.items():
#             for tech in technologies:
#                 key = (country, tech)
#                 if category == 'Generation':
#                     value = v_i_data.get('genInstalledCap', {}).get(str(key), None)
#                 elif category == 'Storage Power':
#                     value = v_i_data.get('storPWInstalledCap', {}).get(str(key), None)
#                 elif category == 'Storage Energy':
#                     value = v_i_data.get('storENInstalledCap', {}).get(str(key), None)
                
#                 if value is not None:
#                     filtered_data.append(value)

#     transmission_data = v_i_data.get('transmissionInstalledCap', {})
#     for value in transmission_data.values():
#         if value is not None:
#             filtered_data.append(value)
    

#     return filtered_data


# no filter
def filter_v_data(data):
    """Extract all data from v_i without parsing."""
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

def filter_xi_data(data):
    """Extract desired variables from xi_i data."""
    filtered_data = []
    xi_i_data = data.get('xi_i', {})
    
    if 'genCapAvail' in xi_i_data:
        for k, v in xi_i_data['genCapAvail'].items():
            try:
                parsed_k = ast.literal_eval(k)
                if len(parsed_k) >= 4:  # (country, tech, period, scenario)
                    country, tech = parsed_k[0], parsed_k[1]
                    for desired_country, desired_tech in desired_data_xi['Generation']:
                        if country == desired_country and tech == desired_tech:
                            filtered_data.append(v)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Failed to parse key {k}: {str(e)}")
    
    if 'sload' in xi_i_data:
        for k, v in xi_i_data['sload'].items():
            try:
                parsed_k = ast.literal_eval(k)
                if len(parsed_k) >= 1:  # (country, tech, period, scenario)
                    country = parsed_k[0]
                    for desired_country, desired_tech in desired_data_xi['Generation']:
                        if country == desired_country:
                            filtered_data.append(v)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Failed to parse key {k}: {str(e)}")
    
    if 'maxRegHydroGen' in xi_i_data:
        for k, v in xi_i_data['maxRegHydroGen'].items():
            try:
                parsed_k = ast.literal_eval(k)
                if len(parsed_k) >= 1:  # (country, tech, period, scenario)
                    country = parsed_k[0]
                    for desired_country, desired_tech in desired_data_xi['Generation']:
                        if country == desired_country:
                            filtered_data.append(v)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Failed to parse key {k}: {str(e)}")
    
    if not filtered_data:
        logging.error("No data was extracted from xi_i_data")
    else:
        logging.info(f"Extracted {len(filtered_data)} features from xi_i_data")
    
    return filtered_data


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
                processed_data[key] = {
                    'period': period,
                    'seed': seed,
                    'filenum': filenum,
                    'xi_i': xi_filtered,
                    'Q_i': list(xi_data.values())[0].get('Q_i', {}).get('scenario1', '')  # Add Q_i here
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
    
    # Remove entries that don't have both v and xi data
    processed_data = {k: v for k, v in processed_data.items() 
                     if 'v_i' in v and 'xi_i' in v}
    
    logging.info(f"Successfully processed {len(processed_data)} matching pairs of files")
    return processed_data

# def perform_pca(X, n_components=0.95):
#     """Perform PCA on the data."""
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     pca = PCA(n_components=n_components)
#     X_pca = pca.fit_transform(X_scaled)
#     return X_pca, pca, scaler

def perform_pca(X, save_prefix, n_components=0.95):
    """Perform PCA on the data and save results with a specified prefix."""
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Save results
    joblib.dump(scaler, f"{save_prefix}_scaler.pkl")
    joblib.dump(pca, f"{save_prefix}_pca.pkl")
    joblib.dump(X_pca, f"{save_prefix}_X_pca.pkl")

    return X_pca, pca, scaler


def ensure_directory_exists(file_path):
    """Ensure the directory for the given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def save_results(processed_data, output_file='training_data.csv'):
    """Save processed data to CSV file."""
    try:
        # Ensure the output directory exists
        ensure_directory_exists(output_file)
        
        # Create new file or overwrite existing one
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['period', 'seed', 'filenum', 'v_i', 'xi_i', 'Q_i']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for key, data in sorted(processed_data.items()):
                if 'v_i' in data and 'xi_i' in data:  # Only save if both v and xi data exist
                    row = {
                        'period': data['period'],
                        'seed': data['seed'],
                        'filenum': data['filenum'],
                        'v_i': json.dumps(data['v_i']),
                        'xi_i': json.dumps(data['xi_i']),
                        'Q_i': data.get('Q_i', '')  # Add Q_i if available
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
    
    base_path = 'results_3_scenarios'
    output_file = 'results/training_data3.csv'
    
    # Process the files
    processed_data = process_files(base_path)
    
    if not processed_data:
        logging.error("No valid data pairs were processed. Exiting.")
        return
    
    ######################### PCA for v ######################################
    # Perform PCA for data v
    v_data = [data['v_i'] for data in processed_data.values() if 'v_i' in data]
    
    if not v_data:
        logging.error("No v data available for PCA. Exiting.")
        return
    
    X = np.array(v_data)
    if X.size == 0 or X.shape[1] == 0:
        logging.error(f"Invalid data shape for PCA: {X.shape}")
        return
        
    logging.info(f"Performing PCA on data with shape: {X.shape}")
    
    try:
        X_pca, pca_model, scaler = perform_pca(X, "v_i_results")
        logging.info(f"PCA completed successfully. Reduced shape: {X_pca.shape}")
        
        # Add PCA results back to processed_data
        for (key, data), pca_vector in zip(processed_data.items(), X_pca):
            if 'v_i' in data:
                data['v_i'] = pca_vector.tolist()

    except Exception as e:
        logging.error(f"Error during PCA or saving results: {str(e)}")
        raise

    ######################### PCA for v ######################################


    ######################### PCA for xi ######################################
    # Prepare data for PCA
    xi_data = [data['xi_i'] for data in processed_data.values() if 'xi_i' in data]
    
    if not xi_data:
        logging.error("No xi data available for PCA. Exiting.")
        return
    
    X = np.array(xi_data)
    if X.size == 0 or X.shape[1] == 0:
        logging.error(f"Invalid data shape for PCA: {X.shape}")
        return
        
    logging.info(f"Performing PCA on data with shape: {X.shape}")
    
    try:
        X_pca, pca_model, scaler = perform_pca(X, "xi_i_results")
        logging.info(f"PCA completed successfully. Reduced shape: {X_pca.shape}")
        
        # Add PCA results back to processed_data
        for (key, data), pca_vector in zip(processed_data.items(), X_pca):
            if 'xi_i' in data:
                data['xi_i'] = pca_vector.tolist()
        
        # Save results
        save_results(processed_data, output_file)
        logging.info("Data processing completed successfully.")
    
    except Exception as e:
        logging.error(f"Error during PCA or saving results: {str(e)}")
        raise

    ######################### PCA for xi ######################################

if __name__ == "__main__":
    main()






