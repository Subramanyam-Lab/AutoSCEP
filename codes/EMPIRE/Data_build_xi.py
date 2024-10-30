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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

desired_generators = ['Solar', 'Windonshore', 'GasCCGT', 'Bio']
desired_countries = ['Germany', 'France']

def include_entry(var, k):
    try:
        parsed_k = ast.literal_eval(k)
    except (ValueError, SyntaxError):
        return False
    
    if var == 'genCapAvail':
        if len(parsed_k) >= 2:
            country, generator = parsed_k[0], parsed_k[1]
            return country in desired_countries and generator in desired_generators
    elif var in ['sload', 'maxRegHydroGen']:
        country = parsed_k[0]
        return country in desired_countries
    return False

def prepare_data(file_paths):
    all_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        scenario_data = list(data.values())[0]
        filtered_data = []
        for var in scenario_data['xi_i'].keys():
            for k, v in scenario_data['xi_i'][var].items():
                if include_entry(var, k):
                    filtered_data.append(v)
        all_data.append(filtered_data)
    return np.array(all_data)

def perform_pca(X, n_components=0.95):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca, scaler

def extract_period(file_path):
    match = re.search(r'xi_Q_(\d+)_period', file_path)
    if match:
        return int(match.group(1))
    return None


def save_pca_data(X_pca, pca_model, scaler, output_dir='pca_results'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/pca_vectors.npy', X_pca)
    
    with open(f'{output_dir}/pca_model.pkl', 'wb') as f:
        pickle.dump(pca_model, f)
    
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    pca_results = {
        'explained_variance_ratio': pca_model.explained_variance_ratio_.tolist(),
        'components': pca_model.components_.tolist(),
        'mean': pca_model.mean_.tolist(),
        'number_of_components': int(pca_model.n_components_),
        'n_samples': X_pca.shape[0],
        'n_features_reduced': X_pca.shape[1]
    }
    
    with open(f'{output_dir}/pca_results.json', 'w') as f:
        json.dump(pca_results, f, indent=2)
    
    logging.info(f"PCA results have been saved to {output_dir}.")


def process_and_save_data(file_paths, X_pca, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['i', 'v_i', 'xi_i', 'Q_i']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file_path, pca_vector in zip(file_paths, X_pca):
            with open(file_path, 'r') as f:
                data = json.load(f)
            period = extract_period(file_path)
            Q_i = list(data.values())[0].get('Q_i', {}).get('scenario1', 'N/A')
            
            csv_data = {
                'i': period,
                'v_i': '',  # Empty as requested
                'xi_i': str(pca_vector.tolist()),  # PCA result as a vector
                'Q_i': Q_i
            }
            writer.writerow(csv_data)
            logging.info(f"Successfully processed file: {file_path}")
            

if __name__ == "__main__":
    file_paths = glob.glob('results_1_scenarios/xi_Q_*_period_*_scenarios_*')
    logging.info(f"Found {len(file_paths)} files")
    
    X = prepare_data(file_paths)
    X_pca, pca_model, scaler = perform_pca(X)
    
    save_pca_data(X_pca, pca_model, scaler)
    
    # loaded_X_pca, loaded_pca_model, loaded_scaler, loaded_results = load_pca_data()

    # Process and save data
    csv_file_path = 'training_data.csv'
    process_and_save_data(file_paths, X_pca, csv_file_path)

    print(f"Data has been processed and saved to {csv_file_path}")

