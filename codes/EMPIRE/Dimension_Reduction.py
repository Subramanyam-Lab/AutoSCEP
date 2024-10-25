import json
import glob
import numpy as np
import re
import os
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

desired_generators = ['Solar', 'Windonshore', 'GasCCGT', 'Bio'] 
desired_countries = ['Germany', 'France']  

def include_entry(var, k):
    try:
        parsed_k = ast.literal_eval(k) 
    except (ValueError, SyntaxError):
        return False  

    if var == 'genCapAvail':
        # Key type: (country, generator, period, scenario)
        if len(parsed_k) >= 2:
            country, generator = parsed_k[0], parsed_k[1]
            return country in desired_countries and generator in desired_generators
    elif var in ['sload', 'maxRegHydroGen']:
        country = parsed_k[0]
        return country in desired_countries
    return False

file_paths = glob.glob('results_1_scenarios/xi_Q_*_period_*_scenarios_*')
logging.info(f"Found {len(file_paths)} files")

period_data = {}

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    period_match = re.search(r'xi_Q_(\d+)_period', file_path)
    if period_match:
        period = int(period_match.group(1))
        
        if period not in period_data:
            period_data[period] = []
        period_data[period].append(data)

logging.info(f"Loaded data for {len(period_data)} periods")

avg_data = {}
for period, datasets in period_data.items():
    avg_data[period] = {"xi_i": {}}
    
    first_dataset = datasets[0]
    first_scenario = list(first_dataset.values())[0]  

    for var in first_scenario['xi_i'].keys():
        keys = list(first_scenario['xi_i'][var].keys())
        first_key = keys[0]
        first_value = first_scenario['xi_i'][var][first_key]
    
    for var in first_scenario['xi_i'].keys():
        for k in first_scenario['xi_i'][var].keys():
            if include_entry(var, k):
                if var not in avg_data[period]["xi_i"]:
                    avg_data[period]["xi_i"][var] = {}
                avg_data[period]["xi_i"][var][k] = 0

    for dataset in datasets:
        scenario_data = list(dataset.values())[0] 
        for var in scenario_data['xi_i'].keys():
            if var in avg_data[period]["xi_i"]:
                for k, v in scenario_data['xi_i'][var].items():
                    if k in avg_data[period]["xi_i"][var]:
                        avg_data[period]["xi_i"][var][k] += v

    num_datasets = len(datasets)
    for var in avg_data[period]["xi_i"].keys():
        for k in avg_data[period]["xi_i"][var].keys():
            avg_data[period]["xi_i"][var][k] /= num_datasets

logging.info("Finished calculating average data for all periods")


included_pairs = set()
for period in avg_data.keys():
    for var in avg_data[period]["xi_i"].keys():
        for k in avg_data[period]["xi_i"][var].keys():
            included_pairs.add((var, k))
included_pairs = sorted(included_pairs)

data_matrix = []
periods = sorted(avg_data.keys())  

for period in periods:
    avg_vector = []
    for var, k in included_pairs:
        value = avg_data[period]["xi_i"].get(var, {}).get(k, 0)
        avg_vector.append(value)
    data_matrix.append(avg_vector)

X = np.array(data_matrix) 

print(f"\nData matrix X shape: {X.shape}")


if X.shape[0] < 2 or X.shape[1] < 1:
    logging.warning("Not enough data to perform PCA due to insufficient features.")
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95)  # 95%의 variance num_component
    X_pca = pca.fit_transform(X_scaled)

    X_reconstructed = pca.inverse_transform(X_pca)
    X_reconstructed_original_scale = scaler.inverse_transform(X_reconstructed)

    # MSE
    reconstruction_error = mean_squared_error(X, X_reconstructed_original_scale)
    logging.info(f"Reconstruction MSE: {reconstruction_error}")

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    logging.info(f"Cumulative explained variance ratio: {cumulative_explained_variance}")

    # Average correlation between original and reconstructed
    correlations = []
    zero_variance_features = 0
    for i in range(X.shape[1]):
        original = X[:, i]
        reconstructed = X_reconstructed_original_scale[:, i]
        
        # Check for zero variance
        if np.std(original) == 0 or np.std(reconstructed) == 0:
            zero_variance_features += 1
            continue  # Skip this feature
        else:
            corr = np.corrcoef(original, reconstructed)[0, 1]
            correlations.append(corr)

    if correlations:
        average_correlation = np.mean(correlations)
    else:
        average_correlation = np.nan  # Assign NaN if no valid correlations

    logging.info(f"Number of features with zero variance: {zero_variance_features}")
    logging.info(f"Average correlation between original and reconstructed data: {average_correlation}")

    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_
    
    logging.info(f"PCA completed. Number of components: {pca.n_components_}")
    logging.info(f"Explained variance ratio: {explained_variance_ratio}")
    
    output_dir = 'average_period_data_with_pca'
    os.makedirs(output_dir, exist_ok=True)
    
    pca_results = {
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_explained_variance': cumulative_explained_variance.tolist(),
        'components': components.tolist(),
        'mean': pca.mean_.tolist(),
        'number_of_components': int(pca.n_components_),
        'reconstruction_error': reconstruction_error,
        'average_correlation': average_correlation 
    }
    with open(f'{output_dir}/pca_results.json', 'w') as f:
        json.dump(pca_results, f, indent=2)
    logging.info(f"PCA results have been saved to '{output_dir}/pca_results.json'")
    
    transformed_data = {
        'periods': periods,
        'X_pca': X_pca.tolist()
    }
    with open(f'{output_dir}/transformed_data.json', 'w') as f:
        json.dump(transformed_data, f, indent=2)
    logging.info(f"Transformed data has been saved to '{output_dir}/transformed_data.json'")

output_dir_avg = 'average_period_data_filtered'
os.makedirs(output_dir_avg, exist_ok=True)

for period, data in avg_data.items():
    filename = f'{output_dir_avg}/average_period_{period}_data.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info(f"Average data for Period {period} has been saved to '{filename}'")

with open(f'{output_dir_avg}/all_periods_average_data.json', 'w') as f:
    json.dump(avg_data, f, indent=2)
logging.info(f"All periods average data has been saved to '{output_dir_avg}/all_periods_average_data.json'")