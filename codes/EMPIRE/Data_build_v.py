import json
import csv
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

desired_data = {
    'Generation': ({'Generation': ['Germany', 'France']}, {'Generation': ['Solar', 'Windonshore', 'GasCCGT', 'Bio']}),
    'Storage Power': ({'Storage Power': ['Germany']}, {'Storage Power': ['Li-Ion_BESS']}),
    'Storage Energy': ({'Storage Energy': ['Germany']}, {'Storage Energy': ['Li-Ion_BESS']})
}

def extract_period(file_path):
    import re
    match = re.search(r'v_(\d+)_period', os.path.basename(file_path))
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Couldn't extract period from filename: {file_path}")

def include_entry(category, key):
    country, technology = key
    if category == 'Generation':
        return (country in desired_data[category][0]['Generation'] and 
                technology in desired_data[category][1]['Generation'])
    elif category in ['Storage Power', 'Storage Energy']:
        return (country in desired_data[category][0][category] and 
                technology in desired_data[category][1][category])
    return False

def prepare_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    extracted_values = []
    scenario_data = list(data.values())[0]['v_i']
    
    # Generation data
    gen_data = scenario_data.get('genInstalledCap', {})
    extracted_values.extend([v for k, v in gen_data.items() if include_entry('Generation', eval(k))])
    
    # Storage Power data
    stor_pw_data = scenario_data.get('storPWInstalledCap', {})
    extracted_values.extend([v for k, v in stor_pw_data.items() if include_entry('Storage Power', eval(k))])
    
    # Storage Energy data
    stor_en_data = scenario_data.get('storENInstalledCap', {})
    extracted_values.extend([v for k, v in stor_en_data.items() if include_entry('Storage Energy', eval(k))])
    
    # Transmission data - include all transmission data
    trans_data = scenario_data.get('transmissionInstalledCap', {})
    extracted_values.extend(list(trans_data.values()))
    
    return extracted_values

def process_all_files(file_paths):
    processed_data = []
    for file_path in sorted(file_paths, key=extract_period):
        try:
            extracted_values = prepare_data(file_path)
            processed_data.append(extracted_values)
            logging.info(f"Successfully processed file: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
    return processed_data

def update_csv(csv_file_path, processed_data):
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_data = list(reader)
    
    if len(csv_data) != len(processed_data):
        raise ValueError(f"Mismatch in number of rows: CSV has {len(csv_data)}, processed data has {len(processed_data)}")
    
    for row, data in zip(csv_data, processed_data):
        row['v_i'] = json.dumps(data)
    
    fieldnames = ['i', 'v_i', 'xi_i', 'Q_i']
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

# Main execution
csv_file_path = 'training_data.csv'
file_paths = glob.glob('results_1_scenarios/v_*_period_*_scenarios_*.json')
logging.info(f"Found {len(file_paths)} files")
processed_data = process_all_files(file_paths)
update_csv(csv_file_path, processed_data)
print(f"Data has been processed and saved to {csv_file_path}")