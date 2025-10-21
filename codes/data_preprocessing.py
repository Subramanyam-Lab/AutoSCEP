# import os
# import glob
# import pandas as pd
# import logging
# import ast
# import re 
# import argparse

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def flatten_v_vector(v_string):
#     try:
#         v_dict = ast.literal_eval(v_string)
#         if 'v_i' in next(iter(v_dict.values())):
#              v_dict = next(iter(v_dict.values()))['v_i']
#         flat_list = []
#         for key in sorted(v_dict.keys()):
#             sub_dict = v_dict[key]
#             for sub_key in sorted(sub_dict.keys()):
#                 flat_list.append(sub_dict[sub_key])
#         return flat_list
#     except (SyntaxError, TypeError, ValueError) as e:
#         logging.warning(f"v_i vector parsing failed: {v_string[:100]}... error: {e}")
#         return []

# def main(data_dir, output_file):
#     logging.info(f"search for 'file_*' folders in '{data_dir}' directory")
#     file_dirs = glob.glob(os.path.join(data_dir, 'file_*'))
#     if not file_dirs:
#         logging.error("'file_*' directory not found")
#         return
        
#     file_nums_to_process = sorted([int(re.findall(r'\d+', os.path.basename(d))[0]) for d in file_dirs])
#     logging.info(f"Total {len(file_nums_to_process)} file_nums to process")

#     try:
#         first_file_dir = os.path.join(data_dir, f'file_{file_nums_to_process[0]}')
#         any_csv_in_first_dir = glob.glob(os.path.join(first_file_dir, 'period_*.csv'))[0]
        
#         sample_df = pd.read_csv(any_csv_in_first_dir)
#         correct_header = sample_df.columns.tolist()
#         logging.info(f"Set correct header: {correct_header}")
#     except IndexError:
#         logging.error("CSV file to get correct header not found. Stopping processing.")
#         return
        
#     logging.info("process data by file_num")
    
#     processed_rows = []
    
#     for i, file_num in enumerate(file_nums_to_process):
#         if (i + 1) % 100 == 0:
#             logging.info(f"Progress: {i+1} / {len(file_nums_to_process)} (file_num: {file_num})")

#         file_path_pattern = os.path.join(data_dir, f'file_{file_num}', 'period_*.csv')
#         period_files = glob.glob(file_path_pattern)

#         if not period_files:
#             logging.warning(f"file_num {file_num} period file not found. Skipping.")
#             continue
            
#         group_df_list = []
#         for f in period_files:
#             try:
#                 with open(f, 'r') as temp_f:
#                     first_line = temp_f.readline()

#                 if 'period' in first_line:
#                     temp_df = pd.read_csv(f)
#                 else:
#                     logging.warning(f"Header not found in {f}. Applying standard header.")
#                     temp_df = pd.read_csv(f, header=None, names=correct_header)
                
#                 group_df_list.append(temp_df)

#             except pd.errors.EmptyDataError:
#                 logging.warning(f"File is empty and will be skipped: {f}")
#             except Exception as e:
#                 logging.error(f"Could not read file {f} due to an error: {e}")

#         if not group_df_list:
#             logging.warning(f"No valid dataframes to concat for file_num {file_num}. Skipping.")
#             continue

#         group_df = pd.concat(group_df_list, ignore_index=True)
#         group_df = group_df.sort_values('period')
        
#         v_vectors = group_df['v_i'].apply(flatten_v_vector).tolist()
#         concatenated_v = [item for sublist in v_vectors for item in sublist]
        
#         summed_eq = group_df['E_Q_i'].sum()
#         summed_c = group_df['c_i'].sum()
        
#         row_data = {
#             'file_num': file_num,
#             'v_concatenated': concatenated_v,
#             'E_Q': summed_eq,
#             'C': summed_c
#         }
#         processed_rows.append(row_data)

#     logging.info("Data reconstruction completed.")

#     logging.info("create final dataframe and save")
    
#     if not processed_rows:
#         logging.warning("No processed data. Creating empty file.")
#         pd.DataFrame().to_csv(output_file, index=False)
#         return

#     final_df = pd.DataFrame(processed_rows)
    
#     v_df = pd.DataFrame(final_df['v_concatenated'].tolist()).add_prefix('v_')
    
#     output_df = pd.concat([
#         final_df[['file_num']],
#         v_df,
#         final_df[['E_Q', 'C']]
#     ], axis=1)

#     output_df.to_csv(output_file, index=False)
#     logging.info(f"Final dataset saved in '{output_file}' (total {len(output_df)} samples)")
#     logging.info(f"Final dataset shape: {output_df.shape}")



# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--numsam', type=int, required=True)
#     parser.add_argument('--seed', type=int, required=True)
#     args = parser.parse_args()
 
#     seed = args.seed
#     numsam = args.numsam
#     data_dir = f"training_full_data_adaptive_{numsam}_{seed}"
#     output_file = f"aggregated_full_dataset_adaptive_{numsam}_{seed}.csv"
#     main(data_dir, output_file)

import os
import glob
import pandas as pd
import logging
import ast
import re 
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def flatten_v_vector_with_keys(v_string):
    """Flatten v_i vector and return both values and corresponding keys"""
    try:
        v_dict = ast.literal_eval(v_string)
        if 'v_i' in next(iter(v_dict.values())):
            v_dict = next(iter(v_dict.values()))['v_i']
        
        flat_list = []
        key_list = []
        
        for category in sorted(v_dict.keys()):
            sub_dict = v_dict[category]
            for sub_key in sorted(sub_dict.keys()):
                flat_list.append(sub_dict[sub_key])
                key_list.append((category, sub_key))
        
        return flat_list, key_list
    except (SyntaxError, TypeError, ValueError) as e:
        logging.warning(f"v_i vector parsing failed: {v_string[:100]}... error: {e}")
        return [], []

def flatten_v_vector(v_string):
    """Original function for flattening without keys"""
    try:
        v_dict = ast.literal_eval(v_string)
        if 'v_i' in next(iter(v_dict.values())):
            v_dict = next(iter(v_dict.values()))['v_i']
        
        flat_list = []
        for key in sorted(v_dict.keys()):
            sub_dict = v_dict[key]
            for sub_key in sorted(sub_dict.keys()):
                flat_list.append(sub_dict[sub_key])
        
        return flat_list
    except (SyntaxError, TypeError, ValueError) as e:
        logging.warning(f"v_i vector parsing failed: {v_string[:100]}... error: {e}")
        return []

def save_key_mapping(key_list, output_file):
    """Save the mapping between categories and their keys"""
    # Group keys by category
    category_dict = {}
    for category, key in key_list:
        if category not in category_dict:
            category_dict[category] = []
        category_dict[category].append(key)
    
    with open(output_file, 'w') as f:
        for category in sorted(category_dict.keys()):
            f.write(f"{category}: {category_dict[category]}\n")

def main(data_dir, output_file):
    logging.info(f"search for 'file_*' folders in '{data_dir}' directory")
    file_dirs = glob.glob(os.path.join(data_dir, 'file_*'))
    if not file_dirs:
        logging.error("'file_*' directory not found")
        return
        
    file_nums_to_process = sorted([int(re.findall(r'\d+', os.path.basename(d))[0]) for d in file_dirs])
    logging.info(f"Total {len(file_nums_to_process)} file_nums to process")

    try:
        first_file_dir = os.path.join(data_dir, f'file_{file_nums_to_process[0]}')
        any_csv_in_first_dir = glob.glob(os.path.join(first_file_dir, 'period_*.csv'))[0]
        
        sample_df = pd.read_csv(any_csv_in_first_dir)
        correct_header = sample_df.columns.tolist()
        logging.info(f"Set correct header: {correct_header}")
    except IndexError:
        logging.error("CSV file to get correct header not found. Stopping processing.")
        return
    
    # Extract key mapping from the first valid sample
    key_mapping_extracted = False
    key_mapping = []
    
    logging.info("process data by file_num")
    
    processed_rows = []
    
    for i, file_num in enumerate(file_nums_to_process):
        if (i + 1) % 100 == 0:
            logging.info(f"Progress: {i+1} / {len(file_nums_to_process)} (file_num: {file_num})")

        file_path_pattern = os.path.join(data_dir, f'file_{file_num}', 'period_*.csv')
        period_files = glob.glob(file_path_pattern)

        if not period_files:
            logging.warning(f"file_num {file_num} period file not found. Skipping.")
            continue
            
        group_df_list = []
        for f in period_files:
            try:
                with open(f, 'r') as temp_f:
                    first_line = temp_f.readline()

                if 'period' in first_line:
                    temp_df = pd.read_csv(f)
                else:
                    logging.warning(f"Header not found in {f}. Applying standard header.")
                    temp_df = pd.read_csv(f, header=None, names=correct_header)
                
                group_df_list.append(temp_df)

            except pd.errors.EmptyDataError:
                logging.warning(f"File is empty and will be skipped: {f}")
            except Exception as e:
                logging.error(f"Could not read file {f} due to an error: {e}")

        if not group_df_list:
            logging.warning(f"No valid dataframes to concat for file_num {file_num}. Skipping.")
            continue

        group_df = pd.concat(group_df_list, ignore_index=True)
        group_df = group_df.sort_values('period')
        
        # Extract key mapping from first period of first file
        if not key_mapping_extracted and len(group_df) > 0:
            first_v_string = group_df['v_i'].iloc[0]
            _, key_mapping = flatten_v_vector_with_keys(first_v_string)
            if key_mapping:
                key_mapping_extracted = True
                logging.info(f"Key mapping extracted from first period (total {len(key_mapping)} keys)")
        
        v_vectors = group_df['v_i'].apply(flatten_v_vector).tolist()
        concatenated_v = [item for sublist in v_vectors for item in sublist]
        
        summed_eq = group_df['E_Q_i'].sum()
        summed_c = group_df['c_i'].sum()
        
        row_data = {
            'file_num': file_num,
            'v_concatenated': concatenated_v,
            'E_Q': summed_eq,
            'C': summed_c
        }
        processed_rows.append(row_data)

    logging.info("Data reconstruction completed.")

    # Save key mapping to file
    if key_mapping:
        key_mapping_file = output_file.replace('.csv', '_key_mapping.txt')
        save_key_mapping(key_mapping, key_mapping_file)
        logging.info(f"Key mapping saved to '{key_mapping_file}'")
    else:
        logging.warning("No key mapping was extracted")

    logging.info("create final dataframe and save")
    
    if not processed_rows:
        logging.warning("No processed data. Creating empty file.")
        pd.DataFrame().to_csv(output_file, index=False)
        return

    final_df = pd.DataFrame(processed_rows)
    
    v_df = pd.DataFrame(final_df['v_concatenated'].tolist()).add_prefix('v_')
    
    output_df = pd.concat([
        final_df[['file_num']],
        v_df,
        final_df[['E_Q', 'C']]
    ], axis=1)

    output_df.to_csv(output_file, index=False)
    logging.info(f"Final dataset saved in '{output_file}' (total {len(output_df)} samples)")
    logging.info(f"Final dataset shape: {output_df.shape}")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--numsam', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
 
    seed = args.seed
    numsam = args.numsam
    data_dir = f"training_full_data_adaptive_{numsam}_{seed}"
    output_file = f"aggregated_full_dataset_adaptive_{numsam}_{seed}.csv"
    main(data_dir, output_file)