# import pandas as pd
# import numpy as np
# from scipy import stats
# import glob
# from FSD_sampling_violation import create_model, load_investment_data, inv_allo
# from pyomo.environ import *
# from sklearn.neighbors import KernelDensity
# import csv
# import io
# import contextlib
# from pyomo.core.expr.visitor import identify_variables
# import warnings
# warnings.filterwarnings('ignore')


# # def read_csv_files(file_pattern):
# #     all_data = []
# #     for file in glob.glob(file_pattern):
# #         df = pd.read_csv(file)
# #         all_data.append(df)
# #     return all_data

# # def build_kde_and_sample(data_list, bandwidth_factor):
# #     combined_data = pd.concat(data_list, ignore_index=True)
# #     grouped = combined_data.groupby(['Node', 'Energy_Type', 'Period', 'Type'])
# #     sampled_data = []
    
# #     for name, group in grouped:
# #         values = group['Value'].values
# #         if len(values) >= 2:
# #             try:
# #                 bw = bandwidth_factor
# #                 kde = stats.gaussian_kde(values, bw_method=bw)
# #                 new_sample = kde.resample(1)[0][0]
# #             except:
# #                 new_sample = np.random.choice(values)
# #         else:
# #             new_sample = np.random.choice(values)
        
# #         sampled_data.append({
# #             'Node': name[0],
# #             'Energy_Type': name[1],
# #             'Period': name[2],
# #             'Type': name[3],
# #             'Value': new_sample
# #         })
# #     return pd.DataFrame(sampled_data)



# import pandas as pd
# import numpy as np
# import glob
# from FSD_sampling_violation import create_model, load_investment_data
# from pyomo.environ import *
# from sklearn.neighbors import KernelDensity
# import io
# import contextlib
# from pyomo.core.expr.visitor import identify_variables


# def read_csv_files(file_pattern):
#     all_samples = []
#     for file in glob.glob(file_pattern):
#         df = pd.read_csv(file)
#         # Assuming 'Value' column contains scalar values
#         sample = df['Value'].values  # Shape: (616,)
#         all_samples.append(sample)
#     return np.array(all_samples)  # Shape: (n_samples, 616)

# def build_kde_and_sample(values, bandwidth_factor):

#     print(f"Data shape: {values.shape}")

#     # Apply log transformation
#     values_log = np.log(values + 1e-6)

#     # Fit the KDE model directly on the high-dimensional data
#     kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth_factor)
#     kde.fit(values_log)

#     # Generate new samples
#     num_samples = 1  # Number of samples to generate
#     samples_log = kde.sample(num_samples, random_state=42)
#     new_samples = np.exp(samples_log)
#     print(f"Generated {num_samples} new samples with shape: {new_samples.shape}")
    
#     return new_samples

# def save_to_csv(df, output_file):
#     df.to_csv(output_file, index=False)

# def check_constraints(instance):
#     violated_constraints = []
#     for constr in instance.component_objects(Constraint, active=True):
#         constr_name = constr.name
#         for index in constr:
#             c = constr[index]
#             try:
#                 body_value = value(c.body)
#                 lower = value(c.lower) if c.lower is not None else None
#                 upper = value(c.upper) if c.upper is not None else None
#                 tol = 1e-6  # Tolerance for floating-point comparisons
#                 violation = None
#                 if lower is not None and body_value < lower - tol:
#                     violation = ('Lower bound violated', body_value, lower)
#                 elif upper is not None and body_value > upper + tol:
#                     violation = ('Upper bound violated', body_value, upper)
#                 if violation:
#                     # Check if the constraint includes slack variables
#                     slack_info = ""
#                     vars_in_constraint = list(identify_variables(c.body))
#                     slack_vars = [v for v in vars_in_constraint if 'Slack' in v.name]
#                     if slack_vars:
#                         slack_values = {v.name: value(v) for v in slack_vars}
#                         slack_info = f", Slack Variables: {slack_values}"
#                     violated_constraints.append((
#                         constr_name,
#                         index,
#                         violation[0],
#                         violation[1],
#                         violation[2],
#                         slack_info
#                     ))
#             except (ValueError, ZeroDivisionError):
#                 # Skip constraints with uninitialized variables
#                 continue
#     return violated_constraints


# def check_model_feasibility(instance):
#     solver = SolverFactory('glpk')
#     try:
#         results = solver.solve(instance, tee=False)
#     except Exception as e:
#         return False

#     if results.solver.termination_condition == TerminationCondition.optimal:
#         return True
#     elif results.solver.termination_condition == TerminationCondition.infeasible:
#         print("Model is infeasible.")
#     else:
#         print(f"Solver Termination Condition: {results.solver.termination_condition}")
#         print("Couldn't evaluate feasibility")
#         return None

# def has_negative_values(fsd_data):
#     for row in fsd_data:
#         cap_value = float(row[4])  # Assuming 'Value' is the 5th column
#         if cap_value < 0:
#             return True
#     return False

# # def generate_feasible_samples(n, input_file_pattern, data_folder, bandwidth_factor):
# #     data_list = read_csv_files(input_file_pattern)
# #     feasible_samples = 0
# #     total_attempts = 0
# #     max_attempts = 1000

# #     while feasible_samples < n:
# #         if total_attempts >= max_attempts:
# #             break
# #         sampled_df = build_kde_and_sample(data_list, bandwidth_factor)
# #         total_attempts += 1

# #         # Convert sampled_df to fsd_data
# #         fsd_data = sampled_df.values.tolist()

# #         # Check for negative values
# #         if has_negative_values(fsd_data):
# #             print(f"Attempt {total_attempts} has negative capacities. Discarding.")
# #             continue

# #         # Load investment data
# #         gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(fsd_data)

# #         # Create model and data
# #         model, data = create_model(data_folder, gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)

# #         # Check feasibility
# #         if check_model_feasibility(model, data):
# #             output_file = f'sampled_data_{feasible_samples + 1}_{bandwidth_factor}.csv'
# #             save_to_csv(sampled_df, output_file)
# #             print(f"Sample {feasible_samples + 1} is feasible. Saved as {output_file}")
# #             feasible_samples += 1
# #         else:
# #             print(f"Attempt {total_attempts} is infeasible. Discarding.")

# #         if total_attempts % 5 == 0:
# #             current_ratio = feasible_samples / total_attempts
# #             print(f"Current accept-rejection ratio: {current_ratio:.6f}")

# #     final_ratio = feasible_samples / total_attempts
# #     print(f"Generated {feasible_samples} feasible samples.")
# #     print(f"Final accept-rejection ratio: {final_ratio:.6f}")
# #     print(f"Total attempts: {total_attempts}")


# def generate_feasible_samples(n, input_file_pattern, data_folder, bandwidth_factor):
#     # Read the data and get a NumPy array of shape (n_samples, 616)
#     values = read_csv_files(input_file_pattern)
#     feasible_samples = 0
#     total_attempts = 0
#     max_attempts = 1000

#     # Read one of the original files to get the metadata
#     sample_template_df = pd.read_csv(glob.glob(input_file_pattern)[0])

#     while feasible_samples < n:
#         if total_attempts >= max_attempts:
#             break

#         # Generate new samples using KDE
#         new_samples = build_kde_and_sample(values, bandwidth_factor)
#         total_attempts += 1

#         # Reconstruct the DataFrame
#         sampled_df = pd.DataFrame()
#         for i in range(new_samples.shape[0]):
#             df = sample_template_df.copy()
#             df['Value'] = new_samples[i]
#             sampled_df = pd.concat([sampled_df, df], ignore_index=True)

#         # Proceed with the rest of your code
#         # Convert sampled_df to fsd_data
#         fsd_data = sampled_df.values.tolist()

#         # Check for negative values
#         if has_negative_values(fsd_data):
#             print(f"Attempt {total_attempts} has negative capacities. Discarding.")
#             continue

#         # Load investment data
#         gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(fsd_data)

#         # Create model and data
#         instance = create_model(data_folder, gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)
#         fsd_instance = inv_allo(instance,gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)
#         # Check feasibility
#         if check_model_feasibility(fsd_instance):
#             output_file = f'sampled_data_{feasible_samples + 1}_{bandwidth_factor}.csv'
#             save_to_csv(sampled_df, output_file)
#             print(f"Sample {feasible_samples + 1} is feasible. Saved as {output_file}")
#             feasible_samples += 1
#         else:
#             print(f"Attempt {total_attempts} is infeasible. Discarding.")

#         if total_attempts % 5 == 0:
#             current_ratio = feasible_samples / total_attempts
#             print(f"Current accept-rejection ratio: {current_ratio:.6f}")

#     final_ratio = feasible_samples / total_attempts
#     print(f"Generated {feasible_samples} feasible samples.")
#     print(f"Final accept-rejection ratio: {final_ratio:.6f}")
#     print(f"Total attempts: {total_attempts}")

# # Main execution
# if __name__ == "__main__":
#     # input_file_pattern = "SeedSamples/reduced/fsd_seed*.csv"  # Adjust this pattern to match your input files
#     input_file_pattern = "SeedSamples/reduced_v2/*_616_seed_*.csv"  # Adjust this pattern to match your input files
#     data_folder = 'Data handler/sampling/reduced'
#     n_samples = 100  # Number of feasible samples you want to generate
#     bandwidth_factor = 1e-7  # Adjust this value to control the bandwidth (0 for exploit, larger for explore)
    
#     generate_feasible_samples(n_samples, input_file_pattern, data_folder, bandwidth_factor)




# import pandas as pd
# import numpy as np
# import glob
# from FSD_sampling_violation import create_model, load_investment_data, inv_allo
# from pyomo.environ import *
# import io
# import contextlib
# from pyomo.core.expr.visitor import identify_variables
# import warnings
# warnings.filterwarnings('ignore')

# def has_negative_values(fsd_data):
#     """
#     Checks if any value in the fsd_data is negative.

#     Parameters:
#     - fsd_data: List of lists representing the FSD data.

#     Returns:
#     - True if any value is negative, False otherwise.
#     """
#     for row in fsd_data:
#         cap_value = float(row[4])  # Assuming 'Value' is the 5th column
#         if cap_value < 0:
#             return True
#     return False

# def save_to_csv(df, output_file):
#     """
#     Saves the DataFrame to a CSV file.

#     Parameters:
#     - df: Pandas DataFrame to save.
#     - output_file: File path for the output CSV.
#     """
#     df.to_csv(output_file, index=False)

# def check_model_feasibility(instance):
#     """
#     Checks the feasibility of the model instance by solving it.

#     Parameters:
#     - instance: Pyomo model instance.

#     Returns:
#     - True if the model is feasible and solved optimally.
#     - False if the model is infeasible.
#     - None if the solver couldn't determine feasibility.
#     """
#     solver = SolverFactory('glpk')  # You can change this to your preferred solver
#     try:
#         results = solver.solve(instance, tee=False)
#     except Exception as e:
#         print(f"Solver exception: {e}")
#         return False

#     if results.solver.termination_condition == TerminationCondition.optimal:
#         return True
#     elif results.solver.termination_condition == TerminationCondition.infeasible:
#         print("Model is infeasible.")
#         return False
#     else:
#         print(f"Solver Termination Condition: {results.solver.termination_condition}")
#         print("Couldn't evaluate feasibility")
#         return None

# # def generate_uniform_fsd_samples(bounds_dict):
# #     """
# #     Generates uniform random samples within the given bounds for FSD variables.

# #     Parameters:
# #     - bounds_dict: Dictionary containing bounds for each FSD variable.

# #     Returns:
# #     - sampled_df: Pandas DataFrame containing the sampled FSD data.
# #     """
# #     sampled_data = []
# #     for var_type, bounds in bounds_dict.items():
# #         for key, (lb, ub) in bounds.items():
# #             if lb == ub:
# #                 value = lb
# #             elif ub is None or ub == float('inf'):
# #                 value = lb  # If no upper bound, set to lower bound
# #             else:
# #                 value = np.random.uniform(lb, ub)
# #             # Depending on var_type, construct the appropriate row
# #             if var_type == 'genInvCap':
# #                 n, g, i = key
# #                 sampled_data.append({
# #                     'Node': n,
# #                     'Energy_Type': g,
# #                     'Period': i,
# #                     'Type': 'Generation',
# #                     'Value': value
# #                 })
# #             elif var_type == 'transmisionInvCap':
# #                 n1, n2, i = key
# #                 sampled_data.append({
# #                     'Node': f"{n1}-{n2}",
# #                     'Energy_Type': 'Transmission',
# #                     'Period': i,
# #                     'Type': 'Transmission',
# #                     'Value': value
# #                 })
# #             elif var_type == 'storPWInvCap':
# #                 n, b, i = key
# #                 sampled_data.append({
# #                     'Node': n,
# #                     'Energy_Type': b,
# #                     'Period': i,
# #                     'Type': 'Storage Power',
# #                     'Value': value
# #                 })
# #             elif var_type == 'storENInvCap':
# #                 n, b, i = key
# #                 sampled_data.append({
# #                     'Node': n,
# #                     'Energy_Type': b,
# #                     'Period': i,
# #                     'Type': 'Storage Energy',
# #                     'Value': value
# #                 })
# #             else:
# #                 continue
# #     df = pd.DataFrame(sampled_data)
# #     return df

# def generate_uniform_fsd_samples(bounds_dict):
#     """
#     Generates truly uniform random samples within the given bounds for FSD variables,
#     ensuring better distribution between lower and upper bounds.
    
#     Parameters:
#     - bounds_dict: Dictionary containing bounds for each FSD variable.
    
#     Returns:
#     - sampled_df: Pandas DataFrame containing the sampled FSD data.
#     """
#     sampled_data = []
    
#     # Define sampling strategies
#     def sample_with_zero_bias(lb, ub, zero_prob=0.9):
#         """Helper function to generate samples with some bias towards zero"""
#         if np.random.random() < zero_prob and lb <= 0:
#             return 0
#         return np.random.uniform(lb, ub)
    
#     def sample_with_range_bias(lb, ub):
#         """Helper function to sample from different ranges with equal probability"""
#         # Split the range into three segments
#         range_size = ub - lb
#         if range_size == 0:
#             return lb
        
#         # Choose which segment to sample from
#         segment = np.random.choice(['low', 'mid', 'high'])
        
#         if segment == 'low':
#             return np.random.uniform(lb, lb + range_size/3)
#         elif segment == 'mid':
#             return np.random.uniform(lb + range_size/3, lb + 2*range_size/3)
#         else:
#             return np.random.uniform(lb + 2*range_size/3, ub)
    
#     for var_type, bounds in bounds_dict.items():
#         for key, (lb, ub) in bounds.items():
#             if lb == ub:
#                 value = lb
#             elif ub is None or ub == float('inf'):
#                 value = lb
#             else:
#                 # Determine sampling strategy based on variable type and bounds
#                 if var_type in ['genInvCap', 'storPWInvCap', 'storENInvCap']:
#                     # For generation and storage capacities, use zero-biased sampling
#                     value = sample_with_zero_bias(lb, ub)
#                 elif var_type == 'transmisionInvCap':
#                     # For transmission, use range-biased sampling
#                     value = sample_with_range_bias(lb, ub)
#                 else:
#                     # Default to simple uniform sampling
#                     value = np.random.uniform(lb, ub)
            
#             # Create the appropriate row based on variable type
#             if var_type == 'genInvCap':
#                 n, g, i = key
#                 sampled_data.append({
#                     'Node': n,
#                     'Energy_Type': g,
#                     'Period': i,
#                     'Type': 'Generation',
#                     'Value': value
#                 })
#             elif var_type == 'transmisionInvCap':
#                 n1, n2, i = key
#                 sampled_data.append({
#                     'Node': f"{n1}-{n2}",
#                     'Energy_Type': 'Transmission',
#                     'Period': i,
#                     'Type': 'Transmission',
#                     'Value': value
#                 })
#             elif var_type == 'storPWInvCap':
#                 n, b, i = key
#                 sampled_data.append({
#                     'Node': n,
#                     'Energy_Type': b,
#                     'Period': i,
#                     'Type': 'Storage Power',
#                     'Value': value
#                 })
#             elif var_type == 'storENInvCap':
#                 n, b, i = key
#                 sampled_data.append({
#                     'Node': n,
#                     'Energy_Type': b,
#                     'Period': i,
#                     'Type': 'Storage Energy',
#                     'Value': value
#                 })

#     return pd.DataFrame(sampled_data)

# def generate_feasible_samples(n, bounds_dict, data_folder, max_attempts=1000):
#     """
#     Generates feasible samples by uniformly sampling within bounds and checking model feasibility.

#     Parameters:
#     - n: Number of feasible samples to generate.
#     - bounds_dict: Dictionary containing bounds for each FSD variable.
#     - data_folder: Path to the data folder required by the model.
#     - max_attempts: Maximum number of attempts for sampling.

#     Returns:
#     - None (Saves feasible samples to CSV files).
#     """
#     feasible_samples = 0
#     total_attempts = 0

#     while feasible_samples < n and total_attempts < max_attempts:
#         total_attempts += 1
#         # Generate uniform samples within bounds
#         sampled_df = generate_uniform_fsd_samples(bounds_dict)
#         # Convert sampled_df to fsd_data
#         fsd_data = sampled_df.values.tolist()

#         # Check for negative values
#         if has_negative_values(fsd_data):
#             print(f"Attempt {total_attempts} has negative capacities. Discarding.")
#             continue

#         # Load investment data
#         gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(fsd_data)

#         # Create model and data
#         instance = create_model(data_folder,gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)
#         fsd_instance = inv_allo(instance, gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)

#         # Check feasibility
#         if check_model_feasibility(fsd_instance):
#             output_file = f'sampled_data_{feasible_samples + 100}.csv'
#             save_to_csv(sampled_df, output_file)
#             print(f"Sample {feasible_samples + 1} is feasible. Saved as {output_file}")
#             feasible_samples += 1
#         else:
#             print(f"Attempt {total_attempts} is infeasible. Discarding.")

#         if total_attempts % 5 == 0:
#             current_ratio = feasible_samples / total_attempts
#             print(f"Current accept-rejection ratio: {current_ratio:.6f}")

#     final_ratio = feasible_samples / total_attempts if total_attempts > 0 else 0
#     print(f"Generated {feasible_samples} feasible samples.")
#     print(f"Final accept-rejection ratio: {final_ratio:.6f}")
#     print(f"Total attempts: {total_attempts}")

# if __name__ == "__main__":
#     # Load bounds from a file or define them directly
#     # For this example, we will assume bounds_dict is provided directly

#     # Replace this with actual code to load bounds from your data
#     bounds_dict = {
#         'genInvCap': {
#             ('Germany', 'Liginiteexisting', 1): (0, 0), ('Germany', 'Liginiteexisting', 2): (0, 0), ('Germany', 'Liginiteexisting', 3): (0, 0), ('Germany', 'Liginiteexisting', 4): (0, 0), 
#             ('Germany', 'Liginiteexisting', 5): (0, 0), ('Germany', 'Liginiteexisting', 6): (0, 0), ('Germany', 'Liginiteexisting', 7): (0, 0), ('Germany', 'Liginiteexisting', 8): (0, 0), 
#             ('Germany', 'Lignite', 1): (0, 180894.0), ('Germany', 'Lignite', 2): (0, 188242.46153846153), ('Germany', 'Lignite', 3): (0, 192100.40384615384), ('Germany', 'Lignite', 4): (0, 194488.65384615384), 
#             ('Germany', 'Lignite', 5): (0, 196178.8), ('Germany', 'Lignite', 6): (0, 199044.7), ('Germany', 'Lignite', 7): (0, 200000.0), ('Germany', 'Lignite', 8): (0, 200000.0), 
#             ('Germany', 'LigniteCCSadv', 1): (0, 0), ('Germany', 'LigniteCCSadv', 2): (0, 0), ('Germany', 'LigniteCCSadv', 3): (0, 0), ('Germany', 'LigniteCCSadv', 4): (0, 0), 
#             ('Germany', 'LigniteCCSadv', 5): (0, 0), ('Germany', 'LigniteCCSadv', 6): (0, 0), ('Germany', 'LigniteCCSadv', 7): (0, 0), ('Germany', 'LigniteCCSadv', 8): (0, 0), 
#             ('Germany', 'Coalexisting', 1): (0, 0), ('Germany', 'Coalexisting', 2): (0, 0), ('Germany', 'Coalexisting', 3): (0, 0), ('Germany', 'Coalexisting', 4): (0, 0), 
#             ('Germany', 'Coalexisting', 5): (0, 0), ('Germany', 'Coalexisting', 6): (0, 0), ('Germany', 'Coalexisting', 7): (0, 0), ('Germany', 'Coalexisting', 8): (0, 0), 
#             ('Denmark', 'Coalexisting', 1): (0, 0), ('Denmark', 'Coalexisting', 2): (0, 0), ('Denmark', 'Coalexisting', 3): (0, 0), ('Denmark', 'Coalexisting', 4): (0, 0), 
#             ('Denmark', 'Coalexisting', 5): (0, 0), ('Denmark', 'Coalexisting', 6): (0, 0), ('Denmark', 'Coalexisting', 7): (0, 0), ('Denmark', 'Coalexisting', 8): (0, 0), 
#             ('France', 'Coalexisting', 1): (0, 0), ('France', 'Coalexisting', 2): (0, 0), ('France', 'Coalexisting', 3): (0, 0), ('France', 'Coalexisting', 4): (0, 0), 
#             ('France', 'Coalexisting', 5): (0, 0), ('France', 'Coalexisting', 6): (0, 0), ('France', 'Coalexisting', 7): (0, 0), ('France', 'Coalexisting', 8): (0, 0), 
#             ('Germany', 'Coal', 1): (0, 181170.0), ('Germany', 'Coal', 2): (0, 188412.3076923077), ('Germany', 'Coal', 3): (0, 192214.51923076922), ('Germany', 'Coal', 4): (0, 194568.26923076922), ('Germany', 'Coal', 5): (0, 196234.0), ('Germany', 'Coal', 6): (0, 199058.5), ('Germany', 'Coal', 7): (0, 200000.0), ('Germany', 'Coal', 8): 
#             (0, 200000.0), ('Denmark', 'Coal', 1): (0, 196835.0), ('Denmark', 'Coal', 2): (0, 198052.3076923077), ('Denmark', 'Coal', 3): (0, 198691.39423076922), ('Denmark', 'Coal', 4): (0, 199087.01923076922), ('Denmark', 'Coal', 5): (0, 199367.0), ('Denmark', 'Coal', 6): (0, 199841.75), ('Denmark', 'Coal', 7): (0, 200000.0), ('Denmark', 'Coal', 8): (0, 200000.0), ('France', 'Coal', 1): (0, 198183.0), ('France', 'Coal', 2): (0, 198881.84615384616), ('France', 'Coal', 3): (0, 199248.74038461538), ('France', 'Coal', 4): (0, 199475.86538461538), ('France', 'Coal', 5): (0, 199636.6), ('France', 'Coal', 6): (0, 199909.15), ('France', 'Coal', 7): (0, 200000.0), ('France', 'Coal', 8): (0, 200000.0), ('Germany', 'CoalCCSadv', 1): (0, 0), ('Germany', 'CoalCCSadv', 2): (0, 0), ('Germany', 'CoalCCSadv', 3): (0, 0), ('Germany', 'CoalCCSadv', 4): (0, 0), ('Germany', 'CoalCCSadv', 5): (0, 0), ('Germany', 'CoalCCSadv', 6): (0, 0), ('Germany', 'CoalCCSadv', 7): (0, 0), ('Germany', 'CoalCCSadv', 8): (0, 0), ('Denmark', 'CoalCCSadv', 1): (0, 0), ('Denmark', 'CoalCCSadv', 2): (0, 0), ('Denmark', 'CoalCCSadv', 3): (0, 0), ('Denmark', 'CoalCCSadv', 4): (0, 0), ('Denmark', 'CoalCCSadv', 5): (0, 0), ('Denmark', 'CoalCCSadv', 6): (0, 0), ('Denmark', 'CoalCCSadv', 7): (0, 0), ('Denmark', 'CoalCCSadv', 8): (0, 0), ('France', 'CoalCCSadv', 1): (0, 0), ('France', 'CoalCCSadv', 2): (0, 0), ('France', 'CoalCCSadv', 3): (0, 0), ('France', 'CoalCCSadv', 4): (0, 0), ('France', 'CoalCCSadv', 5): (0, 0), ('France', 'CoalCCSadv', 6): (0, 0), ('France', 'CoalCCSadv', 7): (0, 0), ('France', 'CoalCCSadv', 8): (0, 0), ('Germany', 'Gasexisting', 1): (0, 0), ('Germany', 'Gasexisting', 2): (0, 0), ('Germany', 'Gasexisting', 3): (0, 0), ('Germany', 'Gasexisting', 4): (0, 0), ('Germany', 'Gasexisting', 5): (0, 0), ('Germany', 'Gasexisting', 6): (0, 0), ('Germany', 'Gasexisting', 7): (0, 0), ('Germany', 'Gasexisting', 8): (0, 0), ('Denmark', 'Gasexisting', 1): (0, 0), ('Denmark', 'Gasexisting', 2): (0, 0), ('Denmark', 'Gasexisting', 3): (0, 0), ('Denmark', 'Gasexisting', 4): (0, 0), ('Denmark', 'Gasexisting', 5): (0, 0), ('Denmark', 'Gasexisting', 6): (0, 0), ('Denmark', 'Gasexisting', 7): (0, 0), ('Denmark', 'Gasexisting', 8): (0, 0), ('France', 'Gasexisting', 1): (0, 0), ('France', 'Gasexisting', 2): (0, 0), ('France', 'Gasexisting', 3): (0, 0), ('France', 'Gasexisting', 4): (0, 0), ('France', 'Gasexisting', 5): (0, 0), ('France', 'Gasexisting', 6): (0, 0), ('France', 'Gasexisting', 7): (0, 0), ('France', 'Gasexisting', 8): (0, 0), ('Germany', 'GasOCGT', 1): (0, 169447.0), ('Germany', 'GasOCGT', 2): (0, 179141.70192307694), ('Germany', 'GasOCGT', 3): (0, 187073.73076923078), ('Germany', 'GasOCGT', 4): (0, 197062.21153846153), ('Germany', 'GasOCGT', 5): (0, 197555.76), ('Germany', 'GasOCGT', 6): (0, 200000.0), ('Germany', 'GasOCGT', 7): (0, 200000.0), ('Germany', 'GasOCGT', 8): (0, 200000.0), ('Denmark', 'GasOCGT', 1): (0, 198379.0), ('Denmark', 'GasOCGT', 2): (0, 198893.35576923078), ('Denmark', 'GasOCGT', 3): (0, 199314.1923076923), ('Denmark', 'GasOCGT', 4): (0, 199844.13461538462), ('Denmark', 'GasOCGT', 5): (0, 199870.32), ('Denmark', 'GasOCGT', 6): (0, 200000.0), ('Denmark', 'GasOCGT', 7): (0, 200000.0), ('Denmark', 'GasOCGT', 8): (0, 200000.0), ('France', 'GasOCGT', 1): (0, 188621.0), ('France', 'GasOCGT', 2): (0, 192231.64423076922), ('France', 'GasOCGT', 3): (0, 195185.8076923077), ('France', 'GasOCGT', 4): (0, 198905.86538461538), ('France', 'GasOCGT', 5): (0, 199089.68), ('France', 'GasOCGT', 6): (0, 200000.0), ('France', 'GasOCGT', 7): (0, 200000.0), ('France', 'GasOCGT', 8): (0, 200000.0), ('Germany', 'GasCCGT', 1): (0, 169447.0), ('Germany', 'GasCCGT', 2): (0, 179141.70192307694), ('Germany', 'GasCCGT', 3): (0, 187073.73076923078), ('Germany', 'GasCCGT', 4): (0, 197062.21153846153), ('Germany', 'GasCCGT', 5): (0, 197555.76), ('Germany', 'GasCCGT', 6): (0, 200000.0), ('Germany', 'GasCCGT', 7): (0, 200000.0), ('Germany', 'GasCCGT', 8): (0, 200000.0), ('Denmark', 'GasCCGT', 1): (0, 198379.0), ('Denmark', 'GasCCGT', 2): (0, 198893.35576923078), ('Denmark', 'GasCCGT', 3): (0, 199314.1923076923), ('Denmark', 'GasCCGT', 4): (0, 199844.13461538462), ('Denmark', 'GasCCGT', 5): (0, 199870.32), ('Denmark', 'GasCCGT', 6): (0, 200000.0), ('Denmark', 'GasCCGT', 7): (0, 200000.0), ('Denmark', 'GasCCGT', 8): (0, 200000.0), ('France', 'GasCCGT', 1): (0, 188621.0), ('France', 'GasCCGT', 2): (0, 192231.64423076922), ('France', 'GasCCGT', 3): (0, 195185.8076923077), ('France', 'GasCCGT', 4): (0, 198905.86538461538), ('France', 'GasCCGT', 5): (0, 199089.68), ('France', 'GasCCGT', 6): (0, 200000.0), ('France', 'GasCCGT', 7): (0, 200000.0), ('France', 'GasCCGT', 8): (0, 200000.0), ('Germany', 'GasCCSadv', 1): (0, 0), ('Germany', 'GasCCSadv', 2): (0, 0), ('Germany', 'GasCCSadv', 3): (0, 0), ('Germany', 'GasCCSadv', 4): (0, 0), ('Germany', 'GasCCSadv', 5): (0, 0), ('Germany', 'GasCCSadv', 6): (0, 0), ('Germany', 'GasCCSadv', 7): (0, 0), ('Germany', 'GasCCSadv', 8): (0, 0), ('Denmark', 'GasCCSadv', 1): (0, 0), ('Denmark', 'GasCCSadv', 2): (0, 0), ('Denmark', 'GasCCSadv', 3): (0, 0), ('Denmark', 'GasCCSadv', 4): (0, 0), ('Denmark', 'GasCCSadv', 5): (0, 0), ('Denmark', 'GasCCSadv', 6): (0, 0), ('Denmark', 'GasCCSadv', 7): (0, 0), ('Denmark', 'GasCCSadv', 8): (0, 0), ('France', 'GasCCSadv', 1): (0, 0), ('France', 'GasCCSadv', 2): (0, 0), ('France', 'GasCCSadv', 3): (0, 0), ('France', 'GasCCSadv', 4): (0, 0), ('France', 'GasCCSadv', 5): (0, 0), ('France', 'GasCCSadv', 6): (0, 0), ('France', 'GasCCSadv', 7): (0, 0), ('France', 'GasCCSadv', 8): (0, 0), ('Germany', 'Oilexisting', 1): (0, 0), ('Germany', 'Oilexisting', 2): (0, 0), ('Germany', 'Oilexisting', 3): (0, 0), ('Germany', 'Oilexisting', 4): (0, 0), ('Germany', 'Oilexisting', 5): (0, 0), ('Germany', 'Oilexisting', 6): (0, 0), ('Germany', 'Oilexisting', 7): (0, 0), ('Germany', 'Oilexisting', 8): (0, 0), ('Denmark', 'Oilexisting', 1): (0, 0), ('Denmark', 'Oilexisting', 2): (0, 0), ('Denmark', 'Oilexisting', 3): (0, 0), ('Denmark', 'Oilexisting', 4): (0, 0), ('Denmark', 'Oilexisting', 5): (0, 0), ('Denmark', 'Oilexisting', 6): (0, 0), ('Denmark', 'Oilexisting', 7): (0, 0), ('Denmark', 'Oilexisting', 8): (0, 0), ('France', 'Oilexisting', 1): (0, 0), ('France', 'Oilexisting', 2): (0, 0), ('France', 'Oilexisting', 3): (0, 0), ('France', 'Oilexisting', 4): (0, 0), ('France', 'Oilexisting', 5): (0, 0), ('France', 'Oilexisting', 6): (0, 0), ('France', 'Oilexisting', 7): (0, 0), ('France', 'Oilexisting', 8): (0, 0), ('Germany', 'Bioexisting', 1): (0, 0), ('Germany', 'Bioexisting', 2): (0, 0), ('Germany', 'Bioexisting', 3): (0, 0), ('Germany', 'Bioexisting', 4): (0, 0), ('Germany', 'Bioexisting', 5): (0, 0), ('Germany', 'Bioexisting', 6): (0, 0), ('Germany', 'Bioexisting', 7): (0, 0), ('Germany', 'Bioexisting', 8): (0, 0), ('Denmark', 'Bioexisting', 1): (0, 0), ('Denmark', 'Bioexisting', 2): (0, 0), ('Denmark', 'Bioexisting', 3): (0, 0), ('Denmark', 'Bioexisting', 4): (0, 0), ('Denmark', 'Bioexisting', 5): (0, 0), ('Denmark', 'Bioexisting', 6): (0, 0), ('Denmark', 'Bioexisting', 7): (0, 0), ('Denmark', 'Bioexisting', 8): (0, 0), ('France', 'Bioexisting', 1): (0, 0), ('France', 'Bioexisting', 2): (0, 0), ('France', 'Bioexisting', 3): (0, 0), ('France', 'Bioexisting', 4): (0, 0), ('France', 'Bioexisting', 5): (0, 0), ('France', 'Bioexisting', 6): (0, 0), ('France', 'Bioexisting', 7): (0, 0), ('France', 'Bioexisting', 8): (0, 0), ('Germany', 'Bio10cofiring', 1): (0, 200000.0), ('Germany', 'Bio10cofiring', 2): (0, 200000.0), ('Germany', 'Bio10cofiring', 3): (0, 200000.0), ('Germany', 'Bio10cofiring', 4): (0, 200000.0), ('Germany', 'Bio10cofiring', 5): (0, 200000.0), ('Germany', 'Bio10cofiring', 6): (0, 200000.0), ('Germany', 'Bio10cofiring', 7): (0, 200000.0), ('Germany', 'Bio10cofiring', 8): (0, 200000.0), ('Denmark', 'Bio10cofiring', 1): (0, 200000.0), ('Denmark', 'Bio10cofiring', 2): (0, 200000.0), ('Denmark', 'Bio10cofiring', 3): (0, 200000.0), ('Denmark', 'Bio10cofiring', 4): (0, 200000.0), ('Denmark', 'Bio10cofiring', 5): (0, 200000.0), ('Denmark', 'Bio10cofiring', 6): (0, 200000.0), ('Denmark', 'Bio10cofiring', 7): (0, 200000.0), ('Denmark', 'Bio10cofiring', 8): (0, 200000.0), ('France', 'Bio10cofiring', 1): (0, 200000.0), ('France', 'Bio10cofiring', 2): (0, 200000.0), ('France', 'Bio10cofiring', 3): (0, 200000.0), ('France', 'Bio10cofiring', 4): (0, 200000.0), ('France', 'Bio10cofiring', 5): (0, 200000.0), ('France', 'Bio10cofiring', 6): (0, 200000.0), ('France', 'Bio10cofiring', 7): (0, 200000.0), ('France', 'Bio10cofiring', 8): (0, 200000.0), ('Germany', 'Nuclear', 1): (0, 0), ('Germany', 'Nuclear', 2): (0, 0), ('Germany', 'Nuclear', 3): (0, 0), ('Germany', 'Nuclear', 4): (0, 0), ('Germany', 'Nuclear', 5): (0, 0), ('Germany', 'Nuclear', 6): (0, 0), ('Germany', 'Nuclear', 7): (0, 0), ('Germany', 'Nuclear', 8): (0, 0), ('France', 'Nuclear', 1): (0, 138630.0), ('France', 'Nuclear', 2): (0, 156164.2857142857), ('France', 'Nuclear', 3): (0, 164931.42857142858), ('France', 'Nuclear', 4): (0, 173698.57142857142), ('France', 'Nuclear', 5): (0, 182465.7142857143), ('France', 'Nuclear', 6): (0, 191232.85714285713), ('France', 'Nuclear', 7): (0, 200000.0), ('France', 'Nuclear', 8): (0, 200000.0), ('France', 'Wave', 1): (0, 207.0), ('France', 'Wave', 2): (0, 207.0), ('France', 'Wave', 3): (0, 207.0), ('France', 'Wave', 4): (0, 207.0), ('France', 'Wave', 5): (0, 207.0), ('France', 'Wave', 6): (0, 207.0), ('France', 'Wave', 7): (0, 207.0), ('France', 'Wave', 8): (0, 207.0), ('Germany', 'Geo', 1): (0, 242.0), ('Germany', 'Geo', 2): (0, 242.0), ('Germany', 'Geo', 3): (0, 242.0), ('Germany', 'Geo', 4): (0, 242.0), ('Germany', 'Geo', 5): (0, 242.0), ('Germany', 'Geo', 6): (0, 242.0), ('Germany', 'Geo', 7): (0, 242.0), ('Germany', 'Geo', 8): (0, 242.0), ('France', 'Geo', 1): (0, 80.0), ('France', 'Geo', 2): (0, 80.0), ('France', 'Geo', 3): (0, 80.0), ('France', 'Geo', 4): (0, 80.0), ('France', 'Geo', 5): (0, 80.0), ('France', 'Geo', 6): (0, 80.0), ('France', 'Geo', 7): (0, 80.0), ('France', 'Geo', 8): (0, 80.0), ('Germany', 'Hydroregulated', 1): (0, 814.1999999999998), ('Germany', 'Hydroregulated', 2): (0, 814.1999999999998), ('Germany', 'Hydroregulated', 3): (0, 814.1999999999998), ('Germany', 'Hydroregulated', 4): (0, 814.1999999999998), ('Germany', 'Hydroregulated', 5): (0, 814.1999999999998), ('Germany', 'Hydroregulated', 6): (0, 814.1999999999998), ('Germany', 'Hydroregulated', 7): (0, 814.1999999999998), ('Germany', 'Hydroregulated', 8): (0, 814.1999999999998), ('France', 'Hydroregulated', 1): (0, 11926.65593789937), ('France', 'Hydroregulated', 2): (0, 11926.65593789937), ('France', 'Hydroregulated', 3): (0, 11926.65593789937), ('France', 'Hydroregulated', 4): (0, 11926.65593789937), ('France', 'Hydroregulated', 5): (0, 11926.65593789937), ('France', 'Hydroregulated', 6): (0, 11926.65593789937), ('France', 'Hydroregulated', 7): (0, 11926.65593789937), ('France', 'Hydroregulated', 8): (0, 11926.65593789937), ('Germany', 'Hydrorun-of-the-river', 1): (0, 433.3590114827766), ('Germany', 'Hydrorun-of-the-river', 2): (0, 433.3590114827766), ('Germany', 'Hydrorun-of-the-river', 3): (0, 433.3590114827766), ('Germany', 'Hydrorun-of-the-river', 4): (0, 433.3590114827766), ('Germany', 'Hydrorun-of-the-river', 5): (0, 433.3590114827766), ('Germany', 'Hydrorun-of-the-river', 6): (0, 433.3590114827766), ('Germany', 'Hydrorun-of-the-river', 7): (0, 433.3590114827766), ('Germany', 'Hydrorun-of-the-river', 8): (0, 433.3590114827766), ('Denmark', 'Hydrorun-of-the-river', 1): (0, 2.0), ('Denmark', 'Hydrorun-of-the-river', 2): (0, 2.0), ('Denmark', 'Hydrorun-of-the-river', 3): (0, 2.0), ('Denmark', 'Hydrorun-of-the-river', 4): (0, 2.0), ('Denmark', 'Hydrorun-of-the-river', 5): (0, 2.0), ('Denmark', 'Hydrorun-of-the-river', 6): (0, 2.0), ('Denmark', 'Hydrorun-of-the-river', 7): (0, 2.0), ('Denmark', 'Hydrorun-of-the-river', 8): (0, 2.0), ('France', 'Hydrorun-of-the-river', 1): (0, 4688.688124201259), ('France', 'Hydrorun-of-the-river', 2): (0, 4688.688124201259), ('France', 'Hydrorun-of-the-river', 3): (0, 4688.688124201259), ('France', 'Hydrorun-of-the-river', 4): (0, 4688.688124201259), ('France', 'Hydrorun-of-the-river', 5): (0, 4688.688124201259), ('France', 'Hydrorun-of-the-river', 6): (0, 4688.688124201259), ('France', 'Hydrorun-of-the-river', 7): (0, 4688.688124201259), ('France', 'Hydrorun-of-the-river', 8): (0, 4688.688124201259), ('Germany', 'Bio', 1): (0, 189778.0), ('Germany', 'Bio', 2): (0, 192698.57142857142), ('Germany', 'Bio', 3): (0, 194158.85714285713), ('Germany', 'Bio', 4): (0, 195619.14285714287), ('Germany', 'Bio', 5): (0, 197079.42857142858), ('Germany', 'Bio', 6): (0, 198539.7142857143), ('Germany', 'Bio', 7): (0, 200000.0), ('Germany', 'Bio', 8): (0, 200000.0), ('France', 'Bio', 1): (0, 197093.0), ('France', 'Bio', 2): (0, 197923.57142857142), ('France', 'Bio', 3): (0, 198338.85714285713), ('France', 'Bio', 4): (0, 198754.14285714287), ('France', 'Bio', 5): (0, 199169.42857142858), ('France', 'Bio', 6): (0, 199584.7142857143), ('France', 'Bio', 7): (0, 200000.0), ('France', 'Bio', 8): (0, 200000.0), ('Germany', 'Windonshore', 1): (0, 51370.0), ('Germany', 'Windonshore', 2): (0, 51370.0), ('Germany', 'Windonshore', 3): (0, 51370.0), ('Germany', 'Windonshore', 4): (0, 51370.0), ('Germany', 'Windonshore', 5): (0, 51370.0), ('Germany', 'Windonshore', 6): (0, 79185.0), ('Germany', 'Windonshore', 7): (0, 107000.0), ('Germany', 'Windonshore', 8): (0, 107000.0), ('Denmark', 'Windonshore', 1): (0, 50356.0), ('Denmark', 'Windonshore', 2): (0, 50356.0), ('Denmark', 'Windonshore', 3): (0, 50356.0), ('Denmark', 'Windonshore', 4): (0, 50356.0), ('Denmark', 'Windonshore', 5): (0, 50356.0), ('Denmark', 'Windonshore', 6): (0, 52678.0), ('Denmark', 'Windonshore', 7): (0, 55000.0), ('Denmark', 'Windonshore', 8): (0, 55000.0), ('France', 'Windonshore', 1): (0, 500000.0), ('France', 'Windonshore', 2): (0, 500000.0), ('France', 'Windonshore', 3): (0, 500000.0), ('France', 'Windonshore', 4): (0, 500000.0), ('France', 'Windonshore', 5): (0, 500000.0), ('France', 'Windonshore', 6): (0, 500000.0), ('France', 'Windonshore', 7): (0, 500000.0), ('France', 'Windonshore', 8): (0, 500000.0), ('Germany', 'Windoffshore', 1): (0, 0), ('Germany', 'Windoffshore', 2): (0, 0), ('Germany', 'Windoffshore', 3): (0, 0), ('Germany', 'Windoffshore', 4): (0, 0), ('Germany', 'Windoffshore', 5): (0, 0), ('Germany', 'Windoffshore', 6): (0, 2260.2599999999998), ('Germany', 'Windoffshore', 7): (0, 2260.2599999999998), ('Germany', 'Windoffshore', 8): (0, 2260.2599999999998), ('Denmark', 'Windoffshore', 1): (0, 9277.579999999996), ('Denmark', 'Windoffshore', 2): (0, 9277.579999999996), ('Denmark', 'Windoffshore', 3): (0, 9277.579999999996), ('Denmark', 'Windoffshore', 4): (0, 9277.579999999996), ('Denmark', 'Windoffshore', 5): (0, 9277.579999999996), ('Denmark', 'Windoffshore', 6): (0, 11582.579999999996), ('Denmark', 'Windoffshore', 7): (0, 11582.579999999996), ('Denmark', 'Windoffshore', 8): (0, 11582.579999999996), ('France', 'Windoffshore', 1): (0, 57900.0), ('France', 'Windoffshore', 2): (0, 57900.0), ('France', 'Windoffshore', 3): (0, 57900.0), ('France', 'Windoffshore', 4): (0, 57900.0), ('France', 'Windoffshore', 5): (0, 57900.0), ('France', 'Windoffshore', 6): (0, 57900.0), ('France', 'Windoffshore', 7): (0, 57900.0), ('France', 'Windoffshore', 8): (0, 57900.0), ('Germany', 'Solar', 1): (0, 500000.0), ('Germany', 'Solar', 2): (0, 500000.0), ('Germany', 'Solar', 3): (0, 500000.0), ('Germany', 'Solar', 4): (0, 500000.0), ('Germany', 'Solar', 5): (0, 500000.0), ('Germany', 'Solar', 6): (0, 500000.0), ('Germany', 'Solar', 7): (0, 500000.0), ('Germany', 'Solar', 8): (0, 500000.0), ('Denmark', 'Solar', 1): (0, 114435.606868), ('Denmark', 'Solar', 2): (0, 114435.606868), ('Denmark', 'Solar', 3): (0, 114435.606868), ('Denmark', 'Solar', 4): (0, 114435.606868), ('Denmark', 'Solar', 5): (0, 114435.606868), ('Denmark', 'Solar', 6): (0, 115203.606868), ('Denmark', 'Solar', 7): (0, 115971.606868), ('Denmark', 'Solar', 8): (0, 115971.606868), ('France', 'Solar', 1): (0, 500000.0), ('France', 'Solar', 2): (0, 500000.0), ('France', 'Solar', 3): (0, 500000.0), ('France', 'Solar', 4): (0, 500000.0), ('France', 'Solar', 5): (0, 500000.0), ('France', 'Solar', 6): (0, 500000.0), ('France', 'Solar', 7): (0, 500000.0), ('France', 'Solar', 8): (0, 500000.0), ('Germany', 'Bio10cofiringCCS', 1): (0, 0), ('Germany', 'Bio10cofiringCCS', 2): (0, 0), ('Germany', 'Bio10cofiringCCS', 3): (0, 0), ('Germany', 'Bio10cofiringCCS', 4): (0, 0), ('Germany', 'Bio10cofiringCCS', 5): (0, 0), ('Germany', 'Bio10cofiringCCS', 6): (0, 0), ('Germany', 'Bio10cofiringCCS', 7): (0, 0), ('Germany', 'Bio10cofiringCCS', 8): (0, 0), ('Denmark', 'Bio10cofiringCCS', 1): (0, 0), ('Denmark', 'Bio10cofiringCCS', 2): (0, 0), ('Denmark', 'Bio10cofiringCCS', 3): (0, 0), ('Denmark', 'Bio10cofiringCCS', 4): (0, 0), ('Denmark', 'Bio10cofiringCCS', 5): (0, 0), ('Denmark', 'Bio10cofiringCCS', 6): (0, 0), ('Denmark', 'Bio10cofiringCCS', 7): (0, 0), ('Denmark', 'Bio10cofiringCCS', 8): (0, 0), ('France', 'Bio10cofiringCCS', 1): (0, 0), ('France', 'Bio10cofiringCCS', 2): (0, 0), ('France', 'Bio10cofiringCCS', 3): (0, 0), ('France', 'Bio10cofiringCCS', 4): (0, 0), ('France', 'Bio10cofiringCCS', 5): (0, 0), ('France', 'Bio10cofiringCCS', 6): (0, 0), ('France', 'Bio10cofiringCCS', 7): (0, 0), ('France', 'Bio10cofiringCCS', 8): (0, 0), ('Germany', 'LigniteCCSsup', 1): (0, 0), ('Germany', 'LigniteCCSsup', 2): (0, 0), ('Germany', 'LigniteCCSsup', 3): (0, 0), ('Germany', 'LigniteCCSsup', 4): (0, 0), ('Germany', 'LigniteCCSsup', 5): (0, 0), ('Germany', 'LigniteCCSsup', 6): (0, 0), ('Germany', 'LigniteCCSsup', 7): (0, 0), ('Germany', 'LigniteCCSsup', 8): (0, 0), ('Germany', 'CoalCCS', 1): (0, 0), ('Germany', 'CoalCCS', 2): (0, 0), ('Germany', 'CoalCCS', 3): (0, 0), ('Germany', 'CoalCCS', 4): (0, 0), ('Germany', 'CoalCCS', 5): (0, 0), ('Germany', 'CoalCCS', 6): (0, 0), ('Germany', 'CoalCCS', 7): (0, 0), ('Germany', 'CoalCCS', 8): (0, 0), ('France', 'CoalCCS', 1): (0, 0), ('France', 'CoalCCS', 2): (0, 0), ('France', 'CoalCCS', 3): (0, 0), ('France', 'CoalCCS', 4): (0, 0), ('France', 'CoalCCS', 5): (0, 0), ('France', 'CoalCCS', 6): (0, 0), ('France', 'CoalCCS', 7): (0, 0), ('France', 'CoalCCS', 8): (0, 0), ('Germany', 'GasCCS', 1): (0, 0), ('Germany', 'GasCCS', 2): (0, 0), ('Germany', 'GasCCS', 3): (0, 0), ('Germany', 'GasCCS', 4): (0, 0), ('Germany', 'GasCCS', 5): (0, 0), ('Germany', 'GasCCS', 6): (0, 0), ('Germany', 'GasCCS', 7): (0, 0), ('Germany', 'GasCCS', 8): (0, 0), ('France', 'GasCCS', 1): (0, 0), ('France', 'GasCCS', 2): (0, 0), ('France', 'GasCCS', 3): (0, 0), ('France', 'GasCCS', 4): (0, 0), ('France', 'GasCCS', 5): (0, 0), ('France', 'GasCCS', 6): (0, 0), ('France', 'GasCCS', 7): (0, 0), ('France', 'GasCCS', 8): (0, 0), ('Germany', 'Waste', 1): (0, 274.20000000000005), ('Germany', 'Waste', 2): (0, 274.20000000000005), ('Germany', 'Waste', 3): (0, 274.20000000000005), ('Germany', 'Waste', 4): (0, 274.20000000000005), ('Germany', 'Waste', 5): (0, 274.20000000000005), ('Germany', 'Waste', 6): (0, 274.20000000000005), ('Germany', 'Waste', 7): (0, 274.20000000000005), ('Germany', 'Waste', 8): (0, 274.20000000000005), ('Denmark', 'Waste', 1): (0, 0), ('Denmark', 'Waste', 2): (0, 0), ('Denmark', 'Waste', 3): (0, 0), ('Denmark', 'Waste', 4): (0, 0), ('Denmark', 'Waste', 5): (0, 0), ('Denmark', 'Waste', 6): (0, 0), ('Denmark', 'Waste', 7): (0, 0), ('Denmark', 'Waste', 8): (0, 0), ('France', 'Waste', 1): (0, 44.15000000000009), ('France', 'Waste', 2): (0, 44.15000000000009), ('France', 'Waste', 3): (0, 44.15000000000009), ('France', 'Waste', 4): (0, 44.15000000000009), ('France', 'Waste', 5): (0, 44.15000000000009), ('France', 'Waste', 6): (0, 44.15000000000009), ('France', 'Waste', 7): (0, 44.15000000000009), ('France', 'Waste', 8): (0, 44.15000000000009)
#         },
#         'transmisionInvCap': {
#             ('Denmark', 'Germany', 1): (0, 0), ('Denmark', 'Germany', 2): (0, 2000), ('Denmark', 'Germany', 3): (0, 2000), ('Denmark', 'Germany', 4): (0, 2000), ('Denmark', 'Germany', 5): (0, 2000), ('Denmark', 'Germany', 6): (0, 2000), ('Denmark', 'Germany', 7): (0, 2000), ('Denmark', 'Germany', 8): (0, 2000), ('France', 'Germany', 1): (0, 0), ('France', 'Germany', 2): (0, 0), ('France', 'Germany', 3): (0, 1800), ('France', 'Germany', 4): (0, 1800), ('France', 'Germany', 5): (0, 1800), ('France', 'Germany', 6): (0, 1800), ('France', 'Germany', 7): (0, 1800), ('France', 'Germany', 8): (0, 1800)
#             # Add other transmission bounds here...
#         },
#         'storPWInvCap': {
#             ('Germany', 'HydroPumpStorage', 1): (0, 4640.0), ('Germany', 'HydroPumpStorage', 2): (0, 4640.0), ('Germany', 'HydroPumpStorage', 3): (0, 4640.0), ('Germany', 'HydroPumpStorage', 4): (0, 4640.0), ('Germany', 'HydroPumpStorage', 5): (0, 4640.0), ('Germany', 'HydroPumpStorage', 6): (0, 4640.0), ('Germany', 'HydroPumpStorage', 7): (0, 4640.0), ('Germany', 'HydroPumpStorage', 8): (0, 4640.0), ('Germany', 'Li-Ion_BESS', 1): (0, 500000), ('Germany', 'Li-Ion_BESS', 2): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 3): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 4): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 5): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 6): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 7): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 8): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 1): (0, 500000), ('Denmark', 'Li-Ion_BESS', 2): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 3): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 4): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 5): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 6): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 7): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 8): (0, 500000.0), ('France', 'HydroPumpStorage', 1): (0, 2524.5), ('France', 'HydroPumpStorage', 2): (0, 2524.5), ('France', 'HydroPumpStorage', 3): (0, 2524.5), ('France', 'HydroPumpStorage', 4): (0, 2524.5), ('France', 'HydroPumpStorage', 5): (0, 2524.5), ('France', 'HydroPumpStorage', 6): (0, 2524.5), ('France', 'HydroPumpStorage', 7): (0, 2524.5), ('France', 'HydroPumpStorage', 8): (0, 2524.5), ('France', 'Li-Ion_BESS', 1): (0, 500000), ('France', 'Li-Ion_BESS', 2): (0, 500000.0), ('France', 'Li-Ion_BESS', 3): (0, 500000.0), ('France', 'Li-Ion_BESS', 4): (0, 500000.0), ('France', 'Li-Ion_BESS', 5): (0, 500000.0), ('France', 'Li-Ion_BESS', 6): (0, 500000.0), ('France', 'Li-Ion_BESS', 7): (0, 500000.0), ('France', 'Li-Ion_BESS', 8): (0, 500000.0)
#             # Add other storage power bounds here...
#         },
#         'storENInvCap': {
#             ('Germany', 'HydroPumpStorage', 1): (0, 3862.600000000006), ('Germany', 'HydroPumpStorage', 2): (0, 3862.600000000006), ('Germany', 'HydroPumpStorage', 3): (0, 3862.600000000006), ('Germany', 'HydroPumpStorage', 4): (0, 3862.600000000006), ('Germany', 'HydroPumpStorage', 5): (0, 3862.600000000006), ('Germany', 'HydroPumpStorage', 6): (0, 3862.600000000006), ('Germany', 'HydroPumpStorage', 7): (0, 3862.600000000006), ('Germany', 'HydroPumpStorage', 8): (0, 3862.600000000006), ('Germany', 'Li-Ion_BESS', 1): (0, 500000), ('Germany', 'Li-Ion_BESS', 2): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 3): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 4): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 5): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 6): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 7): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 8): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 1): (0, 500000), ('Denmark', 'Li-Ion_BESS', 2): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 3): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 4): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 5): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 6): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 7): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 8): (0, 500000.0), ('France', 'HydroPumpStorage', 1): (0, 18429.600000000006), ('France', 'HydroPumpStorage', 2): (0, 18429.600000000006), ('France', 'HydroPumpStorage', 3): (0, 18429.600000000006), ('France', 'HydroPumpStorage', 4): (0, 18429.600000000006), ('France', 'HydroPumpStorage', 5): (0, 18429.600000000006), ('France', 'HydroPumpStorage', 6): (0, 18429.600000000006), ('France', 'HydroPumpStorage', 7): (0, 18429.600000000006), ('France', 'HydroPumpStorage', 8): (0, 18429.600000000006), ('France', 'Li-Ion_BESS', 1): (0, 500000), ('France', 'Li-Ion_BESS', 2): (0, 500000.0), ('France', 'Li-Ion_BESS', 3): (0, 500000.0), ('France', 'Li-Ion_BESS', 4): (0, 500000.0), ('France', 'Li-Ion_BESS', 5): (0, 500000.0), ('France', 'Li-Ion_BESS', 6): (0, 500000.0), ('France', 'Li-Ion_BESS', 7): (0, 500000.0), ('France', 'Li-Ion_BESS', 8): (0, 500000.0)
#             # Add other storage energy bounds here...
#         },
#         # Include all other necessary bounds...
#     }

#     # Ensure that bounds_dict is fully populated with your actual data
#     # You may load bounds_dict from a CSV file or other data source

#     data_folder = 'Data handler/sampling/reduced'
#     n_samples = 10  # Number of feasible samples you want to generate

#     generate_feasible_samples(n_samples, bounds_dict, data_folder)


import pandas as pd
import numpy as np
import glob
from FSD_sampling_violation import create_model, load_investment_data, inv_allo
from pyomo.environ import *
import io
import contextlib
from pyomo.core.expr.visitor import identify_variables
import warnings
warnings.filterwarnings('ignore')
import os
import argparse

def has_negative_values(fsd_data):

    for row in fsd_data:
        cap_value = float(row[4])  # Assuming 'Value' is the 5th column
        if cap_value < 0:
            return True
    return False

def save_to_csv(df, output_file):

    df.to_csv(output_file, index=False)

def check_model_feasibility(instance):

    solver = SolverFactory('glpk')  # You can change this to your preferred solver
    try:
        results = solver.solve(instance, tee=False)
    except Exception as e:
        print(f"Solver exception: {e}")
        return False

    if results.solver.termination_condition == TerminationCondition.optimal:
        return True
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # print("Model is infeasible.")
        return False
    else:
        print(f"Solver Termination Condition: {results.solver.termination_condition}")
        print("Couldn't evaluate feasibility")
        return None


def run_zero_prob_experiment(bounds_dict, data_folder, zero_prob, samples_per_prob=150, max_attempts=100000):
    # Create experiment directory if it doesn't exist
    base_dir = 'DataSamples_zero_prob2'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    print(f"\nTesting zero_probability: {zero_prob:.1f}")
    
    feasible_samples = 0
    total_attempts = 0
    
    while feasible_samples < samples_per_prob and total_attempts < max_attempts:
        total_attempts += 1
        
        # Modified generate_uniform_fsd_samples call with current zero_prob
        sampled_df = generate_uniform_fsd_samples(bounds_dict, zero_prob)
        fsd_data = sampled_df.values.tolist()
        
        # Check for negative values
        if has_negative_values(fsd_data):
            print(f"Attempt {total_attempts} has negative capacities. Discarding.")
            continue
        
        try:
            gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(fsd_data)
            instance = create_model(data_folder, gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)
            fsd_instance = inv_allo(instance, gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)
            
            if check_model_feasibility(fsd_instance):
                output_file = os.path.join(base_dir, f'sample_{int(feasible_samples + 1 + zero_prob*10000)}.csv')
                save_to_csv(sampled_df, output_file)
                print(f"Sample {feasible_samples + 1} is feasible. Saved as {output_file}")
                feasible_samples += 1
            # else:
                # print(f"Attempt {total_attempts} is infeasible. Discarding.")
        
        except Exception as e:
            print(f"Error in attempt {total_attempts}: {e}")
            continue
        
        if total_attempts % 50 == 0:
            current_ratio = feasible_samples / total_attempts
            print(f"Current accept-rejection ratio: {current_ratio:.6f}")
    
    final_ratio = feasible_samples / total_attempts if total_attempts > 0 else 0

    
    print(f"\nResults for zero_probability {zero_prob:.1f}:")
    print(f"Generated {feasible_samples} feasible samples.")
    print(f"Final accept-rejection ratio: {final_ratio:.6f}")
    print(f"Total attempts: {total_attempts}")
    
    print("\nExperiment complete.")

def generate_uniform_fsd_samples(bounds_dict, zero_prob):
    sampled_data = []
    
    def sample_with_zero_bias(lb, ub, zero_prob):
        if np.random.random() < zero_prob and lb <= 0:
            return 0
        return np.random.uniform(lb, ub)
    
    for var_type, bounds in bounds_dict.items():
        for key, (lb, ub) in bounds.items():
            if lb == ub:
                value = lb
            elif ub is None or ub == float('inf'):
                value = lb
            else:
                if var_type in ['genInvCap', 'storPWInvCap', 'storENInvCap','transmisionInvCap']:
                    value = sample_with_zero_bias(lb, ub, zero_prob)
                else:
                    value = np.random.uniform(lb, ub)
            
            if var_type == 'genInvCap':
                n, g, i = key
                sampled_data.append({
                    'Node': n,
                    'Energy_Type': g,
                    'Period': i,
                    'Type': 'Generation',
                    'Value': value
                })
            elif var_type == 'transmisionInvCap':
                n1, n2, i = key
                sampled_data.append({
                    'Node': n1,
                    'Energy_Type': n2,
                    'Period': i,
                    'Type': 'Transmission',
                    'Value': value
                })
            elif var_type == 'storPWInvCap':
                n, b, i = key
                sampled_data.append({
                    'Node': n,
                    'Energy_Type': b,
                    'Period': i,
                    'Type': 'Storage Power',
                    'Value': value
                })
            elif var_type == 'storENInvCap':
                n, b, i = key
                sampled_data.append({
                    'Node': n,
                    'Energy_Type': b,
                    'Period': i,
                    'Type': 'Storage Energy',
                    'Value': value
                })

    return pd.DataFrame(sampled_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prob', type=int, required=True, help='zero_prob')
    args = parser.parse_args()

    zero_prob = (args.prob)*0.1

    data_folder = 'Data handler/sampling/reduced'

    bounds_dict = {
        'genInvCap': {
            ('Germany', 'Liginiteexisting', 1): (0, 0), ('Germany', 'Liginiteexisting', 2): (0, 0), ('Germany', 'Liginiteexisting', 3): (0, 0), ('Germany', 'Liginiteexisting', 4): (0, 0), ('Germany', 'Liginiteexisting', 5): (0, 0), ('Germany', 'Liginiteexisting', 6): (0, 0), ('Germany', 'Liginiteexisting', 7): (0, 0), ('Germany', 'Liginiteexisting', 8): (0, 0), 
            ('Germany', 'Lignite', 1): (0, 500000.0), ('Germany', 'Lignite', 2): (0, 500000.0), ('Germany', 'Lignite', 3): (0, 500000.0), ('Germany', 'Lignite', 4): (0, 500000.0), ('Germany', 'Lignite', 5): (0, 500000.0), ('Germany', 'Lignite', 6): (0, 500000.0), ('Germany', 'Lignite', 7): (0, 500000.0), ('Germany', 'Lignite', 8): (0, 500000.0), 
            ('Germany', 'LigniteCCSadv', 1): (0, 0), ('Germany', 'LigniteCCSadv', 2): (0, 0), ('Germany', 'LigniteCCSadv', 3): (0, 0), ('Germany', 'LigniteCCSadv', 4): (0, 0), ('Germany', 'LigniteCCSadv', 5): (0, 0), ('Germany', 'LigniteCCSadv', 6): (0, 0), ('Germany', 'LigniteCCSadv', 7): (0, 0), ('Germany', 'LigniteCCSadv', 8): (0, 0), 
            ('Germany', 'Coalexisting', 1): (0, 0), ('Germany', 'Coalexisting', 2): (0, 0), ('Germany', 'Coalexisting', 3): (0, 0), ('Germany', 'Coalexisting', 4): (0, 0), ('Germany', 'Coalexisting', 5): (0, 0), ('Germany', 'Coalexisting', 6): (0, 0), ('Germany', 'Coalexisting', 7): (0, 0), ('Germany', 'Coalexisting', 8): (0, 0), 
            ('Denmark', 'Coalexisting', 1): (0, 0), ('Denmark', 'Coalexisting', 2): (0, 0), ('Denmark', 'Coalexisting', 3): (0, 0), ('Denmark', 'Coalexisting', 4): (0, 0), ('Denmark', 'Coalexisting', 5): (0, 0), ('Denmark', 'Coalexisting', 6): (0, 0), ('Denmark', 'Coalexisting', 7): (0, 0), ('Denmark', 'Coalexisting', 8): (0, 0), 
            ('France', 'Coalexisting', 1): (0, 0), ('France', 'Coalexisting', 2): (0, 0), ('France', 'Coalexisting', 3): (0, 0), ('France', 'Coalexisting', 4): (0, 0), ('France', 'Coalexisting', 5): (0, 0), ('France', 'Coalexisting', 6): (0, 0), ('France', 'Coalexisting', 7): (0, 0), ('France', 'Coalexisting', 8): (0, 0), 
            ('Germany', 'Coal', 1): (0, 500000.0), ('Germany', 'Coal', 2): (0, 500000.0), ('Germany', 'Coal', 3): (0, 500000.0), ('Germany', 'Coal', 4): (0, 500000.0), ('Germany', 'Coal', 5): (0, 500000.0), ('Germany', 'Coal', 6): (0, 500000.0), ('Germany', 'Coal', 7): (0, 500000.0), ('Germany', 'Coal', 8): (0, 500000.0), 
            ('Denmark', 'Coal', 1): (0, 500000.0), ('Denmark', 'Coal', 2): (0, 500000.0), ('Denmark', 'Coal', 3): (0, 500000.0), ('Denmark', 'Coal', 4): (0, 500000.0), ('Denmark', 'Coal', 5): (0, 500000.0), ('Denmark', 'Coal', 6): (0, 500000.0), ('Denmark', 'Coal', 7): (0, 500000.0), ('Denmark', 'Coal', 8): (0, 500000.0), 
            ('France', 'Coal', 1): (0, 500000.0), ('France', 'Coal', 2): (0, 500000.0), ('France', 'Coal', 3): (0, 500000.0), ('France', 'Coal', 4): (0, 500000.0), ('France', 'Coal', 5): (0, 500000.0), ('France', 'Coal', 6): (0, 500000.0), ('France', 'Coal', 7): (0, 500000.0), ('France', 'Coal', 8): (0, 500000.0), 
            ('Germany', 'CoalCCSadv', 1): (0, 0), ('Germany', 'CoalCCSadv', 2): (0, 0), ('Germany', 'CoalCCSadv', 3): (0, 0), ('Germany', 'CoalCCSadv', 4): (0, 0), ('Germany', 'CoalCCSadv', 5): (0, 0), ('Germany', 'CoalCCSadv', 6): (0, 0), ('Germany', 'CoalCCSadv', 7): (0, 0), ('Germany', 'CoalCCSadv', 8): (0, 0), 
            ('Denmark', 'CoalCCSadv', 1): (0, 0), ('Denmark', 'CoalCCSadv', 2): (0, 0), ('Denmark', 'CoalCCSadv', 3): (0, 0), ('Denmark', 'CoalCCSadv', 4): (0, 0), ('Denmark', 'CoalCCSadv', 5): (0, 0), ('Denmark', 'CoalCCSadv', 6): (0, 0), ('Denmark', 'CoalCCSadv', 7): (0, 0), ('Denmark', 'CoalCCSadv', 8): (0, 0), 
            ('France', 'CoalCCSadv', 1): (0, 0), ('France', 'CoalCCSadv', 2): (0, 0), ('France', 'CoalCCSadv', 3): (0, 0), ('France', 'CoalCCSadv', 4): (0, 0), ('France', 'CoalCCSadv', 5): (0, 0), ('France', 'CoalCCSadv', 6): (0, 0), ('France', 'CoalCCSadv', 7): (0, 0), ('France', 'CoalCCSadv', 8): (0, 0), 
            ('Germany', 'Gasexisting', 1): (0, 0), ('Germany', 'Gasexisting', 2): (0, 0), ('Germany', 'Gasexisting', 3): (0, 0), ('Germany', 'Gasexisting', 4): (0, 0), ('Germany', 'Gasexisting', 5): (0, 0), ('Germany', 'Gasexisting', 6): (0, 0), ('Germany', 'Gasexisting', 7): (0, 0), ('Germany', 'Gasexisting', 8): (0, 0), 
            ('Denmark', 'Gasexisting', 1): (0, 0), ('Denmark', 'Gasexisting', 2): (0, 0), ('Denmark', 'Gasexisting', 3): (0, 0), ('Denmark', 'Gasexisting', 4): (0, 0), ('Denmark', 'Gasexisting', 5): (0, 0), ('Denmark', 'Gasexisting', 6): (0, 0), ('Denmark', 'Gasexisting', 7): (0, 0), ('Denmark', 'Gasexisting', 8): (0, 0), 
            ('France', 'Gasexisting', 1): (0, 0), ('France', 'Gasexisting', 2): (0, 0), ('France', 'Gasexisting', 3): (0, 0), ('France', 'Gasexisting', 4): (0, 0), ('France', 'Gasexisting', 5): (0, 0), ('France', 'Gasexisting', 6): (0, 0), ('France', 'Gasexisting', 7): (0, 0), ('France', 'Gasexisting', 8): (0, 0), 
            ('Germany', 'GasOCGT', 1): (0, 500000.0), ('Germany', 'GasOCGT', 2): (0, 500000.0), ('Germany', 'GasOCGT', 3): (0, 500000.0), ('Germany', 'GasOCGT', 4): (0, 500000.0), ('Germany', 'GasOCGT', 5): (0, 500000.0), ('Germany', 'GasOCGT', 6): (0, 500000.0), ('Germany', 'GasOCGT', 7): (0, 500000.0), ('Germany', 'GasOCGT', 8): (0, 500000.0), 
            ('Denmark', 'GasOCGT', 1): (0, 500000.0), ('Denmark', 'GasOCGT', 2): (0, 500000.0), ('Denmark', 'GasOCGT', 3): (0, 500000.0), ('Denmark', 'GasOCGT', 4): (0, 500000.0), ('Denmark', 'GasOCGT', 5): (0, 500000.0), ('Denmark', 'GasOCGT', 6): (0, 500000.0), ('Denmark', 'GasOCGT', 7): (0, 500000.0), ('Denmark', 'GasOCGT', 8): (0, 500000.0), 
            ('France', 'GasOCGT', 1): (0, 500000.0), ('France', 'GasOCGT', 2): (0, 500000.0), ('France', 'GasOCGT', 3): (0, 500000.0), ('France', 'GasOCGT', 4): (0, 500000.0), ('France', 'GasOCGT', 5): (0, 500000.0), ('France', 'GasOCGT', 6): (0, 500000.0), ('France', 'GasOCGT', 7): (0, 500000.0), ('France', 'GasOCGT', 8): (0, 500000.0), 
            ('Germany', 'GasCCGT', 1): (0, 500000.0), ('Germany', 'GasCCGT', 2): (0, 500000.0), ('Germany', 'GasCCGT', 3): (0, 500000.0), ('Germany', 'GasCCGT', 4): (0, 500000.0), ('Germany', 'GasCCGT', 5): (0, 500000.0), ('Germany', 'GasCCGT', 6): (0, 500000.0), ('Germany', 'GasCCGT', 7): (0, 500000.0), ('Germany', 'GasCCGT', 8): (0, 500000.0), 
            ('Denmark', 'GasCCGT', 1): (0, 500000.0), ('Denmark', 'GasCCGT', 2): (0, 500000.0), ('Denmark', 'GasCCGT', 3): (0, 500000.0), ('Denmark', 'GasCCGT', 4): (0, 500000.0), ('Denmark', 'GasCCGT', 5): (0, 500000.0), ('Denmark', 'GasCCGT', 6): (0, 500000.0), ('Denmark', 'GasCCGT', 7): (0, 500000.0), ('Denmark', 'GasCCGT', 8): (0, 500000.0), 
            ('France', 'GasCCGT', 1): (0, 500000.0), ('France', 'GasCCGT', 2): (0, 500000.0), ('France', 'GasCCGT', 3): (0, 500000.0), ('France', 'GasCCGT', 4): (0, 500000.0), ('France', 'GasCCGT', 5): (0, 500000.0), ('France', 'GasCCGT', 6): (0, 500000.0), ('France', 'GasCCGT', 7): (0, 500000.0), ('France', 'GasCCGT', 8): (0, 500000.0), 
            ('Germany', 'GasCCSadv', 1): (0, 0), ('Germany', 'GasCCSadv', 2): (0, 0), ('Germany', 'GasCCSadv', 3): (0, 0), ('Germany', 'GasCCSadv', 4): (0, 0), ('Germany', 'GasCCSadv', 5): (0, 0), ('Germany', 'GasCCSadv', 6): (0, 0), ('Germany', 'GasCCSadv', 7): (0, 0), ('Germany', 'GasCCSadv', 8): (0, 0), 
            ('Denmark', 'GasCCSadv', 1): (0, 0), ('Denmark', 'GasCCSadv', 2): (0, 0), ('Denmark', 'GasCCSadv', 3): (0, 0), ('Denmark', 'GasCCSadv', 4): (0, 0), ('Denmark', 'GasCCSadv', 5): (0, 0), ('Denmark', 'GasCCSadv', 6): (0, 0), ('Denmark', 'GasCCSadv', 7): (0, 0), ('Denmark', 'GasCCSadv', 8): (0, 0), 
            ('France', 'GasCCSadv', 1): (0, 0), ('France', 'GasCCSadv', 2): (0, 0), ('France', 'GasCCSadv', 3): (0, 0), ('France', 'GasCCSadv', 4): (0, 0), ('France', 'GasCCSadv', 5): (0, 0), ('France', 'GasCCSadv', 6): (0, 0), ('France', 'GasCCSadv', 7): (0, 0), ('France', 'GasCCSadv', 8): (0, 0), 
            ('Germany', 'Oilexisting', 1): (0, 0), ('Germany', 'Oilexisting', 2): (0, 0), ('Germany', 'Oilexisting', 3): (0, 0), ('Germany', 'Oilexisting', 4): (0, 0), ('Germany', 'Oilexisting', 5): (0, 0), ('Germany', 'Oilexisting', 6): (0, 0), ('Germany', 'Oilexisting', 7): (0, 0), ('Germany', 'Oilexisting', 8): (0, 0), 
            ('Denmark', 'Oilexisting', 1): (0, 0), ('Denmark', 'Oilexisting', 2): (0, 0), ('Denmark', 'Oilexisting', 3): (0, 0), ('Denmark', 'Oilexisting', 4): (0, 0), ('Denmark', 'Oilexisting', 5): (0, 0), ('Denmark', 'Oilexisting', 6): (0, 0), ('Denmark', 'Oilexisting', 7): (0, 0), ('Denmark', 'Oilexisting', 8): (0, 0), 
            ('France', 'Oilexisting', 1): (0, 0), ('France', 'Oilexisting', 2): (0, 0), ('France', 'Oilexisting', 3): (0, 0), ('France', 'Oilexisting', 4): (0, 0), ('France', 'Oilexisting', 5): (0, 0), ('France', 'Oilexisting', 6): (0, 0), ('France', 'Oilexisting', 7): (0, 0), ('France', 'Oilexisting', 8): (0, 0), 
            ('Germany', 'Bioexisting', 1): (0, 0), ('Germany', 'Bioexisting', 2): (0, 0), ('Germany', 'Bioexisting', 3): (0, 0), ('Germany', 'Bioexisting', 4): (0, 0), ('Germany', 'Bioexisting', 5): (0, 0), ('Germany', 'Bioexisting', 6): (0, 0), ('Germany', 'Bioexisting', 7): (0, 0), ('Germany', 'Bioexisting', 8): (0, 0), 
            ('Denmark', 'Bioexisting', 1): (0, 0), ('Denmark', 'Bioexisting', 2): (0, 0), ('Denmark', 'Bioexisting', 3): (0, 0), ('Denmark', 'Bioexisting', 4): (0, 0), ('Denmark', 'Bioexisting', 5): (0, 0), ('Denmark', 'Bioexisting', 6): (0, 0), ('Denmark', 'Bioexisting', 7): (0, 0), ('Denmark', 'Bioexisting', 8): (0, 0), 
            ('France', 'Bioexisting', 1): (0, 0), ('France', 'Bioexisting', 2): (0, 0), ('France', 'Bioexisting', 3): (0, 0), ('France', 'Bioexisting', 4): (0, 0), ('France', 'Bioexisting', 5): (0, 0), ('France', 'Bioexisting', 6): (0, 0), ('France', 'Bioexisting', 7): (0, 0), ('France', 'Bioexisting', 8): (0, 0), 
            ('Germany', 'Bio10cofiring', 1): (0, 500000.0), ('Germany', 'Bio10cofiring', 2): (0, 500000.0), ('Germany', 'Bio10cofiring', 3): (0, 500000.0), ('Germany', 'Bio10cofiring', 4): (0, 500000.0), ('Germany', 'Bio10cofiring', 5): (0, 500000.0), ('Germany', 'Bio10cofiring', 6): (0, 500000.0), ('Germany', 'Bio10cofiring', 7): (0, 500000.0), ('Germany', 'Bio10cofiring', 8): (0, 500000.0), 
            ('Denmark', 'Bio10cofiring', 1): (0, 500000.0), ('Denmark', 'Bio10cofiring', 2): (0, 500000.0), ('Denmark', 'Bio10cofiring', 3): (0, 500000.0), ('Denmark', 'Bio10cofiring', 4): (0, 500000.0), ('Denmark', 'Bio10cofiring', 5): (0, 500000.0), ('Denmark', 'Bio10cofiring', 6): (0, 500000.0), ('Denmark', 'Bio10cofiring', 7): (0, 500000.0), ('Denmark', 'Bio10cofiring', 8): (0, 500000.0), 
            ('France', 'Bio10cofiring', 1): (0, 500000.0), ('France', 'Bio10cofiring', 2): (0, 500000.0), ('France', 'Bio10cofiring', 3): (0, 500000.0), ('France', 'Bio10cofiring', 4): (0, 500000.0), ('France', 'Bio10cofiring', 5): (0, 500000.0), ('France', 'Bio10cofiring', 6): (0, 500000.0), ('France', 'Bio10cofiring', 7): (0, 500000.0), ('France', 'Bio10cofiring', 8): (0, 500000.0), 
            ('Germany', 'Nuclear', 1): (0, 500000.0), ('Germany', 'Nuclear', 2): (0, 500000.0), ('Germany', 'Nuclear', 3): (0, 500000.0), ('Germany', 'Nuclear', 4): (0, 500000.0), ('Germany', 'Nuclear', 5): (0, 500000.0), ('Germany', 'Nuclear', 6): (0, 500000.0), ('Germany', 'Nuclear', 7): (0, 500000.0), ('Germany', 'Nuclear', 8): (0, 500000.0), 
            ('France', 'Nuclear', 1): (0, 500000.0), ('France', 'Nuclear', 2): (0, 500000.0), ('France', 'Nuclear', 3): (0, 500000.0), ('France', 'Nuclear', 4): (0, 500000.0), ('France', 'Nuclear', 5): (0, 500000.0), ('France', 'Nuclear', 6): (0, 500000.0), ('France', 'Nuclear', 7): (0, 500000.0), ('France', 'Nuclear', 8): (0, 500000.0), 
            ('France', 'Wave', 1): (0, 500000.0), ('France', 'Wave', 2): (0, 500000.0), ('France', 'Wave', 3): (0, 500000.0), ('France', 'Wave', 4): (0, 500000.0), ('France', 'Wave', 5): (0, 500000.0), ('France', 'Wave', 6): (0, 500000.0), ('France', 'Wave', 7): (0, 500000.0), ('France', 'Wave', 8): (0, 500000.0), 
            ('Germany', 'Geo', 1): (0, 500000.0), ('Germany', 'Geo', 2): (0, 500000.0), ('Germany', 'Geo', 3): (0, 500000.0), ('Germany', 'Geo', 4): (0, 500000.0), ('Germany', 'Geo', 5): (0, 500000.0), ('Germany', 'Geo', 6): (0, 500000.0), ('Germany', 'Geo', 7): (0, 500000.0), ('Germany', 'Geo', 8): (0, 500000.0), 
            ('France', 'Geo', 1): (0, 500000.0), ('France', 'Geo', 2): (0, 500000.0), ('France', 'Geo', 3): (0, 500000.0), ('France', 'Geo', 4): (0, 500000.0), ('France', 'Geo', 5): (0, 500000.0), ('France', 'Geo', 6): (0, 500000.0), ('France', 'Geo', 7): (0, 500000.0), ('France', 'Geo', 8): (0, 500000.0), 
            ('Germany', 'Hydroregulated', 1): (0, 500000.0), ('Germany', 'Hydroregulated', 2): (0, 500000.0), ('Germany', 'Hydroregulated', 3): (0, 500000.0), ('Germany', 'Hydroregulated', 4): (0, 500000.0), ('Germany', 'Hydroregulated', 5): (0, 500000.0), ('Germany', 'Hydroregulated', 6): (0, 500000.0), ('Germany', 'Hydroregulated', 7): (0, 500000.0), ('Germany', 'Hydroregulated', 8): (0, 500000.0), 
            ('France', 'Hydroregulated', 1): (0, 500000.0), ('France', 'Hydroregulated', 2): (0, 500000.0), ('France', 'Hydroregulated', 3): (0, 500000.0), ('France', 'Hydroregulated', 4): (0, 500000.0), ('France', 'Hydroregulated', 5): (0, 500000.0), ('France', 'Hydroregulated', 6): (0, 500000.0), ('France', 'Hydroregulated', 7): (0, 500000.0), ('France', 'Hydroregulated', 8): (0, 500000.0), 
            ('Germany', 'Hydrorun-of-the-river', 1): (0, 500000.0), ('Germany', 'Hydrorun-of-the-river', 2): (0, 500000.0), ('Germany', 'Hydrorun-of-the-river', 3): (0, 500000.0), ('Germany', 'Hydrorun-of-the-river', 4): (0, 500000.0), ('Germany', 'Hydrorun-of-the-river', 5): (0, 500000.0), ('Germany', 'Hydrorun-of-the-river', 6): (0, 500000.0), ('Germany', 'Hydrorun-of-the-river', 7): (0, 500000.0), ('Germany', 'Hydrorun-of-the-river', 8): (0, 500000.0), 
            ('Denmark', 'Hydrorun-of-the-river', 1): (0, 500000.0), ('Denmark', 'Hydrorun-of-the-river', 2): (0, 500000.0), ('Denmark', 'Hydrorun-of-the-river', 3): (0, 500000.0), ('Denmark', 'Hydrorun-of-the-river', 4): (0, 500000.0), ('Denmark', 'Hydrorun-of-the-river', 5): (0, 500000.0), ('Denmark', 'Hydrorun-of-the-river', 6): (0, 500000.0), ('Denmark', 'Hydrorun-of-the-river', 7): (0, 500000.0), ('Denmark', 'Hydrorun-of-the-river', 8): (0, 500000.0), 
            ('France', 'Hydrorun-of-the-river', 1): (0, 500000.0), ('France', 'Hydrorun-of-the-river', 2): (0, 500000.0), ('France', 'Hydrorun-of-the-river', 3): (0, 500000.0), ('France', 'Hydrorun-of-the-river', 4): (0, 500000.0), ('France', 'Hydrorun-of-the-river', 5): (0, 500000.0), ('France', 'Hydrorun-of-the-river', 6): (0, 500000.0), ('France', 'Hydrorun-of-the-river', 7): (0, 500000.0), ('France', 'Hydrorun-of-the-river', 8): (0, 500000.0), 
            ('Germany', 'Bio', 1): (0, 500000.0), ('Germany', 'Bio', 2): (0, 500000.0), ('Germany', 'Bio', 3): (0, 500000.0), ('Germany', 'Bio', 4): (0, 500000.0), ('Germany', 'Bio', 5): (0, 500000.0), ('Germany', 'Bio', 6): (0, 500000.0), ('Germany', 'Bio', 7): (0, 500000.0), ('Germany', 'Bio', 8): (0, 500000.0), 
            ('France', 'Bio', 1): (0, 500000.0), ('France', 'Bio', 2): (0, 500000.0), ('France', 'Bio', 3): (0, 500000.0), ('France', 'Bio', 4): (0, 500000.0), ('France', 'Bio', 5): (0, 500000.0), ('France', 'Bio', 6): (0, 500000.0), ('France', 'Bio', 7): (0, 500000.0), ('France', 'Bio', 8): (0, 500000.0), 
            ('Germany', 'Windonshore', 1): (0, 500000.0), ('Germany', 'Windonshore', 2): (0, 500000.0), ('Germany', 'Windonshore', 3): (0, 500000.0), ('Germany', 'Windonshore', 4): (0, 500000.0), ('Germany', 'Windonshore', 5): (0, 500000.0), ('Germany', 'Windonshore', 6): (0, 500000.0), ('Germany', 'Windonshore', 7): (0, 500000.0), ('Germany', 'Windonshore', 8): (0, 500000.0), 
            ('Denmark', 'Windonshore', 1): (0, 500000.0), ('Denmark', 'Windonshore', 2): (0, 500000.0), ('Denmark', 'Windonshore', 3): (0, 500000.0), ('Denmark', 'Windonshore', 4): (0, 500000.0), ('Denmark', 'Windonshore', 5): (0, 500000.0), ('Denmark', 'Windonshore', 6): (0, 500000.0), ('Denmark', 'Windonshore', 7): (0, 500000.0), ('Denmark', 'Windonshore', 8): (0, 500000.0), ('France', 'Windonshore', 1): (0, 500000.0), ('France', 'Windonshore', 2): (0, 500000.0), ('France', 'Windonshore', 3): (0, 500000.0), ('France', 'Windonshore', 4): (0, 500000.0), ('France', 'Windonshore', 5): (0, 500000.0), ('France', 'Windonshore', 6): (0, 500000.0), ('France', 'Windonshore', 7): (0, 500000.0), ('France', 'Windonshore', 8): (0, 500000.0), ('Germany', 'Windoffshore', 1): (0, 500000.0), ('Germany', 'Windoffshore', 2): (0, 500000.0), ('Germany', 'Windoffshore', 3): (0, 500000.0), ('Germany', 'Windoffshore', 4): (0, 500000.0), ('Germany', 'Windoffshore', 5): (0, 500000.0), ('Germany', 'Windoffshore', 6): (0, 500000.0), ('Germany', 'Windoffshore', 7): (0, 500000.0), ('Germany', 'Windoffshore', 8): (0, 500000.0), ('Denmark', 'Windoffshore', 1): (0, 500000.0), ('Denmark', 'Windoffshore', 2): (0, 500000.0), ('Denmark', 'Windoffshore', 3): (0, 500000.0), ('Denmark', 'Windoffshore', 4): (0, 500000.0), ('Denmark', 'Windoffshore', 5): (0, 500000.0), ('Denmark', 'Windoffshore', 6): (0, 500000.0), ('Denmark', 'Windoffshore', 7): (0, 500000.0), ('Denmark', 'Windoffshore', 8): (0, 500000.0), ('France', 'Windoffshore', 1): (0, 500000.0), ('France', 'Windoffshore', 2): (0, 500000.0), ('France', 'Windoffshore', 3): (0, 500000.0), ('France', 'Windoffshore', 4): (0, 500000.0), ('France', 'Windoffshore', 5): (0, 500000.0), ('France', 'Windoffshore', 6): (0, 500000.0), ('France', 'Windoffshore', 7): (0, 500000.0), ('France', 'Windoffshore', 8): (0, 500000.0), ('Germany', 'Solar', 1): (0, 500000.0), ('Germany', 'Solar', 2): (0, 500000.0), ('Germany', 'Solar', 3): (0, 500000.0), ('Germany', 'Solar', 4): (0, 500000.0), ('Germany', 'Solar', 5): (0, 500000.0), ('Germany', 'Solar', 6): (0, 500000.0), ('Germany', 'Solar', 7): (0, 500000.0), ('Germany', 'Solar', 8): (0, 500000.0), ('Denmark', 'Solar', 1): (0, 500000.0), ('Denmark', 'Solar', 2): (0, 500000.0), ('Denmark', 'Solar', 3): (0, 500000.0), ('Denmark', 'Solar', 4): (0, 500000.0), ('Denmark', 'Solar', 5): (0, 500000.0), ('Denmark', 'Solar', 6): (0, 500000.0), ('Denmark', 'Solar', 7): (0, 500000.0), ('Denmark', 'Solar', 8): (0, 500000.0), ('France', 'Solar', 1): (0, 500000.0), ('France', 'Solar', 2): (0, 500000.0), ('France', 'Solar', 3): (0, 500000.0), ('France', 'Solar', 4): (0, 500000.0), ('France', 'Solar', 5): (0, 500000.0), ('France', 'Solar', 6): (0, 500000.0), ('France', 'Solar', 7): (0, 500000.0), ('France', 'Solar', 8): (0, 500000.0), ('Germany', 'Bio10cofiringCCS', 1): (0, 0), ('Germany', 'Bio10cofiringCCS', 2): (0, 0), ('Germany', 'Bio10cofiringCCS', 3): (0, 0), ('Germany', 'Bio10cofiringCCS', 4): (0, 0), ('Germany', 'Bio10cofiringCCS', 5): (0, 0), ('Germany', 'Bio10cofiringCCS', 6): (0, 0), ('Germany', 'Bio10cofiringCCS', 7): (0, 0), ('Germany', 'Bio10cofiringCCS', 8): (0, 0), ('Denmark', 'Bio10cofiringCCS', 1): (0, 0), ('Denmark', 'Bio10cofiringCCS', 2): (0, 0), ('Denmark', 'Bio10cofiringCCS', 3): (0, 0), ('Denmark', 'Bio10cofiringCCS', 4): (0, 0), ('Denmark', 'Bio10cofiringCCS', 5): (0, 0), ('Denmark', 'Bio10cofiringCCS', 6): (0, 0), ('Denmark', 'Bio10cofiringCCS', 7): (0, 0), ('Denmark', 'Bio10cofiringCCS', 8): (0, 0), ('France', 'Bio10cofiringCCS', 1): (0, 0), ('France', 'Bio10cofiringCCS', 2): (0, 0), ('France', 'Bio10cofiringCCS', 3): (0, 0), ('France', 'Bio10cofiringCCS', 4): (0, 0), ('France', 'Bio10cofiringCCS', 5): (0, 0), ('France', 'Bio10cofiringCCS', 6): (0, 0), ('France', 'Bio10cofiringCCS', 7): (0, 0), ('France', 'Bio10cofiringCCS', 8): (0, 0), 
            ('Germany', 'LigniteCCSsup', 1): (0, 0), ('Germany', 'LigniteCCSsup', 2): (0, 0), ('Germany', 'LigniteCCSsup', 3): (0, 0), ('Germany', 'LigniteCCSsup', 4): (0, 0), ('Germany', 'LigniteCCSsup', 5): (0, 0), ('Germany', 'LigniteCCSsup', 6): (0, 0), ('Germany', 'LigniteCCSsup', 7): (0, 0), ('Germany', 'LigniteCCSsup', 8): (0, 0), ('Germany', 'CoalCCS', 1): (0, 0), ('Germany', 'CoalCCS', 2): (0, 0), ('Germany', 'CoalCCS', 3): (0, 0), ('Germany', 'CoalCCS', 4): (0, 0), ('Germany', 'CoalCCS', 5): (0, 0), ('Germany', 'CoalCCS', 6): (0, 0), ('Germany', 'CoalCCS', 7): (0, 0), ('Germany', 'CoalCCS', 8): (0, 0), ('France', 'CoalCCS', 1): (0, 0), ('France', 'CoalCCS', 2): (0, 0), ('France', 'CoalCCS', 3): (0, 0), ('France', 'CoalCCS', 4): (0, 0), ('France', 'CoalCCS', 5): (0, 0), ('France', 'CoalCCS', 6): (0, 0), ('France', 'CoalCCS', 7): (0, 0), ('France', 'CoalCCS', 8): (0, 0), ('Germany', 'GasCCS', 1): (0, 0), ('Germany', 'GasCCS', 2): (0, 0), ('Germany', 'GasCCS', 3): (0, 0), ('Germany', 'GasCCS', 4): (0, 0), ('Germany', 'GasCCS', 5): (0, 0), ('Germany', 'GasCCS', 6): (0, 0), ('Germany', 'GasCCS', 7): (0, 0), ('Germany', 'GasCCS', 8): (0, 0), ('France', 'GasCCS', 1): (0, 0), ('France', 'GasCCS', 2): (0, 0), ('France', 'GasCCS', 3): (0, 0), ('France', 'GasCCS', 4): (0, 0), ('France', 'GasCCS', 5): (0, 0), ('France', 'GasCCS', 6): (0, 0), ('France', 'GasCCS', 7): (0, 0), ('France', 'GasCCS', 8): (0, 0), ('Germany', 'Waste', 1): (0, 500000.0), ('Germany', 'Waste', 2): (0, 500000.0), ('Germany', 'Waste', 3): (0, 500000.0), ('Germany', 'Waste', 4): (0, 500000.0), ('Germany', 'Waste', 5): (0, 500000.0), ('Germany', 'Waste', 6): (0, 500000.0), ('Germany', 'Waste', 7): (0, 500000.0), ('Germany', 'Waste', 8): (0, 500000.0), ('Denmark', 'Waste', 1): (0, 500000.0), ('Denmark', 'Waste', 2): (0, 500000.0), ('Denmark', 'Waste', 3): (0, 500000.0), ('Denmark', 'Waste', 4): (0, 500000.0), ('Denmark', 'Waste', 5): (0, 500000.0), ('Denmark', 'Waste', 6): (0, 500000.0), ('Denmark', 'Waste', 7): (0, 500000.0), ('Denmark', 'Waste', 8): (0, 500000.0), 
            ('France', 'Waste', 1): (0, 500000.0), ('France', 'Waste', 2): (0, 500000.0), ('France', 'Waste', 3): (0, 500000.0), ('France', 'Waste', 4): (0, 500000.0), ('France', 'Waste', 5): (0, 500000.0), ('France', 'Waste', 6): (0, 500000.0), ('France', 'Waste', 7): (0, 500000.0), ('France', 'Waste', 8): (0, 500000.0)
        },

        'transmisionInvCap': {
            ('Denmark', 'Germany', 1): (0, 20000), ('Denmark', 'Germany', 2): (0, 20000.0), ('Denmark', 'Germany', 3): (0, 20000.0), ('Denmark', 'Germany', 4): (0, 20000.0), ('Denmark', 'Germany', 5): (0, 20000.0), ('Denmark', 'Germany', 6): (0, 20000.0), ('Denmark', 'Germany', 7): (0, 20000.0), ('Denmark', 'Germany', 8): (0, 20000.0), ('France', 'Germany', 1): (0, 20000), ('France', 'Germany', 2): (0, 20000.0), ('France', 'Germany', 3): (0, 20000.0), ('France', 'Germany', 4): (0, 20000.0), ('France', 'Germany', 5): (0, 20000.0), ('France', 'Germany', 6): (0, 20000.0), ('France', 'Germany', 7): (0, 20000.0), ('France', 'Germany', 8): (0, 20000.0)
        },
        
        'storPWInvCap': {
            ('Germany', 'HydroPumpStorage', 1): (0, 500000), ('Germany', 'HydroPumpStorage', 2): (0, 500000.0), ('Germany', 'HydroPumpStorage', 3): (0, 500000.0), ('Germany', 'HydroPumpStorage', 4): (0, 500000.0), ('Germany', 'HydroPumpStorage', 5): (0, 500000.0), ('Germany', 'HydroPumpStorage', 6): (0, 500000.0), ('Germany', 'HydroPumpStorage', 7): (0, 500000.0), ('Germany', 'HydroPumpStorage', 8): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 1): (0, 500000), ('Germany', 'Li-Ion_BESS', 2): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 3): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 4): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 5): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 6): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 7): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 8): (0, 500000.0), 
            ('Denmark', 'Li-Ion_BESS', 1): (0, 500000), ('Denmark', 'Li-Ion_BESS', 2): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 3): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 4): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 5): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 6): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 7): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 8): (0, 500000.0), ('France', 'HydroPumpStorage', 1): (0, 500000), ('France', 'HydroPumpStorage', 2): (0, 500000.0), ('France', 'HydroPumpStorage', 3): (0, 500000.0), ('France', 'HydroPumpStorage', 4): (0, 500000.0), ('France', 'HydroPumpStorage', 5): (0, 500000.0), ('France', 'HydroPumpStorage', 6): (0, 500000.0), ('France', 'HydroPumpStorage', 7): (0, 500000.0), ('France', 'HydroPumpStorage', 8): (0, 500000.0), 
            ('France', 'Li-Ion_BESS', 1): (0, 500000), ('France', 'Li-Ion_BESS', 2): (0, 500000.0), ('France', 'Li-Ion_BESS', 3): (0, 500000.0), ('France', 'Li-Ion_BESS', 4): (0, 500000.0), ('France', 'Li-Ion_BESS', 5): (0, 500000.0), ('France', 'Li-Ion_BESS', 6): (0, 500000.0), ('France', 'Li-Ion_BESS', 7): (0, 500000.0), ('France', 'Li-Ion_BESS', 8): (0, 500000.0)
        },
        'storENInvCap': {('Germany', 'HydroPumpStorage', 1): (0, 500000), ('Germany', 'HydroPumpStorage', 2): (0, 500000.0), ('Germany', 'HydroPumpStorage', 3): (0, 500000.0), ('Germany', 'HydroPumpStorage', 4): (0, 500000.0), ('Germany', 'HydroPumpStorage', 5): (0, 500000.0), ('Germany', 'HydroPumpStorage', 6): (0, 500000.0), ('Germany', 'HydroPumpStorage', 7): (0, 500000.0), ('Germany', 'HydroPumpStorage', 8): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 1): (0, 500000), ('Germany', 'Li-Ion_BESS', 2): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 3): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 4): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 5): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 6): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 7): (0, 500000.0), ('Germany', 'Li-Ion_BESS', 8): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 1): (0, 500000), ('Denmark', 'Li-Ion_BESS', 2): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 3): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 4): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 5): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 6): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 7): (0, 500000.0), ('Denmark', 'Li-Ion_BESS', 8): (0, 500000.0), ('France', 'HydroPumpStorage', 1): (0, 500000), ('France', 'HydroPumpStorage', 2): (0, 500000.0), ('France', 'HydroPumpStorage', 3): (0, 500000.0), ('France', 'HydroPumpStorage', 4): (0, 500000.0), ('France', 'HydroPumpStorage', 5): (0, 500000.0), ('France', 'HydroPumpStorage', 6): (0, 500000.0), ('France', 'HydroPumpStorage', 7): (0, 500000.0), ('France', 'HydroPumpStorage', 8): (0, 500000.0), 
        ('France', 'Li-Ion_BESS', 1): (0, 500000), ('France', 'Li-Ion_BESS', 2): (0, 500000.0), ('France', 'Li-Ion_BESS', 3): (0, 500000.0), ('France', 'Li-Ion_BESS', 4): (0, 500000.0), ('France', 'Li-Ion_BESS', 5): (0, 500000.0), ('France', 'Li-Ion_BESS', 6): (0, 500000.0), ('France', 'Li-Ion_BESS', 7): (0, 500000.0), ('France', 'Li-Ion_BESS', 8): (0, 500000.0)},
       
    }

    run_zero_prob_experiment(bounds_dict, data_folder, zero_prob, samples_per_prob=1000)
