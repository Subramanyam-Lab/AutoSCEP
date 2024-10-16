import pandas as pd
import numpy as np
from scipy import stats
import glob
from FSD_sampling_violation import create_model, load_investment_data
from pyomo.environ import *
import csv
import io
import contextlib
from pyomo.core.expr.visitor import identify_variables


def read_csv_files(file_pattern):
    all_data = []
    for file in glob.glob(file_pattern):
        df = pd.read_csv(file)
        all_data.append(df)
    return all_data

def build_kde_and_sample(data_list, bandwidth_factor):
    combined_data = pd.concat(data_list, ignore_index=True)
    grouped = combined_data.groupby(['Node', 'Energy_Type', 'Period', 'Type'])
    sampled_data = []
    
    for name, group in grouped:
        values = group['Value'].values
        if len(values) >= 2:
            try:
                bw = bandwidth_factor
                kde = stats.gaussian_kde(values, bw_method=bw)
                new_sample = kde.resample(1)[0][0]
            except:
                new_sample = np.random.choice(values)
        else:
            new_sample = np.random.choice(values)
        
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

# def check_constraints(instance):
#     violated_constraints = []
#     for constr in instance.component_objects(Constraint, active=True):
#         for index in constr:
#             c = constr[index]
#             try:
#                 body_value = value(c.body)
#                 lower = value(c.lower) if c.lower is not None else None
#                 upper = value(c.upper) if c.upper is not None else None
#                 tol = 1e-6  # Tolerance for floating-point comparisons
#                 if lower is not None and body_value < lower - tol:
#                     violated_constraints.append((c.name, index, 'Lower bound violated', body_value, lower))
#                 elif upper is not None and body_value > upper + tol:
#                     violated_constraints.append((c.name, index, 'Upper bound violated', body_value, upper))
#             except ValueError:
#                 # Skip constraints with uninitialized variables
#                 continue
#     return violated_constraints


def check_constraints(instance):
    violated_constraints = []
    for constr in instance.component_objects(Constraint, active=True):
        constr_name = constr.name
        for index in constr:
            c = constr[index]
            try:
                body_value = value(c.body)
                lower = value(c.lower) if c.lower is not None else None
                upper = value(c.upper) if c.upper is not None else None
                tol = 1e-6  # Tolerance for floating-point comparisons
                violation = None
                if lower is not None and body_value < lower - tol:
                    violation = ('Lower bound violated', body_value, lower)
                elif upper is not None and body_value > upper + tol:
                    violation = ('Upper bound violated', body_value, upper)
                if violation:
                    # Check if the constraint includes slack variables
                    slack_info = ""
                    vars_in_constraint = list(identify_variables(c.body))
                    slack_vars = [v for v in vars_in_constraint if 'Slack' in v.name]
                    if slack_vars:
                        slack_values = {v.name: value(v) for v in slack_vars}
                        slack_info = f", Slack Variables: {slack_values}"
                    violated_constraints.append((
                        constr_name,
                        index,
                        violation[0],
                        violation[1],
                        violation[2],
                        slack_info
                    ))
            except (ValueError, ZeroDivisionError):
                # Skip constraints with uninitialized variables
                continue
    return violated_constraints


def check_model_feasibility(model, data):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            instance = model.create_instance(data)
    except ValueError as e:
        return False
    except Exception as e:
        return False

    solver = SolverFactory('glpk')
    try:
        results = solver.solve(instance, tee=False)
    except Exception as e:
        return False

    if results.solver.termination_condition == TerminationCondition.optimal:
        return True
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("Model is infeasible.")
        violated_constraints = check_constraints(instance)
        if violated_constraints:
            print("Violated Constraints:")
            for name, index, violation_type, body_value, bound, slack_info in violated_constraints:
                print(f"{name}[{index}]: {violation_type}. Body value: {body_value}, Bound: {bound}{slack_info}")
        else:
            print("No constraints appear to be violated with current variable values.")
        return False
    else:
        print(f"Solver Termination Condition: {results.solver.termination_condition}")
        print("Couldn't evaluate feasibility")
        return None

def has_negative_values(fsd_data):
    for row in fsd_data:
        cap_value = float(row[4])  # Assuming 'Value' is the 5th column
        if cap_value < 0:
            return True
    return False

def generate_feasible_samples(n, input_file_pattern, data_folder, bandwidth_factor):
    data_list = read_csv_files(input_file_pattern)
    feasible_samples = 0
    total_attempts = 0
    max_attempts = 1000

    while feasible_samples < n:
        if total_attempts >= max_attempts:
            break
        sampled_df = build_kde_and_sample(data_list, bandwidth_factor)
        total_attempts += 1

        # Convert sampled_df to fsd_data
        fsd_data = sampled_df.values.tolist()

        # Check for negative values
        if has_negative_values(fsd_data):
            print(f"Attempt {total_attempts} has negative capacities. Discarding.")
            continue

        # Load investment data
        gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(fsd_data)

        # Create model and data
        model, data = create_model(data_folder, gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap)

        # Check feasibility
        if check_model_feasibility(model, data):
            output_file = f'sampled_data_{feasible_samples + 1}_{bandwidth_factor}.csv'
            save_to_csv(sampled_df, output_file)
            print(f"Sample {feasible_samples + 1} is feasible. Saved as {output_file}")
            feasible_samples += 1
        else:
            print(f"Attempt {total_attempts} is infeasible. Discarding.")

        if total_attempts % 5 == 0:
            current_ratio = feasible_samples / total_attempts
            print(f"Current accept-rejection ratio: {current_ratio:.6f}")

    final_ratio = feasible_samples / total_attempts
    print(f"Generated {n} feasible samples.")
    print(f"Final accept-rejection ratio: {final_ratio:.6f}")
    print(f"Total attempts: {total_attempts}")

# Main execution
if __name__ == "__main__":
    input_file_pattern = "SeedSamples/reduced/fsd_seed*.csv"  # Adjust this pattern to match your input files
    data_folder = 'Data handler/sampling/reduced'
    n_samples = 10  # Number of feasible samples you want to generate
    bandwidth_factor = 5e-5  # Adjust this value to control the bandwidth (0 for exploit, larger for explore)
    
    generate_feasible_samples(n_samples, input_file_pattern, data_folder, bandwidth_factor)