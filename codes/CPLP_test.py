import time
import onnxruntime as ort
from pyomo.environ import *
import pandas as pd
import os
import glob
from func_Q2 import *
import numpy as np
from model_definition import define_model
import re
import random
import multiprocessing
import torch
import pandas as pd
import matplotlib.pyplot as plt


# Define the model
model = AbstractModel()

# Define sets
model.P = Set()  # Set of potential plant locations
model.C = Set()  # Set of customers

# Define parameters
model.f = Param(model.P, within=NonNegativeReals)  # Cost of setting up a plant at each location
model.c = Param(model.P, within=NonNegativeReals)  # Capacity of each plant
model.demands = Param(model.C, mutable=True, within=NonNegativeReals)  # Demand of each customer
model.t = Param(model.C, model.P, within=NonNegativeReals)  # Transport cost from plant to customer


# Define variables
model.x = Var(model.C, model.P, within=NonNegativeReals, bounds=(0, 1))  # Supply ratio from plant to customer
model.y = Var(model.P, within=Binary)  # Binary variable indicating whether a plant is built
model.s = Var(model.P, within=NonNegativeReals)


# Define objective function: minimize setup and transport costs
def objective_rule(model):
    return sum(model.f[p]*model.y[p] for p in model.P) + \
           sum(model.t[c,p]*model.x[c,p] for c in model.C for p in model.P) + sum(model.t[c,p] for c in model.C for p in model.P) * sum(model.s[p] for p in model.P)
model.obj = Objective(rule=objective_rule, sense=minimize)

# Define constraints
def demand_constraint_rule(model, c):
    return sum(model.x[c,p] for p in model.P) == 1  
model.demand_constraint = Constraint(model.C, rule=demand_constraint_rule)

def capacity_constraint_rule(model, p):
    return sum(model.demands[c] * model.x[c,p] for c in model.C) <= model.c[p] * model.y[p] + model.s[p]
model.capacity_constraint = Constraint(model.P, rule=capacity_constraint_rule)


def load_onnx_model(model_path):
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    return ort_session, input_name


# Ensure the CPLP_test directory exists
results_directory = "CPLP_test1"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
    


# Pyomo 모델을 해결하고 ONNX 모델로 예측하는 함수
def solve_and_predict(data_file, size, model, num_tests=1):
    clients, facilities = size
    first_stage_decisions_lst = []
    nn_second_stage_value_lst = []

    problem_size = f"{clients}_{facilities}_{data_file.split('_')[-1].split('.')[0]}"  
    result_filename = f"results_{problem_size}.csv"
    # ONNX 모델 로드
    file_number = data_file.split('_')[-1].split('.')[0]
    onnx_model_path = f'Models_V2/CPLP_{clients}_{facilities}/CPLP_{clients}_{facilities}_{file_number}.onnx'
    onnx_model, input_name = load_onnx_model(onnx_model_path)


    instance = model.create_instance(data_file)
    
    # Initialize an empty DataFrame to store all test results
    all_tests_df = pd.DataFrame()
    
    for test in range(num_tests):
    
        demand_values = {c: random.uniform(5, 35) for c in instance.C}
        for c in instance.C:
            instance.demands[c] = demand_values[c]
        solver = SolverFactory('glpk')
        results = solver.solve(instance, tee=False)

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            # sum(model.f[p]*model.y[p] for p in model.P) 값을 계산
            setup_cost = sum(instance.f[p]*value(instance.y[p]) for p in instance.P)
        else:
            print("Solution is not optimal or solver did not finish successfully.")

        # 첫 번째 단계 결정 추출
        first_stage_decisions = [value(instance.y[p]) for p in instance.P]
        first_stage_decisions_lst.append(first_stage_decisions)

        # ONNX 모델을 사용한 예측
        nn_input = np.array(first_stage_decisions, dtype=np.float32).reshape(1, -1)
        nn_output = onnx_model.run(None, {input_name: nn_input})[0]  # 올바른 입력 이름을 사용합니다.
        nn_expected_second_stage_value = nn_output[0][0]
        nn_second_stage_value_lst.append(nn_expected_second_stage_value)
        
        model_result = setup_cost + nn_expected_second_stage_value
        pyomo_result = value(instance.obj)
        gap = (abs(model_result-pyomo_result)/pyomo_result)*100
    
        # Append the result of the current test to the DataFrame
        df = pd.DataFrame({
            'Pyomo result': [pyomo_result],
            'NN result': [model_result],
            'Gap(%)': [gap]
        })
        all_tests_df = pd.concat([all_tests_df, df], ignore_index=True)

    # After all tests are done, save the DataFrame to a CSV file
    result_filename = os.path.join(results_directory, f"test_results_{problem_size}_tests.csv")
    all_tests_df.to_csv(result_filename, index=False)

    return all_tests_df


if __name__ == '__main__':
    start_time = time.time()
    # problem_sizes = [(10, 10), (25, 25), (50, 50)]
    problem_sizes = [(10,10)]
    num_files_to_load = 30
    num_tests = 1000

    for size in problem_sizes:
        clients, facilities = size
        data_dir = f"data/CPLP_{clients}_{facilities}"
        data_files = glob.glob(os.path.join(data_dir, "*.dat"))
        data_files_sorted = sorted(data_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
        data_files_to_load = data_files_sorted[:num_files_to_load]

        for data_file in data_files_to_load:
            df = solve_and_predict(data_file, size, model, num_tests)

    # CSV 파일을 읽고 박스 플롯 생성
    file_pattern = 'CPLP_test1/test_results*.csv'
    csv_files = glob.glob(file_pattern)
    gap_data_by_size = {}

    for file in csv_files:
        df = pd.read_csv(file)
        size_info = file.split('/')[-1].split('_')[2:4]
        size_key = f"{size_info[0]}_{size_info[1]}"
        # print(size_info[0])
        # print("average gap: ",np.mean(df['Gap(%)'].tolist()))
        
        if size_key not in gap_data_by_size:
            gap_data_by_size[size_key] = []
        gap_data_by_size[size_key].extend(df['Gap(%)'].tolist())

    boxplot_data = [gaps for gaps in gap_data_by_size.values()]
    size_labels = [size for size in gap_data_by_size.keys()]

    plt.figure(figsize=(12, 6))
    plt.boxplot(boxplot_data, labels=size_labels)
    plt.title('Optimality Gap by Problem Size', fontsize=15)
    plt.ylabel('Gap(%)', fontsize=15)
    plt.xlabel('Problem Size ("Number of client"_"Num of plant")', fontsize=15)
    plt.xticks(rotation=0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join("CPLP_test1", "boxplot_gap_percent_by_size.png"))
    plt.show()
    
    for size_key, gaps in gap_data_by_size.items():
        average_gap = np.mean(gaps)
        print(f"Size {size_key} - Average Gap: {average_gap:.2f}%")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")