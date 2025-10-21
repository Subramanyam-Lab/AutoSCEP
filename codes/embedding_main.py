import os
from datetime import datetime
from yaml import safe_load
import time
import warnings
warnings.filterwarnings('ignore')  # To suppress any warnings for cleaner output
import pandas as pd
import argparse
from pyomo.environ import *
from ml_embedding import embed_empire_embedding


def main(model_type, NoOfScenarios, Seed):
    
    UserRunTimeConfig = safe_load(open("config_run.yaml"))

    USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
    temp_dir = UserRunTimeConfig["temp_dir"]
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
    discountrate = UserRunTimeConfig["discountrate"]
    WACC = UserRunTimeConfig["WACC"]
    solver = UserRunTimeConfig["solver"]
    scenariogeneration = UserRunTimeConfig["scenariogeneration"]
    fix_sample = UserRunTimeConfig["fix_sample"]
    LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
    filter_use = UserRunTimeConfig["filter_use"]
    n_cluster = UserRunTimeConfig["n_cluster"]
    moment_matching = UserRunTimeConfig["moment_matching"]
    n_tree_compare = UserRunTimeConfig["n_tree_compare"]
    EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]

    #############################
    ##Non configurable settings##
    #############################

    NoOfRegSeason = 4
    regular_seasons = ["winter", "spring", "summer", "fall"]
    NoOfPeakSeason = 2
    lengthPeakSeason = 24
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    north_sea = False


    #######
    ##RUN##
    #######


    name = version + '_reg' + str(lengthRegSeason) + \
        '_peak' + str(lengthPeakSeason) + \
        '_sce' + str(NoOfScenarios)
    if scenariogeneration and not fix_sample:
            name = name + "_randomSGR"
    else:
        name = name + "_noSGR"
    if filter_use:
        name = name + "_filter" + str(n_cluster)
    if moment_matching:
        name = name + "_moment" + str(n_tree_compare)
    name = name + str(datetime.now().strftime("_%Y%m%d%H%M"))
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name + f'_{0}'
    result_file_path = 'Results/' + name
    FirstHoursOfRegSeason = [lengthRegSeason*i + 1 for i in range(NoOfRegSeason)]
    FirstHoursOfPeakSeason = [lengthRegSeason*NoOfRegSeason + lengthPeakSeason*i + 1 for i in range(NoOfPeakSeason)]
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
    Scenario = ["scenario"+str(i + 1) for i in range(NoOfScenarios)]
    peak_seasons = ['peak'+str(i + 1) for i in range(NoOfPeakSeason)]
    Season = regular_seasons + peak_seasons
    Operationalhour = [i + 1 for i in range(FirstHoursOfPeakSeason[-1] + lengthPeakSeason - 1)]
    HoursOfRegSeason = [(s,h) for s in regular_seasons for h in Operationalhour \
                    if h in list(range(regular_seasons.index(s)*lengthRegSeason+1,
                                regular_seasons.index(s)*lengthRegSeason+lengthRegSeason+1))]
    HoursOfPeakSeason = [(s,h) for s in peak_seasons for h in Operationalhour \
                        if h in list(range(lengthRegSeason*len(regular_seasons)+ \
                                            peak_seasons.index(s)*lengthPeakSeason+1,
                                            lengthRegSeason*len(regular_seasons)+ \
                                                peak_seasons.index(s)*lengthPeakSeason+ \
                                                    lengthPeakSeason+1))]
    HoursOfSeason = HoursOfRegSeason + HoursOfPeakSeason


    instance = embed_empire_embedding(version = version,
            tab_file_path = tab_file_path,
            result_file_path = result_file_path, 
            temp_dir = temp_dir, 
            FirstHoursOfRegSeason = FirstHoursOfRegSeason, 
            FirstHoursOfPeakSeason = FirstHoursOfPeakSeason, 
            lengthRegSeason = lengthRegSeason,
            lengthPeakSeason = lengthPeakSeason,
            Period = Period, 
            Operationalhour = Operationalhour,
            Scenario = Scenario,
            Season = Season,
            HoursOfSeason = HoursOfSeason,
            discountrate = discountrate, 
            WACC = WACC, 
            LeapYearsInvestment = LeapYearsInvestment,
            EMISSION_CAP = EMISSION_CAP,
            USE_TEMP_DIR = USE_TEMP_DIR,
            LOADCHANGEMODULE = LOADCHANGEMODULE,
            north_sea = north_sea,
            NoOfScenarios = NoOfScenarios,
            model_type = model_type,
            Seed = Seed,)
    
    
    if solver == "Gurobi":
        opt = SolverFactory('gurobi', Verbose=True)
        opt.options["Crossover"]=0
        opt.options["Method"]=1
    
    start_time = time.time()
    results = opt.solve(instance, tee=True)
    end_time = time.time()
    solving_time = end_time - start_time
    print("Solver Status:", results.solver.status)
    print("Solver Termination Condition:", results.solver.termination_condition)
    print("ML embedded problem Solving time : ", solving_time)

    instance.solutions.load_from(results)
    objective_value = value(instance.Obj)
    second_stage_cost = value(instance.second_stage_value)     
    first_stage_cost_subtracted = objective_value - second_stage_cost

    print(f"Total cost: {objective_value}")
    print(f"First stage cost: {first_stage_cost_subtracted}")
    print(f"Expected second stage cost: {second_stage_cost}")

    output_dir = "Experiments/MLEMBEDSOLS_adaptive"
    
    save_results_x(instance, NoOfScenarios, model_type, Seed, output_dir)
    save_results_v(instance, NoOfScenarios, model_type, Seed, output_dir)

    return solving_time, first_stage_cost_subtracted


def save_results_x(instance, NoSce, model_type, Seed, output_dir):
    gen_inv_cap = instance.genInvCap.get_values()
    transmision_inv_cap = instance.transmisionInvCap.get_values()
    stor_pw_inv_cap = instance.storPWInvCap.get_values()
    stor_en_inv_cap = instance.storENInvCap.get_values()

    gen_inv_cap = {(k[0], k[1], k[2], 'Generation'): v for k, v in gen_inv_cap.items()}
    transmision_inv_cap = {(k[0], k[1], k[2], 'Transmission'): v for k, v in transmision_inv_cap.items()}
    stor_pw_inv_cap = {(k[0], k[1], k[2], 'Storage Power'): v for k, v in stor_pw_inv_cap.items()}
    stor_en_inv_cap = {(k[0], k[1], k[2], 'Storage Energy'): v for k, v in stor_en_inv_cap.items()}

    inv_cap_data = {**gen_inv_cap, **transmision_inv_cap, **stor_pw_inv_cap, **stor_en_inv_cap}

    data = [(k[0], k[1], k[2], k[3], v) for k, v in inv_cap_data.items()]
    df = pd.DataFrame(data, columns=['Node', 'Energy_Type', 'Period', 'Type', 'Value'])
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, f"full_ML_Embed_solution_{model_type}_{NoSce}_{Seed}.csv")
    df.to_csv(output_file_path, index=False)

    print("DataFrames created and saved successfully.")


def save_results_v(instance, NoSce, model_type, Seed, output_dir):
    gen_installed_cap = instance.genInstalledCap.get_values()
    transmision_installed_cap = instance.transmissionInstalledCap.get_values()
    stor_pw_installed_cap = instance.storPWInstalledCap.get_values()
    stor_en_installed_cap = instance.storENInstalledCap.get_values()

    gen_installed_cap = {(k[0], k[1], k[2], 'Generation'): v for k, v in gen_installed_cap.items()}
    transmision_installed_cap = {(k[0], k[1], k[2], 'Transmission'): v for k, v in transmision_installed_cap.items()}
    stor_pw_installed_cap = {(k[0], k[1], k[2], 'Storage Power'): v for k, v in stor_pw_installed_cap.items()}
    stor_en_installed_cap = {(k[0], k[1], k[2], 'Storage Energy'): v for k, v in stor_en_installed_cap.items()}

    installed_cap_data = {**gen_installed_cap, **transmision_installed_cap, **stor_pw_installed_cap, **stor_en_installed_cap}
    data = [(k[0], k[1], k[2], k[3], v) for k, v in installed_cap_data.items()]
    df = pd.DataFrame(data, columns=['Node', 'Energy_Type', 'Period', 'Type', 'Value'])

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"full_ML_Embed_installed_solution_{model_type}_{NoSce}_{Seed}.csv")
    df.to_csv(output_file_path, index=False)


    print("DataFrames created and saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--num_samples', type=int, required=True, help='num_samples')
    parser.add_argument('--seed', type=int, required=True, help='seed') 
    args = parser.parse_args()
    
    ModelType = ["MLP", "LR"]
    num_samples = args.num_samples
    seed = args.seed
    runtime_csv_file = "full_embedding_log.csv"

    for model_type in ModelType:
        print(f"Running {model_type} with {num_samples} samples with {seed} seed")
        runtime, first_stage_cost_subtracted = main(model_type, num_samples, seed)

        log_data = {
            'ModelType': model_type,
            'NoOfSamples': num_samples,
            'NoOfRuns': seed,
            'Runtime(seconds)': runtime
        }
        
        df_log = pd.DataFrame([log_data])
        file_exists = os.path.exists(runtime_csv_file)
        df_log.to_csv(runtime_csv_file, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
        
        print(f"Runtime of {runtime:.2f} seconds logged to '{runtime_csv_file}'\n")

    print(f"All runs completed. Runtimes are saved in '{runtime_csv_file}'.")
