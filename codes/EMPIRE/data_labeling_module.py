# data_labeling_module.py

import os
import logging
import numpy as np
import pandas as pd
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from FSD_sampling_simple2 import sampling, build_preprocessed_data, build_sample_for_checking
from FSD_sampling_violation import run_first_stage
from label_generation_parallel import scenario_folder_generation, run_single_seed, read_fsd_from_csv

import datetime
from yaml import safe_load
UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
temp_dir = UserRunTimeConfig["temp_dir"]
version = UserRunTimeConfig["version"]
Horizon = UserRunTimeConfig["Horizon"]
NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
discountrate = UserRunTimeConfig["discountrate"]
WACC = UserRunTimeConfig["WACC"]
solver = UserRunTimeConfig["solver"]
scenariogeneration = UserRunTimeConfig["scenariogeneration"]
fix_sample = UserRunTimeConfig["fix_sample"]
LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
filter_make = UserRunTimeConfig["filter_make"] 
filter_use = UserRunTimeConfig["filter_use"]
n_cluster = UserRunTimeConfig["n_cluster"]
moment_matching = UserRunTimeConfig["moment_matching"]
n_tree_compare = UserRunTimeConfig["n_tree_compare"]
EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]
IAMC_PRINT = UserRunTimeConfig["IAMC_PRINT"]
WRITE_LP = UserRunTimeConfig["WRITE_LP"]
PICKLE_INSTANCE = UserRunTimeConfig["PICKLE_INSTANCE"] 


#############################
##Non configurable settings##
#############################

NoOfRegSeason = 4
regular_seasons = ["winter", "spring", "summer", "fall"]
NoOfPeakSeason = 2
lengthPeakSeason = 24 # reduced 7
LeapYearsInvestment = 5
time_format = "%d/%m/%Y %H:%M"
if version in ["europe_v51","europe_reduced_v51"]:
    north_sea = True
else:
    north_sea = False


SEED = 42


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
workbook_path = 'Data handler/' + version
tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name + f'_{SEED}'
scenario_data_path = 'Data handler/' + version + '/ScenarioData'
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
dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}
    

def _get_empire_instance():
    data_folder = f'Data handler/base/{version}'
    model,data = run_first_stage(name = name, 
            tab_file_path = data_folder,
            result_file_path = result_file_path, 
            scenariogeneration = scenariogeneration,
            scenario_data_path = scenario_data_path,
            solver = solver,
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
            IAMC_PRINT = IAMC_PRINT, 
            WRITE_LP = WRITE_LP, 
            PICKLE_INSTANCE = PICKLE_INSTANCE, 
            EMISSION_CAP = EMISSION_CAP,
            USE_TEMP_DIR = USE_TEMP_DIR,
            LOADCHANGEMODULE = LOADCHANGEMODULE,
            north_sea = north_sea,
            scenariopath = tab_file_path)
    
    instance = model.create_instance(data)
    return instance


def _get_fsd_variable_order():
    global _CACHED_FSD_VAR_KEYS
    if _CACHED_FSD_VAR_KEYS is not None:
        return _CACHED_FSD_VAR_KEYS

    logging.info("FSD 변수 순서 목록을 생성합니다 (최초 1회 실행)...")
    instance = _get_empire_instance()
    # FSD_sampling_simple2.py의 build_preprocessed_data 함수를 사용하여 변수 목록 추출
    preprocessed_data = build_preprocessed_data(instance)
    
    # 'var_keys'는 변수 식별자 튜플의 리스트입니다. (예: [('genInvCap', (...)), ...])
    var_keys = preprocessed_data['var_keys']
    
    _CACHED_FSD_VAR_KEYS = var_keys
    logging.info(f"FSD 변수 순서가 생성 및 캐시되었습니다. (총 {len(var_keys)}개 변수)")
    return _CACHED_FSD_VAR_KEYS

def _convert_fsd_array_to_dataframe(fsd_array):
    # 1. 고정된 변수 순서 목록을 가져옵니다.
    var_keys = _get_fsd_variable_order()
    
    # 2. 입력된 배열의 크기가 올바른지 확인합니다.
    if len(fsd_array) != len(var_keys):
        raise ValueError(
            f"FSD 배열의 차원({len(fsd_array)})이 모델의 변수 수({len(var_keys)})와 일치하지 않습니다."
        )
    
    # 3. 변수 순서 목록과 값 배열을 딕셔너리로 매핑합니다.
    fsd_dict = {key: value for key, value in zip(var_keys, fsd_array)}
    
    # 4. FSD_sampling_simple2.py의 build_sample_for_checking 함수를 사용하여 DataFrame 생성
    # 이 함수는 ('var_type', ('N', 'G', 'P')) 형태의 key를 받아 DataFrame으로 변환해줍니다.
    fsd_dataframe = build_sample_for_checking(fsd_dict)
    
    return fsd_dataframe



    

def generate_initial_fsd_samples(num_samples, fsd_dim):
    logging.info(f"{num_samples}개의 초기 FSD 샘플 생성을 시작합니다...")
    instance = _get_empire_instance()
    temp_dir = "initial_fsd_samples"
    sampling(
        instance=instance,
        start_sample_num=0,
        base_dir=temp_dir,
        max_attempts=num_samples
    )
    
    fsd_samples = []
    for i in range(1, num_samples + 1):
        file_path = os.path.join(temp_dir, f"sample_{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            fsd_vector = df['Value'].to_numpy()
            if len(fsd_vector) == fsd_dim:
                 fsd_samples.append(fsd_vector)
            else:
                logging.warning(f"샘플 {file_path}의 차원이 {fsd_dim}과 맞지 않아 건너뜁니다.")

    logging.info(f"{len(fsd_samples)}개의 초기 FSD 샘플을 성공적으로 생성했습니다.")
    return fsd_samples

def generate_targeted_fsd_samples(optimal_fsd_k, num_samples, fsd_dim, alpha):
    logging.info(f"{num_samples}개의 타겟 FSD 샘플 생성을 시작합니다...")
    
    # 1. Convex combination을 위한 랜덤 FSD(x_i) 생성
    # generate_initial_fsd_samples와 동일한 로직 사용
    random_feasible_samples = generate_initial_fsd_samples(num_samples, fsd_dim)
    
    if not random_feasible_samples:
        logging.error("타겟 샘플링을 위한 랜덤 FSD 생성에 실패했습니다.")
        return []

    # 2. 타겟 샘플 계산
    targeted_samples = []
    optimal_fsd_arr = np.array(list(optimal_fsd_k.values())) # 딕셔너리를 배열로 변환
    for random_fsd in random_feasible_samples:
        targeted_sample = alpha * optimal_fsd_arr + (1 - alpha) * random_fsd
        targeted_samples.append(targeted_sample)
        
    logging.info(f"{len(targeted_samples)}개의 타겟 FSD 샘플을 생성했습니다.")
    return targeted_samples



# -------------------------------------------------------------------------
# ### 새로운 작업 관리 및 결과 취합 로직 ###
# -------------------------------------------------------------------------

def _prepare_job_inputs(fsd_samples, jobs_input_dir="temp_fsd_for_jobs"):
    """FSD 샘플들을 개별 CSV 파일로 저장하고, 파일 경로를 반환합니다."""
    os.makedirs(jobs_input_dir, exist_ok=True)
    fsd_file_paths = {}
    for i, fsd_sample in enumerate(fsd_samples):
        fsd_df = _convert_fsd_array_to_dataframe(fsd_sample)
        file_path = os.path.join(jobs_input_dir, f"fsd_sample_{i}.csv")
        fsd_df.to_csv(file_path, index=False)
        fsd_file_paths[i] = file_path
    logging.info(f"{len(fsd_samples)}개의 FSD 샘플이 {jobs_input_dir}에 준비되었습니다.")
    return fsd_file_paths

def _run_single_command(command):
    """단일 쉘 명령어를 실행하는 워커 함수."""
    try:
        # shell=True는 보안에 유의해야 하지만, 여기서는 명령어 생성을 제어하므로 사용
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"명령어 실행 실패: {command}\nError: {e.stderr}")
        return False, e.stderr

def _collect_and_aggregate_results(results_dir, num_expected_jobs):
    """결과 디렉토리에서 모든 결과 파일을 읽고, sample_id 기준으로 집계합니다."""
    logging.info("모든 작업이 완료되기를 기다리는 중...")
    
    # 모든 결과 파일이 생성될 때까지 대기
    while True:
        completed_files = os.listdir(results_dir)
        if len(completed_files) >= num_expected_jobs:
            logging.info("모든 작업 결과 파일이 생성되었습니다. 취합을 시작합니다.")
            break
        time.sleep(10) # 10초마다 확인

    all_results = []
    for f_name in os.listdir(results_dir):
        if f_name.endswith(".csv"):
            file_path = os.path.join(results_dir, f_name)
            all_results.append(pd.read_csv(file_path))
    
    if not all_results:
        logging.warning("취합할 결과 파일이 없습니다.")
        return pd.DataFrame()

    # 데이터프레임 집계
    results_df = pd.concat(all_results, ignore_index=True)
    aggregated_df = results_df.groupby('sample_id')[['c_i', 'E_Q_i']].sum().reset_index()
    aggregated_df.rename(columns={'c_i': 'c_i_total', 'E_Q_i': 'E_Q_i_total'}, inplace=True)
    
    return aggregated_df

def calculate_expected_second_stage_cost(fsd_samples, num_workers):
    """
    작업 분산형 워크플로우를 관리합니다.
    1. 작업 준비 -> 2. 병렬 제출 -> 3. 결과 취합 및 집계
    """
    JOBS_INPUT_DIR = "temp_fsd_for_jobs"
    RESULTS_DIR = "job_results"
    
    # 이전 결과 디렉토리 정리
    if os.path.exists(RESULTS_DIR):
        for f in os.listdir(RESULTS_DIR):
            os.remove(os.path.join(RESULTS_DIR, f))
    else:
        os.makedirs(RESULTS_DIR)

    # 1. 작업 준비: FSD 샘플을 파일로 저장
    fsd_file_paths = _prepare_job_inputs(fsd_samples, JOBS_INPUT_DIR)

    # 2. 작업 제출: 실행할 명령어 목록 생성
    commands = []
    ALL_RELEVANT_PERIODS = [i + 1 for i in range(int((Horizon - 2020) / LeapYearsInvestment))]
    CPUS_PER_JOB = 5 # 사용자가 언급한 작업당 CPU 수
    
    for sample_id, fsd_path in fsd_file_paths.items():
        for period in ALL_RELEVANT_PERIODS:
            command = (
                f"python run_single_job.py "
                f"--fsd-file {fsd_path} "
                f"--sample-id {sample_id} "
                f"--period {period} "
                f"--num-cpus {CPUS_PER_JOB}"
            )
            commands.append(command)
    
    logging.info(f"총 {len(commands)}개의 작업을 {num_workers}개의 동시 프로세스로 실행합니다...")
    
    # 로컬에서 병렬로 명령어 실행 (클러스터에서는 이 부분을 sbatch 루프로 대체)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(_run_single_command, commands))

    # 3. 결과 취합 및 집계
    aggregated_df = _collect_and_aggregate_results(RESULTS_DIR, len(commands))

    if aggregated_df.empty:
        return []

    # 최종 결과를 main_alternating.py가 기대하는 형식으로 변환
    fsd_df = pd.DataFrame({
        'sample_id': range(len(fsd_samples)),
        'fsd_sample': fsd_samples
    })
    final_df = pd.merge(fsd_df, aggregated_df, on='sample_id', how='left').fillna(0)

    final_labeled_results = []
    for _, row in final_df.iterrows():
        output = {'id': row['sample_id']}
        fsd_vector = row['fsd_sample']
        output.update({f'fsd_{i}': val for i, val in enumerate(fsd_vector)})
        output['c_i'] = row['c_i_total']
        output['E_Q_i'] = row['E_Q_i_total']
        final_labeled_results.append(output)

    return final_labeled_results



def create_feature_vector_from_scenario(scenario_files_dict):
    try:
        df_load = pd.read_csv(scenario_files_dict['load'], sep='\t')
        df_avail = pd.read_csv(scenario_files_dict['availability'], sep='\t')
        
        all_features = []
        
        load_features = df_load.groupby(['Node', 'Period'])['ElectricLoadRaw_in_MW'].agg(
            ['sum', 'mean', 'max', 'min', 'std']
        ).fillna(0)
        load_features['load_factor'] = (load_features['mean'] / load_features['max']).fillna(0)
        all_features.append(load_features.values.flatten())

        avail_features = df_avail.groupby(['Node', 'Period', 'IntermitentGenerators'])['GeneratorStochasticAvailabilityRaw'].agg(
            ['mean', 'min', 'std']
        ).fillna(0)
        all_features.append(avail_features.values.flatten())

        feature_vector = np.concatenate(all_features)
        
        return feature_vector

    except Exception as e:
        logging.error(f"특징 벡터 생성 중 오류 발생: {e}")
        SCENARIO_FEATURE_DIM = 100 
        return np.zeros(SCENARIO_FEATURE_DIM)