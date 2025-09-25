#!/usr/bin/env python
# run_single_job.py

import os
import logging
import argparse
import random
import time
import numpy as np
import pandas as pd
from yaml import safe_load
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import norm
from label_generation_parallel import scenario_folder_generation
from second_stage_label import run_second_stage 

# --- 전역 변수 및 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

# data_labeling_module.py에서 가져온 모든 설정값들
USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
temp_dir = UserRunTimeConfig["temp_dir"]
version = UserRunTimeConfig["version"]
Horizon = UserRunTimeConfig["Horizon"]
discountrate = UserRunTimeConfig["discountrate"]
WACC = UserRunTimeConfig["WACC"]
EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]
LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
north_sea = True if version in ["europe_v51", "europe_reduced_v51"] else False
lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
lengthPeakSeason = 24
NoOfRegSeason = 4
NoOfPeakSeason = 2
regular_seasons = ["winter", "spring", "summer", "fall"]
LeapYearsInvestment = 5

# -------------------------------------------------------------------------
# ### LABELING LOGIC ###
# -------------------------------------------------------------------------

def _batch_estimator(sample_id, period, lengthRegSeason, fsd_df, seeds_to_run, num_workers):
    """
    하나의 작업 내에서 여러 시나리오를 병렬로 처리하는 배치 추정기.
    """
    NoOfScenarios = 1  # 각 시뮬레이션은 단일 시나리오
    results = []
    first_stage_obj = None
    
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
    
    # ProcessPoolExecutor를 사용하여 시나리오들을 병렬로 실행
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for seed in seeds_to_run:
            scenario_folder = scenario_folder_generation(lengthRegSeason, seed)
            
            # FSD를 to_dict('records')로 변환하여 전달
            fsd_records = fsd_df.to_dict('records')
            
            future = executor.submit(
                run_second_stage,
                tab_file_path=scenario_folder,temp_dir=temp_dir,FirstHoursOfRegSeason=FirstHoursOfRegSeason,FirstHoursOfPeakSeason=FirstHoursOfPeakSeason,
                lengthRegSeason=lengthRegSeason,lengthPeakSeason=lengthPeakSeason,Period=Period,Operationalhour=Operationalhour,
                Scenario=Scenario,Season=Season,HoursOfSeason=HoursOfSeason,discountrate=discountrate,WACC=WACC,
                LeapYearsInvestment=LeapYearsInvestment,FSD=fsd_records,EMISSION_CAP=EMISSION_CAP,USE_TEMP_DIR=USE_TEMP_DIR,LOADCHANGEMODULE=LOADCHANGEMODULE,
                seed=seed,specific_period=period,file_num=sample_id,north_sea = north_sea,hour_decision = True,version = version)
            futures[future] = seed
        
        for future in as_completed(futures):
            seed = futures[future]
            try:
                f_obj, s_obj, _ = future.result()
                if first_stage_obj is None:
                    first_stage_obj = f_obj
                results.append(f_obj + s_obj)
            except Exception as e:
                logging.error(f"Seed {seed} for sample {sample_id} failed in batch: {e}", exc_info=True)
                
    if first_stage_obj is None:
        first_stage_obj = 0
        logging.warning(f"Sample {sample_id}의 모든 시나리오가 실패하여 1단계 비용을 0으로 설정합니다.")

    return results, first_stage_obj

def calculate_label_for_fsd_data(fsd_dataframe, sample_id, period, num_workers):
    """
    하나의 FSD 데이터에 대해 2단계 기댓값을 계산합니다.
    """
    seeds_pool = list(range(100, 5000))
    random.shuffle(seeds_pool)
    
    start_time = time.time()
    threshold_r, threshold_h = 0.1, 0.1
    current_lengthRegSeason = 6
    h_increment, initial_num_sce = 6, 5
    z = norm.ppf(0.975)
    MAX_L, MAX_N = 72, 500
    
    used_seeds = [seeds_pool.pop() for _ in range(initial_num_sce)]
    results, first_obj = _batch_estimator(sample_id, period, current_lengthRegSeason, fsd_dataframe, used_seeds, num_workers)
    
    h_is_fixed = False
    
    def coefficient_of_variation(costs):
        mean = np.mean(costs)
        std  = np.std(costs, ddof=1)
        return std / mean if mean else np.inf

    # Adaptive loop
    while True:
        N = len(results)
        if N == 0:
            logging.error("배치 추정기가 결과를 반환하지 못했습니다. 종료합니다.")
            break
        if N >= MAX_N:
            logging.warning(f"시나리오 수 N ({N})이 최대치({MAX_N})에 도달했습니다. 종료합니다.")
            break
        
        mean = np.mean(results)
        std = np.std(results, ddof=1) if N > 1 else 0
        r_error = (z * std / np.sqrt(N)) / mean if mean > 0 else float('inf')
        cv = coefficient_of_variation(results)
        
        logging.info(f"N={N}, L={current_lengthRegSeason}, mean={mean:.4f}, r_error={r_error:.4f}, CV={cv:.4f}")

        if not h_is_fixed and cv > threshold_h:
            current_lengthRegSeason += h_increment
            if current_lengthRegSeason > MAX_L:
                logging.warning(f"다음 시즌 길이({current_lengthRegSeason})가 최대치({MAX_L})를 초과합니다. 종료합니다.")
                break
            logging.info(f"변동성이 너무 큼 (CV={cv:.4f}). 시즌 길이를 {current_lengthRegSeason}으로 늘립니다.")
            results, first_obj = _batch_estimator(sample_id, period, current_lengthRegSeason, fsd_dataframe, used_seeds, num_workers)
            continue
        
        h_is_fixed = True
        
        if r_error > threshold_r:
            needed = int(np.ceil((z * std / (threshold_r * mean))**2)) - N if mean > 0 else initial_num_sce
            needed = max(1, min(needed, len(seeds_pool)))
            if needed == 0:
                logging.info("수렴 기준을 만족했습니다.")
                break
            
            logging.info(f"상대 오차가 너무 큼 (r_error={r_error:.4f}). {needed}개의 시나리오를 추가합니다.")
            more_seeds = [seeds_pool.pop() for _ in range(needed)]
            used_seeds.extend(more_seeds)
            more_results, _ = _batch_estimator(sample_id, period, current_lengthRegSeason, fsd_dataframe, more_seeds, num_workers)
            results.extend(more_results)
            continue
        else:
            logging.info("수렴 기준을 만족했습니다.")
            break
            
    exec_time = time.time() - start_time
    label = np.mean(results) - first_obj if results else 0
    
    return {
        'sample_id': sample_id,
        'period': period,
        'c_i': first_obj,
        'E_Q_i': label,
        'N_scenarios': len(results),
        'final_lengthRegSeason': current_lengthRegSeason,
        'execution_time_sec': exec_time,
    }

# -------------------------------------------------------------------------
# ### MAIN EXECUTION BLOCK ###
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single labeling job for a given FSD.")
    parser.add_argument('--fsd-file', type=str, required=True, help="Path to the FSD CSV file.")
    parser.add_argument('--sample-id', type=int, required=True, help="Unique ID for this FSD sample.")
    parser.add_argument('--period', type=int, required=True, help="The period to calculate the label for.")
    parser.add_argument('--num-cpus', type=int, default=5, help="Number of CPUs for parallel scenario processing within this job.")
    args = parser.parse_args()

    job_start_time = time.time()
    logging.info(f"====== Starting Job for (sample_id={args.sample_id}, period={args.period}) with {args.num_cpus} CPUs ======")

    try:
        # 1. FSD 데이터 파일 읽기
        fsd_df = pd.read_csv(args.fsd_file)
        
        # 2. 메인 계산 함수 호출
        result_data = calculate_label_for_fsd_data(
            fsd_dataframe=fsd_df,
            sample_id=args.sample_id,
            period=args.period,
            num_workers=args.num_cpus
        )
        
        # 3. 고유한 파일 경로에 결과 저장
        output_dir = "job_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_sample_{args.sample_id}_period_{args.period}.csv")

        pd.DataFrame([result_data]).to_csv(output_path, index=False)

        total_time = time.time() - job_start_time
        logging.info(f"SUCCESS: Job ({args.sample_id}, {args.period}) finished in {total_time:.2f}s. Result saved to {output_path}")

    except Exception as e:
        logging.error(f"FAILURE: Job ({args.sample_id}, {args.period}) failed.", exc_info=True)