import os
import logging
import torch
import pandas as pd

from nn_module import SimpleNN, train_neural_network
from mip_module import solve_first_stage_with_nn
from data_labeling_module import (
    generate_initial_fsd_samples, 
    generate_targeted_fsd_samples,
    calculate_expected_second_stage_cost
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_ALTERNATING_ITERATIONS = 100
NUM_TARGETED_SAMPLES_PER_ITER = 5
ALPHA = 0.99  # Targeted sampling
INITIAL_SAMPLE_SIZE = 10 

# user defined parameters
FSD_DIM = 616 
NUM_WORKERS_FOR_LABELING = 5 

# File path
OUTPUT_DIR = "alternating_results"
TRAINING_DATA_FILE = os.path.join(OUTPUT_DIR, "training_data.csv")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    logging.info("======= Alternating MIP-NN 2SP Algorithm Start =======")

    # --- 1. 초기 데이터 생성 및 레이블링 ---
    if os.path.exists(TRAINING_DATA_FILE):
        logging.info(f"기존 학습 데이터 파일을 불러옵니다: {TRAINING_DATA_FILE}")
        training_df = pd.read_csv(TRAINING_DATA_FILE)
    else:
        logging.info(f"초기 학습 데이터 {INITIAL_SAMPLE_SIZE}개를 생성합니다...")
        # 1단계 변수(x) 샘플 생성 (0과 1 사이의 임의 값으로 가정)
        initial_fsd_samples = generate_initial_fsd_samples(INITIAL_SAMPLE_SIZE, FSD_DIM)
        
        # 각 샘플에 대해 2단계 문제의 기댓값(레이블) 계산
        # 이 과정이 가장 시간이 많이 소요됩니다.
        labeled_samples = calculate_expected_second_stage_cost(
            fsd_samples=initial_fsd_samples,
            num_workers=NUM_WORKERS_FOR_LABELING
        )
        training_df = pd.DataFrame(labeled_samples)
        training_df.to_csv(TRAINING_DATA_FILE, index=False)
        logging.info("초기 데이터 생성 및 레이블링 완료.")

    exit()
    # --- 2. Alternating Loop 시작 ---
    for k in range(1, NUM_ALTERNATING_ITERATIONS + 1):
        logging.info(f"--- 반복 {k}/{NUM_ALTERNATING_ITERATIONS} 시작 ---")

        # --- 2.1. 신경망 학습 ---
        logging.info("신경망 학습을 시작합니다...")
        
        # 데이터프레임에서 FSD(features)와 E_Q(labels) 분리
        fsd_columns = [col for col in training_df.columns if col.startswith('fsd_')]
        X_train_df = training_df[fsd_columns]
        y_train_df = training_df['E_Q_i']

        # 모델 정의 및 학습
        model = SimpleNN(input_size=FSD_DIM)
        train_neural_network(model, X_train_df, y_train_df, model_save_path=MODEL_SAVE_PATH)
        logging.info(f"신경망 학습 완료. 모델 저장: {MODEL_SAVE_PATH}")

        # --- 2.2. 1단계 MIP 문제 풀이 (NN 내장) ---
        logging.info("학습된 신경망을 내장하여 1단계 MIP 문제를 풉니다...")
        # 학습된 모델 불러오기
        trained_model = SimpleNN(input_size=FSD_DIM)
        trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        
        # MIP 풀이
        # 이 함수는 내부적으로 Pyomo 모델을 만들고, NN을 내장하며, Gurobi로 풉니다.
        optimal_fsd_k = solve_first_stage_with_nn(trained_model)
        
        if optimal_fsd_k is None:
            logging.error("MIP 문제 풀이에 실패했습니다. 알고리즘을 중단합니다.")
            break
        logging.info(f"MIP 문제 풀이 완료. 새로운 해 x_k^*: {optimal_fsd_k}")

        # --- 2.3. 새로운 데이터 샘플링 및 레이블링 ---
        logging.info(f"새로운 해 x_k^* 주변에서 {NUM_TARGETED_SAMPLES_PER_ITER}개의 타겟 샘플을 생성합니다...")
        
        # x_k^* 주변에서 새로운 샘플 생성 (논문의 line 6-10)
        new_fsd_samples = generate_targeted_fsd_samples(
            optimal_fsd_k, NUM_TARGETED_SAMPLES_PER_ITER, FSD_DIM, ALPHA
        )

        logging.info("새로운 샘플에 대한 레이블링을 시작합니다...")
        new_labeled_samples = calculate_expected_second_stage_cost(
            fsd_samples=new_fsd_samples,
            num_workers=NUM_WORKERS_FOR_LABELING
        )
        
        # --- 2.4. 학습 데이터셋 업데이트 ---
        new_training_df = pd.DataFrame(new_labeled_samples)
        training_df = pd.concat([training_df, new_training_df], ignore_index=True)
        training_df.to_csv(TRAINING_DATA_FILE, index=False) # 진행 상황 저장
        
        logging.info(f"학습 데이터셋 업데이트 완료. 현재 데이터 수: {len(training_df)}")

    logging.info("======= Alternating MIP-NN 2SP Algorithm 종료 =======")

if __name__ == "__main__":
    main()