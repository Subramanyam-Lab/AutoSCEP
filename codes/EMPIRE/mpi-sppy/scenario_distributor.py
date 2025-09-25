import os
import pandas as pd
import numpy as np # pandas 내부 의존성으로 보통 함께 설치됩니다.

# --- 설정 ---
# 원본 데이터가 있는 기본 폴더
base_path = 'Data handler/scenarios'
# 결과를 저장할 새로운 폴더 이름
output_base_path = 'Data handler/scenarios_output'

# 처리할 디렉토리 및 파일 이름 정의
scenario_dirs = [str(i) for i in range(1020, 1021)]
file_names = [
    'Stochastic_ElectricLoadRaw.tab',
    'Stochastic_HydroGenMaxSeasonalProduction.tab',
    'Stochastic_StochasticAvailability.tab'
]

# 결과 저장 최상위 폴더 생성
os.makedirs(output_base_path, exist_ok=True)
print(f"결과를 '{output_base_path}' 폴더에 저장합니다.")

# 각 원본 시나리오 디렉토리 순회 (1011 ~ 1020)
for dir_name in scenario_dirs:
    input_dir_path = os.path.join(base_path, dir_name)
    output_dir_path = os.path.join(output_base_path, dir_name)
    
    if not os.path.isdir(input_dir_path):
        print(f"Warning: 원본 디렉토리를 찾을 수 없습니다: {input_dir_path}. 건너뜁니다.")
        continue
        
    print(f"\n[{dir_name}] 폴더 처리 중...")

    # 각 파일 순회
    for file_name in file_names:
        input_file_path = os.path.join(input_dir_path, file_name)
        
        if os.path.exists(input_file_path):
            try:
                # .tab 파일 읽기
                df = pd.read_csv(input_file_path, sep='\t')
                
                if 'Scenario' not in df.columns:
                    print(f"  - Warning: 'Scenario' 열이 없어 {file_name} 파일을 건너뜁니다.")
                    continue

                # --- 핵심 변경 사항 1: 숫자 부분만 추출하여 새로운 열 생성 ---
                # 'scenario1' -> 1, 'scenario2' -> 2 등으로 변환
                # to_numeric의 errors='coerce'는 숫자로 변환할 수 없는 값을 NaN으로 만들어 오류를 방지합니다.
                df['scenario_num'] = pd.to_numeric(df['Scenario'].str.replace('scenario', '', case=False), errors='coerce')
                
                # 숫자로 변환되지 않은 행이 있다면 건너뛰기
                if df['scenario_num'].isnull().any():
                    print(f"  - Warning: '{file_name}' 파일의 'Scenario' 열에 숫자 형식이 아닌 값이 있습니다.")
                    # NaN이 아닌 유효한 데이터만으로 계속 진행
                    df.dropna(subset=['scenario_num'], inplace=True)
                    df['scenario_num'] = df['scenario_num'].astype(int)

                # 10개의 시나리오 세트로 분할
                for set_num in range(1, 11):
                    start_scenario = (set_num - 1) * 10 + 1
                    end_scenario = set_num * 10
                    
                    # 숫자 열을 기준으로 데이터 필터링
                    filtered_df = df[(df['scenario_num'] >= start_scenario) & (df['scenario_num'] <= end_scenario)].copy()
                    
                    if not filtered_df.empty:
                        # --- 핵심 변경 사항 2: Scenario 번호 초기화 ---
                        # (원본 시나리오 번호 - 시작 번호 + 1) 로 계산하여 1~10으로 만듦
                        new_scenario_numbers = filtered_df['scenario_num'] - start_scenario + 1
                        filtered_df['Scenario'] = 'scenario' + new_scenario_numbers.astype(str)

                        # 임시로 사용한 숫자 열 삭제
                        final_df = filtered_df.drop(columns=['scenario_num'])
                        
                        # 시나리오 세트별 결과 폴더 생성 및 저장
                        output_set_dir = os.path.join(output_dir_path, f'scenario_set_{set_num}')
                        os.makedirs(output_set_dir, exist_ok=True)
                        output_file_path = os.path.join(output_set_dir, file_name)
                        final_df.to_csv(output_file_path, sep='\t', index=False)
                
                print(f"  - '{file_name}' 파일을 10개 세트로 분할 및 초기화 완료.")

            except Exception as e:
                print(f"  - Error: {input_file_path} 파일 처리 중 오류 발생: {e}")
        else:
            print(f"  - Warning: 파일 없음: {input_file_path}")

print("\n모든 작업이 성공적으로 완료되었습니다! ✅")