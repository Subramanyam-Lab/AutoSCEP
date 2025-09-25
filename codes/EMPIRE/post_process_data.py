# import os
# import glob
# import pandas as pd
# import numpy as np
# import logging
# import argparse
# import ast  # 문자열을 파이썬 객체(dict, list)로 안전하게 변환

# # 로깅 설정
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    

# def flatten_v_vector(v_string):
#     """
#     문자열 형태의 v_i 딕셔너리를 일관된 순서의 1차원 숫자 리스트로 변환합니다.
#     """
#     try:
#         # 문자열을 실제 딕셔너리로 변환
#         v_dict = ast.literal_eval(v_string)
        
#         # v_dict는 period를 키로 가질 수 있으므로, 내부의 'v_i' 딕셔너리를 추출
#         # 예: {'1': {'v_i': {'genInstalledCap': ...}}}
#         if 'v_i' in next(iter(v_dict.values())):
#              v_dict = next(iter(v_dict.values()))['v_i']

#         flat_list = []
#         # 일관된 순서를 보장하기 위해 최상위 키(genInstalledCap 등)를 정렬
#         for key in sorted(v_dict.keys()):
#             # 각 하위 딕셔너리의 값들을 (키 정렬 후) 순서대로 추가
#             sub_dict = v_dict[key]
#             for sub_key in sorted(sub_dict.keys()):
#                 flat_list.append(sub_dict[sub_key])
#         return flat_list
#     except (SyntaxError, TypeError, ValueError) as e:
#         logging.warning(f"v_i 벡터를 파싱하는 데 실패했습니다: {v_string[:100]}... 오류: {e}")
#         return []

# def main(data_dir, output_file):
#     """
#     데이터를 통합하고 재구성하여 최종 데이터셋을 생성하는 메인 함수
#     """
#     # ## 1. 데이터 로딩 및 통합
#     # ---------------------------------------------------------------------
#     logging.info(f"1. 데이터 로딩 시작: '{data_dir}' 디렉토리에서 데이터를 찾습니다.")
#     all_files = glob.glob(os.path.join(data_dir, 'file_*', 'period_*.csv'))
#     if not all_files:
#         logging.error("데이터 파일을 찾을 수 없습니다.")
#         return

#     df_list = [pd.read_csv(f) for f in all_files]
#     df = pd.concat(df_list, ignore_index=True)
#     logging.info(f"총 {len(df)}개의 (file_num, period) 조합 데이터를 로드했습니다.")


#     # ## 2. 데이터 그룹별 처리 및 재구성
#     # ---------------------------------------------------------------------
#     logging.info("2. file_num 기준으로 데이터 그룹화 및 재구성을 시작합니다.")
    
#     processed_rows = []
    
#     # file_num으로 그룹화
#     for file_num, group_df in df.groupby('file_num'):
        
#         # ★★★ 순서를 보장하기 위해 period 기준으로 정렬 (가장 중요) ★★★
#         group_df = group_df.sort_values('period')
        
#         # v 벡터 처리: 각 period의 v_i를 평탄화하고 순서대로 연결
#         v_vectors = group_df['v_i'].apply(flatten_v_vector).tolist()
#         concatenated_v = [item for sublist in v_vectors for item in sublist]
        
#         try:
#             # avg_features 열의 각 문자열을 실제 리스트로 변환
#             feature_vectors = group_df['avg_features'].apply(ast.literal_eval).tolist()
#         except Exception as e:
#             logging.error(f"file_num {file_num}의 'avg_features' 처리 중 오류 발생: {e}")
            
        
#         # E_Q와 C 값 합산
#         summed_eq = group_df['E_Q_i'].sum()
#         summed_c = group_df['c_i'].sum()
        
#         # 재구성된 데이터를 딕셔너리로 저장
#         row_data = {
#             'file_num': file_num,
#             'v_concatenated': concatenated_v,
#             'E_Q': summed_eq,
#             'C': summed_c
#         }
#         processed_rows.append(row_data)

#     logging.info("데이터 재구성이 완료되었습니다.")

#     # ## 3. 최종 데이터프레임 생성 및 저장
#     # ---------------------------------------------------------------------
#     logging.info("3. 최종 데이터프레임을 생성합니다.")
    
#     # 처리된 행 리스트를 데이터프레임으로 변환
#     final_df = pd.DataFrame(processed_rows)
    
#     # v와 xi 벡터를 개별 컬럼으로 펼치기
#     v_df = pd.DataFrame(final_df['v_concatenated'].tolist()).add_prefix('v_')
    
#     # 최종 데이터프레임 조립
#     output_df = pd.concat([
#         final_df[['file_num']],
#         v_df,
#         final_df[['E_Q', 'C']]
#     ], axis=1)

#     # 최종 데이터셋 저장
#     output_df.to_csv(output_file, index=False)
#     logging.info(f"✅ 최종 데이터셋이 '{output_file}'에 성공적으로 저장되었습니다. (총 {len(output_df)}개 샘플)")
#     logging.info(f"최종 데이터셋 형태: {output_df.shape}")
#     logging.info(f"최종 데이터셋 컬럼 미리보기: {output_df.columns.tolist()[:5]}...{output_df.columns.tolist()[-5:]}")

import os
import glob
import pandas as pd
import numpy as np
import logging
import argparse
import ast
import re # file_num을 추출하기 위해 추가

# 로깅 설정 (기존과 동일)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# flatten_v_vector 함수 (기존과 동일)
def flatten_v_vector(v_string):
    # ... (내용은 변경 없음)
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
        logging.warning(f"v_i 벡터를 파싱하는 데 실패했습니다: {v_string[:100]}... 오류: {e}")
        return []

def main(data_dir, output_file):
    """
    데이터를 통합하고 재구성하여 최종 데이터셋을 생성하는 메인 함수
    (점진적 처리 및 헤더 자동 복구 기능 포함)
    """
    # ## 1. 처리할 작업 목록 탐색
    # ---------------------------------------------------------------------
    logging.info(f"1. 처리할 작업 목록 탐색: '{data_dir}' 디렉토리에서 'file_*' 폴더를 찾습니다.")
    file_dirs = glob.glob(os.path.join(data_dir, 'file_*'))
    if not file_dirs:
        logging.error("처리할 'file_*' 디렉토리를 찾을 수 없습니다.")
        return
        
    file_nums_to_process = sorted([int(re.findall(r'\d+', os.path.basename(d))[0]) for d in file_dirs])
    logging.info(f"총 {len(file_nums_to_process)}개의 file_num을 처리합니다.")

    # --- 1a. 올바른 헤더 정보 미리 저장하기 ---
    try:
        # 정상 파일이 확실한 첫 번째 file_num의 CSV 경로를 찾음
        first_file_dir = os.path.join(data_dir, f'file_{file_nums_to_process[0]}')
        any_csv_in_first_dir = glob.glob(os.path.join(first_file_dir, 'period_*.csv'))[0]
        
        # 정상 파일의 헤더를 읽어서 correct_header 변수에 저장
        sample_df = pd.read_csv(any_csv_in_first_dir)
        correct_header = sample_df.columns.tolist()
        logging.info(f"기준 헤더를 설정했습니다: {correct_header}")
    except IndexError:
        logging.error("기준 헤더를 가져올 CSV 파일을 찾을 수 없습니다. 처리를 중단합니다.")
        return
        
    # ## 2. file_num 단위로 순회하며 데이터 처리 및 재구성
    # ---------------------------------------------------------------------
    logging.info("2. file_num 기준으로 데이터를 순차적으로 처리합니다.")
    
    processed_rows = []
    
    for i, file_num in enumerate(file_nums_to_process):
        if (i + 1) % 100 == 0: # 100개 처리할 때마다 진행 상황 로깅
            logging.info(f"진행 상황: {i+1} / {len(file_nums_to_process)} (file_num: {file_num})")

        file_path_pattern = os.path.join(data_dir, f'file_{file_num}', 'period_*.csv')
        period_files = glob.glob(file_path_pattern)

        if not period_files:
            logging.warning(f"file_num {file_num}에 해당하는 period 파일을 찾을 수 없어 건너뜁니다.")
            continue
            
        group_df_list = []
        for f in period_files:
            try:
                # --- 2a. 파일의 첫 줄을 읽어 헤더 유무 판별 ---
                with open(f, 'r') as temp_f:
                    first_line = temp_f.readline()

                # --- 2b. 조건에 따라 다르게 파일 읽기 ---
                if 'period' in first_line:
                    # 헤더가 있는 경우
                    temp_df = pd.read_csv(f)
                else:
                    # 헤더가 없는 경우, 저장해둔 correct_header를 컬럼 이름으로 지정
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

        # --- 2c. 작은 데이터프레임으로 결합 후 처리 ---
        group_df = pd.concat(group_df_list, ignore_index=True)
        
        # ★★★ 순서를 보장하기 위해 period 기준으로 정렬 (가장 중요) ★★★
        group_df = group_df.sort_values('period')
        
        # v 벡터 처리
        v_vectors = group_df['v_i'].apply(flatten_v_vector).tolist()
        concatenated_v = [item for sublist in v_vectors for item in sublist]
        
        # avg_features 처리
        try:
            feature_vectors = group_df['avg_features'].apply(ast.literal_eval).tolist()
        except Exception as e:
            logging.error(f"file_num {file_num}의 'avg_features' 처리 중 오류 발생: {e}")
            continue
        
        # E_Q와 C 값 합산
        summed_eq = group_df['E_Q_i'].sum()
        summed_c = group_df['c_i'].sum()
        
        # 재구성된 데이터를 딕셔너리로 저장
        row_data = {
            'file_num': file_num,
            'v_concatenated': concatenated_v,
            'E_Q': summed_eq,
            'C': summed_c
        }
        processed_rows.append(row_data)

    logging.info("데이터 재구성이 완료되었습니다.")

    # ## 3. 최종 데이터프레임 생성 및 저장
    # ---------------------------------------------------------------------
    logging.info("3. 최종 데이터프레임을 생성합니다.")
    
    if not processed_rows:
        logging.warning("처리된 데이터가 없어 빈 파일을 생성합니다.")
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
    logging.info(f"✅ 최종 데이터셋이 '{output_file}'에 성공적으로 저장되었습니다. (총 {len(output_df)}개 샘플)")
    logging.info(f"최종 데이터셋 형태: {output_df.shape}")
    logging.info(f"최종 데이터셋 컬럼 미리보기: {output_df.columns.tolist()[:5]}...{output_df.columns.tolist()[-5:]}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="기간별로 분리된 데이터를 file_num 기준으로 통합합니다.")
    parser.add_argument('--data_dir', type=str, default="training_data_distributed_fixed",
                        help="개별 데이터 파일들이 저장된 상위 디렉토리")
    parser.add_argument('--output_file', type=str, default="aggregated_dataset_fixed.csv",
                        help="최종 통합된 데이터셋을 저장할 CSV 파일 경로")
    
    args = parser.parse_args()
    main(args.data_dir, args.output_file)