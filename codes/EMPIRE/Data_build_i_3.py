import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import ast

def merge_csv_files_sequential(output_file, input_pattern):
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print("병합할 CSV 파일이 없습니다.")
        return
    
    # 헤더를 비교하기 위한 변수
    first_header = None
    
    with open(output_file, 'w', newline='') as outfile:
        for i, file in enumerate(csv_files):
            with open(file, 'r') as infile:
                header = infile.readline().strip()
                
                if i == 0:
                    # 첫 번째 파일의 헤더를 저장하고 출력 파일에 기록
                    first_header = header
                    outfile.write(header + '\n')
                else:
                    # 헤더 비교
                    if header != first_header:
                        print(f"파일 {file}의 헤더가 일치하지 않습니다. 건너뜁니다.")
                        continue
                
                # 나머지 데이터를 기록
                for line in infile:
                    outfile.write(line)

    print(f"모든 결과가 {output_file}로 병합되었습니다.")


def merge_data_by_file_num(input_file, output_file):
    data = pd.read_csv(input_file)
    merged_data = []
    grouped = data.groupby('file_num')

    for file_num, group in grouped:
        group = group.sort_values(by='period')
        v = ','.join(map(str, group['v_i'].tolist()))
        E_Q = group['E_Q_i'].sum()
        E_Q_ELSE = group['E_Q_i_else'].sum()
        LL_AMT = group['LL_amt'].sum()
        merged_data.append({'file_num': file_num, 'v': v, 'E_Q': E_Q, 'E_Q_ELSE': E_Q_ELSE, 'LL_AMT': LL_AMT})

    merged_df = pd.DataFrame(merged_data)

    merged_df.to_csv(output_file, index=False)
    print(f"병합된 결과가 {output_file} 파일로 저장되었습니다.")

    plot_E_Q_distribution(merged_df)
    plot_E_Q_ELSE_distribution(merged_df)
    plot_LL_AMT_distribution(merged_df)

    return merged_df


def perform_pca_on_v(data, v_output_file, pca_output_file, scaler_file, pca_file):

    # 'v' 컬럼 문자열 -> 파이썬 객체로 변환
    def safe_eval(s):
        try:
            return ast.literal_eval(s)
        except:
            return None  # 혹은 {} 등

    v_data = data['v'].apply(safe_eval)

    v_vectors = []
    for idx, v_element in enumerate(v_data):
        vector = []

        # 먼저 튜플인지 확인
        if isinstance(v_element, tuple):
            for block_dict in v_element:
                # block_dict 예: {1: {"v_i": {...}}}
                if isinstance(block_dict, dict):
                    # 이제 block_dict.values()를 순회하면
                    # {"v_i": {...}} 와 같은 딕셔너리가 나옵니다.
                    for subdict in block_dict.values():
                        # subdict 예: {"v_i": {...}}
                        if isinstance(subdict, dict):
                            v_i = subdict.get('v_i', {})
                            if isinstance(v_i, dict):
                                # genInstalledCap
                                gen_cap = v_i.get('genInstalledCap', {})
                                vector.extend(gen_cap.values())

                                # transmissionInstalledCap
                                trans_cap = v_i.get('transmissionInstalledCap', {})
                                vector.extend(trans_cap.values())

                                # storPWInstalledCap
                                stor_pw_cap = v_i.get('storPWInstalledCap', {})
                                vector.extend(stor_pw_cap.values())

                                # storENInstalledCap
                                stor_en_cap = v_i.get('storENInstalledCap', {})
                                vector.extend(stor_en_cap.values())

        # 예외적으로, 혹시 딕셔너리(단일)인 경우도 있을 수 있으니 체크
        elif isinstance(v_element, dict):
            # 기존 코드처럼 처리
            for subdict in v_element.values():
                if isinstance(subdict, dict):
                    v_i = subdict.get('v_i', {})
                    if isinstance(v_i, dict):
                        vector.extend(v_i.get('genInstalledCap', {}).values())
                        vector.extend(v_i.get('transmissionInstalledCap', {}).values())
                        vector.extend(v_i.get('storPWInstalledCap', {}).values())
                        vector.extend(v_i.get('storENInstalledCap', {}).values())

        v_vectors.append(vector)

    # 디버깅용: 실제 shape 출력
    print(f"v_vectors shape: {len(v_vectors)} x {len(v_vectors[0]) if v_vectors and len(v_vectors[0])>0 else 0}")

    # 이제 스케일링 & PCA
    scaler = StandardScaler()
    v_scaled = scaler.fit_transform(v_vectors)

    # 결과 저장
    v_scl_df = pd.DataFrame(v_scaled, columns=[f'V{i+1}' for i in range(v_scaled.shape[1])])
    v_scl_df['file_num'] = data['file_num']
    v_scl_df['E_Q'] = data['E_Q']
    v_scl_df['E_Q_ELSE'] = data['E_Q_ELSE']
    v_scl_df['LL_AMT'] = data['LL_AMT']
    v_scl_df.to_csv(v_output_file, index=False)
    print(f"V 결과가 {v_output_file} 파일로 저장되었습니다.")

    joblib.dump(scaler, scaler_file)
    print(f"Scaler 객체가 {scaler_file} 파일로 저장되었습니다.")

    pca = PCA(n_components=0.95)
    v_pca = pca.fit_transform(v_scaled)

    print(f"설정된 주성분 개수: {pca.n_components_}")
    print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
    print(f"누적 설명 분산 비율: {np.cumsum(pca.explained_variance_ratio_)}")

    with open(pca_file, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA 객체가 {pca_file} 파일로 저장되었습니다.")

    # Reconstruction Error 계산
    reconstructed = pca.inverse_transform(v_pca)
    reconstruction_error = np.mean((v_scaled - reconstructed) ** 2)
    print(f"Reconstruction Error: {reconstruction_error}")

    # 결과 저장
    pca_df = pd.DataFrame(v_pca, columns=[f'PC{i+1}' for i in range(v_pca.shape[1])])
    pca_df['file_num'] = data['file_num']
    pca_df['E_Q'] = data['E_Q']
    pca_df['E_Q_ELSE'] = data['E_Q_ELSE']
    pca_df['LL_AMT'] = data['LL_AMT']
    pca_df.to_csv(pca_output_file, index=False)
    print(f"PCA 결과가 {pca_output_file} 파일로 저장되었습니다.")


def plot_E_Q_distribution(data):
    plt.hist(data['E_Q'], bins=20, edgecolor='k', alpha=0.7)
    plt.title('Distribution of E_Q')
    plt.xlabel('E_Q')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('scaler_pca/Dist_E_Q2.png')

def plot_E_Q_ELSE_distribution(data):
    plt.hist(data['E_Q_ELSE'], bins=20, edgecolor='k', alpha=0.7)
    plt.title('Distribution of E_Q_ELSE')
    plt.xlabel('E_Q_ELSE')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('scaler_pca/Dist_E_Q_ELSE2.png')


def plot_LL_AMT_distribution(data):
    plt.hist(data['LL_AMT'], bins=20, edgecolor='k', alpha=0.7)
    plt.title('Distribution of LL_AMT')
    plt.xlabel('LL_AMT')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('scaler_pca/Dist_LL_AMT2.png')


if __name__ == "__main__":
    output_file='scaler_pca/master_experiment_results2.csv'
    input_pattern='results_empire2/experiment_results_task_*.csv'
    merged_file='scaler_pca/merged_results2.csv'
    v_output_file='scaler_pca/v_scl_results2.csv'
    pca_output_file='scaler_pca/pca_results2.csv'
    scaler_file='scaler_pca/scaler2.joblib'
    pca_file='scaler_pca/pca2.pkl'

    merge_csv_files_sequential(output_file, input_pattern)
    merged_data = merge_data_by_file_num(output_file, merged_file)
    perform_pca_on_v(merged_data, v_output_file, pca_output_file, scaler_file, pca_file)
