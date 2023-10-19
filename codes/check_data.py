import pandas as pd
import statsmodels.api as sm

# 데이터 불러오기
data = pd.read_csv('results_10_10.csv')
inputs = data['first stage decision'].apply(eval)  # Assuming the input vectors are stored as strings "[0, 1, 0, ...]"

# 입력 벡터를 여러 컬럼으로 분리
input_df = pd.DataFrame(inputs.tolist(), columns=[f'input_{i}' for i in range(len(inputs[0]))])

# 다중 회귀 분석을 위한 준비
X = input_df
X = sm.add_constant(X)  # 절편(constant) 추가
y = data['expected second stage value']

# 다중 회귀 분석 수행
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())



# import pandas as pd

# # Load the data
# data = pd.read_csv('results_10_10.csv')
# inputs = data['first stage decision'].apply(eval)  # Assuming the input vectors are stored as strings "[0, 1, 0, ...]"
# data['first stage decision'] = inputs

# # Group by unique input values and aggregate their corresponding output values
# grouped = data.groupby('first stage decision').agg({'expected second stage value': list}).reset_index()

# # Print results
# for _, row in grouped.iterrows():
#     input_vector = row['first stage decision']
#     output_values = row['expected second stage value']
#     print(f"Input: {input_vector} | Output Values: {output_values}")
    
# print(f"\nTotal unique input vectors: {len(grouped)}")
