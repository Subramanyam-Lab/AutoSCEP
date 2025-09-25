# nn_module.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from label_generation_parallel import scenario_folder_generation
from data_labeling_module import create_feature_vector_from_scenario

# --- NN-E 모델의 구성 요소 ---

class Psi1_ScenarioEncoder(nn.Module):
    """Ψ¹: 각 시나리오의 특징 벡터를 저차원 임베딩으로 압축"""
    def __init__(self, feature_size, embedding_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.Tanh(),  # ReLU가 아닌 활성화 함수도 사용 가능
            nn.Linear(64, embedding_size)
        )
    def forward(self, x):
        return self.layers(x)

class Psi2_PostAggregation(nn.Module):
    """Ψ²: 집계된 임베딩을 최종 시나리오 대표 벡터로 변환"""
    def __init__(self, embedding_size, final_embedding_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.Tanh(),
            nn.Linear(32, final_embedding_size)
        )
    def forward(self, x):
        return self.layers(x)

class PhiE_FinalPredictor(nn.Module):
    """Φ^E: MIP에 내장될 최종 예측기 (반드시 ReLU만 사용)"""
    def __init__(self, fsd_dim, final_embedding_size):
        super().__init__()
        input_dim = fsd_dim + final_embedding_size
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )
    def forward(self, x):
        return self.layers(x)

# --- 전체 NN-E 모델 ---

class NNE_Model(nn.Module):
    """세 가지 구성 요소를 결합한 전체 NN-E 모델"""
    def __init__(self, fsd_dim, feature_size, embedding_size=32, final_embedding_size=16):
        super().__init__()
        self.psi1 = Psi1_ScenarioEncoder(feature_size, embedding_size)
        self.psi2 = Psi2_PostAggregation(embedding_size, final_embedding_size)
        self.phiE = PhiE_FinalPredictor(fsd_dim, final_embedding_size)

    def forward(self, x_fsd, x_scenarios):
        # x_scenarios의 shape: (batch_size, num_scenarios, feature_size)
        
        # 1. 각 시나리오를 Ψ¹으로 인코딩
        # (batch_size * num_scenarios, feature_size)로 형태 변경
        batch_size, num_scenarios, feature_size = x_scenarios.shape
        scenarios_flat = x_scenarios.view(-1, feature_size)
        scenario_embeddings = self.psi1(scenarios_flat)
        # (batch_size, num_scenarios, embedding_size)로 복원
        scenario_embeddings = scenario_embeddings.view(batch_size, num_scenarios, -1)

        # 2. Mean Aggregation
        aggregated_embedding = torch.mean(scenario_embeddings, dim=1) # (batch_size, embedding_size)

        # 3. Ψ²를 통과시켜 최종 시나리오 대표 벡터 생성
        xi_lambda = self.psi2(aggregated_embedding) # (batch_size, final_embedding_size)

        # 4. 1단계 변수와 시나리오 대표 벡터 결합
        combined_input = torch.cat([x_fsd, xi_lambda], dim=1)

        # 5. Φ^E로 최종 비용 예측
        prediction = self.phiE(combined_input)
        
        return prediction

def train_neural_network(model, training_df, model_save_path, phiE_save_path, epochs=100, lr=0.001, batch_size=32):
    """NN-E 모델을 학습시키는 수정된 함수"""
    
    # ⚠️ training_df를 받아 ScenarioDataset을 생성해야 합니다.
    dataset = ScenarioDataset(training_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- Placeholder: 실제로는 위 주석의 DataLoader를 사용해야 합니다. ---
    # 여기서는 더미 데이터로 학습 과정을 보여줍니다.
    fsd_dim = model.phiE.layers[0].in_features - model.psi2.layers[-1].out_features
    feature_size = model.psi1.layers[0].in_features
    dummy_fsd = torch.randn(len(training_df), fsd_dim)
    dummy_scenarios = torch.randn(len(training_df), 50, feature_size) # 50개 시나리오 가정
    dummy_labels = torch.randn(len(training_df), 1)
    dataset = torch.utils.data.TensorDataset(dummy_fsd, dummy_scenarios, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # --- Placeholder 종료 ---

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for fsd_vector, scenarios, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(fsd_vector, scenarios)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # ⚠️ 중요: 전체 모델과 MIP에 내장할 PhiE 모델을 별도로 저장
    torch.save(model.state_dict(), model_save_path)
    torch.save(model.phiE.state_dict(), phiE_save_path)
    

import torch
from torch.utils.data import Dataset, DataLoader

class ScenarioDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        # FSD, E_Q, used_seeds 컬럼을 미리 추출하여 속도 향상
        self.fsd_cols = [col for col in self.df.columns if col.startswith('fsd_')]
        self.fsd_vectors = torch.tensor(self.df[self.fsd_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(self.df['E_Q_i_total'].values, dtype=torch.float32).view(-1, 1)
        self.seed_lists = self.df['used_seeds'].apply(lambda x: [int(s) for s in str(x).split(',')]).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fsd_vector = self.fsd_vectors[idx]
        label = self.labels[idx]
        seeds = self.seed_lists[idx]
        
        scenario_feature_vectors = []
        for seed in seeds:
            scenario_folder = scenario_folder_generation(lengthRegSeason, seed)
            scenario_files = {
                'load': os.path.join(scenario_folder, 'Stochastic_ElectricLoadRaw.tab'),
                'availability': os.path.join(scenario_folder, 'Stochastic_StochasticAvailability.tab')
            }
            feature_vector = create_feature_vector_from_scenario(scenario_files)
            
            # --- Placeholder (실제 함수로 대체 필요) ---
            SCENARIO_FEATURE_DIM = 100 # 예시 값
            feature_vector = np.random.rand(SCENARIO_FEATURE_DIM)
            # --- Placeholder 종료 ---

            scenario_feature_vectors.append(feature_vector)
        
        # 생성된 특징 벡터들을 numpy 배열로 변환
        scenarios = np.array(scenario_feature_vectors, dtype=np.float32)
        
        return fsd_vector, scenarios, label

def scenario_collate_fn(batch):
    fsd_batch, scenarios_batch, labels_batch = zip(*batch)
    
    # FSD와 레이블은 크기가 고정되어 있으므로 간단히 stack 가능
    fsd_tensors = torch.stack(fsd_batch)
    labels_tensors = torch.stack(labels_batch)
    
    # 시나리오 배치의 최대 시나리오 개수(N) 찾기
    max_n_scenarios = max(s.shape[0] for s in scenarios_batch)
    scenario_feature_dim = scenarios_batch[0].shape[1]
    
    # 패딩을 적용할 텐서 생성 (batch_size, max_N, feature_dim)
    padded_scenarios = torch.zeros(len(batch), max_n_scenarios, scenario_feature_dim)
    
    # 각 샘플의 시나리오를 패딩 텐서에 복사
    for i, scenarios in enumerate(scenarios_batch):
        n_scenarios = scenarios.shape[0]
        padded_scenarios[i, :n_scenarios, :] = torch.from_numpy(scenarios)
        
    return fsd_tensors, padded_scenarios, labels_tensors