# mip_module.py

import pyomo.environ as pyo
import torch
import joblib
import numpy as np
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from omlt.scaling import OffsetScaling

# ⚠️ 중요: 이 함수는 제공해주신 코드의 EMPIRE 모델 정의 부분을 그대로 가져온 것입니다.
# 이 함수는 1단계 최적화 문제의 변수, 파라미터, 제약조건을 정의합니다.
def build_empire_base_model():
    """
    EMPIRE 모델의 기본 구조(Sets, Parameters, Variables, Constraints)를 정의합니다.
    이 함수는 ML과 관련 없는 순수한 1단계 최적화 모델을 반환합니다.
    """
    model = pyo.AbstractModel()

    # ================================================================= #
    # 제공해주신 코드의 모든 Set, Parameter, Var, Constraint 정의를 여기에 붙여넣습니다.
    # 단, ML 임베딩과 관련된 부분(예: nn_inputs, ml_output 등)은 제외합니다.
    # ================================================================= #
    
    # 예시: 제공해주신 코드에서 일부만 가져옴
    # Sets
    model.Generator = pyo.Set(ordered=True)
    model.Technology = pyo.Set(ordered=True)
    model.Storage =  pyo.Set()
    model.Period = pyo.Set(ordered=True)
    model.PeriodActive = pyo.Set(ordered=True)
    model.Node = pyo.Set(ordered=True)
    model.BidirectionalArc = pyo.Set(dimen=2, ordered=True)
    model.GeneratorsOfNode = pyo.Set(dimen=2)
    model.StoragesOfNode = pyo.Set(dimen=2)
    
    # Parameters
    model.discount_multiplier = pyo.Expression(model.PeriodActive) # rule은 instance 생성 시 적용
    model.genInvCost = pyo.Param(model.Generator, model.Period, default=9000000, mutable=True)
    model.transmissionInvCost = pyo.Param(model.BidirectionalArc, model.Period, default=3000000, mutable=True)
    model.storPWInvCost = pyo.Param(model.Storage, model.Period, default=1000000, mutable=True)
    model.storENInvCost = pyo.Param(model.Storage, model.Period, default=800000, mutable=True)
    # ... (제공해주신 코드의 모든 Parameter 정의)...

    # Variables (1단계 결정 변수 x)
    model.genInvCap = pyo.Var(model.GeneratorsOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.transmisionInvCap = pyo.Var(model.BidirectionalArc, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.storPWInvCap = pyo.Var(model.StoragesOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.storENInvCap = pyo.Var(model.StoragesOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)

    # Variables (중간 변수 v, 1단계 결정 변수에 의해 결정됨)
    model.genInstalledCap = pyo.Var(model.GeneratorsOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.transmissionInstalledCap = pyo.Var(model.BidirectionalArc, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.storPWInstalledCap = pyo.Var(model.StoragesOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.storENInstalledCap = pyo.Var(model.StoragesOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)

    # Constraints (1단계 제약 조건)
    # lifetime_rule_gen, installed_gen_cap_rule 등 ML과 무관한 모든 제약조건을 여기에 정의합니다.
    def lifetime_rule_gen(model, n, g, i):
         # ... (제공해주신 코드 내용) ...
         # 간단한 예시로 대체합니다.
        return model.genInstalledCap[n,g,i] == model.genInvCap[n,g,i] # 실제로는 더 복잡한 수명 관련 수식
    model.installedCapDefinitionGen = pyo.Constraint(model.GeneratorsOfNode, model.PeriodActive, rule=lifetime_rule_gen)
    
    # ... (제공해주신 코드의 모든 제약조건 정의)...

    return model

def _convert_pytorch_to_onnx_with_bounds(pytorch_model, onnx_path, fsd_dim):
    """PyTorch 모델을 ONNX로 변환하고, 입력 경계를 계산하여 파일에 저장합니다."""
    # 더미 입력 생성
    dummy_input = torch.randn(1, fsd_dim, requires_grad=True)

    # ONNX로 모델 export
    torch.onnx.export(pytorch_model, dummy_input, onnx_path, export_params=True,
                      opset_version=12, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])

    # TODO: 입력 변수들의 스케일링된 값에 대한 경계를 계산해야 합니다.
    # 제공해주신 `calculate_bounds_from_original_data` 함수를 활용하거나,
    # 정규화된 데이터의 일반적인 범위(예: -5 ~ 5)를 사용할 수 있습니다.
    # 여기서는 예시로 정규분포의 약 99.99%를 포함하는 범위를 사용합니다.
    scaled_input_bounds = {i: (-5.0, 5.0) for i in range(fsd_dim)}

    # ONNX 파일에 경계 정보 추가
    write_onnx_model_with_bounds(onnx_path, None, scaled_input_bounds)

def _define_feature_mapping_and_scaling(instance, x_scaler_path, pca_path):
    """
    Pyomo 모델 변수와 NN 입력 간의 스케일링/PCA 변환 제약식을 정의합니다.
    """
    x_scaler = joblib.load(x_scaler_path)
    pca = joblib.load(pca_path)
    
    n_features = x_scaler.n_features_in_
    n_pca = pca.n_components_
    
    # 1. 스케일링 및 PCA 변환을 위한 Pyomo 변수 선언
    # v_unscaled: 원본 변수 (genInstalledCap 등)
    # v_scaled: x_scaler로 스케일링된 변수
    # nn_inputs: PCA로 변환된 최종 NN 입력 변수 (OMLT 블록의 입력과 연결됨)
    instance.v_unscaled = pyo.Var(range(n_features), within=pyo.Reals)
    instance.v_scaled = pyo.Var(range(n_features), within=pyo.Reals)

    # 2. 원본 변수(v_unscaled)와 Pyomo 모델 변수(genInstalledCap 등) 연결
    # ⚠️ 중요: 이 순서는 학습 시 사용한 순서와 정확히 일치해야 합니다.
    # 제공해주신 `master_feature_order`와 같은 로직이 필요합니다.
    # 여기서는 `x_scaler.feature_names_in_`를 사용하여 순서를 가정합니다.
    feature_map = {name: i for i, name in enumerate(x_scaler.feature_names_in_)}
    
    # 예시: genInstalledCap['DE', 'Solar', 1] -> v_unscaled[0]
    # 이 부분을 실제 변수 이름과 인덱스에 맞게 동적으로 생성해야 합니다.
    # for var_name, idx in feature_map.items():
    #     # var_name을 파싱하여 실제 pyomo 변수와 연결하는 로직 필요
    #     instance.add_component(f"map_{idx}", pyo.Constraint(expr= ... ))

    # 3. 스케일링 제약식: v_scaled = (v_unscaled - mean) / scale
    mean = x_scaler.mean_
    scale = x_scaler.scale_
    @instance.Constraint(range(n_features))
    def scaling_constraint(m, i):
        return m.v_scaled[i] == (m.v_unscaled[i] - mean[i]) / scale[i]

    # 4. PCA 변환 제약식: nn_inputs = pca.transform(v_scaled)
    # nn_inputs는 omlt 블록의 입력(instance.nn.inputs)과 연결됩니다.
    components = pca.components_
    pca_mean = pca.mean_
    @instance.Constraint(range(n_pca))
    def pca_constraint(m, i):
        # PCA transform: (v_scaled - pca_mean) @ components.T
        pca_expr = sum(components[i, j] * (m.v_scaled[j] - pca_mean[j]) for j in range(n_features))
        return m.nn.inputs[i] == pca_expr

def solve_first_stage_with_nn(pytorch_model):
    """
    1단계 MIP 모델을 구성하고, 학습된 NN을 내장하여 최적해 x*를 찾습니다.
    
    Args:
        pytorch_model (torch.nn.Module): 학습이 완료된 PyTorch 모델
    
    Returns:
        dict: 최적화된 1단계 결정 변수(x)들의 값
    """
    # --- 1. 기본 모델 및 데이터 로드 ---
    model = build_empire_base_model()
    # ⚠️ 실제 데이터 경로로 수정해야 합니다.
    data = pyo.DataPortal(model=model)
    # data.load(...) - 제공해주신 코드의 DataPortal 로드 부분
    instance = model.create_instance(data)

    fsd_dim = pytorch_model.layers[0].in_features
    onnx_path = "temp_model.onnx"
    
    # --- 2. PyTorch -> ONNX 변환 및 OMLT 블록 생성 ---
    _convert_pytorch_to_onnx_with_bounds(pytorch_model, onnx_path, fsd_dim)
    network_definition = load_onnx_neural_network_with_bounds(onnx_path)
    formulation = FullSpaceNNFormulation(network_definition)
    instance.nn = OmltBlock()
    instance.nn.build_formulation(formulation)

    # --- 3. 모델 변수와 NN 입/출력 연결 ---
    # ⚠️ 실제 스케일러와 PCA 객체 경로로 수정해야 합니다.
    x_scaler_path = "train_ML/x_scaler.gz" 
    y_scaler_path = "train_ML/y_scaler.gz"
    pca_path = "train_ML/pca.pkl"

    _define_feature_mapping_and_scaling(instance, x_scaler_path, pca_path)
    
    # 출력 스케일링 역변환
    y_scaler = joblib.load(y_scaler_path)
    y_mean = y_scaler.mean_[0]
    y_scale = y_scaler.scale_[0]
    unscaled_output_expr = instance.nn.outputs[0] * y_scale + y_mean

    # --- 4. 최종 목적함수 정의 ---
    # G(x) + E[Q(x)]
    # G(x)는 1단계 투자 비용으로, build_empire_base_model에서 정의된 비용 관련 식을 합산
    first_stage_cost = sum(instance.discount_multiplier[i] * (
        sum(instance.genInvCost[g,i] * instance.genInvCap[n,g,i] for (n, g) in instance.GeneratorsOfNode)
        # ... + transmissionInvCost + storPWInvCost + storENInvCost
    ) for i in instance.PeriodActive)
    
    instance.total_objective = pyo.Objective(
        expr=first_stage_cost + unscaled_output_expr,
        sense=pyo.minimize
    )
    
    # --- 5. MIP 문제 풀이 ---
    solver = pyo.SolverFactory('gurobi')
    solver.options['TimeLimit'] = 600  # 10분 시간 제한
    results = solver.solve(instance, tee=True)

    # --- 6. 결과 추출 ---
    if (results.solver.termination_condition == pyo.TerminationCondition.optimal or
        results.solver.termination_condition == pyo.TerminationCondition.feasible):
        
        # 1단계 '결정' 변수인 InvCap 변수들의 값을 추출하여 반환
        optimal_x = {}
        for (n, g, i) in instance.genInvCap:
            optimal_x[('genInvCap', n, g, i)] = pyo.value(instance.genInvCap[n,g,i])
        # ... transmisionInvCap, storPWInvCap, storENInvCap에 대해서도 동일하게 추출 ...
        
        return optimal_x
    else:
        print("최적해를 찾지 못했습니다. Solver Status:", results.solver.status)
        return None