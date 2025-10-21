from __future__ import division
from pyomo.environ import *
import joblib
import numpy as np
from omlt import OmltBlock
from omlt.io.onnx import (
    load_onnx_neural_network_with_bounds,
)
from omlt.neuralnet import FullSpaceNNFormulation
from first_stage import run_first_stage


########################################### LR #########################################################

def embed_linear_regression_in_pyomo(abstract_model, lr_model, input_vars, output_var_name, scaler_y):

    model = abstract_model
    output_var = Var(domain=Reals, name=output_var_name)
    model.add_component(output_var_name, output_var)
    
    intercept = lr_model.intercept_
    if isinstance(intercept, (np.ndarray, list)):
        intercept = intercept[0]

    coef_vector = lr_model.coef_.flatten()
    target_mean = scaler_y.mean_[0]
    target_std = scaler_y.scale_[0]
    
    def linear_pred_constraint_rule(m):
        linear_expr = intercept + sum(coef_vector[j] * input_vars[j] for j in range(len(coef_vector)))
        return m.component(output_var_name) == (linear_expr * target_std + target_mean)
    
    constraint_name = f"linear_regression_constraint"
    model.add_component(constraint_name, Constraint(rule=linear_pred_constraint_rule))
    
    return model


def integrate_linear_regression_with_pyomo_model(lr_model_path, instance, output_var_name, scaler_y):
    lr_model = joblib.load(lr_model_path)
    n_features = len(list(instance.v_scaled.keys()))
    input_vars = [instance.v_scaled[i] for i in range(1, n_features+1)]
    instance = embed_linear_regression_in_pyomo(instance, lr_model, input_vars, output_var_name, scaler_y)
    return instance



########################################### MLP #########################################################


from omlt import OmltBlock
from omlt.io.onnx import (
    load_onnx_neural_network_with_bounds,
)
from omlt.neuralnet import FullSpaceNNFormulation

def embed_omlt_neural_network(instance, onnx_path, n_features, y_scaler):
    net = load_onnx_neural_network_with_bounds(onnx_path)
    
    instance.nn = OmltBlock()
    formulation = FullSpaceNNFormulation(net)
    instance.nn.build_formulation(formulation)

    @instance.Constraint(RangeSet(1,n_features))
    def connect_nn_inputs(m, i):
        return m.v_scaled[i] == m.nn.inputs[i-1]

    instance.second_stage_value = Var(domain=Reals)
    target_mean = y_scaler.mean_[0]
    target_std = y_scaler.scale_[0]

    @instance.Constraint()
    def connect_nn_output(m):
        return m.second_stage_value == (m.nn.outputs[0] * target_std + target_mean)
        
    return instance

########################################### MLP #########################################################



def order_capacity_indices(instance, n_features, data_mean, data_std, version):
    master_feature_order = []

    if version in ["reduced"]:
        training_data_order ={
            "genInstalledCap": [
                ("Denmark","Bio10cofiring"), ("Denmark","Bio10cofiringCCS"), ("Denmark","Bioexisting"), 
                ("Denmark","Coal"), ("Denmark","CoalCCSadv"), ("Denmark","Coalexisting"),
                ("Denmark","GasCCGT"), ("Denmark","GasCCSadv"), ("Denmark","GasOCGT"),
                ("Denmark","Gasexisting"), ("Denmark","Hydrorun-of-the-river"), ("Denmark","Oilexisting"),
                ("Denmark","Solar"), ("Denmark","Waste"), ("Denmark","Windoffshore"),
                ("Denmark","Windonshore"), ("France","Bio"), ("France","Bio10cofiring"),
                ("France","Bio10cofiringCCS"), ("France","Bioexisting"), ("France","Coal"),
                ("France","CoalCCS"),("France","CoalCCSadv"), ("France","Coalexisting"), ("France","GasCCGT"),
                ("France","GasCCS"),("France","GasCCSadv"), ("France","GasOCGT"), ("France","Gasexisting"),
                ("France","Geo"), ("France","Hydroregulated"), ("France","Hydrorun-of-the-river"),
                ("France","Nuclear"), ("France","Oilexisting"), ("France","Solar"),
                ("France","Waste"), ("France","Wave"), ("France","Windoffshore"),
                ("France","Windonshore"), ("Germany","Bio"), ("Germany","Bio10cofiring"),
                ("Germany","Bio10cofiringCCS"), ("Germany","Bioexisting"), ("Germany","Coal"),
                ("Germany","CoalCCS"), ("Germany","CoalCCSadv"), ("Germany","Coalexisting"), ("Germany","GasCCGT"), ("Germany","GasCCS"),
                ("Germany","GasCCSadv"), ("Germany","GasOCGT"), ("Germany","Gasexisting"),
                ("Germany","Geo"), ("Germany","Hydroregulated"), ("Germany","Hydrorun-of-the-river"),
                ("Germany","Liginiteexisting"), ("Germany","Lignite"), ("Germany","LigniteCCSadv"),
                ("Germany","LigniteCCSsup"), ("Germany","Nuclear"), ("Germany","Oilexisting"),
                ("Germany","Solar"), ("Germany","Waste"), ("Germany","Windoffshore"),
                ("Germany","Windonshore")
            ],
            "storENInstalledCap": [
                ("Denmark","Li-Ion_BESS"), ("France","HydroPumpStorage"), ("France","Li-Ion_BESS"),  
                ("Germany","HydroPumpStorage"), ("Germany","Li-Ion_BESS")
            ],
            "storPWInstalledCap": [
                ("Denmark","Li-Ion_BESS"), ("France","HydroPumpStorage"), ("France","Li-Ion_BESS"),  
                ("Germany","HydroPumpStorage"), ("Germany","Li-Ion_BESS")
            ],
            "transmissionInstalledCap": [
                ("Denmark","Germany"), ("France","Germany")
            ]    
            
        }
    
    else:
        training_data_order ={ "genInstalledCap": [('Austria', 'Bio'), ('Austria', 'Bio10cofiring'), ('Austria', 'Bio10cofiringCCS'), ('Austria', 'Bioexisting'), ('Austria', 'Coal'), ('Austria', 'CoalCCSadv'), ('Austria', 'Coalexisting'), ('Austria', 'GasCCGT'), ('Austria', 'GasCCSadv'), ('Austria', 'GasOCGT'), ('Austria', 'Gasexisting'), ('Austria', 'Geo'), ('Austria', 'Hydroregulated'), ('Austria', 'Hydrorun-of-the-river'), ('Austria', 'Oilexisting'), ('Austria', 'Solar'), ('Austria', 'Waste'), ('Austria', 'Windonshore'), ('Belgium', 'Bio'), ('Belgium', 'Bio10cofiring'), ('Belgium', 'Bio10cofiringCCS'), ('Belgium', 'Bioexisting'), ('Belgium', 'Coal'), ('Belgium', 'CoalCCSadv'), ('Belgium', 'GasCCGT'), ('Belgium', 'GasCCSadv'), ('Belgium', 'GasOCGT'), ('Belgium', 'Gasexisting'), ('Belgium', 'Hydrorun-of-the-river'), ('Belgium', 'Nuclear'), ('Belgium', 'Oilexisting'), ('Belgium', 'Solar'), ('Belgium', 'Waste'), ('Belgium', 'Windoffshore'), ('Belgium', 'Windonshore'), ('BosniaH', 'Bio'), ('BosniaH', 'Bioexisting'), ('BosniaH', 'Coal'), ('BosniaH', 'CoalCCSadv'), ('BosniaH', 'Coalexisting'), ('BosniaH', 'Hydroregulated'), ('BosniaH', 'Liginiteexisting'), ('BosniaH', 'Lignite'), ('BosniaH', 'LigniteCCSadv'), ('BosniaH', 'LigniteCCSsup'), ('BosniaH', 'Oilexisting'), ('BosniaH', 'Solar'), ('BosniaH', 'Waste'), ('BosniaH', 'Windonshore'), ('Bulgaria', 'Bio'), ('Bulgaria', 'Bio10cofiring'), ('Bulgaria', 'Bio10cofiringCCS'), ('Bulgaria', 'Bioexisting'), ('Bulgaria', 'Coal'), ('Bulgaria', 'CoalCCSadv'), ('Bulgaria', 'Coalexisting'), ('Bulgaria', 'GasCCGT'), ('Bulgaria', 'GasCCSadv'), ('Bulgaria', 'GasOCGT'), ('Bulgaria', 'Gasexisting'), ('Bulgaria', 'Hydroregulated'), ('Bulgaria', 'Hydrorun-of-the-river'), ('Bulgaria', 'Liginiteexisting'), ('Bulgaria', 'Lignite'), ('Bulgaria', 'LigniteCCSadv'), ('Bulgaria', 'Nuclear'), ('Bulgaria', 'Oilexisting'), ('Bulgaria', 'Solar'), ('Bulgaria', 'Waste'), ('Bulgaria', 'Windonshore'), ('Croatia', 'Bio'), ('Croatia', 'Bio10cofiring'), ('Croatia', 'Bio10cofiringCCS'), ('Croatia', 'Bioexisting'), ('Croatia', 'Coal'), ('Croatia', 'CoalCCSadv'), ('Croatia', 'Coalexisting'), ('Croatia', 'GasCCGT'), ('Croatia', 'GasCCSadv'), ('Croatia', 'GasOCGT'), ('Croatia', 'Gasexisting'), ('Croatia', 'Hydroregulated'), ('Croatia', 'Hydrorun-of-the-river'), ('Croatia', 'Oilexisting'), ('Croatia', 'Solar'), ('Croatia', 'Waste'), ('Croatia', 'Windonshore'), ('CzechR', 'Bio'), ('CzechR', 'Bio10cofiring'), ('CzechR', 'Bio10cofiringCCS'), ('CzechR', 'Bioexisting'), ('CzechR', 'Coal'), ('CzechR', 'CoalCCSadv'), ('CzechR', 'Coalexisting'), ('CzechR', 'GasCCGT'), ('CzechR', 'GasCCSadv'), ('CzechR', 'GasOCGT'), ('CzechR', 'Gasexisting'), ('CzechR', 'Geo'), ('CzechR', 'Hydroregulated'), ('CzechR', 'Hydrorun-of-the-river'), ('CzechR', 'Liginiteexisting'), ('CzechR', 'Lignite'), ('CzechR', 'LigniteCCSadv'), ('CzechR', 'Nuclear'), ('CzechR', 'Oilexisting'), ('CzechR', 'Solar'), ('CzechR', 'Waste'), ('CzechR', 'Windonshore'), ('Denmark', 'Bio10cofiring'), ('Denmark', 'Bio10cofiringCCS'), ('Denmark', 'Bioexisting'), ('Denmark', 'Coal'), ('Denmark', 'CoalCCSadv'), ('Denmark', 'Coalexisting'), ('Denmark', 'GasCCGT'), ('Denmark', 'GasCCSadv'), ('Denmark', 'GasOCGT'), ('Denmark', 'Gasexisting'), ('Denmark', 'Hydrorun-of-the-river'), ('Denmark', 'Oilexisting'), ('Denmark', 'Solar'), ('Denmark', 'Waste'), ('Denmark', 'Windoffshore'), ('Denmark', 'Windonshore'), ('Estonia', 'Bio10cofiring'), ('Estonia', 'Bio10cofiringCCS'), ('Estonia', 'Bioexisting'), ('Estonia', 'GasCCGT'), ('Estonia', 'GasCCSadv'), ('Estonia', 'GasOCGT'), ('Estonia', 'Gasexisting'), ('Estonia', 'Hydrorun-of-the-river'), ('Estonia', 'Liginiteexisting'), ('Estonia', 'Lignite'), ('Estonia', 'LigniteCCSadv'), ('Estonia', 'Oilexisting'), ('Estonia', 'Solar'), ('Estonia', 'Waste'), ('Estonia', 'Windonshore'), ('Finland', 'Bio10cofiring'), ('Finland', 'Bio10cofiringCCS'), ('Finland', 'Bioexisting'), ('Finland', 'Coal'), ('Finland', 'CoalCCSadv'), ('Finland', 'Coalexisting'), ('Finland', 'GasCCGT'), ('Finland', 'GasCCSadv'), ('Finland', 'GasOCGT'), ('Finland', 'Gasexisting'), ('Finland', 'Hydrorun-of-the-river'), ('Finland', 'Nuclear'), ('Finland', 'Oilexisting'), ('Finland', 'Solar'), ('Finland', 'Waste'), ('Finland', 'Wave'), ('Finland', 'Windoffshore'), ('Finland', 'Windonshore'), ('France', 'Bio'), ('France', 'Bio10cofiring'), ('France', 'Bio10cofiringCCS'), ('France', 'Bioexisting'), ('France', 'Coal'), ('France', 'CoalCCS'), ('France', 'CoalCCSadv'), ('France', 'Coalexisting'), ('France', 'GasCCGT'), ('France', 'GasCCS'), ('France', 'GasCCSadv'), ('France', 'GasOCGT'), ('France', 'Gasexisting'), ('France', 'Geo'), ('France', 'Hydroregulated'), ('France', 'Hydrorun-of-the-river'), ('France', 'Nuclear'), ('France', 'Oilexisting'), ('France', 'Solar'), ('France', 'Waste'), ('France', 'Wave'), ('France', 'Windoffshore'), ('France', 'Windonshore'), ('Germany', 'Bio'), ('Germany', 'Bio10cofiring'), ('Germany', 'Bio10cofiringCCS'), ('Germany', 'Bioexisting'), ('Germany', 'Coal'), ('Germany', 'CoalCCS'), ('Germany', 'CoalCCSadv'), ('Germany', 'Coalexisting'), ('Germany', 'GasCCGT'), ('Germany', 'GasCCS'), ('Germany', 'GasCCSadv'), ('Germany', 'GasOCGT'), ('Germany', 'Gasexisting'), ('Germany', 'Geo'), ('Germany', 'Hydroregulated'), ('Germany', 'Hydrorun-of-the-river'), ('Germany', 'Liginiteexisting'), ('Germany', 'Lignite'), ('Germany', 'LigniteCCSadv'), ('Germany', 'LigniteCCSsup'), ('Germany', 'Nuclear'), ('Germany', 'Oilexisting'), ('Germany', 'Solar'), ('Germany', 'Waste'), ('Germany', 'Windoffshore'), ('Germany', 'Windonshore'), ('GreatBrit.', 'Bio'), ('GreatBrit.', 'Bio10cofiring'), ('GreatBrit.', 'Bio10cofiringCCS'), ('GreatBrit.', 'Bioexisting'), ('GreatBrit.', 'Coal'), ('GreatBrit.', 'CoalCCS'), ('GreatBrit.', 'CoalCCSadv'), ('GreatBrit.', 'Coalexisting'), ('GreatBrit.', 'GasCCGT'), ('GreatBrit.', 'GasCCS'), ('GreatBrit.', 'GasCCSadv'), ('GreatBrit.', 'GasOCGT'), ('GreatBrit.', 'Gasexisting'), ('GreatBrit.', 'Hydroregulated'), ('GreatBrit.', 'Hydrorun-of-the-river'), ('GreatBrit.', 'Nuclear'), ('GreatBrit.', 'Oilexisting'), ('GreatBrit.', 'Solar'), ('GreatBrit.', 'Waste'), ('GreatBrit.', 'Wave'), ('GreatBrit.', 'Windoffshore'), ('GreatBrit.', 'Windonshore'), ('Greece', 'Bio'), ('Greece', 'Bio10cofiring'), ('Greece', 'Bio10cofiringCCS'), ('Greece', 'Bioexisting'), ('Greece', 'GasCCGT'), ('Greece', 'GasCCSadv'), ('Greece', 'GasOCGT'), ('Greece', 'Gasexisting'), ('Greece', 'Geo'), ('Greece', 'Hydroregulated'), ('Greece', 'Hydrorun-of-the-river'), ('Greece', 'Liginiteexisting'), ('Greece', 'Lignite'), ('Greece', 'LigniteCCSadv'), ('Greece', 'Solar'), ('Greece', 'Waste'), ('Greece', 'Windonshore'), ('Hungary', 'Bio'), ('Hungary', 'Bio10cofiring'), ('Hungary', 'Bio10cofiringCCS'), ('Hungary', 'Bioexisting'), ('Hungary', 'Coal'), ('Hungary', 'CoalCCSadv'), ('Hungary', 'Coalexisting'), ('Hungary', 'GasCCGT'), ('Hungary', 'GasCCSadv'), ('Hungary', 'GasOCGT'), ('Hungary', 'Gasexisting'), ('Hungary', 'Geo'), ('Hungary', 'Hydroregulated'), ('Hungary', 'Hydrorun-of-the-river'), ('Hungary', 'Liginiteexisting'), ('Hungary', 'Lignite'), ('Hungary', 'LigniteCCSadv'), ('Hungary', 'Nuclear'), ('Hungary', 'Oilexisting'), ('Hungary', 'Solar'), ('Hungary', 'Waste'), ('Hungary', 'Windonshore'), ('Ireland', 'Bio'), ('Ireland', 'Bio10cofiring'), ('Ireland', 'Bio10cofiringCCS'), ('Ireland', 'Bioexisting'), ('Ireland', 'Coal'), ('Ireland', 'CoalCCSadv'), ('Ireland', 'Coalexisting'), ('Ireland', 'GasCCGT'), ('Ireland', 'GasCCSadv'), ('Ireland', 'GasOCGT'), ('Ireland', 'Gasexisting'), ('Ireland', 'Hydrorun-of-the-river'), ('Ireland', 'Liginiteexisting'), ('Ireland', 'Lignite'), ('Ireland', 'LigniteCCSadv'), ('Ireland', 'Oilexisting'), ('Ireland', 'Solar'), ('Ireland', 'Waste'), ('Ireland', 'Wave'), ('Ireland', 'Windoffshore'), ('Ireland', 'Windonshore'), ('Italy', 'Bio'), ('Italy', 'Bio10cofiring'), ('Italy', 'Bio10cofiringCCS'), ('Italy', 'Bioexisting'), ('Italy', 'Coal'), ('Italy', 'CoalCCS'), ('Italy', 'CoalCCSadv'), ('Italy', 'Coalexisting'), ('Italy', 'GasCCGT'), ('Italy', 'GasCCS'), ('Italy', 'GasCCSadv'), ('Italy', 'GasOCGT'), ('Italy', 'Gasexisting'), ('Italy', 'Geo'), ('Italy', 'Hydroregulated'), ('Italy', 'Hydrorun-of-the-river'), ('Italy', 'Oilexisting'), ('Italy', 'Solar'), ('Italy', 'Waste'), ('Italy', 'Windonshore'), ('Lithuania', 'Bio'), ('Lithuania', 'Bio10cofiring'), ('Lithuania', 'Bio10cofiringCCS'), ('Lithuania', 'Bioexisting'), ('Lithuania', 'GasCCGT'), ('Lithuania', 'GasCCSadv'), ('Lithuania', 'GasOCGT'), ('Lithuania', 'Gasexisting'), ('Lithuania', 'Hydrorun-of-the-river'), ('Lithuania', 'Liginiteexisting'), ('Lithuania', 'Nuclear'), ('Lithuania', 'Oilexisting'), ('Lithuania', 'Solar'), ('Lithuania', 'Waste'), ('Lithuania', 'Windonshore'), ('Luxemb.', 'Bio'), ('Luxemb.', 'Bio10cofiring'), ('Luxemb.', 'Bio10cofiringCCS'), ('Luxemb.', 'Bioexisting'), ('Luxemb.', 'GasCCGT'), ('Luxemb.', 'GasCCSadv'), ('Luxemb.', 'GasOCGT'), ('Luxemb.', 'Gasexisting'), ('Luxemb.', 'Hydroregulated'), ('Luxemb.', 'Hydrorun-of-the-river'), ('Luxemb.', 'Solar'), ('Luxemb.', 'Waste'), ('Luxemb.', 'Windonshore'), ('Spain', 'Bio'), ('Spain', 'Bio10cofiring'), ('Spain', 'Bio10cofiringCCS'), ('Spain', 'Bioexisting'), ('Spain', 'Coal'), ('Spain', 'CoalCCS'), ('Spain', 'CoalCCSadv'), ('Spain', 'Coalexisting'), ('Spain', 'GasCCGT'), ('Spain', 'GasCCS'), ('Spain', 'GasCCSadv'), ('Spain', 'GasOCGT'), ('Spain', 'Gasexisting'), ('Spain', 'Hydroregulated'), ('Spain', 'Hydrorun-of-the-river'), ('Spain', 'Liginiteexisting'), ('Spain', 'Nuclear'), ('Spain', 'Oilexisting'), ('Spain', 'Solar'), ('Spain', 'Waste'), ('Spain', 'Windoffshore'), ('Spain', 'Windonshore'), ('Switzerland', 'Bio'), ('Switzerland', 'Bio10cofiring'), ('Switzerland', 'Bio10cofiringCCS'), ('Switzerland', 'Bioexisting'), ('Switzerland', 'GasCCGT'), ('Switzerland', 'GasCCSadv'), ('Switzerland', 'GasOCGT'), ('Switzerland', 'Hydroregulated'), ('Switzerland', 'Hydrorun-of-the-river'), ('Switzerland', 'Nuclear'), ('Switzerland', 'Oilexisting'), ('Switzerland', 'Solar'), ('Switzerland', 'Waste'), ('Switzerland', 'Windonshore')],
                            "storENInstalledCap" : [('Austria', 'HydroPumpStorage'), ('Austria', 'Li-Ion_BESS'), ('Belgium', 'HydroPumpStorage'), ('Belgium', 'Li-Ion_BESS'), ('BosniaH', 'HydroPumpStorage'), ('BosniaH', 'Li-Ion_BESS'), ('Bulgaria', 'HydroPumpStorage'), ('Bulgaria', 'Li-Ion_BESS'), ('Croatia', 'HydroPumpStorage'), ('Croatia', 'Li-Ion_BESS'), ('CzechR', 'HydroPumpStorage'), ('CzechR', 'Li-Ion_BESS'), ('Denmark', 'Li-Ion_BESS'), ('Estonia', 'Li-Ion_BESS'), ('Finland', 'HydroPumpStorage'), ('Finland', 'Li-Ion_BESS'), ('France', 'HydroPumpStorage'), ('France', 'Li-Ion_BESS'), ('Germany', 'HydroPumpStorage'), ('Germany', 'Li-Ion_BESS'), ('GreatBrit.', 'HydroPumpStorage'), ('GreatBrit.', 'Li-Ion_BESS'), ('Greece', 'HydroPumpStorage'), ('Greece', 'Li-Ion_BESS'), ('Hungary', 'HydroPumpStorage'), ('Hungary', 'Li-Ion_BESS'), ('Ireland', 'HydroPumpStorage'), ('Ireland', 'Li-Ion_BESS'), ('Italy', 'HydroPumpStorage'), ('Italy', 'Li-Ion_BESS'), ('Lithuania', 'HydroPumpStorage'), ('Lithuania', 'Li-Ion_BESS'), ('Luxemb.', 'HydroPumpStorage'), ('Luxemb.', 'Li-Ion_BESS'), ('Spain', 'HydroPumpStorage'), ('Spain', 'Li-Ion_BESS'), ('Switzerland', 'HydroPumpStorage'), ('Switzerland', 'Li-Ion_BESS')],
                            "storPWInstalledCap": [('Austria', 'HydroPumpStorage'), ('Austria', 'Li-Ion_BESS'), ('Belgium', 'HydroPumpStorage'), ('Belgium', 'Li-Ion_BESS'), ('BosniaH', 'HydroPumpStorage'), ('BosniaH', 'Li-Ion_BESS'), ('Bulgaria', 'HydroPumpStorage'), ('Bulgaria', 'Li-Ion_BESS'), ('Croatia', 'HydroPumpStorage'), ('Croatia', 'Li-Ion_BESS'), ('CzechR', 'HydroPumpStorage'), ('CzechR', 'Li-Ion_BESS'), ('Denmark', 'Li-Ion_BESS'), ('Estonia', 'Li-Ion_BESS'), ('Finland', 'HydroPumpStorage'), ('Finland', 'Li-Ion_BESS'), ('France', 'HydroPumpStorage'), ('France', 'Li-Ion_BESS'), ('Germany', 'HydroPumpStorage'), ('Germany', 'Li-Ion_BESS'), ('GreatBrit.', 'HydroPumpStorage'), ('GreatBrit.', 'Li-Ion_BESS'), ('Greece', 'HydroPumpStorage'), ('Greece', 'Li-Ion_BESS'), ('Hungary', 'HydroPumpStorage'), ('Hungary', 'Li-Ion_BESS'), ('Ireland', 'HydroPumpStorage'), ('Ireland', 'Li-Ion_BESS'), ('Italy', 'HydroPumpStorage'), ('Italy', 'Li-Ion_BESS'), ('Lithuania', 'HydroPumpStorage'), ('Lithuania', 'Li-Ion_BESS'), ('Luxemb.', 'HydroPumpStorage'), ('Luxemb.', 'Li-Ion_BESS'), ('Spain', 'HydroPumpStorage'), ('Spain', 'Li-Ion_BESS'), ('Switzerland', 'HydroPumpStorage'), ('Switzerland', 'Li-Ion_BESS')],
                            "transmissionInstalledCap": [('Croatia', 'BosniaH'), ('CzechR', 'Austria'), ('Denmark', 'Germany'), ('Finland', 'Estonia'), ('France', 'Belgium'), ('France', 'Germany'), ('France', 'Spain'), ('France', 'Switzerland'), ('Germany', 'Austria'), ('Germany', 'Belgium'), ('Germany', 'CzechR'), ('Germany', 'Switzerland'), ('GreatBrit.', 'Belgium'), ('GreatBrit.', 'Denmark'), ('GreatBrit.', 'France'), ('Greece', 'Bulgaria'), ('Hungary', 'Austria'), ('Hungary', 'Croatia'), ('Ireland', 'France'), ('Ireland', 'GreatBrit.'), ('Ireland', 'Spain'), ('Italy', 'Austria'), ('Italy', 'BosniaH'), ('Italy', 'Croatia'), ('Italy', 'France'), ('Italy', 'Germany'), ('Italy', 'Greece'), ('Italy', 'Switzerland'), ('Luxemb.', 'Belgium'), ('Luxemb.', 'France'), ('Luxemb.', 'Germany'), ('Switzerland', 'Austria')]
                            }
    
    for i in instance.PeriodActive:
        for key, indices in training_data_order.items():
            if key == 'genInstalledCap':
                for (n, g) in indices:
                    if (n, g) in instance.GeneratorsOfNode:
                        master_feature_order.append(('gen', (n, g, i)))
            elif key == 'storENInstalledCap':
                for (n, b) in indices:
                    if (n, b) in instance.StoragesOfNode:
                        master_feature_order.append(('storEN', (n, b, i)))
            elif key == 'storPWInstalledCap':
                for (n, b) in indices:
                    if (n, b) in instance.StoragesOfNode:
                        master_feature_order.append(('storPW', (n, b, i)))
            elif key == 'transmissionInstalledCap':
                for (n1, n2) in indices:
                    if (n1, n2) in instance.BidirectionalArc:
                        master_feature_order.append(('trans', (n1, n2, i)))
            

    ordered_capacity_indices = master_feature_order
    def v_scaling_rule(m, i):
        feature_index = i - 1  

        var_type, var_indices = ordered_capacity_indices[feature_index]
        capacity_var = None
        if var_type == 'gen':
            capacity_var = m.genInstalledCap[var_indices]
        elif var_type == 'trans':
            capacity_var = m.transmissionInstalledCap[var_indices]
        elif var_type == 'storPW':
            capacity_var = m.storPWInstalledCap[var_indices]
        elif var_type == 'storEN':
            capacity_var = m.storENInstalledCap[var_indices]

        return m.v_scaled[i] == (capacity_var - data_mean[feature_index]) / (data_std[feature_index]+1e-6)
    instance.ml_input_scaled_constraints = Constraint(RangeSet(1, n_features), rule=v_scaling_rule)

    return instance



def embed_empire_embedding(version, tab_file_path, result_file_path, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
                lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
                discountrate, WACC, LeapYearsInvestment, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, north_sea, NoOfScenarios, model_type, Seed):

    first_stage_instance = run_first_stage(version, tab_file_path, result_file_path, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
                lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
                discountrate, WACC, LeapYearsInvestment, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, north_sea)

    
    file_prefix = f"full_s{NoOfScenarios}"

    n_features = 3848 # 616
    scaler_v = joblib.load(f'../models/adaptive/{version}/{file_prefix}_run{Seed}_v_scaler.gz')
    data_mean = scaler_v.mean_
    data_std = scaler_v.scale_

    first_stage_instance.add_component("v_scaled", Var(RangeSet(1, n_features), domain=Reals))
    base_instance = order_capacity_indices(first_stage_instance, n_features, data_mean, data_std, version)

    scaler_y = joblib.load(f'../models/adaptive/{version}/{file_prefix}_run{Seed}_y_scaler.gz')
    onnx_path = f'../models/adaptive/{version}/{file_prefix}_run{Seed}_nn_regressor.onnx'

    output_var_name = 'second_stage_value'
    
    if model_type == "MLP":
        embedded_instance = embed_omlt_neural_network(instance=base_instance,onnx_path=onnx_path,n_features=n_features,y_scaler=scaler_y)

    elif model_type == "LR":
        embedded_instance = integrate_linear_regression_with_pyomo_model(f'../models/adaptive/{version}/{file_prefix}_run{Seed}_lr.joblib', base_instance, output_var_name, scaler_y) # LR



    print("model successfully embedded!")

    def Obj_rule(model):
        first_stage_value = sum(
            model.discount_multiplier[i] * (
                sum(model.genInvCost[g,i] * model.genInvCap[n,g,i] for (n, g) in model.GeneratorsOfNode) +
                sum(model.transmissionInvCost[n1,n2,i] * model.transmisionInvCap[n1,n2,i] for (n1, n2) in model.BidirectionalArc) +
                sum((model.storPWInvCost[b,i] * model.storPWInvCap[n,b,i] + model.storENInvCost[b,i] * model.storENInvCap[n,b,i])
                    for (n, b) in model.StoragesOfNode)
            )
            for i in model.PeriodActive
        )
        Total_cost = first_stage_value + model.second_stage_value
        return Total_cost

    embedded_instance.Obj = Objective(rule=Obj_rule, sense=minimize)

    
    return embedded_instance
