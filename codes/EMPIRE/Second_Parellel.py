import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.common.tempfiles import TempfileManager
import csv
import pandas as pd
import numpy as np
import itertools


def solve_scenario(scenario_data):
    period, scenario, model_data = scenario_data

    model = create_model(model_data, period, scenario)

    set_period_scenario_data(model, period, scenario)

    solver = SolverFactory(model_data['solver'])
    solver.solve(model)
    scenario_results = value(model.Obj)

    return period, scenario, scenario_results

def solve_period(period_data):
    period, scenarios, model_data = period_data
    
    with ProcessPoolExecutor(max_workers=model_data['scenario_workers']) as executor:
        futures = [executor.submit(solve_scenario, (period, scenario, model_data)) for scenario in scenarios]
        results = [future.result() for future in as_completed(futures)]
    
    return period, results

def run_hierarchical_optimization(model_data, periods, scenarios, period_workers=None, scenario_workers=None):
    if period_workers is None:
        period_workers = multiprocessing.cpu_count() // 2  # 예시: CPU 코어의 절반을 period 병렬화에 사용
    
    if scenario_workers is None:
        scenario_workers = 2  # 예시: 각 period 처리에 2개의 worker 할당

    model_data['scenario_workers'] = scenario_workers
    
    period_data = [(period, scenarios, model_data) for period in periods]
    
    with ProcessPoolExecutor(max_workers=period_workers) as executor:
        futures = [executor.submit(solve_period, data) for data in period_data]
        results = [future.result() for future in as_completed(futures)]
    
    return results


def create_model(model_data, period, scenario):
    model = ConcreteModel()
    print("Reading sets...")

    model.Generator = Set(initialize=model_data['Generator'])
    model.Technology = Set(initialize=model_data['Technology'])
    model.Storage = Set(initialize=model_data['Storage'])
    model.Period = Set(initialize=[period])
    model.PeriodActive = Set(initialize=[period])
    model.Operationalhour = Set(initialize=model_data['Operationalhour'])
    model.Season = Set(initialize=model_data['Season'])
    model.Node = Set(initialize=model_data['Node'])
    model.OffshoreNode = Set(initialize=model_data['OffshoreNode'])
    model.DirectionalLink = Set(initialize=model_data['DirectionalLink'])
    model.TransmissionType = Set(initialize=model_data['TransmissionType'])
    model.Scenario = Set(initialize=[scenario])  # Only the specific scenario

    print("Constructing sub sets...")

    # Subsets
    model.GeneratorsOfTechnology = Set(initialize=model_data['GeneratorsOfTechnology'])
    model.GeneratorsOfNode = Set(initialize=model_data['GeneratorsOfNode'])
    model.TransmissionTypeOfDirectionalLink = Set(initialize=model_data['TransmissionTypeOfDirectionalLink'])
    model.ThermalGenerators = Set(initialize=model_data['ThermalGenerators'])
    model.RegHydroGenerator = Set(initialize=model_data['RegHydroGenerator'])
    model.HydroGenerator = Set(initialize=model_data['HydroGenerator'])
    model.StoragesOfNode = Set(initialize=model_data['StoragesOfNode'])
    model.DependentStorage = Set(initialize=model_data['DependentStorage'])
    model.HoursOfSeason = Set(initialize=model_data['HoursOfSeason'])
    model.FirstHoursOfRegSeason = Set(initialize=model_data['FirstHoursOfRegSeason'])
    model.FirstHoursOfPeakSeason = Set(initialize=model_data['FirstHoursOfPeakSeason'])


    #Build arc subsets

    def NodesLinked_init(model, node):
        retval = []
        for (i,j) in model.DirectionalLink:
            if j == node:
                retval.append(i)
        return retval
    model.NodesLinked = Set(model.Node, initialize=NodesLinked_init)

    def BidirectionalArc_init(model):
        retval = []
        for (i,j) in model.DirectionalLink:
            if i != j and (not (j,i) in retval):
                retval.append((i,j))
        return retval
    model.BidirectionalArc = Set(dimen=2, initialize=BidirectionalArc_init, ordered=True) #l


    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    print("Declaring parameters...")

    #Scaling

    # Helper function for parameter initialization
    def init_param(data, *keys):
        return data[keys[0]] if len(keys) == 1 else data[keys[0]][keys[1]]

    # Basic parameters
    model.discountrate = Param(initialize=model_data['discountrate'])
    model.WACC = Param(initialize=model_data['WACC'])
    model.LeapYearsInvestment = Param(initialize=model_data['LeapYearsInvestment'])
    model.operationalDiscountrate = Param(mutable=True)
    model.sceProbab = Param(model.Scenario, initialize={scenario: model_data['sceProbab'][scenario]})
    model.seasScale = Param(model.Season, initialize=model_data['seasScale'])
    model.lengthRegSeason = Param(initialize=model_data['lengthRegSeason'])
    model.lengthPeakSeason = Param(initialize=model_data['lengthPeakSeason'])

    # Cost parameters
    model.genCapitalCost = Param(model.Generator, model.Period, initialize=lambda m, g, i: init_param(model_data['genCapitalCost'], g, i))
    model.transmissionTypeCapitalCost = Param(model.TransmissionType, model.Period, initialize=lambda m, t, i: init_param(model_data['transmissionTypeCapitalCost'], t, i))
    model.storPWCapitalCost = Param(model.Storage, model.Period, initialize=lambda m, b, i: init_param(model_data['storPWCapitalCost'], b, i))
    model.storENCapitalCost = Param(model.Storage, model.Period, initialize=lambda m, b, i: init_param(model_data['storENCapitalCost'], b, i))
    model.genFixedOMCost = Param(model.Generator, model.Period, initialize=lambda m, g, i: init_param(model_data['genFixedOMCost'], g, i))
    model.transmissionTypeFixedOMCost = Param(model.TransmissionType, model.Period, initialize=lambda m, t, i: init_param(model_data['transmissionTypeFixedOMCost'], t, i))
    model.storPWFixedOMCost = Param(model.Storage, model.Period, initialize=lambda m, b, i: init_param(model_data['storPWFixedOMCost'], b, i))
    model.storENFixedOMCost = Param(model.Storage, model.Period, initialize=lambda m, b, i: init_param(model_data['storENFixedOMCost'], b, i))
    model.transmissionLength = Param(model.BidirectionalArc, initialize=lambda m, n1, n2: init_param(model_data['transmissionLength'], (n1, n2)))
    model.genVariableOMCost = Param(model.Generator, initialize=lambda m, g: init_param(model_data['genVariableOMCost'], g))
    model.genFuelCost = Param(model.Generator, model.Period, initialize=lambda m, g, i: init_param(model_data['genFuelCost'], g, i))
    model.genMargCost = Param(model.Generator, model.Period, initialize=lambda m, g, i: init_param(model_data['genMargCost'], g, i))
    model.genCO2TypeFactor = Param(model.Generator, initialize=lambda m, g: init_param(model_data['genCO2TypeFactor'], g))
    model.nodeLostLoadCost = Param(model.Node, model.Period, initialize=lambda m, n, i: init_param(model_data['nodeLostLoadCost'], n, i))
    model.CO2price = Param(model.Period, initialize=lambda m, i: init_param(model_data['CO2price'], i))
    model.CCSCostTSFix = Param(initialize=model_data['CCSCostTSFix'])
    model.CCSCostTSVariable = Param(model.Period, initialize=lambda m, i: init_param(model_data['CCSCostTSVariable'], i))
    model.CCSRemFrac = Param(initialize=model_data['CCSRemFrac'])

    # Technology limitation parameters
    model.genRefInitCap = Param(model.GeneratorsOfNode, initialize=lambda m, n, g: init_param(model_data['genRefInitCap'], (n, g)))
    model.genScaleInitCap = Param(model.Generator, model.Period, initialize=lambda m, g, i: init_param(model_data['genScaleInitCap'], g, i))
    model.genInitCap = Param(model.GeneratorsOfNode, model.Period, initialize=lambda m, n, g, i: init_param(model_data['genInitCap'], (n, g), i))
    model.transmissionInitCap = Param(model.BidirectionalArc, model.Period, initialize=lambda m, n1, n2, i: init_param(model_data['transmissionInitCap'], (n1, n2), i))
    model.storPWInitCap = Param(model.StoragesOfNode, model.Period, initialize=lambda m, n, b, i: init_param(model_data['storPWInitCap'], (n, b), i))
    model.storENInitCap = Param(model.StoragesOfNode, model.Period, initialize=lambda m, n, b, i: init_param(model_data['storENInitCap'], (n, b), i))
    model.genLifetime = Param(model.Generator, initialize=lambda m, g: init_param(model_data['genLifetime'], g))
    model.transmissionLifetime = Param(model.BidirectionalArc, initialize=lambda m, n1, n2: init_param(model_data['transmissionLifetime'], (n1, n2)))
    model.storageLifetime = Param(model.Storage, initialize=lambda m, b: init_param(model_data['storageLifetime'], b))
    model.genEfficiency = Param(model.Generator, model.Period, initialize=lambda m, g, i: init_param(model_data['genEfficiency'], g, i))
    model.lineEfficiency = Param(model.DirectionalLink, initialize=lambda m, n1, n2: init_param(model_data['lineEfficiency'], (n1, n2)))
    model.storageChargeEff = Param(model.Storage, initialize=lambda m, b: init_param(model_data['storageChargeEff'], b))
    model.storageDischargeEff = Param(model.Storage, initialize=lambda m, b: init_param(model_data['storageDischargeEff'], b))
    model.storageBleedEff = Param(model.Storage, initialize=lambda m, b: init_param(model_data['storageBleedEff'], b))
    model.genRampUpCap = Param(model.ThermalGenerators, initialize=lambda m, g: init_param(model_data['genRampUpCap'], g))
    model.storageDiscToCharRatio = Param(model.Storage, initialize=lambda m, b: init_param(model_data['storageDiscToCharRatio'], b))
    model.storagePowToEnergy = Param(model.DependentStorage, initialize=lambda m, b: init_param(model_data['storagePowToEnergy'], b))

    # Stochastic input parameters
    model.sloadRaw = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, 
                           initialize=lambda m, n, h, s, i: init_param(model_data['sloadRaw'], n, h, s, i))
    model.sloadAnnualDemand = Param(model.Node, model.Period, initialize=lambda m, n, i: init_param(model_data['sloadAnnualDemand'], n, i))
    model.sload = Param(model.Node, model.Operationalhour, model.Period, model.Scenario, 
                        initialize=lambda m, n, h, i, s: init_param(model_data['sload'], n, h, i, s))
    model.genCapAvailTypeRaw = Param(model.Generator, initialize=lambda m, g: init_param(model_data['genCapAvailTypeRaw'], g))
    model.genCapAvailStochRaw = Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, 
                                      initialize=lambda m, n, g, h, s, i: init_param(model_data['genCapAvailStochRaw'], (n, g), h, s, i))
    model.genCapAvail = Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, 
                              initialize=lambda m, n, g, h, s, i: init_param(model_data['genCapAvail'], (n, g), h, s, i))
    model.maxRegHydroGenRaw = Param(model.Node, model.Period, model.HoursOfSeason, model.Scenario, 
                                    initialize=lambda m, n, i, s, h, w: init_param(model_data['maxRegHydroGenRaw'], n, i, s, h, w))
    model.maxRegHydroGen = Param(model.Node, model.Period, model.Season, model.Scenario, 
                                 initialize=lambda m, n, i, s, w: init_param(model_data['maxRegHydroGen'], n, i, s, w))
    model.maxHydroNode = Param(model.Node, initialize=lambda m, n: init_param(model_data['maxHydroNode'], n))
    model.storOperationalInit = Param(model.Storage, initialize=lambda m, b: init_param(model_data['storOperationalInit'], b))

    # FSD data parameters
    model.genInvCapParam = Param(model.GeneratorsOfNode, model.PeriodActive, initialize=lambda m, n, g, i: init_param(model_data['genInvCapParam'], (n, g), i))
    model.transmisionInvCapParam = Param(model.BidirectionalArc, model.PeriodActive, initialize=lambda m, n1, n2, i: init_param(model_data['transmisionInvCapParam'], (n1, n2), i))
    model.storPWInvCapParam = Param(model.StoragesOfNode, model.PeriodActive, initialize=lambda m, n, b, i: init_param(model_data['storPWInvCapParam'], (n, b), i))
    model.storENInvCapParam = Param(model.StoragesOfNode, model.PeriodActive, initialize=lambda m, n, b, i: init_param(model_data['storENInvCapParam'], (n, b), i))
    model.genInstalledCap = Param(model.GeneratorsOfNode, model.PeriodActive, initialize=lambda m, n, g, i: init_param(model_data['genInstalledCap'], (n, g), i))
    model.transmissionInstalledCap = Param(model.BidirectionalArc, model.PeriodActive, initialize=lambda m, n1, n2, i: init_param(model_data['transmissionInstalledCap'], (n1, n2), i))
    model.storPWInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, initialize=lambda m, n, b, i: init_param(model_data['storPWInstalledCap'], (n, b), i))
    model.storENInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, initialize=lambda m, n, b, i: init_param(model_data['storENInstalledCap'], (n, b), i))

    # Conditional parameters
    if model_data['EMISSION_CAP']:
        model.CO2cap = Param(model.Period, initialize=lambda m, i: init_param(model_data['CO2cap'], i))
    
    if model_data['LOADCHANGEMODULE']:
        model.sloadMod = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, 
                               initialize=lambda m, n, h, s, i: init_param(model_data['sloadMod'], n, h, s, i))
    
    #############
    ##VARIABLES##
    #############

    model.genOperational = Var(model.GeneratorsOfNode, model.Operationalhour, model.Period, model.Scenario, domain=NonNegativeReals)
    model.storOperational = Var(model.StoragesOfNode, model.Operationalhour, model.Period, model.Scenario, domain=NonNegativeReals)
    model.transmisionOperational = Var(model.DirectionalLink, model.Operationalhour, model.Period, model.Scenario, domain=NonNegativeReals)
    model.storCharge = Var(model.StoragesOfNode, model.Operationalhour, model.Period, model.Scenario, domain=NonNegativeReals)
    model.storDischarge = Var(model.StoragesOfNode, model.Operationalhour, model.Period, model.Scenario, domain=NonNegativeReals)
    model.loadShed = Var(model.Node, model.Operationalhour, model.Period, model.Scenario, domain=NonNegativeReals)

    #############
    #EXPRESSIONS#
    #############

    def multiplier_rule(model, period):
        coeff = 1
        if period > 1:
            coeff = pow(1.0 + model.discountrate, (-model.LeapYearsInvestment * (int(period) - 1)))
        return coeff
    model.discount_multiplier = Expression(model.Period, rule=multiplier_rule)

    def shed_component_rule(model, i):
        return sum(model.operationalDiscountrate * model.seasScale[s] * model.sceProbab[w] * model.nodeLostLoadCost[n, i] * model.loadShed[n, h, i, w] 
                   for n in model.Node for w in model.Scenario for (s, h) in model.HoursOfSeason)
    model.shedcomponent = Expression(model.Period, rule=shed_component_rule)

    def operational_cost_rule(model, i):
        return sum(model.operationalDiscountrate * model.seasScale[s] * model.sceProbab[w] * model.genMargCost[g, i] * model.genOperational[n, g, h, i, w] 
                   for (n, g) in model.GeneratorsOfNode for (s, h) in model.HoursOfSeason for w in model.Scenario)
    model.operationalcost = Expression(model.Period, rule=operational_cost_rule)

    #############
    ##OBJECTIVE##
    #############

    def Obj_rule(model):
        return sum(model.discount_multiplier[i] * (model.shedcomponent[i] + model.operationalcost[i]) for i in model.Period)
    model.Obj = Objective(rule=Obj_rule, sense=minimize)


    ###############
    ##CONSTRAINTS##
    ###############

    def FlowBalance_rule(model, n, h, i, w):
        return sum(model.genOperational[n,g,h,i,w] for g in model.Generator if (n,g) in model.GeneratorsOfNode) \
            + sum((model.storageDischargeEff[b]*model.storDischarge[n,b,h,i,w]-model.storCharge[n,b,h,i,w]) for b in model.Storage if (n,b) in model.StoragesOfNode) \
            + sum((model.lineEfficiency[link,n]*model.transmisionOperational[link,n,h,i,w] - model.transmisionOperational[n,link,h,i,w]) for link in model.NodesLinked[n]) \
            - model.sload[n,h,i,w] + model.loadShed[n,h,i,w] \
            == 0
    model.FlowBalance = Constraint(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, rule=FlowBalance_rule)

    #################################################################

    def genMaxProd_rule(model, n, g, h, i, w):
            return model.genOperational[n,g,h,i,w] - model.genCapAvail[n,g,h,w,i]*model.genInstalledCap[n,g,i] <= 0
    model.maxGenProduction = Constraint(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=genMaxProd_rule)

    #################################################################

    def ramping_rule(model, n, g, h, i, w):
        if h in model.FirstHoursOfRegSeason or h in model.FirstHoursOfPeakSeason:
            return Constraint.Skip
        else:
            if g in model.ThermalGenerators:
                return model.genOperational[n,g,h,i,w]-model.genOperational[n,g,(h-1),i,w] - model.genRampUpCap[g]*model.genInstalledCap[n,g,i] <= 0   #
            else:
                return Constraint.Skip
    model.ramping = Constraint(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=ramping_rule)

    #################################################################

    def storage_energy_balance_rule(model, n, b, h, i, w):
        if h in model.FirstHoursOfRegSeason or h in model.FirstHoursOfPeakSeason:
            return model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] + model.storageChargeEff[b]*model.storCharge[n,b,h,i,w]-model.storDischarge[n,b,h,i,w]-model.storOperational[n,b,h,i,w] == 0   #
        else:
            return model.storageBleedEff[b]*model.storOperational[n,b,(h-1),i,w] + model.storageChargeEff[b]*model.storCharge[n,b,h,i,w]-model.storDischarge[n,b,h,i,w]-model.storOperational[n,b,h,i,w] == 0   #
    model.storage_energy_balance = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_energy_balance_rule)

    #################################################################

    def storage_seasonal_net_zero_balance_rule(model, n, b, h, i, w):
        if h in model.FirstHoursOfRegSeason:
            return model.storOperational[n,b,h+value(model.lengthRegSeason)-1,i,w] - model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] == 0  #
        elif h in model.FirstHoursOfPeakSeason:
            return model.storOperational[n,b,h+value(model.lengthPeakSeason)-1,i,w] - model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] == 0  #
        else:
            return Constraint.Skip
    model.storage_seasonal_net_zero_balance = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_seasonal_net_zero_balance_rule)

    #################################################################

    def storage_operational_cap_rule(model, n, b, h, i, w):
        return model.storOperational[n,b,h,i,w] - model.storENInstalledCap[n,b,i]  <= 0   #
    model.storage_operational_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_operational_cap_rule)

    #################################################################

    def storage_power_discharg_cap_rule(model, n, b, h, i, w):
        return model.storDischarge[n,b,h,i,w] - model.storageDiscToCharRatio[b]*model.storPWInstalledCap[n,b,i] <= 0   #
    model.storage_power_discharg_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_power_discharg_cap_rule)

    #################################################################

    def storage_power_charg_cap_rule(model, n, b, h, i, w):
        return model.storCharge[n,b,h,i,w] - model.storPWInstalledCap[n,b,i] <= 0   #
    model.storage_power_charg_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_power_charg_cap_rule)

    #################################################################

    def hydro_gen_limit_rule(model, n, g, s, i, w):
        if g in model.RegHydroGenerator:
            return sum(model.genOperational[n,g,h,i,w] for h in model.Operationalhour if (s,h) in model.HoursOfSeason) - model.maxRegHydroGen[n,i,s,w] <= 0
        else:
            return Constraint.Skip  #
    model.hydro_gen_limit = Constraint(model.GeneratorsOfNode, model.Season, model.PeriodActive, model.Scenario, rule=hydro_gen_limit_rule)

    #################################################################

    def transmission_cap_rule(model, n1, n2, h, i, w):
        if (n1,n2) in model.BidirectionalArc:
            return model.transmisionOperational[(n1,n2),h,i,w]  - model.transmissionInstalledCap[(n1,n2),i] <= 0
        elif (n2,n1) in model.BidirectionalArc:
            return model.transmisionOperational[(n1,n2),h,i,w]  - model.transmissionInstalledCap[(n2,n1),i] <= 0
    model.transmission_cap = Constraint(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.Scenario, rule=transmission_cap_rule)

    #################################################################

    def wind_farm_tranmission_cap_rule(model, n1, n2, i):
        if n1 in model.OffshoreNode or n2 in model.OffshoreNode:
            if (n1,n2) in model.BidirectionalArc:
                if n1 in model.OffshoreNode:
                    return model.transmissionInstalledCap[(n1,n2),i] <= sum(model.genInstalledCap[n1,g,i] for g in model.Generator if (n1,g) in model.GeneratorsOfNode)
                else:
                    return model.transmissionInstalledCap[(n1,n2),i] <= sum(model.genInstalledCap[n2,g,i] for g in model.Generator if (n2,g) in model.GeneratorsOfNode)
            elif (n2,n1) in model.BidirectionalArc:
                if n1 in model.OffshoreNode:
                    return model.transmissionInstalledCap[(n2,n1),i] <= sum(model.genInstalledCap[n1,g,i] for g in model.Generator if (n1,g) in model.GeneratorsOfNode)
                else:
                    return model.transmissionInstalledCap[(n2,n1),i] <= sum(model.genInstalledCap[n2,g,i] for g in model.Generator if (n2,g) in model.GeneratorsOfNode)
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.wind_farm_transmission_cap = Constraint(model.Node, model.Node, model.PeriodActive, rule=wind_farm_tranmission_cap_rule)
    #################################################################

    if model_data['EMISSION_CAP']:
            def emission_cap_rule(model, i, w):
                return sum(model.seasScale[s] * model.genCO2TypeFactor[g] * (3.6 / model.genEfficiency[g, i]) * model.genOperational[n, g, h, i, w] 
                        for (n, g) in model.GeneratorsOfNode for (s, h) in model.HoursOfSeason) / 1000000 - model.CO2cap[i] <= 0
            model.emission_cap = Constraint(model.Period, model.Scenario, rule=emission_cap_rule)
    
    
    return model

def set_period_scenario_data(model, period, scenario, data):
    def get_data(key, *indices):
        return data[key][indices] if indices else data[key]

    def set_vectorized(param, index_sets, data_key):
        param_data = get_data(data_key)
        if isinstance(index_sets[0], Set):
            for indices in itertools.product(*index_sets):
                param[indices] = param_data[indices]
        else:
            param.update(param_data)

    set_vectorized(model.sload, (model.Node, model.Operationalhour), f'sload_{period}_{scenario}')
    set_vectorized(model.genCapAvail, (model.GeneratorsOfNode, model.Operationalhour), f'genCapAvail_{period}_{scenario}')
    set_vectorized(model.maxRegHydroGen, (model.Node, model.Season), f'maxRegHydroGen_{period}_{scenario}')

    period_params = [
        (model.genMargCost, model.Generator),
        (model.genEfficiency, model.Generator),
        (model.genFuelCost, model.Generator),
        (model.transmissionInstalledCap, model.BidirectionalArc),
        (model.storPWInstalledCap, model.StoragesOfNode),
        (model.storENInstalledCap, model.StoragesOfNode),
        (model.nodeLostLoadCost, model.Node),
        (model.genInvCapParam, model.GeneratorsOfNode),
        (model.genInstalledCap, model.GeneratorsOfNode),
        (model.transmisionInvCapParam, model.BidirectionalArc),
        (model.storPWInvCapParam, model.StoragesOfNode),
        (model.storENInvCapParam, model.StoragesOfNode)
    ]

    for param, index_set in period_params:
        set_vectorized(param, (index_set,), f'{param.name}_{period}')

    single_value_params = [
        (model.CO2price, 'CO2price'),
        (model.sceProbab, 'sceProbab'),
        (model.CCSCostTSVariable, 'CCSCostTSVariable')
    ]

    for param, key in single_value_params:
        param[period] = get_data(key, period)

    if data.get('EMISSION_CAP'):
        model.CO2cap[period] = get_data('CO2cap', period)
    
    if data.get('LOADCHANGEMODULE'):
        set_vectorized(model.sloadMod, (model.Node, model.Operationalhour), f'sloadMod_{period}_{scenario}')

    model.operationalDiscountrate = sum((1 + model.discountrate) ** (-j) for j in range(model.LeapYearsInvestment))

    gen_efficiency = np.array([model.genEfficiency[g, period] for g in model.Generator])
    gen_fuel_cost = np.array([model.genFuelCost[g, period] for g in model.Generator])
    gen_co2_factor = np.array([model.genCO2TypeFactor[g] for g in model.Generator])
    gen_variable_om_cost = np.array([model.genVariableOMCost[g] for g in model.Generator])

    cost_per_energy_unit = (3.6 / gen_efficiency) * (gen_fuel_cost + gen_co2_factor * model.CO2price[period]) + gen_variable_om_cost

    for g, cost in zip(model.Generator, cost_per_energy_unit):
        model.genMargCost[g, period] = cost

    for g in model.Generator:
        if ('CCS', g) in model.GeneratorsOfTechnology:
            model.genMargCost[g, period] += (3.6 / model.genEfficiency[g, period]) * (
                model.CCSRemFrac * model.genCO2TypeFactor[g] * model.CCSCostTSVariable[period]
            )


def run_second_stage(name, tab_file_path, result_file_path, scenariogeneration, scenario_data_path,
                     solver, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
                     lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
                     discountrate, WACC, LeapYearsInvestment, FSD, WRITE_LP,
                     PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE):
    
    model_data = prepare_model_data(name, tab_file_path, result_file_path, scenariogeneration, 
                                    scenario_data_path, solver, temp_dir, FirstHoursOfRegSeason, 
                                    FirstHoursOfPeakSeason, lengthRegSeason, lengthPeakSeason, 
                                    Period, Operationalhour, Scenario, Season, HoursOfSeason,
                                    discountrate, WACC, LeapYearsInvestment, FSD, WRITE_LP,
                                    PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE)
    
    results = run_hierarchical_optimization(model_data, Period, Scenario)
    
    combined_results = combine_results(results)
    
    return combined_results


def combine_results(results):

    total_objective_value = 0

    for period, period_results in results:
        for scenario, scenario_obj_value in period_results:
            total_objective_value += scenario_obj_value

    return total_objective_value

def prepare_model_data(name, tab_file_path, result_file_path, scenariogeneration, scenario_data_path,
                       solver, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
                       lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
                       discountrate, WACC, LeapYearsInvestment, FSD, WRITE_LP,
                       PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE):
    
    data = {}
    
    data['name'] = name
    data['solver'] = solver
    data['discountrate'] = discountrate
    data['WACC'] = WACC
    data['LeapYearsInvestment'] = LeapYearsInvestment
    data['lengthRegSeason'] = lengthRegSeason
    data['lengthPeakSeason'] = lengthPeakSeason
    data['FirstHoursOfRegSeason'] = FirstHoursOfRegSeason
    data['FirstHoursOfPeakSeason'] = FirstHoursOfPeakSeason
    data['Period'] = Period
    data['Operationalhour'] = Operationalhour
    data['Scenario'] = Scenario
    data['Season'] = Season
    data['HoursOfSeason'] = HoursOfSeason
    data['WRITE_LP'] = WRITE_LP
    data['PICKLE_INSTANCE'] = PICKLE_INSTANCE
    data['EMISSION_CAP'] = EMISSION_CAP
    data['USE_TEMP_DIR'] = USE_TEMP_DIR
    data['LOADCHANGEMODULE'] = LOADCHANGEMODULE

    if USE_TEMP_DIR:
        TempfileManager.tempdir = temp_dir

    # Set Data Load
    data['Generator'] = load_set(tab_file_path, 'Sets_Generator.tab')
    data['ThermalGenerators'] = load_set(tab_file_path, 'Sets_ThermalGenerators.tab')
    data['HydroGenerator'] = load_set(tab_file_path, 'Sets_HydroGenerator.tab')
    data['RegHydroGenerator'] = load_set(tab_file_path, 'Sets_HydroGeneratorWithReservoir.tab')
    data['Storage'] = load_set(tab_file_path, 'Sets_Storage.tab')
    data['DependentStorage'] = load_set(tab_file_path, 'Sets_DependentStorage.tab')
    data['Technology'] = load_set(tab_file_path, 'Sets_Technology.tab')
    data['Node'] = load_set(tab_file_path, 'Sets_Node.tab')
    data['OffshoreNode'] = load_set(tab_file_path, 'Sets_OffshoreNode.tab')
    data['DirectionalLink'] = load_set(tab_file_path, 'Sets_DirectionalLines.tab')
    data['TransmissionType'] = load_set(tab_file_path, 'Sets_LineType.tab')
    data['GeneratorsOfTechnology'] = load_set(tab_file_path, 'Sets_GeneratorsOfTechnology.tab')
    data['GeneratorsOfNode'] = load_set(tab_file_path, 'Sets_GeneratorsOfNode.tab')
    data['StoragesOfNode'] = load_set(tab_file_path, 'Sets_StorageOfNodes.tab')

    # Parameter Data Load
    data['genCapitalCost'] = load_parameter(tab_file_path, 'Generator_CapitalCosts.tab')
    data['genFixedOMCost'] = load_parameter(tab_file_path, 'Generator_FixedOMCosts.tab')
    data['genVariableOMCost'] = load_parameter(tab_file_path, 'Generator_VariableOMCosts.tab')
    data['genFuelCost'] = load_parameter(tab_file_path, 'Generator_FuelCosts.tab')
    data['CCSCostTSVariable'] = load_parameter(tab_file_path, 'Generator_CCSCostTSVariable.tab')
    data['genEfficiency'] = load_parameter(tab_file_path, 'Generator_Efficiency.tab')
    data['genRefInitCap'] = load_parameter(tab_file_path, 'Generator_RefInitialCap.tab')
    data['genScaleInitCap'] = load_parameter(tab_file_path, 'Generator_ScaleFactorInitialCap.tab')
    data['genInitCap'] = load_parameter(tab_file_path, 'Generator_InitialCapacity.tab')
    data['genCO2TypeFactor'] = load_parameter(tab_file_path, 'Generator_CO2Content.tab')
    data['genRampUpCap'] = load_parameter(tab_file_path, 'Generator_RampRate.tab')
    data['genCapAvailTypeRaw'] = load_parameter(tab_file_path, 'Generator_GeneratorTypeAvailability.tab')
    data['genLifetime'] = load_parameter(tab_file_path, 'Generator_Lifetime.tab')

    data['transmissionInitCap'] = load_parameter(tab_file_path, 'Transmission_InitialCapacity.tab')
    data['transmissionLength'] = load_parameter(tab_file_path, 'Transmission_Length.tab')
    data['transmissionTypeCapitalCost'] = load_parameter(tab_file_path, 'Transmission_TypeCapitalCost.tab')
    data['transmissionTypeFixedOMCost'] = load_parameter(tab_file_path, 'Transmission_TypeFixedOMCost.tab')
    data['lineEfficiency'] = load_parameter(tab_file_path, 'Transmission_lineEfficiency.tab')
    data['transmissionLifetime'] = load_parameter(tab_file_path, 'Transmission_Lifetime.tab')

    data['storageBleedEff'] = load_parameter(tab_file_path, 'Storage_StorageBleedEfficiency.tab')
    data['storageChargeEff'] = load_parameter(tab_file_path, 'Storage_StorageChargeEff.tab')
    data['storageDischargeEff'] = load_parameter(tab_file_path, 'Storage_StorageDischargeEff.tab')
    data['storagePowToEnergy'] = load_parameter(tab_file_path, 'Storage_StoragePowToEnergy.tab')
    data['storENCapitalCost'] = load_parameter(tab_file_path, 'Storage_EnergyCapitalCost.tab')
    data['storENFixedOMCost'] = load_parameter(tab_file_path, 'Storage_EnergyFixedOMCost.tab')
    data['storENInitCap'] = load_parameter(tab_file_path, 'Storage_EnergyInitialCapacity.tab')
    data['storOperationalInit'] = load_parameter(tab_file_path, 'Storage_StorageInitialEnergyLevel.tab')
    data['storPWCapitalCost'] = load_parameter(tab_file_path, 'Storage_PowerCapitalCost.tab')
    data['storPWFixedOMCost'] = load_parameter(tab_file_path, 'Storage_PowerFixedOMCost.tab')
    data['storPWInitCap'] = load_parameter(tab_file_path, 'Storage_InitialPowerCapacity.tab')
    data['storageLifetime'] = load_parameter(tab_file_path, 'Storage_Lifetime.tab')

    data['nodeLostLoadCost'] = load_parameter(tab_file_path, 'Node_NodeLostLoadCost.tab')
    data['sloadAnnualDemand'] = load_parameter(tab_file_path, 'Node_ElectricAnnualDemand.tab')
    data['maxHydroNode'] = load_parameter(tab_file_path, 'Node_HydroGenMaxAnnualProduction.tab')

    data['seasScale'] = load_parameter(tab_file_path, 'General_seasonScale.tab')

    if EMISSION_CAP:
        data['CO2cap'] = load_parameter(tab_file_path, 'General_CO2Cap.tab')
    else:
        data['CO2price'] = load_parameter(tab_file_path, 'General_CO2Price.tab')

    scenariopath = tab_file_path if scenariogeneration else scenario_data_path
    data['maxRegHydroGenRaw'] = load_parameter(scenariopath, 'Stochastic_HydroGenMaxSeasonalProduction.tab')
    data['genCapAvailStochRaw'] = load_parameter(scenariopath, 'Stochastic_StochasticAvailability.tab')
    data['sloadRaw'] = load_parameter(scenariopath, 'Stochastic_ElectricLoadRaw.tab')

    if LOADCHANGEMODULE:
        data['sloadMod'] = load_parameter(scenariopath, 'LoadchangeModule/Stochastic_ElectricLoadMod.tab')

    preprocess_data(data)
    process_fsd_data(data, FSD)

    return data

def load_set(file_path, file_name):
    with open(file_path + '/' + file_name, 'r') as f:
        return set(line.strip() for line in f)

def load_parameter(file_path, file_name):
    return pd.read_csv(file_path + '/' + file_name, sep='\t', index_col=0)


def preprocess_data(data):
    data['sceProbab'] = {sce: 1.0 / len(data['Scenario']) for sce in data['Scenario']}
    data['genMargCost'] = calculate_gen_marginal_cost(data)
    process_load_data(data)
    process_hydro_data(data)
    process_gen_availability_data(data)

def calculate_gen_marginal_cost(data):
    genMargCost = {}
    for g in data['Generator']:
        for i in data['Period']:
            if ('CCS', g) in data['GeneratorsOfTechnology']:
                costperenergyunit = (3.6 / data['genEfficiency'][g][i]) * (data['genFuelCost'][g][i] + (1 - data['CCSRemFrac']) * data['genCO2TypeFactor'][g] * data['CO2price'][i]) + \
                                    (3.6 / data['genEfficiency'][g][i]) * (data['CCSRemFrac'] * data['genCO2TypeFactor'][g] * data['CCSCostTSVariable'][i]) + \
                                    data['genVariableOMCost'][g]
            else:
                costperenergyunit = (3.6 / data['genEfficiency'][g][i]) * (data['genFuelCost'][g][i] + data['genCO2TypeFactor'][g] * data['CO2price'][i]) + \
                                    data['genVariableOMCost'][g]
            genMargCost[g, i] = costperenergyunit
    return genMargCost


def process_load_data(data):
    sload = {}
    for n in data['Node']:
        for i in data['Period']:
            noderawdemand = 0
            for s in data['Season']:
                for h in data['Operationalhour']:
                    if (s, h) in data['HoursOfSeason']:
                        if h < data['FirstHoursOfRegSeason'][-1] + data['lengthRegSeason']:
                            for sce in data['Scenario']:
                                noderawdemand += data['sceProbab'][sce] * data['seasScale'][s] * data['sloadRaw'][n][h][sce][i]
            
            if data['sloadAnnualDemand'][n][i] < 1:
                hourlyscale = 0
            else:
                hourlyscale = data['sloadAnnualDemand'][n][i] / noderawdemand

            for h in data['Operationalhour']:
                for sce in data['Scenario']:
                    sload[n, h, i, sce] = data['sloadRaw'][n][h][sce][i] * hourlyscale
                    if data['LOADCHANGEMODULE']:
                        sload[n, h, i, sce] += data['sloadMod'][n][h][sce][i]
                    if sload[n, h, i, sce] < 0:
                        sload[n, h, i, sce] = 10  # Minimum load

    data['sload'] = sload

def process_hydro_data(data):
    maxRegHydroGen = {}
    for n in data['Node']:
        for s in data['Season']:
            for i in data['Period']:
                for sce in data['Scenario']:
                    maxRegHydroGen[n, i, s, sce] = sum(data['maxRegHydroGenRaw'][n][i][s][h][sce] 
                                                       for h in data['Operationalhour'] 
                                                       if (s, h) in data['HoursOfSeason'])
    data['maxRegHydroGen'] = maxRegHydroGen

def process_gen_availability_data(data):
    genCapAvail = {}
    for (n, g) in data['GeneratorsOfNode']:
        for h in data['Operationalhour']:
            for s in data['Scenario']:
                for i in data['Period']:
                    if data['genCapAvailTypeRaw'][g] == 0:
                        genCapAvail[n, g, h, s, i] = data['genCapAvailStochRaw'][n][g][h][s][i]
                    else:
                        genCapAvail[n, g, h, s, i] = data['genCapAvailTypeRaw'][g]
    data['genCapAvail'] = genCapAvail


def process_fsd_data(data, FSD):
    gen_inv_cap = {}
    transmission_inv_cap = {}
    stor_pw_inv_cap = {}
    stor_en_inv_cap = {}
    
    for row in FSD:
        country, energy_type, period, type_, cap_value = row
        period = int(period)
        cap_value = float(cap_value)
        
        if type_ == 'Generation':
            if (country, energy_type) not in gen_inv_cap:
                gen_inv_cap[(country, energy_type)] = {}
            gen_inv_cap[(country, energy_type)][period] = cap_value
        elif type_ == 'Transmission':
            if country not in transmission_inv_cap:
                transmission_inv_cap[country] = {}
            transmission_inv_cap[country][period] = cap_value
        elif type_ == 'Storage Power':
            if (country, energy_type) not in stor_pw_inv_cap:
                stor_pw_inv_cap[(country, energy_type)] = {}
            stor_pw_inv_cap[(country, energy_type)][period] = cap_value
        elif type_ == 'Storage Energy':
            if (country, energy_type) not in stor_en_inv_cap:
                stor_en_inv_cap[(country, energy_type)] = {}
            stor_en_inv_cap[(country, energy_type)][period] = cap_value

    data['genInvCapParam'] = {}
    for (n, g) in data['GeneratorsOfNode']:
        for i in data['Period']:
            if (n, g) in gen_inv_cap and i in gen_inv_cap[(n, g)]:
                data['genInvCapParam'][(n, g, i)] = gen_inv_cap[(n, g)][i]
            else:
                data['genInvCapParam'][(n, g, i)] = 0.0

    data['transmissionInvCapParam'] = {}
    for (n1, n2) in data['DirectionalLink']:
        for i in data['Period']:
            if n1 in transmission_inv_cap and i in transmission_inv_cap[n1]:
                data['transmissionInvCapParam'][(n1, n2, i)] = transmission_inv_cap[n1][i]
            elif n2 in transmission_inv_cap and i in transmission_inv_cap[n2]:
                data['transmissionInvCapParam'][(n1, n2, i)] = transmission_inv_cap[n2][i]
            else:
                data['transmissionInvCapParam'][(n1, n2, i)] = 0.0

    data['storPWInvCapParam'] = {}
    data['storENInvCapParam'] = {}
    for (n, b) in data['StoragesOfNode']:
        for i in data['Period']:
            if (n, b) in stor_pw_inv_cap and i in stor_pw_inv_cap[(n, b)]:
                data['storPWInvCapParam'][(n, b, i)] = stor_pw_inv_cap[(n, b)][i]
            else:
                data['storPWInvCapParam'][(n, b, i)] = 0.0
            
            if (n, b) in stor_en_inv_cap and i in stor_en_inv_cap[(n, b)]:
                data['storENInvCapParam'][(n, b, i)] = stor_en_inv_cap[(n, b)][i]
            else:
                data['storENInvCapParam'][(n, b, i)] = 0.0

    calculate_installed_capacities(data)

def calculate_installed_capacities(data):

    data['genInstalledCap'] = {}
    for (n, g) in data['GeneratorsOfNode']:
        for i in data['Period']:
            start_period = max(1, i - data['genLifetime'][g] // data['LeapYearsInvestment'] + 1)
            data['genInstalledCap'][(n, g, i)] = sum(data['genInvCapParam'].get((n, g, j), 0) 
                                                     for j in range(start_period, i + 1)) + data['genInitCap'][n][g][i]

    data['transmissionInstalledCap'] = {}
    for (n1, n2) in data['DirectionalLink']:
        for i in data['Period']:
            start_period = max(1, i - data['transmissionLifetime'][n1, n2] // data['LeapYearsInvestment'] + 1)
            data['transmissionInstalledCap'][(n1, n2, i)] = sum(data['transmissionInvCapParam'].get((n1, n2, j), 0) 
                                                                for j in range(start_period, i + 1)) + data['transmissionInitCap'][n1][n2][i]

    data['storPWInstalledCap'] = {}
    data['storENInstalledCap'] = {}
    for (n, b) in data['StoragesOfNode']:
        for i in data['Period']:
            start_period = max(1, i - data['storageLifetime'][b] // data['LeapYearsInvestment'] + 1)
            data['storPWInstalledCap'][(n, b, i)] = sum(data['storPWInvCapParam'].get((n, b, j), 0) 
                                                        for j in range(start_period, i + 1)) + data['storPWInitCap'][n][b][i]
            data['storENInstalledCap'][(n, b, i)] = sum(data['storENInvCapParam'].get((n, b, j), 0) 
                                                        for j in range(start_period, i + 1)) + data['storENInitCap'][n][b][i]

