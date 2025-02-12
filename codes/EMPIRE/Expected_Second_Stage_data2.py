from __future__ import division
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager
import csv
import sys
import cloudpickle
import time
import os
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import multiprocessing
from datetime import datetime
import json
import shutil
import psutil
import logging
from multiprocessing import Pool





# def run_second_stage_valid(name, tab_file_path, result_file_path, scenariogeneration, scenario_data_path,
#                solver, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
#                lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
#                discountrate, WACC, LeapYearsInvestment, FSD, WRITE_LP,
#                PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE,seed,specific_period,file_num):
    
#     start_time = time.time()
#     if USE_TEMP_DIR:
#         TempfileManager.tempdir = temp_dir
    
#     model = AbstractModel()

#     ##########
#     ##MODULE##
#     ##########
    
#     # scenariopath = tab_file_path
#     scenariopath = f"Data handler/random_sce_original/{seed}"
#     tab_file_path = "Data handler/sampling/full"
    
#     def period_filter(model):
#         return [specific_period]

#     ########
#     ##SETS##
#     ########

#     #Define the sets

#     #Supply technology sets
#     model.Generator = Set(ordered=True) #g
#     model.Technology = Set(ordered=True) #t
#     model.Storage =  Set() #b

#     # Temporal sets
#     model.Period = Set(ordered=True) #max period
#     # model.PeriodActive = Set(initialize=period_filter)
#     model.PeriodActive = Set(ordered=True, initialize=Period) #i
#     model.Operationalhour = Set(ordered=True, initialize=Operationalhour) #h
#     model.Season = Set(ordered=True, initialize=Season) #s

#     #Spatial sets
#     model.Node = Set(ordered=True) #n
#     model.OffshoreNode = Set(ordered=True, within=model.Node) #n
#     model.DirectionalLink = Set(dimen=2, within=model.Node*model.Node, ordered=True) #a
#     model.TransmissionType = Set(ordered=True)

#     #Stochastic sets
#     model.Scenario = Set(ordered=True, initialize=Scenario) #w

#     #Subsets
#     model.GeneratorsOfTechnology=Set(dimen=2) #(t,g) for all t in T, g in G_t
#     model.GeneratorsOfNode = Set(dimen=2) #(n,g) for all n in N, g in G_n
#     model.TransmissionTypeOfDirectionalLink = Set(dimen=3) #(n1,n2,t) for all (n1,n2) in L, t in T
#     model.ThermalGenerators = Set(within=model.Generator) #g_ramp
#     model.RegHydroGenerator = Set(within=model.Generator) #g_reghyd
#     model.HydroGenerator = Set(within=model.Generator) #g_hyd
#     model.StoragesOfNode = Set(dimen=2) #(n,b) for all n in N, b in B_n
#     model.DependentStorage = Set() #b_dagger
#     model.HoursOfSeason = Set(dimen=2, ordered=True, initialize=HoursOfSeason) #(s,h) for all s in S, h in H_s
#     model.FirstHoursOfRegSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfRegSeason)
#     model.FirstHoursOfPeakSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfPeakSeason)

# #    print("Reading sets...")

#     #Load the data

#     data = DataPortal()
#     data.load(filename=tab_file_path + "/" + 'Sets_Generator.tab',format="set", set=model.Generator)
#     data.load(filename=tab_file_path + "/" + 'Sets_ThermalGenerators.tab',format="set", set=model.ThermalGenerators)
#     data.load(filename=tab_file_path + "/" + 'Sets_HydroGenerator.tab',format="set", set=model.HydroGenerator)
#     data.load(filename=tab_file_path + "/" + 'Sets_HydroGeneratorWithReservoir.tab',format="set", set=model.RegHydroGenerator)
#     data.load(filename=tab_file_path + "/" + 'Sets_Storage.tab',format="set", set=model.Storage)
#     data.load(filename=tab_file_path + "/" + 'Sets_DependentStorage.tab',format="set", set=model.DependentStorage)
#     data.load(filename=tab_file_path + "/" + 'Sets_Technology.tab',format="set", set=model.Technology)
#     data.load(filename=tab_file_path + "/" + 'Sets_Node.tab',format="set", set=model.Node)
#     data.load(filename=tab_file_path + "/" + 'Sets_OffshoreNode.tab',format="set", set=model.OffshoreNode)
#     data.load(filename=tab_file_path + "/" + 'Sets_Horizon.tab',format="set", set=model.Period)
#     data.load(filename=tab_file_path + "/" + 'Sets_DirectionalLines.tab',format="set", set=model.DirectionalLink)
#     data.load(filename=tab_file_path + "/" + 'Sets_LineType.tab',format="set", set=model.TransmissionType)
#     data.load(filename=tab_file_path + "/" + 'Sets_LineTypeOfDirectionalLines.tab',format="set", set=model.TransmissionTypeOfDirectionalLink)
#     data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfTechnology.tab',format="set", set=model.GeneratorsOfTechnology)
#     data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfNode.tab',format="set", set=model.GeneratorsOfNode)
#     data.load(filename=tab_file_path + "/" + 'Sets_StorageOfNodes.tab',format="set", set=model.StoragesOfNode)

#     #Build arc subsets

#     def NodesLinked_init(model, node):
#         retval = []
#         for (i,j) in model.DirectionalLink:
#             if j == node:
#                 retval.append(i)
#         return retval
#     model.NodesLinked = Set(model.Node, initialize=NodesLinked_init)

#     def BidirectionalArc_init(model):
#         retval = []
#         for (i,j) in model.DirectionalLink:
#             if i != j and (not (j,i) in retval):
#                 retval.append((i,j))
#         return retval
#     model.BidirectionalArc = Set(dimen=2, initialize=BidirectionalArc_init, ordered=True) #l

    
    
#     ##############
#     ##PARAMETERS##
#     ##############

#     #Define the parameters

#     #Scaling

#     model.discountrate = Param(initialize=discountrate)  
#     model.LeapYearsInvestment = Param(initialize=LeapYearsInvestment)
#     model.operationalDiscountrate = Param(mutable=True)
#     model.sceProbab = Param(model.Scenario, mutable=True)
#     model.seasScale = Param(model.Season, initialize=1.0, mutable=True)
#     model.lengthRegSeason = Param(initialize=lengthRegSeason) 
#     model.lengthPeakSeason = Param(initialize=lengthPeakSeason) 

#     #Cost
#     model.genInvCost = Param(model.Generator, model.Period, default=9000000, mutable=True)
#     model.transmissionInvCost = Param(model.BidirectionalArc, model.Period, default=3000000, mutable=True)
#     model.storPWInvCost = Param(model.Storage, model.Period, default=1000000, mutable=True)
#     model.storENInvCost = Param(model.Storage, model.Period, default=800000, mutable=True)
    

#     model.genVariableOMCost = Param(model.Generator, default=0.0, mutable=True)
#     model.genFuelCost = Param(model.Generator, model.Period, default=0.0, mutable=True)
#     model.genMargCost = Param(model.Generator, model.Period, default=600, mutable=True)
#     model.genCO2TypeFactor = Param(model.Generator, default=0.0, mutable=True)
#     model.nodeLostLoadCost = Param(model.Node, model.Period, default=22000.0)
#     # model.nodeLostLoadCost = Param(model.Node, model.Period, default=10.0)
#     model.CO2price = Param(model.Period, default=0.0, mutable=True)
#     model.CCSCostTSFix = Param(initialize=1149873.72) #NB! Hard-coded
#     model.CCSCostTSVariable = Param(model.Period, default=0.0, mutable=True)
#     model.CCSRemFrac = Param(initialize=0.9)

#     #Node dependent technology limitations

#     model.genRefInitCap = Param(model.GeneratorsOfNode, default=0.0, mutable=True)
#     model.genScaleInitCap = Param(model.Generator, model.Period, default=0.0, mutable=True)
#     model.genInitCap = Param(model.GeneratorsOfNode, model.Period, default=0.0, mutable=True)
#     model.transmissionInitCap = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
#     model.storPWInitCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
#     model.storENInitCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)

#     #Type dependent technology limitations

#     model.genLifetime = Param(model.Generator, default=0.0, mutable=True)
#     model.transmissionLifetime = Param(model.BidirectionalArc, default=40.0, mutable=True)
#     model.storageLifetime = Param(model.Storage, default=0.0, mutable=True)
#     model.genEfficiency = Param(model.Generator, model.Period, default=1.0, mutable=True)
#     model.lineEfficiency = Param(model.DirectionalLink, default=0.97, mutable=True)
#     model.storageChargeEff = Param(model.Storage, default=1.0, mutable=True)
#     model.storageDischargeEff = Param(model.Storage, default=1.0, mutable=True)
#     model.storageBleedEff = Param(model.Storage, default=1.0, mutable=True)
#     model.genRampUpCap = Param(model.ThermalGenerators, default=0.0, mutable=True)
#     model.storageDiscToCharRatio = Param(model.Storage, default=1.0, mutable=True) #NB! Hard-coded
#     model.storagePowToEnergy = Param(model.DependentStorage, default=1.0, mutable=True)

#     #Stochastic input

#     model.sloadRaw = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
#     model.sloadAnnualDemand = Param(model.Node, model.Period, default=0.0, mutable=True)
#     model.sload = Param(model.Node, model.Operationalhour, model.Period, model.Scenario, default=0.0, mutable=True)
#     model.genCapAvailTypeRaw = Param(model.Generator, default=1.0, mutable=True)
#     model.genCapAvailStochRaw = Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
#     model.genCapAvail = Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
#     model.maxRegHydroGenRaw = Param(model.Node, model.Period, model.HoursOfSeason, model.Scenario, default=0.0, mutable=True)
#     model.maxRegHydroGen = Param(model.Node, model.Period, model.Season, model.Scenario, default=0.0, mutable=True)
#     model.maxHydroNode = Param(model.Node, default=0.0, mutable=True)
#     model.storOperationalInit = Param(model.Storage, default=0.0, mutable=True) #Percentage of installed energy capacity initially
    
#     # Define new parameters for FSD data
#     model.genInvCapParam = Param(model.GeneratorsOfNode, model.PeriodActive, default=0.0, mutable=True)
#     model.transmisionInvCapParam = Param(model.BidirectionalArc, model.PeriodActive, default=0.0, mutable=True)
#     model.storPWInvCapParam = Param(model.StoragesOfNode, model.PeriodActive, default=0.0, mutable=True)
#     model.storENInvCapParam = Param(model.StoragesOfNode, model.PeriodActive, default=0.0, mutable=True)
#     model.genInstalledCap = Param(model.GeneratorsOfNode, model.PeriodActive, default=0.0, mutable=True)
#     model.transmissionInstalledCap = Param(model.BidirectionalArc, model.PeriodActive, default=0.0, mutable=True)
#     model.storPWInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, default=0.0, mutable=True)
#     model.storENInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, default=0.0, mutable=True)

#     if EMISSION_CAP:
#         model.CO2cap = Param(model.Period, default=5000.0, mutable=True)
    
#     if LOADCHANGEMODULE:
#         model.sloadMod = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    
#     #Load the parameters

#     data.load(filename=tab_file_path + "/" + 'Generator_VariableOMCosts.tab', param=model.genVariableOMCost, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_FuelCosts.tab', param=model.genFuelCost, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_CCSCostTSVariable.tab', param=model.CCSCostTSVariable, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_Efficiency.tab', param=model.genEfficiency, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_RefInitialCap.tab', param=model.genRefInitCap, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_ScaleFactorInitialCap.tab', param=model.genScaleInitCap, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_InitialCapacity.tab', param=model.genInitCap, format="table") #node_generator_intial_capacity.xlsx
#     data.load(filename=tab_file_path + "/" + 'Generator_CO2Content.tab', param=model.genCO2TypeFactor, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_RampRate.tab', param=model.genRampUpCap, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_GeneratorTypeAvailability.tab', param=model.genCapAvailTypeRaw, format="table")
#     data.load(filename=tab_file_path + "/" + 'Generator_Lifetime.tab', param=model.genLifetime, format="table") 

#     data.load(filename=tab_file_path + "/" + 'Transmission_InitialCapacity.tab', param=model.transmissionInitCap, format="table")
#     data.load(filename=tab_file_path + "/" + 'Transmission_lineEfficiency.tab', param=model.lineEfficiency, format="table")
#     data.load(filename=tab_file_path + "/" + 'Transmission_Lifetime.tab', param=model.transmissionLifetime, format="table")

#     data.load(filename=tab_file_path + "/" + 'Storage_StorageBleedEfficiency.tab', param=model.storageBleedEff, format="table")
#     data.load(filename=tab_file_path + "/" + 'Storage_StorageChargeEff.tab', param=model.storageChargeEff, format="table")
#     data.load(filename=tab_file_path + "/" + 'Storage_StorageDischargeEff.tab', param=model.storageDischargeEff, format="table")
#     data.load(filename=tab_file_path + "/" + 'Storage_StoragePowToEnergy.tab', param=model.storagePowToEnergy, format="table")
    
#     data.load(filename=tab_file_path + "/" + 'Storage_EnergyInitialCapacity.tab', param=model.storENInitCap, format="table")
#     data.load(filename=tab_file_path + "/" + 'Storage_StorageInitialEnergyLevel.tab', param=model.storOperationalInit, format="table")
#     data.load(filename=tab_file_path + "/" + 'Storage_InitialPowerCapacity.tab', param=model.storPWInitCap, format="table")
#     data.load(filename=tab_file_path + "/" + 'Storage_Lifetime.tab', param=model.storageLifetime, format="table")

#     data.load(filename=tab_file_path + "/" + 'Node_NodeLostLoadCost.tab', param=model.nodeLostLoadCost, format="table")
#     data.load(filename=tab_file_path + "/" + 'Node_ElectricAnnualDemand.tab', param=model.sloadAnnualDemand, format="table") 
#     data.load(filename=tab_file_path + "/" + 'Node_HydroGenMaxAnnualProduction.tab', param=model.maxHydroNode, format="table") 
    
#     # It should be passed by inputs of this function
#     data.load(filename=scenariopath + "/" + 'Stochastic_HydroGenMaxSeasonalProduction.tab', param=model.maxRegHydroGenRaw, format="table")
#     data.load(filename=scenariopath + "/" + 'Stochastic_StochasticAvailability.tab', param=model.genCapAvailStochRaw, format="table") 
#     data.load(filename=scenariopath + "/" + 'Stochastic_ElectricLoadRaw.tab', param=model.sloadRaw, format="table") 
#     ##############################################################

#     data.load(filename=tab_file_path + "/" + 'General_seasonScale.tab', param=model.seasScale, format="table") 

#     if EMISSION_CAP:
#         data.load(filename=tab_file_path + "/" + 'General_CO2Cap.tab', param=model.CO2cap, format="table")
#     else:
#         data.load(filename=tab_file_path + "/" + 'General_CO2Price.tab', param=model.CO2price, format="table")

#     if LOADCHANGEMODULE:
#         data.load(filename=scenariopath + "/" + 'LoadchangeModule/Stochastic_ElectricLoadMod.tab', param=model.sloadMod, format="table")

# #    print("Constructing parameter values...")

#     def prepSceProbab_rule(model):
#         #Build an equiprobable probability distribution for scenarios
#         for sce in model.Scenario:
#             model.sceProbab[sce] = value(1/len(model.Scenario))

#     model.build_SceProbab = BuildAction(rule=prepSceProbab_rule)

#     def prepOperationalCostGen_rule(model):
#         #Build generator short term marginal costs

#         for g in model.Generator:
#             for i in model.PeriodActive:
#                 if ('CCS',g) in model.GeneratorsOfTechnology:
#                     costperenergyunit=(3.6/model.genEfficiency[g,i])*(model.genFuelCost[g,i]+(1-model.CCSRemFrac)*model.genCO2TypeFactor[g]*model.CO2price[i])+ \
#                     (3.6/model.genEfficiency[g,i])*(model.CCSRemFrac*model.genCO2TypeFactor[g]*model.CCSCostTSVariable[i])+ \
#                     model.genVariableOMCost[g]
#                 else:
#                     costperenergyunit=(3.6/model.genEfficiency[g,i])*(model.genFuelCost[g,i]+model.genCO2TypeFactor[g]*model.CO2price[i])+ \
#                     model.genVariableOMCost[g]
#                 model.genMargCost[g,i]=costperenergyunit

#     model.build_OperationalCostGen = BuildAction(rule=prepOperationalCostGen_rule)

#     def prepInitialCapacityNodeGen_rule(model):
#         #Build initial capacity for generator type in node

#         for (n,g) in model.GeneratorsOfNode:
#             for i in model.PeriodActive:
#                 if value(model.genInitCap[n,g,i]) == 0:
#                     model.genInitCap[n,g,i] = model.genRefInitCap[n,g]*(1-model.genScaleInitCap[g,i])

#     model.build_InitialCapacityNodeGen = BuildAction(rule=prepInitialCapacityNodeGen_rule)

#     def prepOperationalDiscountrate_rule(model):
#         #Build operational discount rate

#         model.operationalDiscountrate = sum((1+model.discountrate)**(-j) for j in list(range(0,value(model.LeapYearsInvestment))))

#     model.build_operationalDiscountrate = BuildAction(rule=prepOperationalDiscountrate_rule)     

#     def prepRegHydro_rule(model):
#         #Build hydrolimits for all periods

#         for n in model.Node:
#             for s in model.Season:
#                 for i in model.PeriodActive:
#                     for sce in model.Scenario:
#                         model.maxRegHydroGen[n,i,s,sce]=sum(model.maxRegHydroGenRaw[n,i,s,h,sce] for h in model.Operationalhour if (s,h) in model.HoursOfSeason)

#     model.build_maxRegHydroGen = BuildAction(rule=prepRegHydro_rule)

#     def prepGenCapAvail_rule(model):
#         #Build generator availability for all periods

#         for (n,g) in model.GeneratorsOfNode:
#             for h in model.Operationalhour:
#                 for s in model.Scenario:
#                     for i in model.PeriodActive:
#                         if value(model.genCapAvailTypeRaw[g]) == 0:
#                             model.genCapAvail[n,g,h,s,i]=model.genCapAvailStochRaw[n,g,h,s,i]
#                         else:
#                             model.genCapAvail[n,g,h,s,i]=model.genCapAvailTypeRaw[g]

#     model.build_genCapAvail = BuildAction(rule=prepGenCapAvail_rule)

#     def prepSload_rule(model):
#         #Build load profiles for all periods

#         counter = 0
# #       f = open(result_file_path + '/AdjustedNegativeLoad_' + name + '.txt', 'w')
#         for n in model.Node:
#             for i in model.PeriodActive:
#                 noderawdemand = 0
#                 for (s,h) in model.HoursOfSeason:
#                     if value(h) < value(FirstHoursOfRegSeason[-1] + model.lengthRegSeason):
#                         for sce in model.Scenario:
#                                 noderawdemand += value(model.sceProbab[sce]*model.seasScale[s]*model.sloadRaw[n,h,sce,i])
#                 if value(model.sloadAnnualDemand[n,i]) < 1:
#                     hourlyscale = 0
#                 else:
#                     hourlyscale = value(model.sloadAnnualDemand[n,i]) / noderawdemand
#                 for h in model.Operationalhour:
#                     for sce in model.Scenario:
#                         model.sload[n, h, i, sce] = model.sloadRaw[n,h,sce,i]*hourlyscale
#                         if LOADCHANGEMODULE:
#                             model.sload[n,h,i,sce] = model.sload[n,h,i,sce] + model.sloadMod[n,h,sce,i]
#                         if value(model.sload[n,h,i,sce]) < 0:
# #                            f.write('Adjusted electricity load: ' + str(value(model.sload[n,h,i,sce])) + ', 10 MW for hour ' + str(h) + ' and scenario ' + str(sce) + ' in ' + str(n) + "\n")
#                             model.sload[n,h,i,sce] = 10
#                             counter += 1

# #        f.write('Hours with too small raw electricity load: ' + str(counter))
# #        f.close()
    
#     model.build_sload = BuildAction(rule=prepSload_rule)
    
    
#     #############
#     ##VARIABLES##
#     #############

#     ## Second Stage Decisions ##
#     model.genOperational = Var(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
#     model.storOperational = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
#     model.transmisionOperational = Var(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals) #flow
#     model.storCharge = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
#     model.storDischarge = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
#     model.loadShed = Var(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)

#     def multiplier_rule(model,period):
#         coeff=1
#         if period>1:
#             coeff=pow(1.0+model.discountrate,(-LeapYearsInvestment*(int(period)-1)))
#         return coeff
#     model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)

#     def shed_component_rule(model,i):
#         return sum(model.operationalDiscountrate*model.seasScale[s]*model.sceProbab[w]*model.nodeLostLoadCost[n,i]*model.loadShed[n,h,i,w] for n in model.Node for w in model.Scenario for (s,h) in model.HoursOfSeason)
#     model.shedcomponent=Expression(model.PeriodActive,rule=shed_component_rule)

#     def operational_cost_rule(model,i):
#         return sum(model.operationalDiscountrate*model.seasScale[s]*model.sceProbab[w]*model.genMargCost[g,i]*model.genOperational[n,g,h,i,w] for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason for w in model.Scenario)
#     model.operationalcost=Expression(model.PeriodActive,rule=operational_cost_rule)

#     #############
#     ##OBJECTIVE##
#     #############

#     def Obj_rule(model):
#         return sum(model.discount_multiplier[i]*(model.shedcomponent[i] + model.operationalcost[i]) for i in model.PeriodActive)
#     model.Obj = Objective(rule=Obj_rule, sense=minimize)

#     ###############
#     ##CONSTRAINTS##
#     ###############

#     def FlowBalance_rule(model, n, h, i, w):
#         return sum(model.genOperational[n,g,h,i,w] for g in model.Generator if (n,g) in model.GeneratorsOfNode) \
#             + sum((model.storageDischargeEff[b]*model.storDischarge[n,b,h,i,w]-model.storCharge[n,b,h,i,w]) for b in model.Storage if (n,b) in model.StoragesOfNode) \
#             + sum((model.lineEfficiency[link,n]*model.transmisionOperational[link,n,h,i,w] - model.transmisionOperational[n,link,h,i,w]) for link in model.NodesLinked[n]) \
#             - model.sload[n,h,i,w] + model.loadShed[n,h,i,w] \
#             == 0
#     model.FlowBalance = Constraint(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, rule=FlowBalance_rule)

#     #################################################################

#     def genMaxProd_rule(model, n, g, h, i, w):
#             return model.genOperational[n,g,h,i,w] - model.genCapAvail[n,g,h,w,i]*model.genInstalledCap[n,g,i] <= 0
#     model.maxGenProduction = Constraint(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=genMaxProd_rule)

#     #################################################################

#     def ramping_rule(model, n, g, h, i, w):
#         if h in model.FirstHoursOfRegSeason or h in model.FirstHoursOfPeakSeason:
#             return Constraint.Skip
#         else:
#             if g in model.ThermalGenerators:
#                 return model.genOperational[n,g,h,i,w]-model.genOperational[n,g,(h-1),i,w] - model.genRampUpCap[g]*model.genInstalledCap[n,g,i] <= 0   #
#             else:
#                 return Constraint.Skip
#     model.ramping = Constraint(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=ramping_rule)

#     #################################################################

#     def storage_energy_balance_rule(model, n, b, h, i, w):
#         if h in model.FirstHoursOfRegSeason or h in model.FirstHoursOfPeakSeason:
#             return model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] + model.storageChargeEff[b]*model.storCharge[n,b,h,i,w]-model.storDischarge[n,b,h,i,w]-model.storOperational[n,b,h,i,w] == 0   #
#         else:
#             return model.storageBleedEff[b]*model.storOperational[n,b,(h-1),i,w] + model.storageChargeEff[b]*model.storCharge[n,b,h,i,w]-model.storDischarge[n,b,h,i,w]-model.storOperational[n,b,h,i,w] == 0   #
#     model.storage_energy_balance = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_energy_balance_rule)

#     #################################################################

#     def storage_seasonal_net_zero_balance_rule(model, n, b, h, i, w):
#         if h in model.FirstHoursOfRegSeason:
#             return model.storOperational[n,b,h+value(model.lengthRegSeason)-1,i,w] - model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] == 0  #
#         elif h in model.FirstHoursOfPeakSeason:
#             return model.storOperational[n,b,h+value(model.lengthPeakSeason)-1,i,w] - model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] == 0  #
#         else:
#             return Constraint.Skip
#     model.storage_seasonal_net_zero_balance = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_seasonal_net_zero_balance_rule)

#     #################################################################

#     def storage_operational_cap_rule(model, n, b, h, i, w):
#         return model.storOperational[n,b,h,i,w] - model.storENInstalledCap[n,b,i]  <= 0   #
#     model.storage_operational_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_operational_cap_rule)

#     #################################################################

#     def storage_power_discharg_cap_rule(model, n, b, h, i, w):
#         return model.storDischarge[n,b,h,i,w] - model.storageDiscToCharRatio[b]*model.storPWInstalledCap[n,b,i] <= 0   #
#     model.storage_power_discharg_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_power_discharg_cap_rule)

#     #################################################################

#     def storage_power_charg_cap_rule(model, n, b, h, i, w):
#         return model.storCharge[n,b,h,i,w] - model.storPWInstalledCap[n,b,i] <= 0   #
#     model.storage_power_charg_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_power_charg_cap_rule)

#     #################################################################

#     def hydro_node_limit_rule(model, n, i):
#         return sum(model.genOperational[n,g,h,i,w]*model.seasScale[s]*model.sceProbab[w] for g in model.HydroGenerator if (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason for w in model.Scenario) - model.maxHydroNode[n] <= 0   #
#     model.hydro_node_limit = Constraint(model.Node, model.PeriodActive, rule=hydro_node_limit_rule)

#     #################################################################

#     def hydro_gen_limit_rule(model, n, g, s, i, w):
#         if g in model.RegHydroGenerator:
#             return sum(model.genOperational[n,g,h,i,w] for h in model.Operationalhour if (s,h) in model.HoursOfSeason) - model.maxRegHydroGen[n,i,s,w] <= 0
#         else:
#             return Constraint.Skip  #
#     model.hydro_gen_limit = Constraint(model.GeneratorsOfNode, model.Season, model.PeriodActive, model.Scenario, rule=hydro_gen_limit_rule)

#     #################################################################

#     def transmission_cap_rule(model, n1, n2, h, i, w):
#         if (n1,n2) in model.BidirectionalArc:
#             return model.transmisionOperational[(n1,n2),h,i,w]  - model.transmissionInstalledCap[(n1,n2),i] <= 0
#         elif (n2,n1) in model.BidirectionalArc:
#             return model.transmisionOperational[(n1,n2),h,i,w]  - model.transmissionInstalledCap[(n2,n1),i] <= 0
#     model.transmission_cap = Constraint(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.Scenario, rule=transmission_cap_rule)

#     #################################################################

#     if EMISSION_CAP:
#     	def emission_cap_rule(model, i, w):
#     	    return sum(model.seasScale[s]*model.genCO2TypeFactor[g]*(3.6/model.genEfficiency[g,i])*model.genOperational[n,g,h,i,w] for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason)/1000000 \
#     	        - model.CO2cap[i] <= 0   #
#     	model.emission_cap = Constraint(model.PeriodActive, model.Scenario, rule=emission_cap_rule)



#     instance = model.create_instance(data) #, report_timing=True)
#     instance.dual = Suffix(direction=Suffix.IMPORT) #Make sure the dual value is collected into solver results (if solver supplies dual information)



#     # Load FSD data
#     gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(FSD)

#     for (n, g) in instance.GeneratorsOfNode:
#             if (n, g) in gen_inv_cap:
#                 for i in instance.PeriodActive:
#                     if i in gen_inv_cap[(n, g)]:
#                         cap_value = gen_inv_cap[(n, g)][i]
#                         instance.genInvCapParam[n, g, i] = cap_value
#             else:
#                 print(f"(n, g) = ({n}, {g}): Not found in gen_inv_cap")

#     # Transmission
#     for (n1, n2) in instance.BidirectionalArc:
#         if (n1, n2) in transmission_inv_cap:
#             for i in instance.PeriodActive:
#                 if i in transmission_inv_cap[(n1, n2)]:
#                     cap_value = transmission_inv_cap[(n1, n2)][i]
#                     instance.transmisionInvCapParam[n1, n2, i] = cap_value
#         else:
#             print(f"(n1, n2) = ({n1}, {n2}): Not found in transmission_inv_cap")

#     # Storage
#     for (n, b) in instance.StoragesOfNode:
#         if (n, b) in stor_pw_inv_cap:
#             for i in instance.PeriodActive:
#                 if i in stor_pw_inv_cap[(n, b)]:
#                     cap_value = stor_pw_inv_cap[(n, b)][i]
#                     instance.storPWInvCapParam[n, b, i] = cap_value
#         else:
#             print(f"(n, b) = ({n}, {b}): Not found in stor_pw_inv_cap")

#         if (n, b) in stor_en_inv_cap:
#             for i in instance.PeriodActive:
#                 if i in stor_en_inv_cap[(n, b)]:
#                     cap_value = stor_en_inv_cap[(n, b)][i]
#                     instance.storENInvCapParam[n, b, i] = cap_value
#         else:
#             print(f"(n, b) = ({n}, {b}): Not found in stor_en_inv_cap")



#     calculate_gen_installed_cap(instance)
#     calculate_transmission_installed_cap(instance)
#     calculate_storage_installed_cap(instance)
    

#     opt = SolverFactory('gurobi', Verbose=True)
#     opt.options["Crossover"]=0
#     # opt.options["Method"]=2
#     opt.options['threads'] = 1
#     opt.options['BarConvTol'] = 1e-4

#     results = opt.solve(instance, tee=True) 

#     obj_value = value(instance.Obj)
#     shed_cost_lst = [value(instance.shedcomponent[i]) for i in instance.PeriodActive]
#     shed_cost = np.mean(shed_cost_lst)

#     print(f"original (accross period) obj_value : {obj_value}, shed cost : {shed_cost}")
    
#     # Let's also calculate what the sum should be based on components
#     total_by_components = sum(
#         value(instance.discount_multiplier[i]) * 
#         (value(instance.shedcomponent[i]) + value(instance.operationalcost[i]))
#         for i in instance.PeriodActive
#     )
#     print(f"Calculated from components: {total_by_components}")




#     return obj_value









def calculate_gen_installed_cap(instance):
    for (n, g) in instance.GeneratorsOfNode:
        for i in instance.Period:
            startPeriod = max(1, value(1 + i - instance.genLifetime[g] / instance.LeapYearsInvestment))
            instance.genInstalledCap[n, g, i] = sum(instance.genInvCapParam[n, g, j]
                                                for j in instance.Period
                                                if startPeriod <= j <= i) + instance.genInitCap[n, g, i]

def calculate_transmission_installed_cap(instance):
    for (n1, n2) in instance.BidirectionalArc:
        for i in instance.Period:
            startPeriod = max(1, value(1 + i - instance.transmissionLifetime[n1, n2] / instance.LeapYearsInvestment))
            instance.transmissionInstalledCap[n1, n2, i] = sum(instance.transmisionInvCapParam[n1, n2, j]
                                                            for j in instance.Period
                                                            if startPeriod <= j <= i) + instance.transmissionInitCap[n1, n2, i]

def calculate_storage_installed_cap(instance):
    for (n, b) in instance.StoragesOfNode:
        for i in instance.Period:
            startPeriod = max(1, value(1 + i - instance.storageLifetime[b] / instance.LeapYearsInvestment))
            instance.storPWInstalledCap[n, b, i] = sum(instance.storPWInvCapParam[n, b, j]
                                                    for j in instance.Period
                                                    if startPeriod <= j <= i) + instance.storPWInitCap[n, b, i]
            instance.storENInstalledCap[n, b, i] = sum(instance.storENInvCapParam[n, b, j]
                                                    for j in instance.Period
                                                    if startPeriod <= j <= i) + instance.storENInitCap[n, b, i]


def run_second_stage(name, tab_file_path, result_file_path, scenariogeneration, scenario_data_path,
               solver, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
               lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
               discountrate, WACC, LeapYearsInvestment, FSD, WRITE_LP,
               PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE,seed,specific_period,file_num):

    start_time = time.time()
    if USE_TEMP_DIR:
        TempfileManager.tempdir = temp_dir

    model = AbstractModel()

    ##########
    ##MODULE##
    ##########
    
    # scenariopath = tab_file_path
    scenariopath = f"Data handler/random_sce_reduced/{seed}"
    tab_file_path = "Data handler/sampling/reduced"

    # scenariopath = f"Data handler/random_sce_original/{seed}"
    # tab_file_path = "Data handler/sampling/full"
    
    def period_filter(model):
        return [specific_period]

    ########
    ##SETS##
    ########

    #Define the sets

    #Supply technology sets
    model.Generator = Set(ordered=True) #g
    model.Technology = Set(ordered=True) #t
    model.Storage =  Set() #b

    # Temporal sets
    model.Period = Set(ordered=True) #max period
    model.PeriodActive = Set(initialize=period_filter)
    # model.PeriodActive = Set(ordered=True, initialize=Period) #i
    model.Operationalhour = Set(ordered=True, initialize=Operationalhour) #h
    model.Season = Set(ordered=True, initialize=Season) #s

    #Spatial sets
    model.Node = Set(ordered=True) #n
    model.OffshoreNode = Set(ordered=True, within=model.Node) #n
    model.DirectionalLink = Set(dimen=2, within=model.Node*model.Node, ordered=True) #a
    model.TransmissionType = Set(ordered=True)

    #Stochastic sets
    model.Scenario = Set(ordered=True, initialize=Scenario) #w

    #Subsets
    model.GeneratorsOfTechnology=Set(dimen=2) #(t,g) for all t in T, g in G_t
    model.GeneratorsOfNode = Set(dimen=2) #(n,g) for all n in N, g in G_n
    model.TransmissionTypeOfDirectionalLink = Set(dimen=3) #(n1,n2,t) for all (n1,n2) in L, t in T
    model.ThermalGenerators = Set(within=model.Generator) #g_ramp
    model.RegHydroGenerator = Set(within=model.Generator) #g_reghyd
    model.HydroGenerator = Set(within=model.Generator) #g_hyd
    model.StoragesOfNode = Set(dimen=2) #(n,b) for all n in N, b in B_n
    model.DependentStorage = Set() #b_dagger
    model.HoursOfSeason = Set(dimen=2, ordered=True, initialize=HoursOfSeason) #(s,h) for all s in S, h in H_s
    model.FirstHoursOfRegSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfRegSeason)
    model.FirstHoursOfPeakSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfPeakSeason)

#    print("Reading sets...")

    #Load the data

    data = DataPortal()
    data.load(filename=tab_file_path + "/" + 'Sets_Generator.tab',format="set", set=model.Generator)
    data.load(filename=tab_file_path + "/" + 'Sets_ThermalGenerators.tab',format="set", set=model.ThermalGenerators)
    data.load(filename=tab_file_path + "/" + 'Sets_HydroGenerator.tab',format="set", set=model.HydroGenerator)
    data.load(filename=tab_file_path + "/" + 'Sets_HydroGeneratorWithReservoir.tab',format="set", set=model.RegHydroGenerator)
    data.load(filename=tab_file_path + "/" + 'Sets_Storage.tab',format="set", set=model.Storage)
    data.load(filename=tab_file_path + "/" + 'Sets_DependentStorage.tab',format="set", set=model.DependentStorage)
    data.load(filename=tab_file_path + "/" + 'Sets_Technology.tab',format="set", set=model.Technology)
    data.load(filename=tab_file_path + "/" + 'Sets_Node.tab',format="set", set=model.Node)
    data.load(filename=tab_file_path + "/" + 'Sets_OffshoreNode.tab',format="set", set=model.OffshoreNode)
    data.load(filename=tab_file_path + "/" + 'Sets_Horizon.tab',format="set", set=model.Period)
    data.load(filename=tab_file_path + "/" + 'Sets_DirectionalLines.tab',format="set", set=model.DirectionalLink)
    data.load(filename=tab_file_path + "/" + 'Sets_LineType.tab',format="set", set=model.TransmissionType)
    data.load(filename=tab_file_path + "/" + 'Sets_LineTypeOfDirectionalLines.tab',format="set", set=model.TransmissionTypeOfDirectionalLink)
    data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfTechnology.tab',format="set", set=model.GeneratorsOfTechnology)
    data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfNode.tab',format="set", set=model.GeneratorsOfNode)
    data.load(filename=tab_file_path + "/" + 'Sets_StorageOfNodes.tab',format="set", set=model.StoragesOfNode)

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

    #Scaling

    model.discountrate = Param(initialize=discountrate)  
    model.LeapYearsInvestment = Param(initialize=LeapYearsInvestment)
    model.operationalDiscountrate = Param(mutable=True)
    model.sceProbab = Param(model.Scenario, mutable=True)
    model.seasScale = Param(model.Season, initialize=1.0, mutable=True)
    model.lengthRegSeason = Param(initialize=lengthRegSeason) 
    model.lengthPeakSeason = Param(initialize=lengthPeakSeason) 

    #Cost
    model.genInvCost = Param(model.Generator, model.Period, default=9000000, mutable=True)
    model.transmissionInvCost = Param(model.BidirectionalArc, model.Period, default=3000000, mutable=True)
    model.storPWInvCost = Param(model.Storage, model.Period, default=1000000, mutable=True)
    model.storENInvCost = Param(model.Storage, model.Period, default=800000, mutable=True)
    

    model.genVariableOMCost = Param(model.Generator, default=0.0, mutable=True)
    model.genFuelCost = Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genMargCost = Param(model.Generator, model.Period, default=600, mutable=True)
    model.genCO2TypeFactor = Param(model.Generator, default=0.0, mutable=True)
    model.nodeLostLoadCost = Param(model.Node, model.Period, default=22000.0)
    model.CO2price = Param(model.Period, default=0.0, mutable=True)
    model.CCSCostTSFix = Param(initialize=1149873.72) #NB! Hard-coded
    model.CCSCostTSVariable = Param(model.Period, default=0.0, mutable=True)
    model.CCSRemFrac = Param(initialize=0.9)

    #Node dependent technology limitations

    model.genRefInitCap = Param(model.GeneratorsOfNode, default=0.0, mutable=True)
    model.genScaleInitCap = Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genInitCap = Param(model.GeneratorsOfNode, model.Period, default=0.0, mutable=True)
    model.transmissionInitCap = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWInitCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENInitCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)

    #Type dependent technology limitations

    model.genLifetime = Param(model.Generator, default=0.0, mutable=True)
    model.transmissionLifetime = Param(model.BidirectionalArc, default=40.0, mutable=True)
    model.storageLifetime = Param(model.Storage, default=0.0, mutable=True)
    model.genEfficiency = Param(model.Generator, model.Period, default=1.0, mutable=True)
    model.lineEfficiency = Param(model.DirectionalLink, default=0.97, mutable=True)
    model.storageChargeEff = Param(model.Storage, default=1.0, mutable=True)
    model.storageDischargeEff = Param(model.Storage, default=1.0, mutable=True)
    model.storageBleedEff = Param(model.Storage, default=1.0, mutable=True)
    model.genRampUpCap = Param(model.ThermalGenerators, default=0.0, mutable=True)
    model.storageDiscToCharRatio = Param(model.Storage, default=1.0, mutable=True) #NB! Hard-coded
    model.storagePowToEnergy = Param(model.DependentStorage, default=1.0, mutable=True)

    #Stochastic input

    model.sloadRaw = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    model.sloadAnnualDemand = Param(model.Node, model.Period, default=0.0, mutable=True)
    model.sload = Param(model.Node, model.Operationalhour, model.Period, model.Scenario, default=0.0, mutable=True)
    model.genCapAvailTypeRaw = Param(model.Generator, default=1.0, mutable=True)
    model.genCapAvailStochRaw = Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    model.genCapAvail = Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    model.maxRegHydroGenRaw = Param(model.Node, model.Period, model.HoursOfSeason, model.Scenario, default=0.0, mutable=True)
    model.maxRegHydroGen = Param(model.Node, model.Period, model.Season, model.Scenario, default=0.0, mutable=True)
    model.maxHydroNode = Param(model.Node, default=0.0, mutable=True)
    model.storOperationalInit = Param(model.Storage, default=0.0, mutable=True) #Percentage of installed energy capacity initially
    
    # Define new parameters for FSD data
    model.genInvCapParam = Param(model.GeneratorsOfNode, model.Period, default=0.0, mutable=True)
    model.transmisionInvCapParam = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWInvCapParam = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENInvCapParam = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.genInstalledCap = Param(model.GeneratorsOfNode, model.Period, default=0.0, mutable=True)
    model.transmissionInstalledCap = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWInstalledCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENInstalledCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)

    if EMISSION_CAP:
        model.CO2cap = Param(model.Period, default=5000.0, mutable=True)
    
    if LOADCHANGEMODULE:
        model.sloadMod = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    
    #Load the parameters

    data.load(filename=tab_file_path + "/" + 'Generator_VariableOMCosts.tab', param=model.genVariableOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_FuelCosts.tab', param=model.genFuelCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_CCSCostTSVariable.tab', param=model.CCSCostTSVariable, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_Efficiency.tab', param=model.genEfficiency, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_RefInitialCap.tab', param=model.genRefInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_ScaleFactorInitialCap.tab', param=model.genScaleInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_InitialCapacity.tab', param=model.genInitCap, format="table") #node_generator_intial_capacity.xlsx
    data.load(filename=tab_file_path + "/" + 'Generator_CO2Content.tab', param=model.genCO2TypeFactor, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_RampRate.tab', param=model.genRampUpCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_GeneratorTypeAvailability.tab', param=model.genCapAvailTypeRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_Lifetime.tab', param=model.genLifetime, format="table") 

    data.load(filename=tab_file_path + "/" + 'Transmission_InitialCapacity.tab', param=model.transmissionInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_lineEfficiency.tab', param=model.lineEfficiency, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_Lifetime.tab', param=model.transmissionLifetime, format="table")

    data.load(filename=tab_file_path + "/" + 'Storage_StorageBleedEfficiency.tab', param=model.storageBleedEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageChargeEff.tab', param=model.storageChargeEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageDischargeEff.tab', param=model.storageDischargeEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StoragePowToEnergy.tab', param=model.storagePowToEnergy, format="table")
    
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyInitialCapacity.tab', param=model.storENInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageInitialEnergyLevel.tab', param=model.storOperationalInit, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_InitialPowerCapacity.tab', param=model.storPWInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_Lifetime.tab', param=model.storageLifetime, format="table")

    data.load(filename=tab_file_path + "/" + 'Node_NodeLostLoadCost.tab', param=model.nodeLostLoadCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Node_ElectricAnnualDemand.tab', param=model.sloadAnnualDemand, format="table") 
    data.load(filename=tab_file_path + "/" + 'Node_HydroGenMaxAnnualProduction.tab', param=model.maxHydroNode, format="table") 
    
    # It should be passed by inputs of this function
    data.load(filename=scenariopath + "/" + 'Stochastic_HydroGenMaxSeasonalProduction.tab', param=model.maxRegHydroGenRaw, format="table")
    data.load(filename=scenariopath + "/" + 'Stochastic_StochasticAvailability.tab', param=model.genCapAvailStochRaw, format="table") 
    data.load(filename=scenariopath + "/" + 'Stochastic_ElectricLoadRaw.tab', param=model.sloadRaw, format="table") 
    ##############################################################

    data.load(filename=tab_file_path + "/" + 'General_seasonScale.tab', param=model.seasScale, format="table") 

    if EMISSION_CAP:
        data.load(filename=tab_file_path + "/" + 'General_CO2Cap.tab', param=model.CO2cap, format="table")
    else:
        data.load(filename=tab_file_path + "/" + 'General_CO2Price.tab', param=model.CO2price, format="table")

    if LOADCHANGEMODULE:
        data.load(filename=scenariopath + "/" + 'LoadchangeModule/Stochastic_ElectricLoadMod.tab', param=model.sloadMod, format="table")

#    print("Constructing parameter values...")

    def prepSceProbab_rule(model):
        #Build an equiprobable probability distribution for scenarios
        for sce in model.Scenario:
            model.sceProbab[sce] = value(1/len(model.Scenario))

    model.build_SceProbab = BuildAction(rule=prepSceProbab_rule)

    def prepOperationalCostGen_rule(model):
        #Build generator short term marginal costs

        for g in model.Generator:
            for i in model.PeriodActive:
                if ('CCS',g) in model.GeneratorsOfTechnology:
                    costperenergyunit=(3.6/model.genEfficiency[g,i])*(model.genFuelCost[g,i]+(1-model.CCSRemFrac)*model.genCO2TypeFactor[g]*model.CO2price[i])+ \
                    (3.6/model.genEfficiency[g,i])*(model.CCSRemFrac*model.genCO2TypeFactor[g]*model.CCSCostTSVariable[i])+ \
                    model.genVariableOMCost[g]
                else:
                    costperenergyunit=(3.6/model.genEfficiency[g,i])*(model.genFuelCost[g,i]+model.genCO2TypeFactor[g]*model.CO2price[i])+ \
                    model.genVariableOMCost[g]
                model.genMargCost[g,i]=costperenergyunit

    model.build_OperationalCostGen = BuildAction(rule=prepOperationalCostGen_rule)

    def prepInitialCapacityNodeGen_rule(model):
        #Build initial capacity for generator type in node

        for (n,g) in model.GeneratorsOfNode:
            for i in model.PeriodActive:
                if value(model.genInitCap[n,g,i]) == 0:
                    model.genInitCap[n,g,i] = model.genRefInitCap[n,g]*(1-model.genScaleInitCap[g,i])

    model.build_InitialCapacityNodeGen = BuildAction(rule=prepInitialCapacityNodeGen_rule)

    def prepOperationalDiscountrate_rule(model):
        #Build operational discount rate

        model.operationalDiscountrate = sum((1+model.discountrate)**(-j) for j in list(range(0,value(model.LeapYearsInvestment))))

    model.build_operationalDiscountrate = BuildAction(rule=prepOperationalDiscountrate_rule)     

    def prepRegHydro_rule(model):
        #Build hydrolimits for all periods

        for n in model.Node:
            for s in model.Season:
                for i in model.PeriodActive:
                    for sce in model.Scenario:
                        model.maxRegHydroGen[n,i,s,sce]=sum(model.maxRegHydroGenRaw[n,i,s,h,sce] for h in model.Operationalhour if (s,h) in model.HoursOfSeason)

    model.build_maxRegHydroGen = BuildAction(rule=prepRegHydro_rule)

    def prepGenCapAvail_rule(model):
        #Build generator availability for all periods

        for (n,g) in model.GeneratorsOfNode:
            for h in model.Operationalhour:
                for s in model.Scenario:
                    for i in model.PeriodActive:
                        if value(model.genCapAvailTypeRaw[g]) == 0:
                            model.genCapAvail[n,g,h,s,i]=model.genCapAvailStochRaw[n,g,h,s,i]
                        else:
                            model.genCapAvail[n,g,h,s,i]=model.genCapAvailTypeRaw[g]

    model.build_genCapAvail = BuildAction(rule=prepGenCapAvail_rule)

    def prepSload_rule(model):
        #Build load profiles for all periods

        counter = 0
#       f = open(result_file_path + '/AdjustedNegativeLoad_' + name + '.txt', 'w')
        for n in model.Node:
            for i in model.PeriodActive:
                noderawdemand = 0
                for (s,h) in model.HoursOfSeason:
                    if value(h) < value(FirstHoursOfRegSeason[-1] + model.lengthRegSeason):
                        for sce in model.Scenario:
                                noderawdemand += value(model.sceProbab[sce]*model.seasScale[s]*model.sloadRaw[n,h,sce,i])
                if value(model.sloadAnnualDemand[n,i]) < 1:
                    hourlyscale = 0
                else:
                    hourlyscale = value(model.sloadAnnualDemand[n,i]) / noderawdemand
                for h in model.Operationalhour:
                    for sce in model.Scenario:
                        model.sload[n, h, i, sce] = model.sloadRaw[n,h,sce,i]*hourlyscale
                        if LOADCHANGEMODULE:
                            model.sload[n,h,i,sce] = model.sload[n,h,i,sce] + model.sloadMod[n,h,sce,i]
                        if value(model.sload[n,h,i,sce]) < 0:
#                            f.write('Adjusted electricity load: ' + str(value(model.sload[n,h,i,sce])) + ', 10 MW for hour ' + str(h) + ' and scenario ' + str(sce) + ' in ' + str(n) + "\n")
                            model.sload[n,h,i,sce] = 10
                            counter += 1

#        f.write('Hours with too small raw electricity load: ' + str(counter))
#        f.close()
    
    model.build_sload = BuildAction(rule=prepSload_rule)
    
    
    #############
    ##VARIABLES##
    #############

    ## Second Stage Decisions ##
    model.genOperational = Var(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.storOperational = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.transmisionOperational = Var(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals) #flow
    model.storCharge = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.storDischarge = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.loadShed = Var(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
   


    def multiplier_rule(model,period):
        coeff=1
        if period>1:
            coeff=pow(1.0+model.discountrate,(-LeapYearsInvestment*(int(period)-1)))
        return coeff
    model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)

    def shed_component_rule(model,i):
        return sum(model.operationalDiscountrate*model.seasScale[s]*model.sceProbab[w]*model.nodeLostLoadCost[n,i]*model.loadShed[n,h,i,w] for n in model.Node for w in model.Scenario for (s,h) in model.HoursOfSeason)
    model.shedcomponent=Expression(model.PeriodActive,rule=shed_component_rule)

    def operational_cost_rule(model,i):
        return sum(model.operationalDiscountrate*model.seasScale[s]*model.sceProbab[w]*model.genMargCost[g,i]*model.genOperational[n,g,h,i,w] for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason for w in model.Scenario)
    model.operationalcost=Expression(model.PeriodActive,rule=operational_cost_rule)

    #############
    ##OBJECTIVE##
    #############

    def Obj_rule(model):
        i = list(model.PeriodActive)[0]  # specific period
        return model.discount_multiplier[i]*(model.shedcomponent[i] + model.operationalcost[i])
    model.Obj = Objective(rule=Obj_rule, sense=minimize)

    # model.operationalDiscountrate*
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

    
    if EMISSION_CAP:
    	def emission_cap_rule(model, i, w):
    	    return sum(model.seasScale[s]*model.genCO2TypeFactor[g]*(3.6/model.genEfficiency[g,i])*model.genOperational[n,g,h,i,w] for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason)/1000000 \
    	        - model.CO2cap[i] <= 0   #
    	model.emission_cap = Constraint(model.PeriodActive, model.Scenario, rule=emission_cap_rule)

    
    #######
    ##RUN##
    #######

    instance = model.create_instance(data) #, report_timing=True)
    instance.dual = Suffix(direction=Suffix.IMPORT) #Make sure the dual value is collected into solver results (if solver supplies dual information)

    gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(FSD)

    for (n, g) in instance.GeneratorsOfNode:
        if (n, g) in gen_inv_cap:
            for i in instance.Period:
                if i in gen_inv_cap[(n, g)]:
                    cap_value = gen_inv_cap[(n, g)][i]
                    instance.genInvCapParam[n, g, i] = cap_value
        else:
            print(f"(n, g) = ({n}, {g}): Not found in gen_inv_cap")

    # Transmission
    for (n1, n2) in instance.BidirectionalArc:
        if (n1, n2) in transmission_inv_cap:
            for i in instance.Period:
                if i in transmission_inv_cap[(n1, n2)]:
                    cap_value = transmission_inv_cap[(n1, n2)][i]
                    instance.transmisionInvCapParam[n1, n2, i] = cap_value
        else:
            print(f"(n1, n2) = ({n1}, {n2}): Not found in transmission_inv_cap")

    # Storage
    for (n, b) in instance.StoragesOfNode:
        if (n, b) in stor_pw_inv_cap:
            for i in instance.Period:
                if i in stor_pw_inv_cap[(n, b)]:
                    cap_value = stor_pw_inv_cap[(n, b)][i]
                    instance.storPWInvCapParam[n, b, i] = cap_value
        else:
            print(f"(n, b) = ({n}, {b}): Not found in stor_pw_inv_cap")

        if (n, b) in stor_en_inv_cap:
            for i in instance.Period:
                if i in stor_en_inv_cap[(n, b)]:
                    cap_value = stor_en_inv_cap[(n, b)][i]
                    instance.storENInvCapParam[n, b, i] = cap_value
        else:
            print(f"(n, b) = ({n}, {b}): Not found in stor_en_inv_cap")


    calculate_gen_installed_cap(instance)
    calculate_transmission_installed_cap(instance)
    calculate_storage_installed_cap(instance)


    opt = SolverFactory('gurobi', Verbose=True)
    opt.options["Crossover"]=0
    # opt.options["Method"]=2
    opt.options['threads'] = 1
    opt.options['BarConvTol'] = 1e-4

    results = opt.solve(instance, tee=False) 
    first_stage_val = calculate_f_x(instance)
    v_dict = results_saving(instance)

    obj_value = value(instance.Obj)
    # shed_cost = value(instance.shedcomponent[specific_period])

    # if shed_cost > 0:
    #     infeasi_idx = 1
    # else: 
    #     infeasi_idx = 0
    
    # return first_stage_val, obj_value, v_dict, infeasi_idx 

    second_else_value, ll_amt = calculate_second_stage(instance)

    return first_stage_val, obj_value, second_else_value, v_dict, ll_amt

    
    # # if os.path.exists(scenariopath):
    # #     shutil.rmtree(scenariopath)






def load_investment_data(fsd_data):
    gen_inv_cap = {}
    transmission_inv_cap = {}
    stor_pw_inv_cap = {}
    stor_en_inv_cap = {}
    
    for row in fsd_data:
        country, energy_type, period, type_, cap_value = row
        period = int(period)
        cap_value = float(cap_value)
        
        if type_ == 'Generation':
            if (country, energy_type) not in gen_inv_cap:
                gen_inv_cap[(country, energy_type)] = {}
            gen_inv_cap[(country, energy_type)][period] = cap_value
        elif type_ == 'Transmission':
            if (country, energy_type) not in transmission_inv_cap:
                transmission_inv_cap[(country, energy_type)] = {}
            transmission_inv_cap[(country, energy_type)][period] = cap_value
        elif type_ == 'Storage Power':
            if (country, energy_type) not in stor_pw_inv_cap:
                stor_pw_inv_cap[(country, energy_type)] = {}
            stor_pw_inv_cap[(country, energy_type)][period] = cap_value
        elif type_ == 'Storage Energy':
            if (country, energy_type) not in stor_en_inv_cap:
                stor_en_inv_cap[(country, energy_type)] = {}
            stor_en_inv_cap[(country, energy_type)][period] = cap_value

    return gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap


def set_fixed_capacities(instance, FSD):
    """Set fixed capacities for the period from FSD data"""
    gen_inv_cap, transmission_inv_cap, stor_pw_inv_cap, stor_en_inv_cap = load_investment_data(FSD)

    for (n, g) in instance.GeneratorsOfNode:
        if (n, g) in gen_inv_cap:
            for i in instance.PeriodActive:
                if i in gen_inv_cap[(n, g)]:
                    cap_value = gen_inv_cap[(n, g)][i]
                    instance.genInvCapParam[n, g, i] = cap_value
        else:
            print(f"(n, g) = ({n}, {g}): Not found in gen_inv_cap")

    # Transmission
    for (n1, n2) in instance.BidirectionalArc:
        if (n1, n2) in transmission_inv_cap:
            for i in instance.PeriodActive:
                if i in transmission_inv_cap[(n1, n2)]:
                    cap_value = transmission_inv_cap[(n1, n2)][i]
                    instance.transmisionInvCapParam[n1, n2, i] = cap_value
        else:
            print(f"(n1, n2) = ({n1}, {n2}): Not found in transmission_inv_cap")

    # Storage
    for (n, b) in instance.StoragesOfNode:
        if (n, b) in stor_pw_inv_cap:
            for i in instance.PeriodActive:
                if i in stor_pw_inv_cap[(n, b)]:
                    cap_value = stor_pw_inv_cap[(n, b)][i]
                    instance.storPWInvCapParam[n, b, i] = cap_value
        else:
            print(f"(n, b) = ({n}, {b}): Not found in stor_pw_inv_cap")

        if (n, b) in stor_en_inv_cap:
            for i in instance.PeriodActive:
                if i in stor_en_inv_cap[(n, b)]:
                    cap_value = stor_en_inv_cap[(n, b)][i]
                    instance.storENInvCapParam[n, b, i] = cap_value
        else:
            print(f"(n, b) = ({n}, {b}): Not found in stor_en_inv_cap")


    return instance


def results_saving(instance):
    v = {}
    for i in instance.PeriodActive:
        v = {i: {'v_i': {}}}
            
        v[i]['v_i']['genInstalledCap'] = {str((n, g)): get_value(instance.genInstalledCap[n, g, i])
                                                for n, g in instance.GeneratorsOfNode}
        v[i]['v_i']['transmissionInstalledCap'] = {str((n1, n2)): get_value(instance.transmissionInstalledCap[n1, n2, i])
                                                        for n1, n2 in instance.BidirectionalArc}
        v[i]['v_i']['storPWInstalledCap'] = {str((n, b)): get_value(instance.storPWInstalledCap[n, b, i])
                                                    for n, b in instance.StoragesOfNode}
        v[i]['v_i']['storENInstalledCap'] = {str((n, b)): get_value(instance.storENInstalledCap[n, b, i])
                                                    for n, b in instance.StoragesOfNode} 
    return v



def infeasible_index(instance):
    infeasi_idx_lst = []
    for w in instance.Scenario:
        for i in instance.PeriodActive:
            shed_cost = sum(get_value(instance.operationalDiscountrate)*get_value(instance.seasScale[s]) * get_value(instance.nodeLostLoadCost[n,i]) * get_value(instance.loadShed[n,h,i,w])
                            for n in instance.Node 
                            for (s,h) in instance.HoursOfSeason)
            if shed_cost > 0:
                infeasi_idx = 1
                infeasi_idx_lst.append(infeasi_idx)
            else:
                infeasi_idx = 0
                infeasi_idx_lst.append(infeasi_idx)
    
    return infeasi_idx_lst



def get_value(param):
    return param.value if hasattr(param, 'value') else param

def calculate_Q(instance):
    q_lst = []
    for w in instance.Scenario: 
        for i in instance.PeriodActive:
            # Calculate Q_i for a specific period i and scenario w
            operational_cost = sum(get_value(instance.operationalDiscountrate)*get_value(instance.seasScale[s]) * get_value(instance.genMargCost[g,i]) * get_value(instance.genOperational[n,g,h,i,w])
                                for (n,g) in instance.GeneratorsOfNode 
                                for (s,h) in instance.HoursOfSeason) 
            
            shed_cost = sum(get_value(instance.operationalDiscountrate)*get_value(instance.seasScale[s]) * get_value(instance.nodeLostLoadCost[n,i]) * get_value(instance.loadShed[n,h,i,w])
                            for n in instance.Node 
                            for (s,h) in instance.HoursOfSeason)

            scaled_q_i = value(instance.discount_multiplier[i]) * (operational_cost + shed_cost)

            q_lst.append(scaled_q_i)
    return q_lst



def calculate_second_stage(instance):
    q_lst = []
    ll_lst = []
    for w in instance.Scenario: 
        for i in instance.PeriodActive:
            # Calculate Q_i for a specific period i and scenario w
            operational_cost = sum(get_value(instance.operationalDiscountrate)*get_value(instance.seasScale[s]) * get_value(instance.genMargCost[g,i]) * get_value(instance.genOperational[n,g,h,i,w])
                                for (n,g) in instance.GeneratorsOfNode 
                                for (s,h) in instance.HoursOfSeason) 
            
            shed_amt = sum(get_value(instance.operationalDiscountrate)*get_value(instance.seasScale[s]) * get_value(instance.loadShed[n,h,i,w]) for n in instance.Node for (s,h) in instance.HoursOfSeason)

            scaled_q_i = value(instance.discount_multiplier[i]) * (operational_cost)
            scaled_shed_amt = value(instance.discount_multiplier[i]) * (shed_amt)
            q_lst.append(scaled_q_i)
            ll_lst.append(scaled_shed_amt)

    return q_lst,ll_lst


def calculate_f_x(instance):
    # Calculate Q_i for a specific period i and scenario w
    first_stage_val_lst = []
    for i in instance.PeriodActive:
        first_stage_value = value(instance.discount_multiplier[i]) * (
            sum(
                get_value(instance.genInvCost[g, i]) * get_value(instance.genInvCapParam[n, g, i])
                for (n, g) in instance.GeneratorsOfNode
            ) +
            sum(
                get_value(instance.transmissionInvCost[n1, n2, i]) * get_value(instance.transmisionInvCapParam[n1, n2, i])
                for (n1, n2) in instance.BidirectionalArc
            ) +
            sum(
                (get_value(instance.storPWInvCost[b, i]) * get_value(instance.storPWInvCapParam[n, b, i]) +
                get_value(instance.storENInvCost[b, i]) * get_value(instance.storENInvCapParam[n, b, i]))
                for (n, b) in instance.StoragesOfNode
            )
        )
        first_stage_val_lst.append(first_stage_value)
    return first_stage_val_lst



