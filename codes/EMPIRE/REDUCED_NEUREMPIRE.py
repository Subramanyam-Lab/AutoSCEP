from __future__ import division
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager
import csv
import sys
import cloudpickle
import time
from datetime import datetime
import os
import joblib
import pandas as pd
import numpy as np
import multiprocessing
import json
from scenario_random import generate_random_scenario
from reader import generate_tab_files
from yaml import safe_load
import argparse
import shutil

activate_rule = "version1"

__author__ = "Stian Backe"
__license__ = "MIT"
__maintainer__ = "Stian Backe"
__email__ = "stian.backe@ntnu.no"

def run_empire(name, tab_file_path, result_file_path, scenariogeneration, scenario_data_path,
               solver, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
               lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
               discountrate, WACC, LeapYearsInvestment, IAMC_PRINT, WRITE_LP,
               PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, seed,north_sea,version):

    if USE_TEMP_DIR:
        TempfileManager.tempdir = temp_dir

    # if not os.path.exists(result_file_path):
    #     os.makedirs(result_file_path)

    model = AbstractModel()

    scenariopath = tab_file_path
    tab_file_path = f'Data handler/base/{version}'
    
    ###########
    ##SOLVERS##
    ###########

    if solver == "CPLEX":
        print("Solver: CPLEX")
    elif solver == "Xpress":
        print("Solver: Xpress")
    elif solver == "Gurobi":
        print("Solver: Gurobi")
    elif solver == "GLPK":
        print("Solver: GLPK")
    else:
        sys.exit("ERROR! Invalid solver! Options: CPLEX, Xpress, Gurobi")

    ##########
    ##MODULE##
    ##########

    if WRITE_LP:
        print("Will write LP-file...")

    if PICKLE_INSTANCE:
        print("Will pickle instance...")

    if EMISSION_CAP:
        print("Absolute emission cap in each scenario...")
    else:
        print("No absolute emission cap...")
    ########
    ##SETS##
    ########

    #Define the sets

    print("Declaring sets...")

    #Supply technology sets
    model.Generator = Set(ordered=True) #g
    model.Technology = Set(ordered=True) #t
    model.Storage =  Set() #b

    #Temporal sets
    model.Period = Set(ordered=True) #max period,|I|,it becomes 8
    model.PeriodActive = Set(ordered=True, initialize=Period) #i
    model.Operationalhour = Set(ordered=True, initialize=Operationalhour) #h
    model.Season = Set(ordered=True, initialize=Season) #s

    #Spatial sets
    model.Node = Set(ordered=True) #n
    if north_sea:
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
    model.FirstHoursOfRegSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfRegSeason) # begining hour of the Regular Season
    model.FirstHoursOfPeakSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfPeakSeason) # begining hour of the Peak Season

    print("Reading sets...")

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
    if north_sea:
        data.load(filename=tab_file_path + "/" + 'Sets_OffshoreNode.tab',format="set", set=model.OffshoreNode)
    data.load(filename=tab_file_path + "/" + 'Sets_Horizon.tab',format="set", set=model.Period)
    data.load(filename=tab_file_path + "/" + 'Sets_DirectionalLines.tab',format="set", set=model.DirectionalLink)
    data.load(filename=tab_file_path + "/" + 'Sets_LineType.tab',format="set", set=model.TransmissionType)
    data.load(filename=tab_file_path + "/" + 'Sets_LineTypeOfDirectionalLines.tab',format="set", set=model.TransmissionTypeOfDirectionalLink)
    data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfTechnology.tab',format="set", set=model.GeneratorsOfTechnology)
    data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfNode.tab',format="set", set=model.GeneratorsOfNode)
    data.load(filename=tab_file_path + "/" + 'Sets_StorageOfNodes.tab',format="set", set=model.StoragesOfNode)

    print("Constructing sub sets...")

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

    model.discountrate = Param(initialize=discountrate) 
    model.WACC = Param(initialize=WACC) 
    model.LeapYearsInvestment = Param(initialize=LeapYearsInvestment) # 5 year
    model.operationalDiscountrate = Param(mutable=True) 
    model.sceProbab = Param(model.Scenario, mutable=True) # pi_w, \sum pi_w =1 
    model.seasScale = Param(model.Season, initialize=1.0, mutable=True) # \alpha_s
    model.lengthRegSeason = Param(initialize=lengthRegSeason) # 168
    model.lengthPeakSeason = Param(initialize=lengthPeakSeason) # 48 

    #Cost

    model.genCapitalCost = Param(model.Generator, model.Period, default=0, mutable=True)
    model.transmissionTypeCapitalCost = Param(model.TransmissionType, model.Period, default=0, mutable=True)
    model.storPWCapitalCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.storENCapitalCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.genFixedOMCost = Param(model.Generator, model.Period, default=0, mutable=True)
    model.transmissionTypeFixedOMCost = Param(model.TransmissionType, model.Period, default=0, mutable=True)
    model.storPWFixedOMCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.storENFixedOMCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.genInvCost = Param(model.Generator, model.Period, default=9000000, mutable=True)
    model.transmissionInvCost = Param(model.BidirectionalArc, model.Period, default=3000000, mutable=True)
    model.storPWInvCost = Param(model.Storage, model.Period, default=1000000, mutable=True)
    model.storENInvCost = Param(model.Storage, model.Period, default=800000, mutable=True)
    model.transmissionLength = Param(model.BidirectionalArc, default=0, mutable=True)
    model.genVariableOMCost = Param(model.Generator, default=0.0, mutable=True)
    model.genFuelCost = Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genMargCost = Param(model.Generator, model.Period, default=600, mutable=True)
    model.genCO2TypeFactor = Param(model.Generator, default=0.0, mutable=True)
    # model.nodeLostLoadCost = Param(model.Node, model.Period, default=22000.0)
    model.nodeLostLoadCost = Param(model.Node, model.Period, default=1e+8, mutable=False)
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
    model.genMaxBuiltCap = Param(model.Node, model.Technology, model.Period, default=500000.0, mutable=True)
    model.transmissionMaxBuiltCap = Param(model.BidirectionalArc, model.Period, default=20000.0, mutable=True)
    model.storPWMaxBuiltCap = Param(model.StoragesOfNode, model.Period, default=500000.0, mutable=True)
    model.storENMaxBuiltCap = Param(model.StoragesOfNode, model.Period, default=500000.0, mutable=True)
    model.genMaxInstalledCapRaw = Param(model.Node, model.Technology, default=0.0, mutable=True)
    model.genMaxInstalledCap = Param(model.Node, model.Technology, model.Period, default=0.0, mutable=True)
    model.transmissionMaxInstalledCapRaw = Param(model.BidirectionalArc, model.Period, default=0.0)
    model.transmissionMaxInstalledCap = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWMaxInstalledCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storPWMaxInstalledCapRaw = Param(model.StoragesOfNode, default=0.0, mutable=True)
    model.storENMaxInstalledCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENMaxInstalledCapRaw = Param(model.StoragesOfNode, default=0.0, mutable=True)

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
    

    if EMISSION_CAP:
        model.CO2cap = Param(model.Period, default=5000.0, mutable=True)
    
    if LOADCHANGEMODULE:
        model.sloadMod = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)



    #Load the parameters

    print("Reading parameters...")

    data.load(filename=tab_file_path + "/" + 'Generator_CapitalCosts.tab', param=model.genCapitalCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_FixedOMCosts.tab', param=model.genFixedOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_VariableOMCosts.tab', param=model.genVariableOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_FuelCosts.tab', param=model.genFuelCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_CCSCostTSVariable.tab', param=model.CCSCostTSVariable, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_Efficiency.tab', param=model.genEfficiency, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_RefInitialCap.tab', param=model.genRefInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_ScaleFactorInitialCap.tab', param=model.genScaleInitCap, format="table")
    if version in ["europe_v51","europe_reduced_v51", "europe_v50", "reduced"]:
        data.load(filename=tab_file_path + "/" + 'Generator_MaxBuiltCapacity.tab', param=model.genMaxBuiltCap, format="table")#?
        data.load(filename=tab_file_path + "/" + 'Generator_InitialCapacity.tab', param=model.genInitCap, format="table") #node_generator_intial_capacity.xlsx
    data.load(filename=tab_file_path + "/" + 'Generator_MaxInstalledCapacity.tab', param=model.genMaxInstalledCapRaw, format="table")#maximum_capacity_constraint_040317_high
    data.load(filename=tab_file_path + "/" + 'Generator_CO2Content.tab', param=model.genCO2TypeFactor, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_RampRate.tab', param=model.genRampUpCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_GeneratorTypeAvailability.tab', param=model.genCapAvailTypeRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_Lifetime.tab', param=model.genLifetime, format="table") 

    data.load(filename=tab_file_path + "/" + 'Transmission_InitialCapacity.tab', param=model.transmissionInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_MaxBuiltCapacity.tab', param=model.transmissionMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_MaxInstallCapacityRaw.tab', param=model.transmissionMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_Length.tab', param=model.transmissionLength, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_TypeCapitalCost.tab', param=model.transmissionTypeCapitalCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_TypeFixedOMCost.tab', param=model.transmissionTypeFixedOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_lineEfficiency.tab', param=model.lineEfficiency, format="table")
    data.load(filename=tab_file_path + "/" + 'Transmission_Lifetime.tab', param=model.transmissionLifetime, format="table")

    data.load(filename=tab_file_path + "/" + 'Storage_StorageBleedEfficiency.tab', param=model.storageBleedEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageChargeEff.tab', param=model.storageChargeEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageDischargeEff.tab', param=model.storageDischargeEff, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StoragePowToEnergy.tab', param=model.storagePowToEnergy, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyCapitalCost.tab', param=model.storENCapitalCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyFixedOMCost.tab', param=model.storENFixedOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyInitialCapacity.tab', param=model.storENInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyMaxBuiltCapacity.tab', param=model.storENMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_EnergyMaxInstalledCapacity.tab', param=model.storENMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_StorageInitialEnergyLevel.tab', param=model.storOperationalInit, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_PowerCapitalCost.tab', param=model.storPWCapitalCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_PowerFixedOMCost.tab', param=model.storPWFixedOMCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_InitialPowerCapacity.tab', param=model.storPWInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_PowerMaxBuiltCapacity.tab', param=model.storPWMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_PowerMaxInstalledCapacity.tab', param=model.storPWMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + 'Storage_Lifetime.tab', param=model.storageLifetime, format="table")
    # new
    # data.load(filename=f'Data handler/europe_reduced_v50/Average_Cap_Avail_{len(Period)}.tab', param=model.avg_cap_avail, format="table")

    #Temporarily 
    # data.load(filename=tab_file_path + "/" + 'Node_NodeLostLoadCost.tab', param=model.nodeLostLoadCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Node_ElectricAnnualDemand.tab', param=model.sloadAnnualDemand, format="table") 
    data.load(filename=tab_file_path + "/" + 'Node_HydroGenMaxAnnualProduction.tab', param=model.maxHydroNode, format="table") 
    
    # if scenariogeneration:
    #     scenariopath = tab_file_path
    # else:
    #     scenariopath = scenario_data_path

    data.load(filename=scenariopath + "/" + 'Stochastic_HydroGenMaxSeasonalProduction.tab', param=model.maxRegHydroGenRaw, format="table")
    data.load(filename=scenariopath + "/" + 'Stochastic_StochasticAvailability.tab', param=model.genCapAvailStochRaw, format="table") 
    data.load(filename=scenariopath + "/" + 'Stochastic_ElectricLoadRaw.tab', param=model.sloadRaw, format="table") 

    data.load(filename=tab_file_path + "/" + 'General_seasonScale.tab', param=model.seasScale, format="table") 

    if EMISSION_CAP:
        data.load(filename=tab_file_path + "/" + 'General_CO2Cap.tab', param=model.CO2cap, format="table")
    else:
        data.load(filename=tab_file_path + "/" + 'General_CO2Price.tab', param=model.CO2price, format="table")

    if LOADCHANGEMODULE:
        data.load(filename=scenariopath + "/" + 'LoadchangeModule/Stochastic_ElectricLoadMod.tab', param=model.sloadMod, format="table")

    print("Constructing parameter values...")



    def adjust_season_scale_rule(model):
        # Regular seasons are defined as:
        regular_seasons = ["winter", "spring", "summer", "fall"]
        # Calculate the common value for regular seasons:
        regular_scale = float((8760 - 48) / (4 * model.lengthRegSeason))
        for s in model.Season:
            if s in regular_seasons:
                model.seasScale[s] = regular_scale
            else:
                # For peak seasons (assumed to be not in regular_seasons)
                model.seasScale[s] = 1.0
    model.adjust_season_scale = BuildAction(rule=adjust_season_scale_rule)



    # It means that the probability of scenarios is equally same regardless of scenario, and the expected second stage value is just average of second stage value. 
    def prepSceProbab_rule(model):
        #Build an equiprobable probability distribution for scenarios

        for sce in model.Scenario:
            model.sceProbab[sce] = value(1/len(model.Scenario))

    model.build_SceProbab = BuildAction(rule=prepSceProbab_rule)



    # This function consists the costs per period for each generator, storage, transmission
    def prepInvCost_rule(model):
        #Build investment cost for generators, storages and transmission. Annual cost is calculated for the lifetime of the generator and discounted for a year.
        #Then cost is discounted for the investment period (or the remaining lifetime). CCS generators has additional fixed costs depending on emissions. 

        #Generator 
        for g in model.Generator:
            for i in model.PeriodActive:
                costperyear=(model.WACC/(1-((1+model.WACC)**(-model.genLifetime[g]))))*model.genCapitalCost[g,i]+model.genFixedOMCost[g,i]
                costperperiod=costperyear*1000*(1-(1+model.discountrate)**-(min(value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), value(model.genLifetime[g]))))/(1-(1/(1+model.discountrate)))
                if ('CCS',g) in model.GeneratorsOfTechnology:
                    costperperiod+=model.CCSCostTSFix*model.CCSRemFrac*model.genCO2TypeFactor[g]*(3.6/model.genEfficiency[g,i])
                model.genInvCost[g,i]=costperperiod

        #Storage
        for b in model.Storage:
            for i in model.PeriodActive:
                costperyearPW=(model.WACC/(1-((1+model.WACC)**(-model.storageLifetime[b]))))*model.storPWCapitalCost[b,i]+model.storPWFixedOMCost[b,i]
                costperperiodPW=costperyearPW*1000*(1-(1+model.discountrate)**-(min(value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), value(model.storageLifetime[b]))))/(1-(1/(1+model.discountrate)))
                model.storPWInvCost[b,i]=costperperiodPW
                costperyearEN=(model.WACC/(1-((1+model.WACC)**(-model.storageLifetime[b]))))*model.storENCapitalCost[b,i]+model.storENFixedOMCost[b,i]
                costperperiodEN=costperyearEN*1000*(1-(1+model.discountrate)**-(min(value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), value(model.storageLifetime[b]))))/(1-(1/(1+model.discountrate)))
                model.storENInvCost[b,i]=costperperiodEN

        #Transmission
        for (n1,n2) in model.BidirectionalArc:
            for i in model.PeriodActive:
                for t in model.TransmissionType:
                    if (n1,n2,t) in model.TransmissionTypeOfDirectionalLink:
                        costperyear=(model.WACC/(1-((1+model.WACC)**(-model.transmissionLifetime[n1,n2]))))*model.transmissionLength[n1,n2]*model.transmissionTypeCapitalCost[t,i]+model.transmissionTypeFixedOMCost[t,i]
                        costperperiod=costperyear*(1-(1+model.discountrate)**-(min(value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), value(model.transmissionLifetime[n1,n2]))))/(1-(1/(1+model.discountrate)))
                        model.transmissionInvCost[n1,n2,i]=costperperiod

    model.build_InvCost = BuildAction(rule=prepInvCost_rule) # This is the cost vector of first stage. 

    # This function consists of cost vector for generation, such as q^{gen}
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

    # This is \bar_{x}
    def prepInitialCapacityNodeGen_rule(model):
        #Build initial capacity for generator type in node

        for (n,g) in model.GeneratorsOfNode:
            for i in model.PeriodActive:
                if value(model.genInitCap[n,g,i]) == 0:
                    model.genInitCap[n,g,i] = model.genRefInitCap[n,g]*(1-model.genScaleInitCap[g,i])

    model.build_InitialCapacityNodeGen = BuildAction(rule=prepInitialCapacityNodeGen_rule)

    # This is \bar_{x}
    def prepInitialCapacityTransmission_rule(model):
        #Build initial capacity for transmission lines to ensure initial capacity is the upper installation bound if infeasible

        for (n1,n2) in model.BidirectionalArc:
            for i in model.PeriodActive:
                if value(model.transmissionMaxInstalledCapRaw[n1,n2,i]) <= value(model.transmissionInitCap[n1,n2,i]):
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionInitCap[n1,n2,i]
                else:
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionMaxInstalledCapRaw[n1,n2,i]

    model.build_InitialCapacityTransmission = BuildAction(rule=prepInitialCapacityTransmission_rule)

    # This is V on the mathematical obejctive function
    def prepOperationalDiscountrate_rule(model):
        #Build operational discount rate

        model.operationalDiscountrate = sum((1+model.discountrate)**(-j) for j in list(range(0,value(model.LeapYearsInvestment))))

    model.build_operationalDiscountrate = BuildAction(rule=prepOperationalDiscountrate_rule)     


    # Following functions are represent \bar_{V}
    def prepGenMaxInstalledCap_rule(model):
        #Build resource limit (installed limit) for all periods. Avoid infeasibility if installed limit lower than initially installed cap.

        for t in model.Technology:
            for n in model.Node:
                for i in model.PeriodActive:
                    if value(model.genMaxInstalledCapRaw[n,t] <= sum(model.genInitCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology)):
                        model.genMaxInstalledCap[n,t,i]=sum(model.genInitCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology)
                    else:
                        model.genMaxInstalledCap[n,t,i]=model.genMaxInstalledCapRaw[n,t] 
                        
    model.build_genMaxInstalledCap = BuildAction(rule=prepGenMaxInstalledCap_rule)

    def storENMaxInstalledCap_rule(model):
        #Build installed limit (resource limit) for storEN

        for (n,b) in model.StoragesOfNode:
            for i in model.PeriodActive:
                model.storENMaxInstalledCap[n,b,i]=model.storENMaxInstalledCapRaw[n,b]

    model.build_storENMaxInstalledCap = BuildAction(rule=storENMaxInstalledCap_rule)

    def storPWMaxInstalledCap_rule(model):
        #Build installed limit (resource limit) for storPW

        for (n,b) in model.StoragesOfNode:
            for i in model.PeriodActive:
                model.storPWMaxInstalledCap[n,b,i]=model.storPWMaxInstalledCapRaw[n,b]

    model.build_storPWMaxInstalledCap = BuildAction(rule=storPWMaxInstalledCap_rule)


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
        # f = open(result_file_path + '/AdjustedNegativeLoad_' + name + '.txt', 'w')
        for n in model.Node:
            for i in model.PeriodActive:
                noderawdemand = 0
                for (s,h) in model.HoursOfSeason:
                    # if value(h) < value(FirstHoursOfRegSeason[-1] + model.lengthRegSeason):
                        for sce in model.Scenario:
                                noderawdemand += value(model.sceProbab[sce]*model.seasScale[s]*model.sloadRaw[n,h,sce,i])
                if value(model.sloadAnnualDemand[n,i]) < 1:
                    hourlyscale = 0
                else:
                    hourlyscale = value(model.sloadAnnualDemand[n,i]) / noderawdemand
                for h in model.Operationalhour:
                    for sce in model.Scenario:
                        model.sload[n, h, i, sce] = (model.sloadRaw[n,h,sce,i]*hourlyscale)
                        if LOADCHANGEMODULE:
                            model.sload[n,h,i,sce] = model.sload[n,h,i,sce] + model.sloadMod[n,h,sce,i]
                        if value(model.sload[n,h,i,sce]) < 0:
                            f.write('Adjusted electricity load: ' + str(value(model.sload[n,h,i,sce])) + ', 10 MW for hour ' + str(h) + ' and scenario ' + str(sce) + ' in ' + str(n) + "\n")
                            model.sload[n,h,i,sce] = 10
                            counter += 1

        # f.write('Hours with too small raw electricity load: ' + str(counter))
        # f.close()

    model.build_sload = BuildAction(rule=prepSload_rule)

    print("Sets and parameters declared and read...")

    #############
    ##VARIABLES##
    #############

    print("Declaring variables...")

    # First Stage Decisions (x)
    model.genInvCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmisionInvCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)

    # First Stage Decisions (v)
    model.genInstalledCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmissionInstalledCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInstalledCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInstalledCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)

    # Second Stage Decisions (y,w)
    model.genOperational = Var(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.storOperational = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.transmisionOperational = Var(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals) #flow
    model.storCharge = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.storDischarge = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.loadShed = Var(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    
    ###############
    ##EXPRESSIONS##
    ###############

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
        return sum(model.discount_multiplier[i]*(
            sum(model.genInvCost[g,i]* model.genInvCap[n,g,i] for (n,g) in model.GeneratorsOfNode ) + \
            sum(model.transmissionInvCost[n1,n2,i]*model.transmisionInvCap[n1,n2,i] for (n1,n2) in model.BidirectionalArc ) + \
            sum((model.storPWInvCost[b,i]*model.storPWInvCap[n,b,i]+model.storENInvCost[b,i]*model.storENInvCap[n,b,i]) for (n,b) in model.StoragesOfNode ) + \
            model.shedcomponent[i] + model.operationalcost[i]) for i in model.PeriodActive)
    model.Obj = Objective(rule=Obj_rule, sense=minimize)

    ###############
    ##CONSTRAINTS##
    ###############

    #### First Stage Constraints ####

    def lifetime_rule_gen(model, n, g, i):
        startPeriod=1
        if value(1+i-(model.genLifetime[g]/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.genLifetime[g]/model.LeapYearsInvestment)
        return sum(model.genInvCap[n,g,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.genInstalledCap[n,g,i] + model.genInitCap[n,g,i]== 0   #
    model.installedCapDefinitionGen = Constraint(model.GeneratorsOfNode, model.PeriodActive, rule=lifetime_rule_gen)

    #################################################################

    def lifetime_rule_storEN(model, n, b, i):
        startPeriod=1
        if value(1+i-model.storageLifetime[b]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.storageLifetime[b]/model.LeapYearsInvestment)
        return sum(model.storENInvCap[n,b,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.storENInstalledCap[n,b,i] + model.storENInitCap[n,b,i]== 0   #
    model.installedCapDefinitionStorEN = Constraint(model.StoragesOfNode, model.PeriodActive, rule=lifetime_rule_storEN)

    #################################################################

    def lifetime_rule_storPOW(model, n, b, i):
        startPeriod=1
        if value(1+i-model.storageLifetime[b]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.storageLifetime[b]/model.LeapYearsInvestment)
        return sum(model.storPWInvCap[n,b,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.storPWInstalledCap[n,b,i] + model.storPWInitCap[n,b,i]== 0   #
    model.installedCapDefinitionStorPOW = Constraint(model.StoragesOfNode, model.PeriodActive, rule=lifetime_rule_storPOW)

    #################################################################

    def lifetime_rule_trans(model, n1, n2, i):
        startPeriod=1
        if value(1+i-model.transmissionLifetime[n1,n2]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.transmissionLifetime[n1,n2]/model.LeapYearsInvestment)
        return sum(model.transmisionInvCap[n1,n2,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.transmissionInstalledCap[n1,n2,i] + model.transmissionInitCap[n1,n2,i] == 0   #
    model.installedCapDefinitionTrans = Constraint(model.BidirectionalArc, model.PeriodActive, rule=lifetime_rule_trans)

    #################################################################

    def investment_gen_cap_rule(model, t, n, i):
        return sum(model.genInvCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology) - model.genMaxBuiltCap[n,t,i] <= 0
    model.investment_gen_cap = Constraint(model.Technology, model.Node, model.PeriodActive, rule=investment_gen_cap_rule)

    #################################################################

    def investment_trans_cap_rule(model, n1, n2, i):
        return model.transmisionInvCap[n1,n2,i] - model.transmissionMaxBuiltCap[n1,n2,i] <= 0
    model.investment_trans_cap = Constraint(model.BidirectionalArc, model.PeriodActive, rule=investment_trans_cap_rule)

    #################################################################

    def investment_storage_power_cap_rule(model, n, b, i):
        return model.storPWInvCap[n,b,i] - model.storPWMaxBuiltCap[n,b,i] <= 0
    model.investment_storage_power_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=investment_storage_power_cap_rule)

    #################################################################

    def investment_storage_energy_cap_rule(model, n, b, i):
        return model.storENInvCap[n,b,i] - model.storENMaxBuiltCap[n,b,i] <= 0
    model.investment_storage_energy_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=investment_storage_energy_cap_rule)

    ################################################################

    def installed_gen_cap_rule(model, t, n, i):
        return sum(model.genInstalledCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology) - model.genMaxInstalledCap[n,t,i] <= 0
    model.installed_gen_cap = Constraint(model.Technology, model.Node, model.PeriodActive, rule=installed_gen_cap_rule)

    #################################################################

    def installed_trans_cap_rule(model, n1, n2, i):
        return model.transmissionInstalledCap[n1,n2,i] - model.transmissionMaxInstalledCap[n1,n2,i] <= 0
    model.installed_trans_cap = Constraint(model.BidirectionalArc, model.PeriodActive, rule=installed_trans_cap_rule)

    #################################################################

    def installed_storage_power_cap_rule(model, n, b, i):
        return model.storPWInstalledCap[n,b,i] - model.storPWMaxInstalledCap[n,b,i] <= 0
    model.installed_storage_power_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=installed_storage_power_cap_rule)

    #################################################################

    def installed_storage_energy_cap_rule(model, n, b, i):
        return model.storENInstalledCap[n,b,i] - model.storENMaxInstalledCap[n,b,i] <= 0
    model.installed_storage_energy_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=installed_storage_energy_cap_rule)

    #################################################################
    # if north_sea:
    #     def wind_farm_tranmission_cap_rule(model, n1, n2, i):
    #         if n1 in model.OffshoreNode or n2 in model.OffshoreNode:
    #             if (n1,n2) in model.BidirectionalArc:
    #                 if n1 in model.OffshoreNode:
    #                     return model.transmissionInstalledCap[(n1,n2),i] <= sum(model.genInstalledCap[n1,g,i] for g in model.Generator if (n1,g) in model.GeneratorsOfNode)
    #                 else:
    #                     return model.transmissionInstalledCap[(n1,n2),i] <= sum(model.genInstalledCap[n2,g,i] for g in model.Generator if (n2,g) in model.GeneratorsOfNode)
    #             elif (n2,n1) in model.BidirectionalArc:
    #                 if n1 in model.OffshoreNode:
    #                     return model.transmissionInstalledCap[(n2,n1),i] <= sum(model.genInstalledCap[n1,g,i] for g in model.Generator if (n1,g) in model.GeneratorsOfNode)
    #                 else:
    #                     return model.transmissionInstalledCap[(n2,n1),i] <= sum(model.genInstalledCap[n2,g,i] for g in model.Generator if (n2,g) in model.GeneratorsOfNode)
    #             else:
    #                 return Constraint.Skip
    #         else:
    #             return Constraint.Skip
    #     model.wind_farm_transmission_cap = Constraint(model.Node, model.Node, model.PeriodActive, rule=wind_farm_tranmission_cap_rule)

    #################################################################

    def power_energy_relate_rule(model, n, b, i):
        if b in model.DependentStorage:
            return model.storPWInstalledCap[n,b,i] - model.storagePowToEnergy[b]*model.storENInstalledCap[n,b,i] == 0   #
        else:
            return Constraint.Skip
    model.power_energy_relate = Constraint(model.StoragesOfNode, model.PeriodActive, rule=power_energy_relate_rule)

    #################################################################

    if activate_rule == "version1": 
        def version1_rule(model, h, i):
            gen_avail_capacity = sum(model.sceProbab[w]*sum(model.genInstalledCap[n, g, i] * model.genCapAvail[n,g,h,w,i] for (n, g) in model.GeneratorsOfNode) for w in model.Scenario)
            average_sload = sum(model.sceProbab[w]*sum(model.sload[n,h,i,w] for n in model.Node) for w in model.Scenario)
            return average_sload-gen_avail_capacity <= 0 
        model.version1 = Constraint(model.Operationalhour, model.PeriodActive,rule=version1_rule)

    if activate_rule == "version2":
        def version2_rule(model, s, i):
            num_hours = len([h for (season, h) in model.HoursOfSeason if season == s])
            gen_avail_capacity = sum(model.sceProbab[w]*sum(sum(model.genInstalledCap[n, g, i] * model.genCapAvail[n,g,h,w,i] for (n, g) in model.GeneratorsOfNode) for h in model.Operationalhour if (s,h) in model.HoursOfSeason) for w in model.Scenario)
            average_gen_avail_capacity = gen_avail_capacity / num_hours
            average_sload = sum(model.sceProbab[w]*sum(sum(model.sload[n,h,i,w] for n in model.Node)for h in model.Operationalhour if (s,h) in model.HoursOfSeason)for w in model.Scenario)
            hourly_average_sload = average_sload / num_hours
            return hourly_average_sload - average_gen_avail_capacity <= 0 
        model.version2 = Constraint(model.Season, model.PeriodActive,rule=version2_rule)

    if activate_rule == "version3":
        model.peakLoad = Var(model.Season, model.Scenario, model.PeriodActive, domain=NonNegativeReals)
        # Define peak load for each season, scenario, and period:
        def peak_load_definition_rule(model, s, w, i, h):
            if (s, h) not in model.HoursOfSeason:
                return Constraint.Skip
            return model.peakLoad[s, w, i] >= sum(model.sload[n, h, i, w] for n in model.Node)

        model.peakLoadDefinition = Constraint(
            model.Season, model.Scenario, model.PeriodActive, model.Operationalhour, 
            rule=peak_load_definition_rule
        )

        # Use the peak load variable in a constraint comparing capacity and peak demand:
        def version3_rule(model, s, i):
            average_peak_sload = sum(model.sceProbab[w] * model.peakLoad[s, w, i] for w in model.Scenario)
            relevant_hours = [h for (season, h) in model.HoursOfSeason if season == s]
            gen_avail_capacity = sum(
                model.sceProbab[w] *
                sum(model.genInstalledCap[n, g, i] * model.genCapAvail[n, g, h, w, i] for (n, g) in model.GeneratorsOfNode)for w in model.Scenario for h in relevant_hours
            )
            average_gen_avail_capacity = gen_avail_capacity / len(relevant_hours)
            return average_peak_sload - average_gen_avail_capacity <= 0

        model.version3 = Constraint(model.Season, model.PeriodActive, rule=version3_rule)

    #### Second Stage Constraints ####

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

    #################################################################

    # def soft_loadshed_rule(model, n, h, i, w):
    #     return model.loadShed[n, h, i, w] <= 0
    # model.SoftLoadShed = Constraint(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, rule=soft_loadshed_rule)

    print("Objective and constraints read...")

    #######
    ##RUN##
    #######

    print("Building instance...")
    
    start = time.time()
    instance = model.create_instance(data) #, report_timing=True)
    instance.dual = Suffix(direction=Suffix.IMPORT) #Make sure the dual value is collected into solver results (if solver supplies dual information)



    
    
    # print("Computed genMargCost values:")
    # for g in instance.Generator:
    #     for i in instance.PeriodActive:
    #         print(f"genMargCost[{g}, {i}] = {value(instance.genMargCost[g,i])}, genInvCost[{g}, {i}] = {value(instance.genInvCost[g,i])}")
            
    # print("Computed genMargCost values:")
    # for g in instance.Generator:
    #     for i in instance.PeriodActive:
    #         print(f"genInvCost[{g}, {i}] = {value(instance.genInvCost[g,i])}")

    # def calculate_avg_sload_period(instance):
    #     avg_sload = {}
    #     for n in instance.Node:
    #         for i in instance.PeriodActive:
    #             for h in instance.Operationalhour:            
    #                 total_load = sum(
    #                     value(instance.sceProbab[w]) * value(instance.sload[n, h, i, w])
    #                     for w in instance.Scenario
    #                 )
    #                 avg_sload[n,i,h] = total_load
    #     return avg_sload

    # avg_sload = calculate_avg_sload_period(instance)

    # data = []
    # for (node, period, hour), sload in avg_sload.items():
    #     data.append([node, period, hour, sload])

    # df = pd.DataFrame(data, columns=["Unnamed:_1", "Unnamed:_2", "Unnamed:_3", "Unnamed:_4"])
    # header = ["Unnamed:_1", "Unnamed:_2", "Unnamed:_3", "Unnamed:_4"]
    # tab_file_path = f"Data handler/{version}/Average_sload_{str(lengthRegSeason)}_2.tab"

    # with open(tab_file_path, "w") as f:
    #     f.write("\t".join(header) + "\n")  # 첫 번째 줄에 헤더 작성
    #     df.to_csv(f, sep="\t", index=False, header=False)  # 데이터 저장

    # print(f"파일이 저장되었습니다: {tab_file_path}")

    # raise 3


    # def calculate_cap_avail(instance):
    #     cap_avail = {}
    #     for i in instance.PeriodActive:
    #         for h in instance.Operationalhour:
    #             for (n,g) in instance.GeneratorsOfNode:
    #                 avg_avail = sum(value(instance.sceProbab[w]) * value(instance.genCapAvail[n, g, h, w, i]) for w in instance.Scenario)
    #                 cap_avail[n,g,h,i] = avg_avail
    #                 if cap_avail[n, g, h, i] > 1:
    #                     print(f"WARNING: cap_avail[{n}, {g}, {h}, {i}] = {cap_avail[n, g, h, i]}")
    #                     cap_avail[n, g, h, i] = 1.0
    #     return cap_avail

    # cap_avail_dict = calculate_cap_avail(instance)
    
    # data = []
    # for (country, generator, hour, period), cap_avail in cap_avail_dict.items():
    #     data.append([country, generator, hour, period, cap_avail])

    # df = pd.DataFrame(data, columns=["Unnamed:_1", "Unnamed:_2", "Unnamed:_3", "Unnamed:_4","Unnamed:_5"])
    # header = ["Unnamed:_1", "Unnamed:_2", "Unnamed:_3", "Unnamed:_4", "Unnamed:_5"]
    # tab_file_path = f"Data handler/{version}/Average_Cap_Avail_{str(lengthRegSeason)}.tab"

    # with open(tab_file_path, "w") as f:
    #     f.write("\t".join(header) + "\n")  # 첫 번째 줄에 헤더 작성
    #     df.to_csv(f, sep="\t", index=False, header=False)  # 데이터 저장

    # print(f"파일이 저장되었습니다: {tab_file_path}")

    # raise 3


    # def create_sload_by_season_and_scenario_csv(instance, filename="sload_values_by_season_and_scenario_whole.csv"):
    #     with open(filename, "w", newline="") as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Season", "Hour", "Scenario", "Average_sload"])
            
    #         # Iterate over all (season, hour) pairs from the model's HoursOfSeason set.
    #         for (season, hour) in instance.HoursOfSeason:
    #             for w in instance.Scenario:
    #                 total = 0.0
    #                 count = 0
    #                 for i in instance.PeriodActive:
    #                     for n in instance.Node:
    #                         total += value(instance.sload[n, hour, i, w])
    #                         count += 1
    #                 avg_sload = total / count if count > 0 else 0.0
    #                 writer.writerow([season, hour, w, avg_sload])
    #     print(f"CSV file '{filename}' created successfully.")

    # create_sload_by_season_and_scenario_csv(instance)
    # raise 3



    # def create_sload_by_season_and_scenario_and_period_csv(instance, filename="sload_values_by_season_and_scenario_and_period.csv"):
    #     with open(filename, "w", newline="") as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["Period", "Scenario", "Season", "Hour", "Total_sload"])
            
    #         # Iterate over all (season, hour) pairs from the model's HoursOfSeason set.
    #         for i in instance.PeriodActive:
    #             for w in instance.Scenario:
    #                 for (season, hour) in instance.HoursOfSeason:
    #                     total_sload = 0.0
    #                     for n in instance.Node:
    #                         total_sload += value(instance.sload[n, hour, i, w])
    #                     writer.writerow([i, w, season, hour, total_sload])
    #     print(f"CSV file '{filename}' created successfully.")

    # create_sload_by_season_and_scenario_and_period_csv(instance)

    # raise 4



    
    end = time.time()
    print(f"Building instance took [sec]: {end - start}")
    

    
    print("----------------------Problem Statistics---------------------")
    print("Nodes: "+ str(len(instance.Node)))
    print("Lines: "+str(len(instance.BidirectionalArc)))
    print("")
    print("GeneratorTypes: "+str(len(instance.Generator)))
    print("TotalGenerators: "+str(len(instance.GeneratorsOfNode)))
    print("StorageTypes: "+str(len(instance.Storage)))
    print("TotalStorages: "+str(len(instance.StoragesOfNode)))
    print("")
    print("InvestmentPeriod: ", instance.PeriodActive)
    print("InvestmentUntil: "+str(value(2020+int(len(instance.PeriodActive)*LeapYearsInvestment))))
    print("Scenarios: "+str(len(instance.Scenario)))
    print("TotalOperationalHoursPerScenario: "+str(len(instance.Operationalhour)))
    print("TotalOperationalHoursPerInvYear: "+str(len(instance.Operationalhour)*len(instance.Scenario)))
    print("Seasons: "+str(len(instance.Season)))
    print("RegularSeasons: "+str(len(instance.FirstHoursOfRegSeason)))
    print("LengthRegSeason: "+str(value(instance.lengthRegSeason)))
    print("PeakSeasons: "+str(len(instance.FirstHoursOfPeakSeason)))
    print("LengthPeakSeason: "+str(value(instance.lengthPeakSeason)))
    print("")
    print("Discount rate: "+str(value(instance.discountrate)))
    print("Operational discount scale: "+str(value(instance.operationalDiscountrate)))
    print("--------------------------------------------------------------")
    

    # Count variables
    variable_count = sum(1 for _ in instance.component_data_objects(Var))
    print("Number of Variables:", variable_count)

    # Count constraints
    constraint_count = sum(1 for _ in instance.component_data_objects(Constraint))
    print("Number of Constraints:", constraint_count)
    

    print("Solving...")

    if solver == "CPLEX":
        opt = SolverFactory("cplex", Verbose=True)
        opt.options["lpmethod"] = 4
        opt.options["solutiontype"] = 2
        #instance.display('outputs_cplex.txt')
    if solver == "Xpress":
        opt = SolverFactory("xpress") #Verbose=True
        opt.options["defaultAlg"] = 4
        opt.options["crossover"] = 0
        opt.options["lpLog"] = 1
        opt.options["Trace"] = 1
        #instance.display('outputs_xpress.txt')
    if solver == "Gurobi":
        opt = SolverFactory('gurobi', Verbose=True)
        opt.options["Crossover"]=0
        opt.options["Method"]=2
        # opt.options['BarConvTol'] = 1e-4
    if solver == "GLPK":
        opt = SolverFactory("glpk", Verbose=True)
    
    # for (n,i) in instance.nodeLostLoadCost:
    #     instance.nodeLostLoadCost[n,i] = 1e6
    results = opt.solve(instance, tee=True)# , logfile=result_file_path + '\logfile_' + name + '.log' , keepfiles=True, symbolic_solver_labels=True)
    # bounds_dict = get_inv_cap_bounds(instance)
    # print(bounds_dict)
    # num_scenarios = len(instance.Scenario)
    instance.solutions.load_from(results)
    objective_value = value(instance.Obj)
    NoSce = len(Scenario)
    get_results(instance, seed,NoSce,lengthRegSeason)
    get_results_v(instance, seed,NoSce,lengthRegSeason)
    expected_second_stage_value, total_ll_amt = compute_expected_second_stage_value(instance)
    print(f"total_ll_amt: {total_ll_amt}")
    
    return objective_value, expected_second_stage_value


def compute_expected_second_stage_value(instance):
    # Calculate the total operational cost over all periods
    expected_second_stage_value = 0
    total_ll_amt = 0
    for i in instance.PeriodActive:
        second_stage_value = value(instance.discount_multiplier[i]) * (
            value(instance.shedcomponent[i]) + value(instance.operationalcost[i])
        )
        expected_second_stage_value += second_stage_value

        # Iterate over nodes, season-hour pairs, and scenarios
        for n in instance.Node:
            for (s, h) in instance.HoursOfSeason:
                for w in instance.Scenario:
                    ls_val = value(instance.loadShed[n, h, i, w])
                    # Print if load shed is greater than or equal to zero
                    # (Change the condition to > 0 if you only want strictly positive values.)
                    if ls_val > 1:
                        print(f"Period {i}, Season: {s}, Hour: {h}, Scenario: {w}, LoadShed: {ls_val}")

        shed_amt = sum(
            value(instance.sceProbab[w]) * value(instance.loadShed[n, h, i, w])
            for n in instance.Node
            for (s, h) in instance.HoursOfSeason
            for w in instance.Scenario
        )
        total_ll_amt += shed_amt

    return expected_second_stage_value, total_ll_amt



def get_results(instance, seed, NoSce,lengthRegSeason):
    # Retrieve relevant data from the instance
    gen_inv_cap = instance.genInvCap.get_values()
    transmision_inv_cap = instance.transmisionInvCap.get_values()
    stor_pw_inv_cap = instance.storPWInvCap.get_values()
    stor_en_inv_cap = instance.storENInvCap.get_values()

    total_fsd_length = len(gen_inv_cap) + len(transmision_inv_cap) + len(stor_pw_inv_cap) + len(stor_en_inv_cap)

    # Add a generator type label to each entry
    gen_inv_cap = {(k[0], k[1], k[2], 'Generation'): v for k, v in gen_inv_cap.items()}
    transmision_inv_cap = {(k[0], k[1], k[2], 'Transmission'): v for k, v in transmision_inv_cap.items()}
    stor_pw_inv_cap = {(k[0], k[1], k[2], 'Storage Power'): v for k, v in stor_pw_inv_cap.items()}
    stor_en_inv_cap = {(k[0], k[1], k[2], 'Storage Energy'): v for k, v in stor_en_inv_cap.items()}

    # Combine all investment capacities and costs into dictionaries
    inv_cap_data = {**gen_inv_cap, **transmision_inv_cap, **stor_pw_inv_cap, **stor_en_inv_cap}

    # Convert the capacity data into a DataFrame
    data = [(k[0], k[1], k[2], k[3], v) for k, v in inv_cap_data.items()]
    df = pd.DataFrame(data, columns=['Node', 'Energy_Type', 'Period', 'Type', 'Value'])

    # Create output directories if they don't exist
    output_dir = "ORIFSD"
    os.makedirs(output_dir, exist_ok=True)

    # Save capacity and cost data to CSV
    # output_file_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d%H%M')}_{total_fsd_length}_seed_{seed}_inv_cap.csv")
    output_file_path = os.path.join(output_dir, f"{NoSce}_seed_{seed}_len_{lengthRegSeason}_inv_cap.csv")
    df.to_csv(output_file_path, index=False)

    print("DataFrames created and saved successfully.")


def get_results_v(instance, seed,NoSce,lengthRegSeason):
    # Retrieve relevant data from the instance
    gen_installed_cap = instance.genInstalledCap.get_values()
    transmision_installed_cap = instance.transmissionInstalledCap.get_values()
    stor_pw_installed_cap = instance.storPWInstalledCap.get_values()
    stor_en_installed_cap = instance.storENInstalledCap.get_values()

    total_fsd_length = len(gen_installed_cap) + len(transmision_installed_cap) + len(stor_pw_installed_cap) + len(stor_en_installed_cap)

    # Add a generator type label to each entry
    gen_installed_cap = {(k[0], k[1], k[2], 'Generation'): v for k, v in gen_installed_cap.items()}
    transmision_installed_cap = {(k[0], k[1], k[2], 'Transmission'): v for k, v in transmision_installed_cap.items()}
    stor_pw_installed_cap = {(k[0], k[1], k[2], 'Storage Power'): v for k, v in stor_pw_installed_cap.items()}
    stor_en_installed_cap = {(k[0], k[1], k[2], 'Storage Energy'): v for k, v in stor_en_installed_cap.items()}

    # Combine all investment capacities and costs into dictionaries
    installed_cap_data = {**gen_installed_cap, **transmision_installed_cap, **stor_pw_installed_cap, **stor_en_installed_cap}

    # Convert the capacity data into a DataFrame
    data = [(k[0], k[1], k[2], k[3], v) for k, v in installed_cap_data.items()]
    df = pd.DataFrame(data, columns=['Node', 'Energy_Type', 'Period', 'Type', 'Value'])

    # Create output directories if they don't exist
    output_dir = "ORIFSD"
    os.makedirs(output_dir, exist_ok=True)

    # Save capacity and cost data to CSV
    # output_file_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d%H%M')}_{total_fsd_length}_seed_{seed}_inv_cap.csv")
    output_file_path = os.path.join(output_dir, f"{NoSce}_seed_{seed}_len_{lengthRegSeason}_installed_cap.csv")
    df.to_csv(output_file_path, index=False)

    print("DataFrames created and saved successfully.")





def analyze_load_shedding(instance, load_shed_threshold=1.0, output_file=None):
    records = []

    # Loop over all period/scenario/hour/node combinations.
    for i in instance.PeriodActive:
        for w in instance.Scenario:
            for (s, h) in instance.HoursOfSeason:
                for n in instance.Node:
                    ls_val = value(instance.loadShed[n, h, i, w])
                    # Focus on entries where load shedding is significant
                    if ls_val > load_shed_threshold:
                        # Basic info
                        record = {
                            'Period': i,
                            'Scenario': w,
                            'Season': s,
                            'Hour': h,
                            'Node': n,
                            'LoadShed': ls_val,
                            'Load': value(instance.sload[n, h, i, w]),
                        }

                        # Sum of available generation capacity at node n
                        # = sum over g: installed capacity * capacity-availability factor
                        # for the hour h and scenario w
                        sum_avail_cap = 0.0
                        sum_dispatched = 0.0
                        for g in instance.Generator:
                            if (n, g) in instance.GeneratorsOfNode:
                                installed_cap = value(instance.genInstalledCap[n, g, i])
                                avail_factor = value(instance.genCapAvail[n, g, h, w, i])
                                sum_avail_cap += installed_cap * avail_factor

                                # Actual dispatch
                                sum_dispatched += value(instance.genOperational[n, g, h, i, w])

                        record['TotalGenCap_Available'] = sum_avail_cap
                        record['TotalGen_Dispatched'] = sum_dispatched


                        total_gen_max_installed = 0.0
                        for t in instance.Technology: 
                            for g in instance.Generator: 
                                if (n,g) in instance.GeneratorsOfNode and (t,g) in instance.GeneratorsOfTechnology:
                                    total_gen_max_installed += value(instance.genMaxInstalledCap[n, t, i])
                            record['TotalGen_MaxInstalledCap'] = total_gen_max_installed


                        # Sum of stored energy in each storage at node n
                        # plus charge/discharge rates
                        stor_prev_level = 0.0
                        stor_level = 0.0
                        stor_charge = 0.0
                        stor_discharge = 0.0
                        if hasattr(instance, 'StoragesOfNode'):
                            for b in instance.Storage:
                                if (n, b) in instance.StoragesOfNode:
                                    stor_prev_level += value(instance.storOperational[n, b, (h-1), i, w])
                                    stor_level += value(instance.storOperational[n, b, h, i, w])
                                    stor_charge += value(instance.storCharge[n, b, h, i, w])
                                    stor_discharge += value(instance.storDischarge[n, b, h, i, w])

                        record['StoragePrevLevel'] = stor_prev_level
                        record['StorageLevel'] = stor_level
                        record['StorageCharge'] = stor_charge
                        record['StorageDischarge'] = stor_discharge

                        # Incoming flow from other nodes
                        # Summation of lineEfficiency[m,n]*transmisionOperational[m,n,h,i,w]
                        # where m is any node that has (m,n) in model.DirectionalLink
                        # or uses NodesLinked[n] if you have that set defined
                        incoming_flow = 0.0
                        if hasattr(instance, 'NodesLinked'):
                            # For each node 'm' s.t. (m,n) is a directional link
                            for m in instance.NodesLinked[n]:
                                # Some models store lineEfficiency with key (m,n), so check carefully
                                eff = value(instance.lineEfficiency[(m, n)])
                                flow = value(instance.transmisionOperational[(m, n), h, i, w])
                                incoming_flow += eff * flow
                        else:
                            # Alternative approach if not using NodesLinked
                            # You'd iterate over instance.DirectionalLink to find pairs (m,n)
                            # that match the target node n
                            for (m, nd) in instance.DirectionalLink:
                                if nd == n:
                                    eff = value(instance.lineEfficiency[(m, nd)])
                                    flow = value(instance.transmisionOperational[(m, nd, h, i, w)])
                                    incoming_flow += eff * flow

                        record['IncomingFlow'] = incoming_flow

                        # Append final record
                        records.append(record)

    # Build a DataFrame
    df = pd.DataFrame(records)

    # Optionally save to CSV
    if output_file:
        df.to_csv(output_file, index=False)

    return df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Specific seed')
    parser.add_argument('--lenreg', type=int, required=True, help='Len Reg')
    args = parser.parse_args()
    SEED = args.seed
    lengthRegSeason = args.lenreg

    UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

    USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
    temp_dir = UserRunTimeConfig["temp_dir"]
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
    # lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
    discountrate = UserRunTimeConfig["discountrate"]
    WACC = UserRunTimeConfig["WACC"]
    solver = UserRunTimeConfig["solver"]
    scenariogeneration = UserRunTimeConfig["scenariogeneration"]
    fix_sample = UserRunTimeConfig["fix_sample"]
    LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
    filter_make = UserRunTimeConfig["filter_make"] 
    filter_use = UserRunTimeConfig["filter_use"]
    n_cluster = UserRunTimeConfig["n_cluster"]
    moment_matching = UserRunTimeConfig["moment_matching"]
    n_tree_compare = UserRunTimeConfig["n_tree_compare"]
    EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]
    IAMC_PRINT = UserRunTimeConfig["IAMC_PRINT"]
    WRITE_LP = UserRunTimeConfig["WRITE_LP"]
    PICKLE_INSTANCE = UserRunTimeConfig["PICKLE_INSTANCE"] 

    #############################
    ##Non configurable settings##
    #############################

    NoOfRegSeason = 4
    regular_seasons = ["winter", "spring", "summer", "fall"]
    NoOfPeakSeason = 2
    lengthPeakSeason = 24
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    if version in ["europe_v51","europe_reduced_v51"]:
        north_sea = True
    else:
        north_sea = False

    #######
    ##RUN##
    #######

    name = version + '_reg' + str(lengthRegSeason) + \
        '_peak' + str(lengthPeakSeason) + \
        '_sce' + str(NoOfScenarios)
    if scenariogeneration and not fix_sample:
            name = name + "_randomSGR"
    else:
        name = name + "_noSGR"
    if filter_use:
        name = name + "_filter" + str(n_cluster)
    if moment_matching:
        name = name + "_moment" + str(n_tree_compare)
    name = name + str(datetime.now().strftime("_%Y%m%d%H%M"))
    workbook_path = 'Data handler/' + version
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name + f'_{lengthRegSeason}' + f'_{SEED}'
    scenario_data_path = 'Data handler/' + version + '/ScenarioData'
    result_file_path = 'Results/' + name
    FirstHoursOfRegSeason = [lengthRegSeason*i + 1 for i in range(NoOfRegSeason)]
    FirstHoursOfPeakSeason = [lengthRegSeason*NoOfRegSeason + lengthPeakSeason*i + 1 for i in range(NoOfPeakSeason)]
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
    Scenario = ["scenario"+str(i + 1) for i in range(NoOfScenarios)]
    peak_seasons = ['peak'+str(i + 1) for i in range(NoOfPeakSeason)]
    Season = regular_seasons + peak_seasons
    Operationalhour = [i + 1 for i in range(FirstHoursOfPeakSeason[-1] + lengthPeakSeason - 1)]
    HoursOfRegSeason = [(s,h) for s in regular_seasons for h in Operationalhour \
                    if h in list(range(regular_seasons.index(s)*lengthRegSeason+1,
                                regular_seasons.index(s)*lengthRegSeason+lengthRegSeason+1))]
    HoursOfPeakSeason = [(s,h) for s in peak_seasons for h in Operationalhour \
                        if h in list(range(lengthRegSeason*len(regular_seasons)+ \
                                            peak_seasons.index(s)*lengthPeakSeason+1,
                                            lengthRegSeason*len(regular_seasons)+ \
                                                peak_seasons.index(s)*lengthPeakSeason+ \
                                                    lengthPeakSeason+1))]
    HoursOfSeason = HoursOfRegSeason + HoursOfPeakSeason

    if version in ["europe_v51","europe_reduced_v51"]:
        dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
                      "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
                      "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
                      "ES": "Spain", "FI": "Finland", "FR": "France",
                      "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
                      "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
                      "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
                      "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
                      "PL": "Poland", "PT": "Portugal", "RO": "Romania",
                      "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
                      "SK": "Slovakia", "MF": "MorayFirth", "FF": "FirthofForth",
                      "DB": "DoggerBank", "HS": "Hornsea", "OD": "OuterDowsing",
                      "NF": "Norfolk", "EA": "EastAnglia", "BS": "Borssele",
                      "HK": "HollandseeKust", "HB": "HelgolanderBucht", "NS": "Nordsoen",
                      "UN": "UtsiraNord", "SN1": "SorligeNordsjoI", "SN2": "SorligeNordsjoII"}
    elif version in ["reduced"]:
        dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}
    else :
        dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
                        "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
                        "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
                        "ES": "Spain", "FI": "Finland", "FR": "France",
                        "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
                        "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
                        "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
                        "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
                        "PL": "Poland", "PT": "Portugal", "RO": "Romania",
                        "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
                        "SK": "Slovakia"}

    start_time = time.time()
    if scenariogeneration:
        generate_random_scenario(filepath = scenario_data_path,
                                tab_file_path = tab_file_path,
                                scenarios = NoOfScenarios,
                                seasons = regular_seasons,
                                Periods = len(Period),
                                regularSeasonHours = lengthRegSeason,
                                peakSeasonHours = lengthPeakSeason,
                                dict_countries = dict_countries,
                                time_format = time_format,
                                filter_make = filter_make,
                                filter_use = filter_use,
                                n_cluster = n_cluster,
                                moment_matching = moment_matching,
                                n_tree_compare = n_tree_compare,
                                fix_sample = fix_sample,
                                north_sea = north_sea,
                                LOADCHANGEMODULE = LOADCHANGEMODULE,
                                seed = SEED)

    generate_tab_files(filepath = workbook_path, tab_file_path = tab_file_path)

    # tab_file_path = f'Data handler/scenarios_PH/{NoOfScenarios}/{SEED}' # 다시해야함

    objective_value, expected_second_stage_value = run_empire(name = name, 
            tab_file_path = tab_file_path,
            result_file_path = result_file_path, 
            scenariogeneration = scenariogeneration,
            scenario_data_path = scenario_data_path,
            solver = solver,
            temp_dir = temp_dir, 
            FirstHoursOfRegSeason = FirstHoursOfRegSeason, 
            FirstHoursOfPeakSeason = FirstHoursOfPeakSeason, 
            lengthRegSeason = lengthRegSeason,
            lengthPeakSeason = lengthPeakSeason,
            Period = Period, 
            Operationalhour = Operationalhour,
            Scenario = Scenario,
            Season = Season,
            HoursOfSeason = HoursOfSeason,
            discountrate = discountrate, 
            WACC = WACC, 
            LeapYearsInvestment = LeapYearsInvestment,
            IAMC_PRINT = IAMC_PRINT, 
            WRITE_LP = WRITE_LP, 
            PICKLE_INSTANCE = PICKLE_INSTANCE, 
            EMISSION_CAP = EMISSION_CAP,
            USE_TEMP_DIR = USE_TEMP_DIR,
            LOADCHANGEMODULE = LOADCHANGEMODULE,
            seed = SEED,
            north_sea = north_sea,
            version = version)
    end_time = time.time()
    print("Objective Value :", objective_value)
    print("Expected Second Stage Value :", expected_second_stage_value)
    print("Total Solving Time :", end_time - start_time)