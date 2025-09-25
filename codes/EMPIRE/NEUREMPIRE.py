from __future__ import division
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager
import sys
import time
import os
import pandas as pd
import numpy as np
import multiprocessing



__author__ = "Stian Backe"
__license__ = "MIT"
__maintainer__ = "Stian Backe"
__email__ = "stian.backe@ntnu.no"

def run_empire(name, tab_file_path, result_file_path, scenariogeneration, scenario_data_path,
               solver, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
               lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
               discountrate, WACC, LeapYearsInvestment, IAMC_PRINT, WRITE_LP,
               PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, seed,north_sea):

    if USE_TEMP_DIR:
        TempfileManager.tempdir = temp_dir

    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    model = AbstractModel()

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
    model.nodeLostLoadCost = Param(model.Node, model.Period, default=1e+10, mutable=False)
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
    data.load(filename=tab_file_path + "/" + 'Generator_InitialCapacity.tab', param=model.genInitCap, format="table") #node_generator_intial_capacity.xlsx
    data.load(filename=tab_file_path + "/" + 'Generator_MaxBuiltCapacity.tab', param=model.genMaxBuiltCap, format="table")#?
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
    
    #Temporarily 
    # data.load(filename=tab_file_path + "/" + 'Node_NodeLostLoadCost.tab', param=model.nodeLostLoadCost, format="table")
    data.load(filename=tab_file_path + "/" + 'Node_ElectricAnnualDemand.tab', param=model.sloadAnnualDemand, format="table") 
    data.load(filename=tab_file_path + "/" + 'Node_HydroGenMaxAnnualProduction.tab', param=model.maxHydroNode, format="table") 
    
    if scenariogeneration:
        scenariopath = tab_file_path
    else:
        scenariopath = scenario_data_path

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
        f = open(result_file_path + '/AdjustedNegativeLoad_' + name + '.txt', 'w')
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
                        model.sload[n, h, i, sce] = model.sloadRaw[n,h,sce,i]*hourlyscale
                        if LOADCHANGEMODULE:
                            model.sload[n,h,i,sce] = model.sload[n,h,i,sce] + model.sloadMod[n,h,sce,i]
                        if value(model.sload[n,h,i,sce]) < 0:
                            f.write('Adjusted electricity load: ' + str(value(model.sload[n,h,i,sce])) + ', 10 MW for hour ' + str(h) + ' and scenario ' + str(sce) + ' in ' + str(n) + "\n")
                            model.sload[n,h,i,sce] = 10
                            counter += 1

        f.write('Hours with too small raw electricity load: ' + str(counter))
        f.close()

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
        first_stage_value = sum(model.discount_multiplier[i]*(sum(model.genInvCost[g,i]* model.genInvCap[n,g,i] for (n,g) in model.GeneratorsOfNode ) + \
            sum(model.transmissionInvCost[n1,n2,i]*model.transmisionInvCap[n1,n2,i] for (n1,n2) in model.BidirectionalArc ) + \
            sum((model.storPWInvCost[b,i]*model.storPWInvCap[n,b,i]+model.storENInvCost[b,i]*model.storENInvCap[n,b,i]) for (n,b) in model.StoragesOfNode ) 
            ) for i in model.PeriodActive)
        expected_second_stage_value = sum(model.discount_multiplier[i]*(model.shedcomponent[i] + model.operationalcost[i]) for i in model.PeriodActive)
        return first_stage_value + expected_second_stage_value
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
    if north_sea:
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

    def power_energy_relate_rule(model, n, b, i):
        if b in model.DependentStorage:
            return model.storPWInstalledCap[n,b,i] - model.storagePowToEnergy[b]*model.storENInstalledCap[n,b,i] == 0   #
        else:
            return Constraint.Skip
    model.power_energy_relate = Constraint(model.StoragesOfNode, model.PeriodActive, rule=power_energy_relate_rule)

    #################################################################

    def version1_rule(model, h, i):
            gen_avail_capacity = sum(model.sceProbab[w]*sum(model.genInstalledCap[n, g, i] * model.genCapAvail[n,g,h,w,i] for (n, g) in model.GeneratorsOfNode) for w in model.Scenario)
            average_sload = sum(model.sceProbab[w]*sum(model.sload[n,h,i,w] for n in model.Node) for w in model.Scenario)
            return average_sload-gen_avail_capacity <= 0 
    model.version1 = Constraint(model.Operationalhour, model.PeriodActive,rule=version1_rule)

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


    def soft_loadshed_rule(model, n, h, i, w):
        return model.loadShed[n, h, i, w] <= 0
    model.SoftLoadShed = Constraint(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, rule=soft_loadshed_rule)

    
    print("Objective and constraints read...")

    #######
    ##RUN##
    #######

    print("Building instance...")
    
    start = time.time()
    instance = model.create_instance(data) #, report_timing=True)
    instance.dual = Suffix(direction=Suffix.IMPORT) #Make sure the dual value is collected into solver results (if solver supplies dual information)
    
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
    

    if WRITE_LP:
        print("Writing LP-file...")
        start = time.time()
        lpstring = 'LP_' + name + '.lp'
        if USE_TEMP_DIR:
            lpstring = temp_dir + '/LP_'+ name + '.lp'
        instance.write(lpstring, io_options={'symbolic_solver_labels': True})
        end = time.time()
        print("Writing LP-file took [sec]:")
        print(end - start)

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
    if solver == "GLPK":
        opt = SolverFactory("glpk", Verbose=True)
    
    # for (n,i) in instance.nodeLostLoadCost:
    #     instance.nodeLostLoadCost[n,i] = 1e6
    results = opt.solve(instance, tee=True)# , logfile=result_file_path + '\logfile_' + name + '.log' , keepfiles=True, symbolic_solver_labels=True)
    # bounds_dict = get_inv_cap_bounds(instance)
    # print(bounds_dict)
    # num_scenarios = len(instance.Scenario)
    instance.solutions.load_from(results)
    get_results(instance, seed)
    get_results_v(instance, seed)
    expected_second_stage_value, total_ll_amt = compute_expected_second_stage_value(instance)
    objective_value = value(instance.Obj)
    print(f"total_ll_amt: {total_ll_amt}")

    return objective_value, expected_second_stage_value


def compute_expected_second_stage_value(instance):
    # Calculate the total operational cost over all periods
    expected_second_stage_value = 0
    total_ll_amt = 0
    for i in instance.PeriodActive:
        second_stage_value = value(instance.discount_multiplier[i]) * (value(instance.shedcomponent[i]) + value(instance.operationalcost[i]))
        expected_second_stage_value += second_stage_value
        shed_amt = sum(value(instance.operationalDiscountrate)*value(instance.seasScale[s])*value(instance.sceProbab[w])*value(instance.loadShed[n,h,i,w]) for n in instance.Node for w in instance.Scenario for (s,h) in instance.HoursOfSeason)
        total_ll_amt += value(instance.discount_multiplier[i]) * (shed_amt)
    return expected_second_stage_value, total_ll_amt



def compute_scenario_second_stage_values(instance):
    scenario_values = {}
    
    for w in instance.Scenario:
        second_stage_value = 0

        for i in instance.PeriodActive:
            # shedcomponent의 시나리오별 값 계산
            shedcomponent_value = sum(
                value(instance.operationalDiscountrate *
                      instance.seasScale[s] *
                      instance.sceProbab[w] *
                      instance.nodeLostLoadCost[n, i] *
                      instance.loadShed[n, h, i, w])
                for n in instance.Node
                for (s, h) in instance.HoursOfSeason
            )

            # operationalcost의 시나리오별 값 계산
            operational_cost = sum(
                value(instance.operationalDiscountrate *
                      instance.seasScale[s] *
                      instance.sceProbab[w] *
                      instance.genMargCost[g, i] *
                      instance.genOperational[n, g, h, i, w])
                for (n, g) in instance.GeneratorsOfNode
                for (s, h) in instance.HoursOfSeason
            )

            # 시나리오별 second stage 값을 합산
            second_stage_value += value(instance.discount_multiplier[i]) * (shedcomponent_value + operational_cost)

        # 각 시나리오의 값을 저장
        scenario_values[w] = second_stage_value


    values = list(scenario_values.values())

    # 평균 계산
    mean_value = sum(values) / len(values)
    
    # 분산 계산
    variance = sum((x - mean_value) ** 2 for x in values) / len(values)

    return variance




def get_results(instance, seed):
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
    output_dir = "FSD"
    os.makedirs(output_dir, exist_ok=True)

    # Save capacity and cost data to CSV
    # output_file_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d%H%M')}_{total_fsd_length}_seed_{seed}_inv_cap.csv")
    output_file_path = os.path.join(output_dir, f"{total_fsd_length}_seed_{seed}_inv_cap.csv")
    df.to_csv(output_file_path, index=False)

    print("DataFrames created and saved successfully.")


def get_results_v(instance, seed):
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
    output_dir = "FSD"
    os.makedirs(output_dir, exist_ok=True)

    # Save capacity and cost data to CSV
    # output_file_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d%H%M')}_{total_fsd_length}_seed_{seed}_inv_cap.csv")
    output_file_path = os.path.join(output_dir, f"{total_fsd_length}_seed_{seed}_installed_cap.csv")
    df.to_csv(output_file_path, index=False)

    print("DataFrames created and saved successfully.")



def get_inv_cap_bounds(instance):
    bounds_dict = {}

    # Generator Investment Capacity Bounds
    gen_inv_cap_bounds = {}
    for (n, g, i) in instance.genInvCap:
        lb = 0
        # Find the technology t corresponding to generator g
        t_list = [t for (t, g1) in instance.GeneratorsOfTechnology if g1 == g]
        if not t_list:
            ub = None  # No upper bound
        else:
            t = t_list[0]
            # Maximum built capacity in period i
            max_built_cap = value(instance.genMaxBuiltCap[n, t, i])
            # Upper bound is the minimum of max_additional_cap and max_built_cap
            ub = max(0, max_built_cap)
        gen_inv_cap_bounds[(n, g, i)] = (lb, ub)

    # Transmission Investment Capacity Bounds
    transmision_inv_cap_bounds = {}
    for (n1, n2, i) in instance.transmisionInvCap:
        lb = 0
        max_built_cap = value(instance.transmissionMaxBuiltCap[n1, n2, i])
        ub = max(0, max_built_cap)
        transmision_inv_cap_bounds[(n1, n2, i)] = (lb, ub)

    # Storage Power Investment Capacity Bounds
    stor_pw_inv_cap_bounds = {}
    for (n, b, i) in instance.storPWInvCap:
        lb = 0
        max_built_cap = value(instance.storPWMaxBuiltCap[n, b, i])
        ub = max(0, max_built_cap)
        stor_pw_inv_cap_bounds[(n, b, i)] = (lb, ub)

    # Storage Energy Investment Capacity Bounds
    stor_en_inv_cap_bounds = {}
    for (n, b, i) in instance.storENInvCap:
        lb = 0
        max_built_cap = value(instance.storENMaxBuiltCap[n, b, i])
        ub = max(0,max_built_cap)
        stor_en_inv_cap_bounds[(n, b, i)] = (lb, ub)

    bounds_dict['genInvCap'] = gen_inv_cap_bounds
    bounds_dict['transmisionInvCap'] = transmision_inv_cap_bounds
    bounds_dict['storPWInvCap'] = stor_pw_inv_cap_bounds
    bounds_dict['storENInvCap'] = stor_en_inv_cap_bounds

    return bounds_dict