from __future__ import division
from pyomo.environ import *
from pyomo.common.tempfiles import TempfileManager

def run_first_stage(version, tab_file_path, result_file_path, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
               lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
               discountrate, WACC, LeapYearsInvestment, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, north_sea):

    if USE_TEMP_DIR:
        TempfileManager.tempdir = temp_dir

    model = AbstractModel()

    tab_file_path = f'Data handler/base/{version}'
    
    ########
    ##SETS##
    ########


    #Define the sets


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

    data.load(filename=tab_file_path + "/" + 'Node_ElectricAnnualDemand.tab', param=model.sloadAnnualDemand, format="table")
    data.load(filename=tab_file_path + "/" + 'Node_HydroGenMaxAnnualProduction.tab', param=model.maxHydroNode, format="table")
    data.load(filename=tab_file_path + "/" + 'General_seasonScale.tab', param=model.seasScale, format="table")

    
    model.exp_sload = Param(model.Node, model.Period, model.Operationalhour, default=0.0, mutable = True)
    model.avg_cap_avail = Param(model.GeneratorsOfNode, model.Operationalhour, model.Period, default=0.0, mutable=True)
    data.load(filename=f'Data handler/base/reduced/Average_sload.tab', param=model.exp_sload, format="table")
    data.load(filename=f'Data handler/base/reduced/Average_cap_avail.tab', param=model.avg_cap_avail, format="table")
    
    if EMISSION_CAP:
        data.load(filename=tab_file_path + "/" + 'General_CO2Cap.tab', param=model.CO2cap, format="table")
    else:
        data.load(filename=tab_file_path + "/" + 'General_CO2Price.tab', param=model.CO2price, format="table")

    def prepSceProbab_rule(model):
        for sce in model.Scenario:
            model.sceProbab[sce] = value(1/len(model.Scenario))

    model.build_SceProbab = BuildAction(rule=prepSceProbab_rule)


    def prepInvCost_rule(model):

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

    def prepOperationalCostGen_rule(model):
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
        for (n,g) in model.GeneratorsOfNode:
            for i in model.PeriodActive:
                if value(model.genInitCap[n,g,i]) == 0:
                    model.genInitCap[n,g,i] = model.genRefInitCap[n,g]*(1-model.genScaleInitCap[g,i])

    model.build_InitialCapacityNodeGen = BuildAction(rule=prepInitialCapacityNodeGen_rule)

    def prepInitialCapacityTransmission_rule(model):
        for (n1,n2) in model.BidirectionalArc:
            for i in model.PeriodActive:
                if value(model.transmissionMaxInstalledCapRaw[n1,n2,i]) <= value(model.transmissionInitCap[n1,n2,i]):
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionInitCap[n1,n2,i]
                else:
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionMaxInstalledCapRaw[n1,n2,i]

    model.build_InitialCapacityTransmission = BuildAction(rule=prepInitialCapacityTransmission_rule)

    def prepOperationalDiscountrate_rule(model):
        model.operationalDiscountrate = sum((1+model.discountrate)**(-j) for j in list(range(0,value(model.LeapYearsInvestment))))

    model.build_operationalDiscountrate = BuildAction(rule=prepOperationalDiscountrate_rule)


    def prepGenMaxInstalledCap_rule(model):
        for t in model.Technology:
            for n in model.Node:
                for i in model.PeriodActive:
                    if value(model.genMaxInstalledCapRaw[n,t] <= sum(model.genInitCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology)):
                        model.genMaxInstalledCap[n,t,i]=sum(model.genInitCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology)
                    else:
                        model.genMaxInstalledCap[n,t,i]=model.genMaxInstalledCapRaw[n,t]

    model.build_genMaxInstalledCap = BuildAction(rule=prepGenMaxInstalledCap_rule)

    def storENMaxInstalledCap_rule(model):
        for (n,b) in model.StoragesOfNode:
            for i in model.PeriodActive:
                model.storENMaxInstalledCap[n,b,i]=model.storENMaxInstalledCapRaw[n,b]

    model.build_storENMaxInstalledCap = BuildAction(rule=storENMaxInstalledCap_rule)

    def storPWMaxInstalledCap_rule(model):
        for (n,b) in model.StoragesOfNode:
            for i in model.PeriodActive:
                model.storPWMaxInstalledCap[n,b,i]=model.storPWMaxInstalledCapRaw[n,b]

    model.build_storPWMaxInstalledCap = BuildAction(rule=storPWMaxInstalledCap_rule)


    #############
    ##VARIABLES##
    #############

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

    ###############
    ##EXPRESSIONS##
    ###############

    def multiplier_rule(model,period):
        coeff=1
        if period>1:
            coeff=pow(1.0+model.discountrate,(-LeapYearsInvestment*(int(period)-1)))
        return coeff
    model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)


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
    
    #################### NEW CONSTRAINTS #############################################
    
    def version1_rule(model, n, h, i):
        gen_avail_capacity = sum(model.genInstalledCap[n, g, i] * model.avg_cap_avail[n,g,h,i] for g in model.Generator if (n, g) in model.GeneratorsOfNode)
        return model.exp_sload[n,i,h]*1.3-gen_avail_capacity <= 0 
    model.version1 = Constraint(model.Node, model.Operationalhour, model.PeriodActive,rule=version1_rule)

    alpha_dict = {1: 0.5, 2: 0.55, 3: 0.6, 4: 0.65, 5: 0.78, 6: 0.78, 7: 0.80, 8: 0.90}
    model.alpha = Param(model.PeriodActive, initialize=alpha_dict)

    low_cost_tech = ['Solar', 'Windonshore', 'Windoffshore', 'Hydroregulated', 'Hydrorun-of-the-river', 'Geo', 'Wave', 'Nuclear', 'Bio', 'LigniteCCSadv', 'CoalCCSadv', 'GasCCSadv']
    def low_cost_init(model):
        return [(n,g) for (n,g) in model.GeneratorsOfNode if g in low_cost_tech]
    model.GeneratorsOfLowCost = Set(within=model.GeneratorsOfNode, initialize=low_cost_init)

    def low_cost_share_rule(model, n, i):
        total_cap = sum(model.genInstalledCap[n, g, i] 
                        for (n_tmp, g) in model.GeneratorsOfNode if n_tmp == n)
        low_cost_cap = sum(model.genInstalledCap[n, g, i] 
                        for (n_tmp, g) in model.GeneratorsOfLowCost if n_tmp == n)
        return (model.alpha[i] * total_cap - low_cost_cap <= 0)

    model.low_cost_share_constraint = Constraint(model.Node, model.PeriodActive, rule=low_cost_share_rule)

    model.beta = Param(initialize=0.35)
    def dynamic_storage_rule(model, n, i):
        low_cost_capacity_built = sum(model.genInstalledCap[n, g, i]
                                    for (n_tmp, g) in model.GeneratorsOfLowCost
                                    if n_tmp == n)

        storage_power_built = sum(model.storPWInstalledCap[n, b, i]
                                for (n2, b) in model.StoragesOfNode
                                if n2 == n and b in model.DependentStorage)

        return (model.beta * low_cost_capacity_built - storage_power_built) <= 0

    model.dynamic_storage_constraint = Constraint(model.Node, model.PeriodActive, rule=dynamic_storage_rule)


    model.max_avg_marginal_cost = Param(initialize=50.0, mutable=True)
    def avg_marginal_cost_rule(model, n, i):
        
        total_capacity = sum(model.genInstalledCap[n, g, i]  for g in model.Generator if (n, g) in model.GeneratorsOfNode)
        
        weighted_marginal_cost_sum = sum(model.genInstalledCap[n, g, i] * model.genMargCost[g, i]
                                        for g in model.Generator if (n, g) in model.GeneratorsOfNode)
        return weighted_marginal_cost_sum <= model.max_avg_marginal_cost * total_capacity

    model.avg_marginal_cost_constraint = Constraint(model.Node, model.PeriodActive, rule=avg_marginal_cost_rule)


    #########################################################################################



    # # variable for embedding
    # n_features = 616
    # model.v_scaled = Var(RangeSet(1, n_features), domain=Reals)

    instance = model.create_instance(data)
    instance.dual = Suffix(direction=Suffix.IMPORT) 


    return instance

