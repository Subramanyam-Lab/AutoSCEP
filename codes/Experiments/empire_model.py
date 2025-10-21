import pyomo.environ as pyo
from yaml import safe_load


def run_empire(scenariopath):
    UserRunTimeConfig = safe_load(open("config_run.yaml"))
    

    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
    discountrate = UserRunTimeConfig["discountrate"]
    WACC = UserRunTimeConfig["WACC"]
    EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]

    #############################
    ##Non configurable settings##
    #############################

    NoOfRegSeason = 4
    regular_seasons = ["winter", "spring", "summer", "fall"]
    NoOfPeakSeason = 2
    lengthPeakSeason = 24
    LeapYearsInvestment = 5
    north_sea = False


    FirstHoursOfRegSeason = [lengthRegSeason*i + 1 for i in range(NoOfRegSeason)]
    FirstHoursOfPeakSeason = [lengthRegSeason*NoOfRegSeason + lengthPeakSeason*i + 1 for i in range(NoOfPeakSeason)]
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
    Scenario = ["scenario1"]
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

    
    model = pyo.AbstractModel()

    # scenariopath = f'Data handler/base/{scenario[0]}'
    tab_file_path = f'Data handler/base/{version}'

    ########
    ##SETS##
    ########


    #Supply technology sets
    model.Generator = pyo.Set(ordered=True) #g
    model.Technology = pyo.Set(ordered=True) #t
    model.Storage =  pyo.Set() #b

    #Temporal sets
    model.Period = pyo.Set(ordered=True) #max period,|I|,it becomes 8
    model.PeriodActive = pyo.Set(ordered=True, initialize=Period) #i
    model.Operationalhour = pyo.Set(ordered=True, initialize=Operationalhour) #h
    model.Season = pyo.Set(ordered=True, initialize=Season) #s

    #Spatial sets
    model.Node = pyo.Set(ordered=True) #n
    if north_sea:
        model.OffshoreNode = pyo.Set(ordered=True, within=model.Node) #n
    model.DirectionalLink = pyo.Set(dimen=2, within=model.Node*model.Node, ordered=True) #a
    model.TransmissionType = pyo.Set(ordered=True)

    #Stochastic sets
    model.Scenario = pyo.Set(ordered=True, initialize=Scenario) #w
    model.ScenarioActive = pyo.Set(ordered=True, initialize=Scenario) #w

    #Subsets
    model.GeneratorsOfTechnology=pyo.Set(dimen=2) #(t,g) for all t in T, g in G_t
    model.GeneratorsOfNode = pyo.Set(dimen=2) #(n,g) for all n in N, g in G_n
    model.TransmissionTypeOfDirectionalLink = pyo.Set(dimen=3) #(n1,n2,t) for all (n1,n2) in L, t in T
    model.ThermalGenerators = pyo.Set(within=model.Generator) #g_ramp
    model.RegHydroGenerator = pyo.Set(within=model.Generator) #g_reghyd
    model.HydroGenerator = pyo.Set(within=model.Generator) #g_hyd
    model.StoragesOfNode = pyo.Set(dimen=2) #(n,b) for all n in N, b in B_n
    model.DependentStorage = pyo.Set() #b_dagger
    model.HoursOfSeason = pyo.Set(dimen=2, ordered=True, initialize=HoursOfSeason) #(s,h) for all s in S, h in H_s
    model.FirstHoursOfRegSeason = pyo.Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfRegSeason) # begining hour of the Regular Season
    model.FirstHoursOfPeakSeason = pyo.Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfPeakSeason) # begining hour of the Peak Season


    #Load the data

    data = pyo.DataPortal()
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
    model.NodesLinked = pyo.Set(model.Node, initialize=NodesLinked_init)

    def BidirectionalArc_init(model):
        retval = []
        for (i,j) in model.DirectionalLink:
            if i != j and (not (j,i) in retval):
                retval.append((i,j))
        return retval
    model.BidirectionalArc = pyo.Set(dimen=2, initialize=BidirectionalArc_init, ordered=True) #l

    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters


    #Scaling

    model.discountrate = pyo.Param(initialize=discountrate) 
    model.WACC = pyo.Param(initialize=WACC) 
    model.LeapYearsInvestment = pyo.Param(initialize=LeapYearsInvestment) # 5 year
    model.operationalDiscountrate = pyo.Param(mutable=True) 
    # model.sceProbab = Param(model.Scenario, mutable=True) # pi_w, \sum pi_w =1 
    model.seasScale = pyo.Param(model.Season, initialize=1.0, mutable=True) # \alpha_s
    model.lengthRegSeason = pyo.Param(initialize=lengthRegSeason) # 168
    model.lengthPeakSeason = pyo.Param(initialize=lengthPeakSeason) # 48 

    #Cost

    model.genCapitalCost = pyo.Param(model.Generator, model.Period, default=0, mutable=True)
    model.transmissionTypeCapitalCost = pyo.Param(model.TransmissionType, model.Period, default=0, mutable=True)
    model.storPWCapitalCost = pyo.Param(model.Storage, model.Period, default=0, mutable=True)
    model.storENCapitalCost = pyo.Param(model.Storage, model.Period, default=0, mutable=True)
    model.genFixedOMCost = pyo.Param(model.Generator, model.Period, default=0, mutable=True)
    model.transmissionTypeFixedOMCost = pyo.Param(model.TransmissionType, model.Period, default=0, mutable=True)
    model.storPWFixedOMCost = pyo.Param(model.Storage, model.Period, default=0, mutable=True)
    model.storENFixedOMCost = pyo.Param(model.Storage, model.Period, default=0, mutable=True)
    model.genInvCost = pyo.Param(model.Generator, model.Period, default=9000000, mutable=True)
    model.transmissionInvCost = pyo.Param(model.BidirectionalArc, model.Period, default=3000000, mutable=True)
    model.storPWInvCost = pyo.Param(model.Storage, model.Period, default=1000000, mutable=True)
    model.storENInvCost = pyo.Param(model.Storage, model.Period, default=800000, mutable=True)
    model.transmissionLength = pyo.Param(model.BidirectionalArc, default=0, mutable=True)
    model.genVariableOMCost = pyo.Param(model.Generator, default=0.0, mutable=True)
    model.genFuelCost = pyo.Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genMargCost = pyo.Param(model.Generator, model.Period, default=600, mutable=True)
    model.genCO2TypeFactor = pyo.Param(model.Generator, default=0.0, mutable=True)
    model.nodeLostLoadCost = pyo.Param(model.Node, model.Period, default=22000.0)
    # model.nodeLostLoadCost = pyo.Param(model.Node, model.Period, default=1e+6, mutable=False)
    model.CO2price = pyo.Param(model.Period, default=0.0, mutable=True)
    model.CCSCostTSFix = pyo.Param(initialize=1149873.72) #NB! Hard-coded
    model.CCSCostTSVariable = pyo.Param(model.Period, default=0.0, mutable=True)
    model.CCSRemFrac = pyo.Param(initialize=0.9)

    #Node dependent technology limitations

    model.genRefInitCap = pyo.Param(model.GeneratorsOfNode, default=0.0, mutable=True)
    model.genScaleInitCap = pyo.Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genInitCap = pyo.Param(model.GeneratorsOfNode, model.Period, default=0.0, mutable=True)
    model.transmissionInitCap = pyo.Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWInitCap = pyo.Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENInitCap = pyo.Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.genMaxBuiltCap = pyo.Param(model.Node, model.Technology, model.Period, default=500000.0, mutable=True)
    model.transmissionMaxBuiltCap = pyo.Param(model.BidirectionalArc, model.Period, default=20000.0, mutable=True)
    model.storPWMaxBuiltCap = pyo.Param(model.StoragesOfNode, model.Period, default=500000.0, mutable=True)
    model.storENMaxBuiltCap = pyo.Param(model.StoragesOfNode, model.Period, default=500000.0, mutable=True)
    model.genMaxInstalledCapRaw = pyo.Param(model.Node, model.Technology, default=0.0, mutable=True)
    model.genMaxInstalledCap = pyo.Param(model.Node, model.Technology, model.Period, default=0.0, mutable=True)
    model.transmissionMaxInstalledCapRaw = pyo.Param(model.BidirectionalArc, model.Period, default=0.0)
    model.transmissionMaxInstalledCap = pyo.Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWMaxInstalledCap = pyo.Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storPWMaxInstalledCapRaw = pyo.Param(model.StoragesOfNode, default=0.0, mutable=True)
    model.storENMaxInstalledCap = pyo.Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENMaxInstalledCapRaw = pyo.Param(model.StoragesOfNode, default=0.0, mutable=True)

    #Type dependent technology limitations

    model.genLifetime = pyo.Param(model.Generator, default=0.0, mutable=True)
    model.transmissionLifetime = pyo.Param(model.BidirectionalArc, default=40.0, mutable=True)
    model.storageLifetime = pyo.Param(model.Storage, default=0.0, mutable=True)
    model.genEfficiency = pyo.Param(model.Generator, model.Period, default=1.0, mutable=True)
    model.lineEfficiency = pyo.Param(model.DirectionalLink, default=0.97, mutable=True)
    model.storageChargeEff = pyo.Param(model.Storage, default=1.0, mutable=True)
    model.storageDischargeEff = pyo.Param(model.Storage, default=1.0, mutable=True)
    model.storageBleedEff = pyo.Param(model.Storage, default=1.0, mutable=True)
    model.genRampUpCap = pyo.Param(model.ThermalGenerators, default=0.0, mutable=True)
    model.storageDiscToCharRatio = pyo.Param(model.Storage, default=1.0, mutable=True) #NB! Hard-coded
    model.storagePowToEnergy = pyo.Param(model.DependentStorage, default=1.0, mutable=True)

    #Stochastic input

    model.sloadRaw = pyo.Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    model.sloadAnnualDemand = pyo.Param(model.Node, model.Period, default=0.0, mutable=True)
    model.sload = pyo.Param(model.Node, model.Operationalhour, model.Period, model.ScenarioActive, default=0.0, mutable=True)
    model.genCapAvailTypeRaw = pyo.Param(model.Generator, default=1.0, mutable=True)
    model.genCapAvailStochRaw = pyo.Param(model.GeneratorsOfNode, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)
    model.genCapAvail = pyo.Param(model.GeneratorsOfNode, model.Operationalhour, model.ScenarioActive, model.Period, default=0.0, mutable=True)
    model.maxRegHydroGenRaw = pyo.Param(model.Node, model.Period, model.HoursOfSeason, model.Scenario, default=0.0, mutable=True)
    model.maxRegHydroGen = pyo.Param(model.Node, model.Period, model.Season, model.ScenarioActive, default=0.0, mutable=True)
    model.maxHydroNode = pyo.Param(model.Node, default=0.0, mutable=True)
    model.storOperationalInit = pyo.Param(model.Storage, default=0.0, mutable=True) #Percentage of installed energy capacity initially
    

    if EMISSION_CAP:
        model.CO2cap = pyo.Param(model.Period, default=5000.0, mutable=True)
    


    #Load the parameters

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
    data.load(filename=tab_file_path + "/" + 'Node_ElectricAnnualDemand.tab', param=model.sloadAnnualDemand, format="table") 
    data.load(filename=tab_file_path + "/" + 'Node_HydroGenMaxAnnualProduction.tab', param=model.maxHydroNode, format="table") 
    

    data.load(filename=scenariopath + "/" + 'Stochastic_HydroGenMaxSeasonalProduction.tab', param=model.maxRegHydroGenRaw, format="table")
    data.load(filename=scenariopath + "/" + 'Stochastic_StochasticAvailability.tab', param=model.genCapAvailStochRaw, format="table") 
    data.load(filename=scenariopath + "/" + 'Stochastic_ElectricLoadRaw.tab', param=model.sloadRaw, format="table") 

    data.load(filename=tab_file_path + "/" + 'General_seasonScale.tab', param=model.seasScale, format="table") 

    if EMISSION_CAP:
        data.load(filename=tab_file_path + "/" + 'General_CO2Cap.tab', param=model.CO2cap, format="table")
    else:
        data.load(filename=tab_file_path + "/" + 'General_CO2Price.tab', param=model.CO2price, format="table")



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
    model.adjust_season_scale = pyo.BuildAction(rule=adjust_season_scale_rule)
    

    # This function consists the costs per period for each generator, storage, transmission
    def prepInvCost_rule(model):
        #Build investment cost for generators, storages and transmission. Annual cost is calculated for the lifetime of the generator and discounted for a year.
        #Then cost is discounted for the investment period (or the remaining lifetime). CCS generators has additional fixed costs depending on emissions. 

        #Generator 
        for g in model.Generator:
            for i in model.PeriodActive:
                costperyear=(model.WACC/(1-((1+model.WACC)**(-model.genLifetime[g]))))*model.genCapitalCost[g,i]+model.genFixedOMCost[g,i]
                costperperiod=costperyear*1000*(1-(1+model.discountrate)**-(min(pyo.value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), pyo.value(model.genLifetime[g]))))/(1-(1/(1+model.discountrate)))
                if ('CCS',g) in model.GeneratorsOfTechnology:
                    costperperiod+=model.CCSCostTSFix*model.CCSRemFrac*model.genCO2TypeFactor[g]*(3.6/model.genEfficiency[g,i])
                model.genInvCost[g,i]=costperperiod

        #Storage
        for b in model.Storage:
            for i in model.PeriodActive:
                costperyearPW=(model.WACC/(1-((1+model.WACC)**(-model.storageLifetime[b]))))*model.storPWCapitalCost[b,i]+model.storPWFixedOMCost[b,i]
                costperperiodPW=costperyearPW*1000*(1-(1+model.discountrate)**-(min(pyo.value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), pyo.value(model.storageLifetime[b]))))/(1-(1/(1+model.discountrate)))
                model.storPWInvCost[b,i]=costperperiodPW
                costperyearEN=(model.WACC/(1-((1+model.WACC)**(-model.storageLifetime[b]))))*model.storENCapitalCost[b,i]+model.storENFixedOMCost[b,i]
                costperperiodEN=costperyearEN*1000*(1-(1+model.discountrate)**-(min(pyo.value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), pyo.value(model.storageLifetime[b]))))/(1-(1/(1+model.discountrate)))
                model.storENInvCost[b,i]=costperperiodEN

        #Transmission
        for (n1,n2) in model.BidirectionalArc:
            for i in model.PeriodActive:
                for t in model.TransmissionType:
                    if (n1,n2,t) in model.TransmissionTypeOfDirectionalLink:
                        costperyear=(model.WACC/(1-((1+model.WACC)**(-model.transmissionLifetime[n1,n2]))))*model.transmissionLength[n1,n2]*model.transmissionTypeCapitalCost[t,i]+model.transmissionTypeFixedOMCost[t,i]
                        costperperiod=costperyear*(1-(1+model.discountrate)**-(min(pyo.value((len(model.PeriodActive)-i+1)*LeapYearsInvestment), pyo.value(model.transmissionLifetime[n1,n2]))))/(1-(1/(1+model.discountrate)))
                        model.transmissionInvCost[n1,n2,i]=costperperiod

    model.build_InvCost = pyo.BuildAction(rule=prepInvCost_rule) # This is the cost vector of first stage. 

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

    model.build_OperationalCostGen = pyo.BuildAction(rule=prepOperationalCostGen_rule)

    # This is \bar_{x}
    def prepInitialCapacityNodeGen_rule(model):
        #Build initial capacity for generator type in node

        for (n,g) in model.GeneratorsOfNode:
            for i in model.PeriodActive:
                if pyo.value(model.genInitCap[n,g,i]) == 0:
                    model.genInitCap[n,g,i] = model.genRefInitCap[n,g]*(1-model.genScaleInitCap[g,i])

    model.build_InitialCapacityNodeGen = pyo.BuildAction(rule=prepInitialCapacityNodeGen_rule)

    # This is \bar_{x}
    def prepInitialCapacityTransmission_rule(model):
        #Build initial capacity for transmission lines to ensure initial capacity is the upper installation bound if infeasible

        for (n1,n2) in model.BidirectionalArc:
            for i in model.PeriodActive:
                if pyo.value(model.transmissionMaxInstalledCapRaw[n1,n2,i]) <= pyo.value(model.transmissionInitCap[n1,n2,i]):
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionInitCap[n1,n2,i]
                else:
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionMaxInstalledCapRaw[n1,n2,i]

    model.build_InitialCapacityTransmission = pyo.BuildAction(rule=prepInitialCapacityTransmission_rule)

    # This is V on the mathematical obejctive function
    def prepOperationalDiscountrate_rule(model):
        #Build operational discount rate

        model.operationalDiscountrate = sum((1+model.discountrate)**(-j) for j in list(range(0,pyo.value(model.LeapYearsInvestment))))

    model.build_operationalDiscountrate = pyo.BuildAction(rule=prepOperationalDiscountrate_rule)     


    # Following functions are represent \bar_{V}
    def prepGenMaxInstalledCap_rule(model):
        #Build resource limit (installed limit) for all periods. Avoid infeasibility if installed limit lower than initially installed cap.

        for t in model.Technology:
            for n in model.Node:
                for i in model.PeriodActive:
                    if pyo.value(model.genMaxInstalledCapRaw[n,t] <= sum(model.genInitCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology)):
                        model.genMaxInstalledCap[n,t,i]=sum(model.genInitCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology)
                    else:
                        model.genMaxInstalledCap[n,t,i]=model.genMaxInstalledCapRaw[n,t] 
                        
    model.build_genMaxInstalledCap = pyo.BuildAction(rule=prepGenMaxInstalledCap_rule)

    def storENMaxInstalledCap_rule(model):
        #Build installed limit (resource limit) for storEN

        for (n,b) in model.StoragesOfNode:
            for i in model.PeriodActive:
                model.storENMaxInstalledCap[n,b,i]=model.storENMaxInstalledCapRaw[n,b]

    model.build_storENMaxInstalledCap = pyo.BuildAction(rule=storENMaxInstalledCap_rule)

    def storPWMaxInstalledCap_rule(model):
        #Build installed limit (resource limit) for storPW

        for (n,b) in model.StoragesOfNode:
            for i in model.PeriodActive:
                model.storPWMaxInstalledCap[n,b,i]=model.storPWMaxInstalledCapRaw[n,b]

    model.build_storPWMaxInstalledCap = pyo.BuildAction(rule=storPWMaxInstalledCap_rule)


    def prepRegHydro_rule(model):
        #Build hydrolimits for all periods

        for n in model.Node:
            for s in model.Season:
                for i in model.PeriodActive:
                    for sce in model.ScenarioActive:
                        model.maxRegHydroGen[n,i,s,sce]=sum(model.maxRegHydroGenRaw[n,i,s,h,sce] for h in model.Operationalhour if (s,h) in model.HoursOfSeason)

    model.build_maxRegHydroGen = pyo.BuildAction(rule=prepRegHydro_rule)

    def prepGenCapAvail_rule(model):
        #Build generator availability for all periods

        for (n,g) in model.GeneratorsOfNode:
            for h in model.Operationalhour:
                for s in model.ScenarioActive:
                    for i in model.PeriodActive:
                        if pyo.value(model.genCapAvailTypeRaw[g]) == 0:
                            model.genCapAvail[n,g,h,s,i]=model.genCapAvailStochRaw[n,g,h,s,i]
                        else:
                            model.genCapAvail[n,g,h,s,i]=model.genCapAvailTypeRaw[g]

    model.build_genCapAvail = pyo.BuildAction(rule=prepGenCapAvail_rule)

    def prepSload_rule(model):
        #Build load profiles for all periods

        counter = 0
        # f = open(result_file_path + '/AdjustedNegativeLoad_' + name + '.txt', 'w')
        for n in model.Node:
            for i in model.PeriodActive:
                noderawdemand = 0
                for (s,h) in model.HoursOfSeason:
                    # if value(h) < value(FirstHoursOfRegSeason[-1] + model.lengthRegSeason):
                        for sce in model.ScenarioActive:
                                # noderawdemand += value(model.sceProbab[sce]*model.seasScale[s]*model.sloadRaw[n,h,sce,i])
                                noderawdemand += pyo.value(model.seasScale[s]*model.sloadRaw[n,h,sce,i])
                if pyo.value(model.sloadAnnualDemand[n,i]) < 1:
                    hourlyscale = 0
                else:
                    hourlyscale = pyo.value(model.sloadAnnualDemand[n,i]) / noderawdemand
                for h in model.Operationalhour:
                    for sce in model.ScenarioActive:
                        model.sload[n, h, i, sce] = (model.sloadRaw[n,h,sce,i]*hourlyscale)
                        # if LOADCHANGEMODULE:
                        #     model.sload[n,h,i,sce] = model.sload[n,h,i,sce] + model.sloadMod[n,h,sce,i]
                        if pyo.value(model.sload[n,h,i,sce]) < 0:
                            # f.write('Adjusted electricity load: ' + str(value(model.sload[n,h,i,sce])) + ', 10 MW for hour ' + str(h) + ' and scenario ' + str(sce) + ' in ' + str(n) + "\n")
                            model.sload[n,h,i,sce] = 10
                            counter += 1

        # f.write('Hours with too small raw electricity load: ' + str(counter))
        # f.close()

    model.build_sload = pyo.BuildAction(rule=prepSload_rule)

 
    #############
    ##VARIABLES##
    #############

 
    # First Stage Decisions (x)
    model.genInvCap = pyo.Var(model.GeneratorsOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.transmisionInvCap = pyo.Var(model.BidirectionalArc, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.storPWInvCap = pyo.Var(model.StoragesOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.storENInvCap = pyo.Var(model.StoragesOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)

    # First Stage Decisions (v)
    model.genInstalledCap = pyo.Var(model.GeneratorsOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.transmissionInstalledCap = pyo.Var(model.BidirectionalArc, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.storPWInstalledCap = pyo.Var(model.StoragesOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)
    model.storENInstalledCap = pyo.Var(model.StoragesOfNode, model.PeriodActive, domain=pyo.NonNegativeReals)

    # Second Stage Decisions (y,w)
    model.genOperational = pyo.Var(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, domain=pyo.NonNegativeReals)
    model.storOperational = pyo.Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, domain=pyo.NonNegativeReals)
    model.transmisionOperational = pyo.Var(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.ScenarioActive, domain=pyo.NonNegativeReals) #flow
    model.storCharge = pyo.Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, domain=pyo.NonNegativeReals)
    model.storDischarge = pyo.Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, domain=pyo.NonNegativeReals)
    model.loadShed = pyo.Var(model.Node, model.Operationalhour, model.PeriodActive, model.ScenarioActive, domain=pyo.NonNegativeReals)
    
    ###############
    ##EXPRESSIONS##
    ###############

    def multiplier_rule(model,period):
        coeff=1
        if period>1:
            coeff=pow(1.0+model.discountrate,(-LeapYearsInvestment*(int(period)-1)))
        return coeff
    model.discount_multiplier=pyo.Expression(model.PeriodActive, rule=multiplier_rule)

    def shed_component_rule(model):
        return sum(model.discount_multiplier[i]*sum(model.operationalDiscountrate*model.seasScale[s]*model.nodeLostLoadCost[n,i]*model.loadShed[n,h,i,w] for n in model.Node for w in model.ScenarioActive for (s,h) in model.HoursOfSeason) for i in model.PeriodActive)
    model.shedcomponent=pyo.Expression(rule=shed_component_rule)

    def operational_cost_rule(model):
        return sum(model.discount_multiplier[i]*sum(model.operationalDiscountrate*model.seasScale[s]*model.genMargCost[g,i]*model.genOperational[n,g,h,i,w] for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason for w in model.ScenarioActive)for i in model.PeriodActive)
    model.operationalcost=pyo.Expression(rule=operational_cost_rule)

    def investment_cost_rule(model):
        return sum(model.discount_multiplier[i]*(
            sum(model.genInvCost[g,i]* model.genInvCap[n,g,i] for (n,g) in model.GeneratorsOfNode ) + \
            sum(model.transmissionInvCost[n1,n2,i]*model.transmisionInvCap[n1,n2,i] for (n1,n2) in model.BidirectionalArc ) + \
            sum((model.storPWInvCost[b,i]*model.storPWInvCap[n,b,i]+model.storENInvCost[b,i]*model.storENInvCap[n,b,i]) for (n,b) in model.StoragesOfNode))for i in model.PeriodActive) 
    model.investcost=pyo.Expression(rule=investment_cost_rule)

    #############
    ##OBJECTIVE##
    #############

    model.Obj = pyo.Objective(expr=model.investcost + model.operationalcost + model.shedcomponent , sense=pyo.minimize)

    ###############
    ##CONSTRAINTS##
    ###############

    #### First Stage Constraints ####

    def lifetime_rule_gen(model, n, g, i):
        startPeriod=1
        if pyo.value(1+i-(model.genLifetime[g]/model.LeapYearsInvestment))>startPeriod:
            startPeriod=pyo.value(1+i-model.genLifetime[g]/model.LeapYearsInvestment)
        return sum(model.genInvCap[n,g,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.genInstalledCap[n,g,i] + model.genInitCap[n,g,i]== 0   #
    model.installedCapDefinitionGen = pyo.Constraint(model.GeneratorsOfNode, model.PeriodActive, rule=lifetime_rule_gen)

    #################################################################

    def lifetime_rule_storEN(model, n, b, i):
        startPeriod=1
        if pyo.value(1+i-model.storageLifetime[b]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=pyo.value(1+i-model.storageLifetime[b]/model.LeapYearsInvestment)
        return sum(model.storENInvCap[n,b,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.storENInstalledCap[n,b,i] + model.storENInitCap[n,b,i]== 0   #
    model.installedCapDefinitionStorEN = pyo.Constraint(model.StoragesOfNode, model.PeriodActive, rule=lifetime_rule_storEN)

    #################################################################

    def lifetime_rule_storPOW(model, n, b, i):
        startPeriod=1
        if pyo.value(1+i-model.storageLifetime[b]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=pyo.value(1+i-model.storageLifetime[b]/model.LeapYearsInvestment)
        return sum(model.storPWInvCap[n,b,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.storPWInstalledCap[n,b,i] + model.storPWInitCap[n,b,i]== 0   #
    model.installedCapDefinitionStorPOW = pyo.Constraint(model.StoragesOfNode, model.PeriodActive, rule=lifetime_rule_storPOW)

    #################################################################

    def lifetime_rule_trans(model, n1, n2, i):
        startPeriod=1
        if pyo.value(1+i-model.transmissionLifetime[n1,n2]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=pyo.value(1+i-model.transmissionLifetime[n1,n2]/model.LeapYearsInvestment)
        return sum(model.transmisionInvCap[n1,n2,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.transmissionInstalledCap[n1,n2,i] + model.transmissionInitCap[n1,n2,i] == 0   #
    model.installedCapDefinitionTrans = pyo.Constraint(model.BidirectionalArc, model.PeriodActive, rule=lifetime_rule_trans)

    #################################################################

    def investment_gen_cap_rule(model, t, n, i):
        return sum(model.genInvCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology) - model.genMaxBuiltCap[n,t,i] <= 0
    model.investment_gen_cap = pyo.Constraint(model.Technology, model.Node, model.PeriodActive, rule=investment_gen_cap_rule)

    #################################################################

    def investment_trans_cap_rule(model, n1, n2, i):
        return model.transmisionInvCap[n1,n2,i] - model.transmissionMaxBuiltCap[n1,n2,i] <= 0
    model.investment_trans_cap = pyo.Constraint(model.BidirectionalArc, model.PeriodActive, rule=investment_trans_cap_rule)

    #################################################################

    def investment_storage_power_cap_rule(model, n, b, i):
        return model.storPWInvCap[n,b,i] - model.storPWMaxBuiltCap[n,b,i] <= 0
    model.investment_storage_power_cap = pyo.Constraint(model.StoragesOfNode, model.PeriodActive, rule=investment_storage_power_cap_rule)

    #################################################################

    def investment_storage_energy_cap_rule(model, n, b, i):
        return model.storENInvCap[n,b,i] - model.storENMaxBuiltCap[n,b,i] <= 0
    model.investment_storage_energy_cap = pyo.Constraint(model.StoragesOfNode, model.PeriodActive, rule=investment_storage_energy_cap_rule)

    ################################################################

    def installed_gen_cap_rule(model, t, n, i):
        return sum(model.genInstalledCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology) - model.genMaxInstalledCap[n,t,i] <= 0
    model.installed_gen_cap = pyo.Constraint(model.Technology, model.Node, model.PeriodActive, rule=installed_gen_cap_rule)

    #################################################################

    def installed_trans_cap_rule(model, n1, n2, i):
        return model.transmissionInstalledCap[n1,n2,i] - model.transmissionMaxInstalledCap[n1,n2,i] <= 0
    model.installed_trans_cap = pyo.Constraint(model.BidirectionalArc, model.PeriodActive, rule=installed_trans_cap_rule)

    #################################################################

    def installed_storage_power_cap_rule(model, n, b, i):
        return model.storPWInstalledCap[n,b,i] - model.storPWMaxInstalledCap[n,b,i] <= 0
    model.installed_storage_power_cap = pyo.Constraint(model.StoragesOfNode, model.PeriodActive, rule=installed_storage_power_cap_rule)

    #################################################################

    def installed_storage_energy_cap_rule(model, n, b, i):
        return model.storENInstalledCap[n,b,i] - model.storENMaxInstalledCap[n,b,i] <= 0
    model.installed_storage_energy_cap = pyo.Constraint(model.StoragesOfNode, model.PeriodActive, rule=installed_storage_energy_cap_rule)

    #################################################################

    def power_energy_relate_rule(model, n, b, i):
        if b in model.DependentStorage:
            return model.storPWInstalledCap[n,b,i] - model.storagePowToEnergy[b]*model.storENInstalledCap[n,b,i] == 0   #
        else:
            return pyo.Constraint.Skip
    model.power_energy_relate = pyo.Constraint(model.StoragesOfNode, model.PeriodActive, rule=power_energy_relate_rule)

    #################################################################

    #### Second Stage Constraints ####

    def FlowBalance_rule(model, n, h, i, w):
        return sum(model.genOperational[n,g,h,i,w] for g in model.Generator if (n,g) in model.GeneratorsOfNode) \
            + sum((model.storageDischargeEff[b]*model.storDischarge[n,b,h,i,w]-model.storCharge[n,b,h,i,w]) for b in model.Storage if (n,b) in model.StoragesOfNode) \
            + sum((model.lineEfficiency[link,n]*model.transmisionOperational[link,n,h,i,w] - model.transmisionOperational[n,link,h,i,w]) for link in model.NodesLinked[n]) \
            - model.sload[n,h,i,w] + model.loadShed[n,h,i,w] \
            == 0
    model.FlowBalance = pyo.Constraint(model.Node, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=FlowBalance_rule)

    #################################################################

    def genMaxProd_rule(model, n, g, h, i, w):
            return model.genOperational[n,g,h,i,w] - model.genCapAvail[n,g,h,w,i]*model.genInstalledCap[n,g,i] <= 0
    model.maxGenProduction = pyo.Constraint(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=genMaxProd_rule)

    #################################################################

    def ramping_rule(model, n, g, h, i, w):
        if h in model.FirstHoursOfRegSeason or h in model.FirstHoursOfPeakSeason:
            return pyo.Constraint.Skip
        else:
            if g in model.ThermalGenerators:
                return model.genOperational[n,g,h,i,w]-model.genOperational[n,g,(h-1),i,w] - model.genRampUpCap[g]*model.genInstalledCap[n,g,i] <= 0   #
            else:
                return pyo.Constraint.Skip
    model.ramping = pyo.Constraint(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=ramping_rule)

    #################################################################

    def storage_energy_balance_rule(model, n, b, h, i, w):
        if h in model.FirstHoursOfRegSeason or h in model.FirstHoursOfPeakSeason:
            return model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] + model.storageChargeEff[b]*model.storCharge[n,b,h,i,w]-model.storDischarge[n,b,h,i,w]-model.storOperational[n,b,h,i,w] == 0   #
        else:
            return model.storageBleedEff[b]*model.storOperational[n,b,(h-1),i,w] + model.storageChargeEff[b]*model.storCharge[n,b,h,i,w]-model.storDischarge[n,b,h,i,w]-model.storOperational[n,b,h,i,w] == 0   #
    model.storage_energy_balance = pyo.Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=storage_energy_balance_rule)

    #################################################################

    def storage_seasonal_net_zero_balance_rule(model, n, b, h, i, w):
        if h in model.FirstHoursOfRegSeason:
            return model.storOperational[n,b,h+pyo.value(model.lengthRegSeason)-1,i,w] - model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] == 0  #
        elif h in model.FirstHoursOfPeakSeason:
            return model.storOperational[n,b,h+pyo.value(model.lengthPeakSeason)-1,i,w] - model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] == 0  #
        else:
            return pyo.Constraint.Skip
    model.storage_seasonal_net_zero_balance = pyo.Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=storage_seasonal_net_zero_balance_rule)

    #################################################################

    def storage_operational_cap_rule(model, n, b, h, i, w):
        return model.storOperational[n,b,h,i,w] - model.storENInstalledCap[n,b,i]  <= 0   #
    model.storage_operational_cap = pyo.Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=storage_operational_cap_rule)

    #################################################################

    def storage_power_discharg_cap_rule(model, n, b, h, i, w):
        return model.storDischarge[n,b,h,i,w] - model.storageDiscToCharRatio[b]*model.storPWInstalledCap[n,b,i] <= 0   #
    model.storage_power_discharg_cap = pyo.Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=storage_power_discharg_cap_rule)

    #################################################################

    def storage_power_charg_cap_rule(model, n, b, h, i, w):
        return model.storCharge[n,b,h,i,w] - model.storPWInstalledCap[n,b,i] <= 0   #
    model.storage_power_charg_cap = pyo.Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=storage_power_charg_cap_rule)

    #################################################################

    def hydro_gen_limit_rule(model, n, g, s, i, w):
        if g in model.RegHydroGenerator:
            return sum(model.genOperational[n,g,h,i,w] for h in model.Operationalhour if (s,h) in model.HoursOfSeason) - model.maxRegHydroGen[n,i,s,w] <= 0
        else:
            return pyo.Constraint.Skip  #
    model.hydro_gen_limit = pyo.Constraint(model.GeneratorsOfNode, model.Season, model.PeriodActive, model.ScenarioActive, rule=hydro_gen_limit_rule)

    #################################################################

    def transmission_cap_rule(model, n1, n2, h, i, w):
        if (n1,n2) in model.BidirectionalArc:
            return model.transmisionOperational[(n1,n2),h,i,w]  - model.transmissionInstalledCap[(n1,n2),i] <= 0
        elif (n2,n1) in model.BidirectionalArc:
            return model.transmisionOperational[(n1,n2),h,i,w]  - model.transmissionInstalledCap[(n2,n1),i] <= 0
    model.transmission_cap = pyo.Constraint(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.ScenarioActive, rule=transmission_cap_rule)

    #################################################################

    if EMISSION_CAP:
        def emission_cap_rule(model, i, w):
            return sum(model.seasScale[s]*model.genCO2TypeFactor[g]*(3.6/model.genEfficiency[g,i])*model.genOperational[n,g,h,i,w] for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason)/1000000 \
                - model.CO2cap[i] <= 0   #
        model.emission_cap = pyo.Constraint(model.PeriodActive, model.ScenarioActive, rule=emission_cap_rule)

    #################################################################

    
    return model,data
