from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
import os
import time

def run_empire(name, tab_file_path, result_file_path, scenariogeneration, scenario_data_path,
               solver, temp_dir, FirstHoursOfRegSeason, FirstHoursOfPeakSeason, lengthRegSeason,
               lengthPeakSeason, Period, Operationalhour, Scenario, Season, HoursOfSeason,
               discountrate, WACC, LeapYearsInvestment, IAMC_PRINT, WRITE_LP,
               PICKLE_INSTANCE, EMISSION_CAP, USE_TEMP_DIR, LOADCHANGEMODULE, specific_period):

    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    model = Model("empire")

    # Load Sets using Pandas
    def load_set(filename):
        df = pd.read_csv(filename, sep='\t', header=None)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    def load_set_dim2(filename):
        df = pd.read_csv(filename, sep='\t', header=None)
        return list(df.iloc[:, :2].dropna().apply(tuple, axis=1))

    def load_set_dim3(filename):
        df = pd.read_csv(filename, sep='\t', header=None)
        return list(df.iloc[:, :3].dropna().apply(tuple, axis=1))

    # Supply technology sets
    Generator = load_set(tab_file_path + "/" + 'Sets_Generator.tab')
    Technology = load_set(tab_file_path + "/" + 'Sets_Technology.tab')
    Storage = load_set(tab_file_path + "/" + 'Sets_Storage.tab')

    # Thermal and Hydro Generators
    ThermalGenerators = load_set(tab_file_path + "/" + 'Sets_ThermalGenerators.tab')
    RegHydroGenerator = load_set(tab_file_path + "/" + 'Sets_HydroGeneratorWithReservoir.tab')
    HydroGenerator = load_set(tab_file_path + "/" + 'Sets_HydroGenerator.tab')
    DependentStorage = load_set(tab_file_path + "/" + 'Sets_DependentStorage.tab')

    # Temporal sets
    PeriodSet = Period
    OperationalhourSet = Operationalhour
    SeasonSet = Season
    ScenarioSet = Scenario
    HoursOfSeasonSet = HoursOfSeason

    # Spatial sets
    Node = load_set(tab_file_path + "/" + 'Sets_Node.tab')
    OffshoreNode = load_set(tab_file_path + "/" + 'Sets_OffshoreNode.tab')
    DirectionalLink = load_set_dim2(tab_file_path + "/" + 'Sets_DirectionalLines.tab')
    TransmissionType = load_set(tab_file_path + "/" + 'Sets_LineType.tab')

    # Subsets
    GeneratorsOfTechnology = load_set_dim2(tab_file_path + "/" + 'Sets_GeneratorsOfTechnology.tab')
    GeneratorsOfNode = load_set_dim2(tab_file_path + "/" + 'Sets_GeneratorsOfNode.tab')
    TransmissionTypeOfDirectionalLink = load_set_dim3(tab_file_path + "/" + 'Sets_LineTypeOfDirectionalLines.tab')
    StoragesOfNode = load_set_dim2(tab_file_path + "/" + 'Sets_StorageOfNodes.tab')

    # Build arc subsets
    NodesLinked = {node: [] for node in Node}
    for (i, j) in DirectionalLink:
        if j in NodesLinked:
            NodesLinked[j].append(i)
        else:
            NodesLinked[j] = [i]

    BidirectionalArc = []
    seen_arcs = set()
    for (i, j) in DirectionalLink:
        if (j, i) not in seen_arcs and i != j:
            BidirectionalArc.append((i, j))
            seen_arcs.add((i, j))
            seen_arcs.add((j, i))

    # Load Parameters using Pandas
    def load_param(filename, index_cols, param_col='Value'):
        df = pd.read_csv(filename, sep='\t')
        df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace from column names
        print(f"Columns in {filename}: {df.columns.tolist()}")  # Print columns for debugging
        df.set_index(index_cols, inplace=True)
        param_dict = df[param_col].to_dict()
        return param_dict

    # Generator parameters
    genCapitalCost = load_param(tab_file_path + "/" + 'Generator_CapitalCosts.tab', ['Generator', 'Period'])
    genFixedOMCost = load_param(tab_file_path + "/" + 'Generator_FixedOMCosts.tab', ['Generator', 'Period'])
    genVariableOMCost = load_param(tab_file_path + "/" + 'Generator_VariableOMCosts.tab', ['Generator'])
    genFuelCost = load_param(tab_file_path + "/" + 'Generator_FuelCosts.tab', ['Generator', 'Period'])
    CCSCostTSVariable = load_param(tab_file_path + "/" + 'Generator_CCSCostTSVariable.tab', ['Period'])
    genEfficiency = load_param(tab_file_path + "/" + 'Generator_Efficiency.tab', ['Generator', 'Period'])
    genRefInitCap = load_param(tab_file_path + "/" + 'Generator_RefInitialCap.tab', ['Node', 'Generator'])
    genScaleInitCap = load_param(tab_file_path + "/" + 'Generator_ScaleFactorInitialCap.tab', ['Generator', 'Period'])
    genInitCap = load_param(tab_file_path + "/" + 'Generator_InitialCapacity.tab', ['Node', 'Generator', 'Period'])
    genMaxBuiltCap = load_param(tab_file_path + "/" + 'Generator_MaxBuiltCapacity.tab', ['Node', 'Technology', 'Period'])
    genMaxInstalledCapRaw = load_param(tab_file_path + "/" + 'Generator_MaxInstalledCapacity.tab', ['Node', 'Technology'])
    genCO2TypeFactor = load_param(tab_file_path + "/" + 'Generator_CO2Content.tab', ['Generator'])
    genRampUpCap = load_param(tab_file_path + "/" + 'Generator_RampRate.tab', ['Generator'])
    genCapAvailTypeRaw = load_param(tab_file_path + "/" + 'Generator_GeneratorTypeAvailability.tab', ['Generator'])
    genLifetime = load_param(tab_file_path + "/" + 'Generator_Lifetime.tab', ['Generator'])

    # Transmission parameters
    transmissionInitCap = load_param(tab_file_path + "/" + 'Transmission_InitialCapacity.tab', ['Node1', 'Node2', 'Period'])
    transmissionMaxBuiltCap = load_param(tab_file_path + "/" + 'Transmission_MaxBuiltCapacity.tab', ['Node1', 'Node2', 'Period'])
    transmissionMaxInstalledCapRaw = load_param(tab_file_path + "/" + 'Transmission_MaxInstallCapacityRaw.tab', ['Node1', 'Node2'])
    transmissionLength = load_param(tab_file_path + "/" + 'Transmission_Length.tab', ['Node1', 'Node2'])
    transmissionTypeCapitalCost = load_param(tab_file_path + "/" + 'Transmission_TypeCapitalCost.tab', ['TransmissionType', 'Period'])
    transmissionTypeFixedOMCost = load_param(tab_file_path + "/" + 'Transmission_TypeFixedOMCost.tab', ['TransmissionType', 'Period'])
    lineEfficiency = load_param(tab_file_path + "/" + 'Transmission_lineEfficiency.tab', ['Node1', 'Node2'])
    transmissionLifetime = load_param(tab_file_path + "/" + 'Transmission_Lifetime.tab', ['Node1', 'Node2'])

    # Storage parameters
    storageBleedEff = load_param(tab_file_path + "/" + 'Storage_StorageBleedEfficiency.tab', ['Storage'])
    storageChargeEff = load_param(tab_file_path + "/" + 'Storage_StorageChargeEff.tab', ['Storage'])
    storageDischargeEff = load_param(tab_file_path + "/" + 'Storage_StorageDischargeEff.tab', ['Storage'])
    storagePowToEnergy = load_param(tab_file_path + "/" + 'Storage_StoragePowToEnergy.tab', ['Storage'])
    storENCapitalCost = load_param(tab_file_path + "/" + 'Storage_EnergyCapitalCost.tab', ['Storage', 'Period'])
    storENFixedOMCost = load_param(tab_file_path + "/" + 'Storage_EnergyFixedOMCost.tab', ['Storage', 'Period'])
    storENInitCap = load_param(tab_file_path + "/" + 'Storage_EnergyInitialCapacity.tab', ['Node', 'Storage', 'Period'])
    storENMaxBuiltCap = load_param(tab_file_path + "/" + 'Storage_EnergyMaxBuiltCapacity.tab', ['Node', 'Storage', 'Period'])
    storENMaxInstalledCapRaw = load_param(tab_file_path + "/" + 'Storage_EnergyMaxInstalledCapacity.tab', ['Node', 'Storage'])
    storOperationalInit = load_param(tab_file_path + "/" + 'Storage_StorageInitialEnergyLevel.tab', ['Storage'])
    storPWCapitalCost = load_param(tab_file_path + "/" + 'Storage_PowerCapitalCost.tab', ['Storage', 'Period'])
    storPWFixedOMCost = load_param(tab_file_path + "/" + 'Storage_PowerFixedOMCost.tab', ['Storage', 'Period'])
    storPWInitCap = load_param(tab_file_path + "/" + 'Storage_InitialPowerCapacity.tab', ['Node', 'Storage', 'Period'])
    storPWMaxBuiltCap = load_param(tab_file_path + "/" + 'Storage_PowerMaxBuiltCapacity.tab', ['Node', 'Storage', 'Period'])
    storPWMaxInstalledCapRaw = load_param(tab_file_path + "/" + 'Storage_PowerMaxInstalledCapacity.tab', ['Node', 'Storage'])
    storageLifetime = load_param(tab_file_path + "/" + 'Storage_Lifetime.tab', ['Storage'])

    # Node parameters
    nodeLostLoadCost = load_param(tab_file_path + "/" + 'Node_NodeLostLoadCost.tab', ['Node', 'Period'])
    sloadAnnualDemand = load_param(tab_file_path + "/" + 'Node_ElectricAnnualDemand.tab', ['Node', 'Period'])
    maxHydroNode = load_param(tab_file_path + "/" + 'Node_HydroGenMaxAnnualProduction.tab', ['Node'])

    # General parameters
    seasScale = load_param(tab_file_path + "/" + 'General_seasonScale.tab', ['Season'])

    if EMISSION_CAP:
        CO2cap = load_param(tab_file_path + "/" + 'General_CO2Cap.tab', ['Period'])
    else:
        CO2price = load_param(tab_file_path + "/" + 'General_CO2Price.tab', ['Period'])

    # Stochastic parameters
    if scenariogeneration:
        scenariopath = tab_file_path
    else:
        scenariopath = scenario_data_path

    maxRegHydroGenRaw = load_param(scenariopath + "/" + 'Stochastic_HydroGenMaxSeasonalProduction.tab',
                                   ['Node', 'Period', 'Season', 'Hour', 'Scenario'])
    genCapAvailStochRaw = load_param(scenariopath + "/" + 'Stochastic_StochasticAvailability.tab',
                                     ['Node', 'Generator', 'Operationalhour', 'Scenario', 'Period'])
    sloadRaw = load_param(scenariopath + "/" + 'Stochastic_ElectricLoadRaw.tab',
                          ['Node', 'Operationalhour', 'Scenario', 'Period'])

    if LOADCHANGEMODULE:
        sloadMod = load_param(scenariopath + "/" + 'LoadchangeModule/Stochastic_ElectricLoadMod.tab',
                              ['Node', 'Operationalhour', 'Scenario', 'Period'])

    # Scenario probabilities
    sceProbab = {s: 1 / len(ScenarioSet) for s in ScenarioSet}

    # Precompute Investment Costs
    # Generator investment cost
    genInvCost = {}
    CCSCostTSFix = 1149873.72  # Hard-coded value
    CCSRemFrac = 0.9  # Hard-coded value

    for g in Generator:
        genLifetime_g = float(genLifetime.get(g, 20))  # Default lifetime if not specified
        for i in PeriodSet:
            i_int = int(i)
            WACC_term = WACC / (1 - (1 + WACC) ** (-genLifetime_g))
            genCapitalCost_gi = float(genCapitalCost.get((g, i), 0.0))
            genFixedOMCost_gi = float(genFixedOMCost.get((g, i), 0.0))
            costperyear = WACC_term * genCapitalCost_gi + genFixedOMCost_gi
            num_periods_remaining = (len(PeriodSet) - i_int + 1) * LeapYearsInvestment
            min_lifetime = min(num_periods_remaining, genLifetime_g)
            costperperiod = costperyear * 1000 * (1 - (1 + discountrate) ** (-min_lifetime)) / (
                        1 - (1 / (1 + discountrate)))
            # Check if ('CCS', g) in GeneratorsOfTechnology
            if any(t == 'CCS' and gg == g for (t, gg) in GeneratorsOfTechnology):
                genCO2TypeFactor_g = float(genCO2TypeFactor.get(g, 0.0))
                genEfficiency_gi = float(genEfficiency.get((g, i), 1.0))
                costperperiod += CCSCostTSFix * CCSRemFrac * genCO2TypeFactor_g * (3.6 / genEfficiency_gi)
            genInvCost[g, i] = costperperiod

    # Storage investment cost
    storPWInvCost = {}
    storENInvCost = {}
    for b in Storage:
        storageLifetime_b = float(storageLifetime.get(b, 0.0))
        for i in PeriodSet:
            i_int = int(i)
            WACC_term = WACC / (1 - (1 + WACC) ** (-storageLifetime_b))
            storPWCapitalCost_bi = float(storPWCapitalCost.get((b, i), 0.0))
            storPWFixedOMCost_bi = float(storPWFixedOMCost.get((b, i), 0.0))
            costperyearPW = WACC_term * storPWCapitalCost_bi + storPWFixedOMCost_bi
            num_periods_remaining = (len(PeriodSet) - i_int + 1) * LeapYearsInvestment
            min_lifetime = min(num_periods_remaining, storageLifetime_b)
            costperperiodPW = costperyearPW * 1000 * (1 - (1 + discountrate) ** (-min_lifetime)) / (
                        1 - (1 / (1 + discountrate)))
            storPWInvCost[b, i] = costperperiodPW

            storENCapitalCost_bi = float(storENCapitalCost.get((b, i), 0.0))
            storENFixedOMCost_bi = float(storENFixedOMCost.get((b, i), 0.0))
            costperyearEN = WACC_term * storENCapitalCost_bi + storENFixedOMCost_bi
            costperperiodEN = costperyearEN * 1000 * (1 - (1 + discountrate) ** (-min_lifetime)) / (
                        1 - (1 / (1 + discountrate)))
            storENInvCost[b, i] = costperperiodEN

    # Transmission investment cost
    transmissionInvCost = {}
    for (n1, n2) in BidirectionalArc:
        transmissionLifetime_n1n2 = float(transmissionLifetime.get((n1, n2), 40.0))
        for i in PeriodSet:
            i_int = int(i)
            num_periods_remaining = (len(PeriodSet) - i_int + 1) * LeapYearsInvestment
            min_lifetime = min(num_periods_remaining, transmissionLifetime_n1n2)
            for t in TransmissionType:
                if (n1, n2, t) in TransmissionTypeOfDirectionalLink:
                    transmissionLength_n1n2 = float(transmissionLength.get((n1, n2), 0.0))
                    transmissionTypeCapitalCost_ti = float(transmissionTypeCapitalCost.get((t, i), 0.0))
                    transmissionTypeFixedOMCost_ti = float(transmissionTypeFixedOMCost.get((t, i), 0.0))
                    WACC_term = WACC / (1 - (1 + WACC) ** (-transmissionLifetime_n1n2))
                    costperyear = WACC_term * transmissionLength_n1n2 * transmissionTypeCapitalCost_ti + transmissionTypeFixedOMCost_ti
                    costperperiod = costperyear * (1 - (1 + discountrate) ** (-min_lifetime)) / (
                                1 - (1 / (1 + discountrate)))
                    transmissionInvCost[n1, n2, i] = costperperiod

    # Variables
    genInvCap = model.addVars(GeneratorsOfNode, PeriodSet, name='genInvCap', lb=0)
    transmisionInvCap = model.addVars(BidirectionalArc, PeriodSet, name='transmisionInvCap', lb=0)
    storPWInvCap = model.addVars(StoragesOfNode, PeriodSet, name='storPWInvCap', lb=0)
    storENInvCap = model.addVars(StoragesOfNode, PeriodSet, name='storENInvCap', lb=0)

    genInstalledCap = model.addVars(GeneratorsOfNode, PeriodSet, name='genInstalledCap', lb=0)
    transmissionInstalledCap = model.addVars(BidirectionalArc, PeriodSet, name='transmissionInstalledCap', lb=0)
    storPWInstalledCap = model.addVars(StoragesOfNode, PeriodSet, name='storPWInstalledCap', lb=0)
    storENInstalledCap = model.addVars(StoragesOfNode, PeriodSet, name='storENInstalledCap', lb=0)

    # Discount multiplier
    discount_multiplier = {}
    for period in PeriodSet:
        period_int = int(period)
        coeff = 1
        if period_int > 1:
            coeff = pow(1.0 + discountrate, -LeapYearsInvestment * (period_int - 1))
        discount_multiplier[period] = coeff

    # Objective Function
    obj_expr = quicksum(
        discount_multiplier[i] * (
            quicksum(genInvCost[g, i] * genInvCap[n, g, i] for (n, g) in GeneratorsOfNode if (g, i) in genInvCost) +
            quicksum(transmissionInvCost[n1, n2, i] * transmisionInvCap[n1, n2, i] for (n1, n2) in BidirectionalArc if (n1, n2, i) in transmissionInvCost) +
            quicksum(
                storPWInvCost[b, i] * storPWInvCap[n, b, i] + storENInvCost[b, i] * storENInvCap[n, b, i]
                for (n, b) in StoragesOfNode if (b, i) in storPWInvCost and (b, i) in storENInvCost
            )
        )
        for i in PeriodSet
    )
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints
    # Installed capacity definition for generators
    for (n, g) in GeneratorsOfNode:
        genLifetime_g = float(genLifetime.get(g, 20.0))
        for i in PeriodSet:
            i_int = int(i)
            startPeriod = 1
            computed_startPeriod = int(1 + i_int - (genLifetime_g / LeapYearsInvestment))
            if computed_startPeriod > startPeriod:
                startPeriod = computed_startPeriod
            model.addConstr(
                quicksum(
                    genInvCap[n, g, str(j)] for j in PeriodSet if int(j) >= startPeriod and int(j) <= i_int
                ) - genInstalledCap[n, g, i] + float(genInitCap.get((n, g, i), 0.0)) == 0,
                name=f"installedCapDefinitionGen_{n}_{g}_{i}"
            )

    # Installed capacity definition for storage energy
    for (n, b) in StoragesOfNode:
        storageLifetime_b = float(storageLifetime.get(b, 20.0))
        for i in PeriodSet:
            i_int = int(i)
            startPeriod = 1
            computed_startPeriod = int(1 + i_int - (storageLifetime_b / LeapYearsInvestment))
            if computed_startPeriod > startPeriod:
                startPeriod = computed_startPeriod
            model.addConstr(
                quicksum(
                    storENInvCap[n, b, str(j)] for j in PeriodSet if int(j) >= startPeriod and int(j) <= i_int
                ) - storENInstalledCap[n, b, i] + float(storENInitCap.get((n, b, i), 0.0)) == 0,
                name=f"installedCapDefinitionStorEN_{n}_{b}_{i}"
            )

    # Installed capacity definition for storage power
    for (n, b) in StoragesOfNode:
        storageLifetime_b = float(storageLifetime.get(b, 20.0))
        for i in PeriodSet:
            i_int = int(i)
            startPeriod = 1
            computed_startPeriod = int(1 + i_int - (storageLifetime_b / LeapYearsInvestment))
            if computed_startPeriod > startPeriod:
                startPeriod = computed_startPeriod
            model.addConstr(
                quicksum(
                    storPWInvCap[n, b, str(j)] for j in PeriodSet if int(j) >= startPeriod and int(j) <= i_int
                ) - storPWInstalledCap[n, b, i] + float(storPWInitCap.get((n, b, i), 0.0)) == 0,
                name=f"installedCapDefinitionStorPW_{n}_{b}_{i}"
            )

    # Installed capacity definition for transmission lines
    for (n1, n2) in BidirectionalArc:
        transmissionLifetime_n1n2 = float(transmissionLifetime.get((n1, n2), 40.0))
        for i in PeriodSet:
            i_int = int(i)
            startPeriod = 1
            computed_startPeriod = int(1 + i_int - (transmissionLifetime_n1n2 / LeapYearsInvestment))
            if computed_startPeriod > startPeriod:
                startPeriod = computed_startPeriod
            model.addConstr(
                quicksum(
                    transmisionInvCap[n1, n2, str(j)] for j in PeriodSet if int(j) >= startPeriod and int(j) <= i_int
                ) - transmissionInstalledCap[n1, n2, i] + float(transmissionInitCap.get((n1, n2, i), 0.0)) == 0,
                name=f"installedCapDefinitionTrans_{n1}_{n2}_{i}"
            )

    # Investment constraints for generator capacity
    for t in Technology:
        for n in Node:
            for i in PeriodSet:
                max_built_cap = float(genMaxBuiltCap.get((n, t, i), 0.0))
                model.addConstr(
                    quicksum(
                        genInvCap[n, g, i]
                        for g in Generator
                        if (n, g) in GeneratorsOfNode and (t, g) in GeneratorsOfTechnology
                    ) - max_built_cap <= 0,
                    name=f"investment_gen_cap_{t}_{n}_{i}"
                )

    # Investment constraints for transmission capacity
    for (n1, n2) in BidirectionalArc:
        for i in PeriodSet:
            max_built_cap = float(transmissionMaxBuiltCap.get((n1, n2, i), 0.0))
            model.addConstr(
                transmisionInvCap[n1, n2, i] - max_built_cap <= 0,
                name=f"investment_trans_cap_{n1}_{n2}_{i}"
            )

    # Investment constraints for storage power capacity
    for (n, b) in StoragesOfNode:
        for i in PeriodSet:
            max_built_cap = float(storPWMaxBuiltCap.get((n, b, i), 0.0))
            model.addConstr(
                storPWInvCap[n, b, i] - max_built_cap <= 0,
                name=f"investment_storage_power_cap_{n}_{b}_{i}"
            )

    # Investment constraints for storage energy capacity
    for (n, b) in StoragesOfNode:
        for i in PeriodSet:
            max_built_cap = float(storENMaxBuiltCap.get((n, b, i), 0.0))
            model.addConstr(
                storENInvCap[n, b, i] - max_built_cap <= 0,
                name=f"investment_storage_energy_cap_{n}_{b}_{i}"
            )

    # Installed capacity constraints for generators
    for t in Technology:
        for n in Node:
            for i in PeriodSet:
                max_installed_cap = float(genMaxInstalledCapRaw.get((n, t), 0.0))
                model.addConstr(
                    quicksum(
                        genInstalledCap[n, g, i]
                        for g in Generator
                        if (n, g) in GeneratorsOfNode and (t, g) in GeneratorsOfTechnology
                    ) - max_installed_cap <= 0,
                    name=f"installed_gen_cap_{t}_{n}_{i}"
                )

    # Installed capacity constraints for transmission
    for (n1, n2) in BidirectionalArc:
        for i in PeriodSet:
            max_installed_cap = float(transmissionMaxInstalledCapRaw.get((n1, n2), 0.0))
            model.addConstr(
                transmissionInstalledCap[n1, n2, i] - max_installed_cap <= 0,
                name=f"installed_trans_cap_{n1}_{n2}_{i}"
            )

    # Installed capacity constraints for storage power
    for (n, b) in StoragesOfNode:
        for i in PeriodSet:
            max_installed_cap = float(storPWMaxInstalledCapRaw.get((n, b), 0.0))
            model.addConstr(
                storPWInstalledCap[n, b, i] - max_installed_cap <= 0,
                name=f"installed_storage_power_cap_{n}_{b}_{i}"
            )

    # Installed capacity constraints for storage energy
    for (n, b) in StoragesOfNode:
        for i in PeriodSet:
            max_installed_cap = float(storENMaxInstalledCapRaw.get((n, b), 0.0))
            model.addConstr(
                storENInstalledCap[n, b, i] - max_installed_cap <= 0,
                name=f"installed_storage_energy_cap_{n}_{b}_{i}"
            )

    # Power to energy relationship for dependent storage
    for (n, b) in StoragesOfNode:
        if b in DependentStorage:
            for i in PeriodSet:
                pow_to_energy_ratio = float(storagePowToEnergy.get(b, 1.0))
                model.addConstr(
                    storPWInstalledCap[n, b, i] - pow_to_energy_ratio * storENInstalledCap[n, b, i] == 0,
                    name=f"power_energy_relate_{n}_{b}_{i}"
                )

    # Return the model instance
    return model