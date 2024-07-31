import os
from reader import generate_tab_files
import time
import pandas as pd
import numpy as np
import logging
from yaml import safe_load
from pyomo.environ import *
from datetime import datetime

start = time.time()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
temp_dir = UserRunTimeConfig["temp_dir"]
version = UserRunTimeConfig["version"]
Horizon = UserRunTimeConfig["Horizon"]
NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
scenariogeneration = UserRunTimeConfig["scenariogeneration"]
fix_sample = UserRunTimeConfig["fix_sample"]
filter_make = UserRunTimeConfig["filter_make"] 
filter_use = UserRunTimeConfig["filter_use"]
n_cluster = UserRunTimeConfig["n_cluster"]
moment_matching = UserRunTimeConfig["moment_matching"]
n_tree_compare = UserRunTimeConfig["n_tree_compare"]

#############################
##Non configurable settings##
#############################
NoOfRegSeason = 4
regular_seasons = ["winter", "spring", "summer", "fall"]
NoOfPeakSeason = 2
lengthPeakSeason = 7
LeapYearsInvestment = 5



# Generate name for this run
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

# Set up paths
workbook_path = 'Data handler/' + version
tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name
scenario_data_path = 'Data handler/' + version + '/ScenarioData'
result_file_path = 'Results/' + name

# Create necessary directories
os.makedirs(tab_file_path, exist_ok=True)
os.makedirs(result_file_path, exist_ok=True)

# Generate tab files
generate_tab_files(filepath=workbook_path, tab_file_path=tab_file_path)


def sample_model(tab_file_path):
    # Define the Pyomo model
    model = ConcreteModel()

    # DataPortal instance to load data
    data_portal = DataPortal()

    Period = list(range(1, int((2060 - 2020) / LeapYearsInvestment) + 1))

    #Supply technology sets
    model.Generator = Set(ordered=True) #g
    model.Technology = Set(ordered=True) #t
    model.Storage =  Set() #b

    #Temporal sets
    model.Period = Set(ordered=True) #max period
    model.PeriodActive = Set(ordered=True, initialize=Period) #i

    #Spatial sets
    model.Node = Set(ordered=True) #n
    model.OffshoreNode = Set(ordered=True, within=model.Node) #n
    model.DirectionalLink = Set(dimen=2, within=model.Node*model.Node, ordered=True) #a
    model.TransmissionType = Set(ordered=True)

    #Subsets
    model.GeneratorsOfTechnology=Set(dimen=2) #(t,g) for all t in T, g in G_t
    model.GeneratorsOfNode = Set(dimen=2) #(n,g) for all n in N, g in G_n
    model.TransmissionTypeOfDirectionalLink = Set(dimen=3) #(n1,n2,t) for all (n1,n2) in L, t in T
    model.StoragesOfNode = Set(dimen=2) #(n,b) for all n in N, b in B_n


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


    # Define sets based on the loaded data
    data_portal.load(filename=os.path.join(tab_file_path, "Sets_Generator.tab"), set=model.Generator)
    data_portal.load(filename=os.path.join(tab_file_path, "Sets_Node.tab"), set=model.Node)
    data_portal.load(filename=os.path.join(tab_file_path, "Sets_Storage.tab"), set=model.Storage)
    data_portal.load(filename=os.path.join(tab_file_path, "Sets_Technology.tab"), set=model.Technology)
    data_portal.load(filename=os.path.join(tab_file_path, "Sets_DirectionalLines.tab"), set=model.DirectionalLink)
    data_portal.load(filename=os.path.join(tab_file_path, "Sets_GeneratorsOfNode.tab"), set=model.GeneratorsOfNode)
    data_portal.load(filename=os.path.join(tab_file_path, "Sets_StorageOfNodes.tab"), set=model.StoragesOfNode)
    data_portal.load(filename=os.path.join(tab_file_path, "Sets_GeneratorsOfTechnology.tab"), set=model.GeneratorsOfTechnology)


    print("Sets are ready!")

    # Initialize any additional sets or parameters if needed
    model.LeapYearsInvestment = Param(initialize=LeapYearsInvestment)
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
    model.genLifetime = Param(model.Generator, default=0.0, mutable=True)
    model.transmissionLifetime = Param(model.BidirectionalArc, default=40.0, mutable=True)
    model.storageLifetime = Param(model.Storage, default=0.0, mutable=True)

    # Define parameters with loaded data
    data_portal.load(filename=os.path.join(tab_file_path, "Generator_MaxBuiltCapacity.tab"), param=model.genMaxBuiltCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Transmission_MaxBuiltCapacity.tab"), param=model.transmissionMaxBuiltCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Storage_PowerMaxBuiltCapacity.tab"), param=model.storPWMaxBuiltCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Storage_EnergyMaxBuiltCapacity.tab"), param=model.storENMaxBuiltCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Generator_MaxInstalledCapacity.tab"), param=model.genMaxInstalledCapRaw, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Transmission_MaxInstallCapacityRaw.tab"), param=model.transmissionMaxInstalledCapRaw, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Storage_PowerMaxInstalledCapacity.tab"), param=model.storPWMaxInstalledCapRaw, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Storage_EnergyMaxInstalledCapacity.tab"), param=model.storENMaxInstalledCapRaw, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Generator_InitialCapacity.tab"), param=model.genInitCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Transmission_InitialCapacity.tab"), param=model.transmissionInitCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Storage_InitialPowerCapacity.tab"), param=model.storPWInitCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Storage_EnergyInitialCapacity.tab"), param=model.storENInitCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Generator_Lifetime.tab"), param=model.genLifetime, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Storage_Lifetime.tab"), param=model.storageLifetime, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, "Transmission_Lifetime.tab"), param=model.transmissionLifetime, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, 'Generator_RefInitialCap.tab'), param=model.genRefInitCap, format="table")
    data_portal.load(filename=os.path.join(tab_file_path, 'Generator_ScaleFactorInitialCap.tab'), param=model.genScaleInitCap, format="table")

    print("Parameters are ready!")


    # Define variables, constraints, and other model components as needed
    model.genInvCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmisionInvCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.genInstalledCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmissionInstalledCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInstalledCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInstalledCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)


    print("Variables are ready!")


    ##############
    ####RULE######
    ##############

    def prepInitialCapacityNodeGen_rule(model):
        #Build initial capacity for generator type in node

        for (n,g) in model.GeneratorsOfNode:
            for i in model.PeriodActive:
                if value(model.genInitCap[n,g,i]) == 0:
                    model.genInitCap[n,g,i] = model.genRefInitCap[n,g]*(1-model.genScaleInitCap[g,i])

    model.build_InitialCapacityNodeGen = BuildAction(rule=prepInitialCapacityNodeGen_rule)

    def prepInitialCapacityTransmission_rule(model):
        #Build initial capacity for transmission lines to ensure initial capacity is the upper installation bound if infeasible

        for (n1,n2) in model.BidirectionalArc:
            for i in model.PeriodActive:
                if value(model.transmissionMaxInstalledCapRaw[n1,n2,i]) <= value(model.transmissionInitCap[n1,n2,i]):
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionInitCap[n1,n2,i]
                else:
                    model.transmissionMaxInstalledCap[n1,n2,i] = model.transmissionMaxInstalledCapRaw[n1,n2,i]

    model.build_InitialCapacityTransmission = BuildAction(rule=prepInitialCapacityTransmission_rule)

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


    print("Preprocessing all done!")

    ###############
    ##CONSTRAINTS##
    ###############


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

    return model


def validate_sample(model, sample):
    # Reset all investment variables
    for var in [model.genInvCap, model.storENInvCap, model.storPWInvCap, model.transmisionInvCap]:
        for key in var.keys():
            var[key].set_value(0)

    # Set values from the sample
    for _, row in sample.iterrows():
        if row['Type'] == 'Generation':
            model.genInvCap[row['Country'], row['Energy_Type'], row['Period']].set_value(row['Value'])
        elif row['Type'] == 'Storage Energy':
            model.storENInvCap[row['Country'], row['Energy_Type'], row['Period']].set_value(row['Value'])
        elif row['Type'] == 'Storage Power':
            model.storPWInvCap[row['Country'], row['Energy_Type'], row['Period']].set_value(row['Value'])
        elif row['Type'] == 'Transmission':
            model.transmisionInvCap[row['Country'], row['Energy_Type'], row['Period']].set_value(row['Value'])

    # Check constraints
    for const in model.component_objects(Constraint):
        for index in const:
            body_value = value(const[index].body())
            lower_value = value(const[index].lower) if const[index].lower is not None else None
            upper_value = value(const[index].upper) if const[index].upper is not None else None
            if (lower_value is not None and body_value < lower_value - 1e-6) or \
               (upper_value is not None and body_value > upper_value + 1e-6):
                logging.debug(f"Constraint {const.name}[{index}] violated: {body_value} not in [{lower_value}, {upper_value}]")
                return False
            if abs(value(const[index].body()) - value(const[index].upper)) > 1e-6 and \
               abs(value(const[index].body()) - value(const[index].lower)) > 1e-6:
                logging.info(f"Constraint {const.name}[{index}] violated: {const[index].body()} not in [{const[index].lower}, {const[index].upper}]")
                return False
    return True


def sample_generation(model, num_samples=1):
    print("Sample Generating!")
    samples = []
    for sample_num in range(num_samples):
        try:
            sample = []
            # Sample values for genInvCap
            for n, g in model.GeneratorsOfNode:
                print(n)
                print(g)
                for i in model.PeriodActive:
                    print(i)
                    max_cap = max(value(model.genMaxBuiltCap[n, t, i]) 
                                  for t in model.Technology if (t, g) in model.GeneratorsOfTechnology)
                    sample_value = np.random.uniform(0, max_cap)
                    sample.append({'Country': n, 'Energy_Type': g, 'Period': i, 'Type': 'Generation', 'Value': sample_value})

            # Sample values for storage
            for n, b in model.StoragesOfNode:
                for i in model.PeriodActive:
                    en_sample_value = np.random.uniform(0, value(model.storENMaxBuiltCap[n, b, i]))
                    pw_sample_value = np.random.uniform(0, value(model.storPWMaxBuiltCap[n, b, i]))
                    sample.append({'Country': n, 'Energy_Type': b, 'Period': i, 'Type': 'Storage Energy', 'Value': en_sample_value})
                    sample.append({'Country': n, 'Energy_Type': b, 'Period': i, 'Type': 'Storage Power', 'Value': pw_sample_value})

            # Sample values for transmission
            for n1, n2 in model.BidirectionalArc:
                for i in model.PeriodActive:
                    sample_value = np.random.uniform(0, value(model.transmissionMaxBuiltCap[n1, n2, i]))
                    sample.append({'Country': n1, 'Energy_Type': n2, 'Period': i, 'Type': 'Transmission', 'Value': sample_value})

            samples.append(pd.DataFrame(sample))
            print(f"Sample {sample_num + 1} generated successfully")
        except Exception as e:
            print(f"Error generating sample {sample_num + 1}: {str(e)}")

    return samples

if __name__ == "__main__":
    num_samples = 10
    max_attempts = 100  
    model = sample_model(tab_file_path)
    
    valid_samples = []
    attempts = 0

    while len(valid_samples) < num_samples and attempts < max_attempts:
        samples = sample_generation(model, 1) 

        if samples and not samples[0].empty:
            sample = samples[0]
            logging.info(f"Validating sample {attempts + 1}")
            if validate_sample(model, sample):
                valid_samples.append(sample)
                logging.info(f"Sample {attempts + 1} is valid")
            else:
                logging.info(f"Sample {attempts + 1} is invalid")
        else:
            logging.warning(f"Failed to generate sample on attempt {attempts + 1}")
        
        attempts += 1

    if valid_samples:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"valid_samples_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        pd.concat(valid_samples).to_csv(output_path, index=False)
        logging.info(f"Saved {len(valid_samples)} valid samples out of {attempts} attempts to {output_path}")
    else:
        logging.warning(f"No valid samples found after {attempts} attempts. Check constraints and sampling ranges.")

    for const in model.component_objects(Constraint):
        for index in const:
            lower = value(const[index].lower) if const[index].lower is not None else "-inf"
            upper = value(const[index].upper) if const[index].upper is not None else "inf"
            logging.debug(f"Constraint {const.name}[{index}] range: [{lower}, {upper}]")