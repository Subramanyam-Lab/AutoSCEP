import os
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
NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
scenariogeneration = UserRunTimeConfig["scenariogeneration"]
n_cluster = UserRunTimeConfig["n_cluster"]
moment_matching = UserRunTimeConfig["moment_matching"]
n_tree_compare = UserRunTimeConfig["n_tree_compare"]

#############################
##Non configurable settings##
#############################
LeapYearsInvestment = 5



def sample_model(tab_file_path):

    print("Loading complete. Model setup starting...")
    # Define the Pyomo model

    model = AbstractModel()

    Period = [i + 1 for i in range(int((2060-2020)/LeapYearsInvestment))]

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

    data = DataPortal()
    data.load(filename=tab_file_path + "/" + 'Sets_Generator.tab',format="set", set=model.Generator)
    data.load(filename=tab_file_path + "/" + 'Sets_Storage.tab',format="set", set=model.Storage)
    data.load(filename=tab_file_path + "/" + 'Sets_Technology.tab',format="set", set=model.Technology)
    data.load(filename=tab_file_path + "/" + 'Sets_Node.tab',format="set", set=model.Node)
    data.load(filename=tab_file_path + "/" + 'Sets_Horizon.tab',format="set", set=model.Period)
    data.load(filename=tab_file_path + "/" + 'Sets_DirectionalLines.tab',format="set", set=model.DirectionalLink)
    data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfTechnology.tab',format="set", set=model.GeneratorsOfTechnology)
    data.load(filename=tab_file_path + "/" + 'Sets_GeneratorsOfNode.tab',format="set", set=model.GeneratorsOfNode)
    data.load(filename=tab_file_path + "/" + 'Sets_StorageOfNodes.tab',format="set", set=model.StoragesOfNode)
    data.load(filename=tab_file_path + "/" + 'Sets_LineType.tab',format="set", set=model.TransmissionType)
    data.load(filename=tab_file_path + "/" + 'Sets_LineTypeOfDirectionalLines.tab',format="set", set=model.TransmissionTypeOfDirectionalLink)


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

    # Load parameters
    data.load(filename=tab_file_path + "/" + "Generator_MaxBuiltCapacity.tab", param=model.genMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + "Transmission_MaxBuiltCapacity.tab", param=model.transmissionMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + "Storage_PowerMaxBuiltCapacity.tab", param=model.storPWMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + "Storage_EnergyMaxBuiltCapacity.tab", param=model.storENMaxBuiltCap, format="table")
    data.load(filename=tab_file_path + "/" + "Generator_MaxInstalledCapacity.tab", param=model.genMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + "Transmission_MaxInstallCapacityRaw.tab", param=model.transmissionMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + "Storage_PowerMaxInstalledCapacity.tab", param=model.storPWMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + "Storage_EnergyMaxInstalledCapacity.tab", param=model.storENMaxInstalledCapRaw, format="table")
    data.load(filename=tab_file_path + "/" + "Generator_InitialCapacity.tab", param=model.genInitCap, format="table")
    data.load(filename=tab_file_path + "/" + "Transmission_InitialCapacity.tab", param=model.transmissionInitCap, format="table")
    data.load(filename=tab_file_path + "/" + "Storage_InitialPowerCapacity.tab", param=model.storPWInitCap, format="table")
    data.load(filename=tab_file_path + "/" + "Storage_EnergyInitialCapacity.tab", param=model.storENInitCap, format="table")
    data.load(filename=tab_file_path + "/" + "Generator_Lifetime.tab", param=model.genLifetime, format="table")
    data.load(filename=tab_file_path + "/" + "Storage_Lifetime.tab", param=model.storageLifetime, format="table")
    data.load(filename=tab_file_path + "/" + "Transmission_Lifetime.tab", param=model.transmissionLifetime, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_RefInitialCap.tab', param=model.genRefInitCap, format="table")
    data.load(filename=tab_file_path + "/" + 'Generator_ScaleFactorInitialCap.tab', param=model.genScaleInitCap, format="table")

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

    return model, data




def load_multiple_best_samples(directory_path):
    all_samples = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            samples = pd.read_csv(file_path)
            all_samples.append(samples)
    return all_samples

def calculate_statistics(samples_df):
    # Calculate mean and std dev for each type by Country, Energy_Type, and Period
    statistics_df = samples_df.groupby(['Country', 'Energy_Type', 'Period', 'Type']).agg(['mean', 'std']).reset_index()
    statistics_df.columns = ['Country', 'Energy_Type', 'Period', 'Type', 'Mean', 'Std']
    
    return statistics_df

def validate_sample(instance, sample):
    # Reset all investment variables
    for var in [instance.genInvCap, instance.storENInvCap, instance.storPWInvCap, instance.transmisionInvCap]:
        for key in var.keys():
            var[key].set_value(0)

    # Set values from the sample
    for _, row in sample.iterrows():
        if row['Type'] == 'Generation':
            instance.genInvCap[row['Country'], row['Energy_Type'], row['Period']].set_value(row['Value'])
        elif row['Type'] == 'Transmission':
            instance.transmisionInvCap[row['Country'], row['Energy_Type'], row['Period']].set_value(row['Value'])
        elif row['Type'] == 'Storage Energy':
            instance.storENInvCap[row['Country'], row['Energy_Type'], row['Period']].set_value(row['Value'])
        elif row['Type'] == 'Storage Power':
            instance.storPWInvCap[row['Country'], row['Energy_Type'], row['Period']].set_value(row['Value'])
        
    # Initialize installed capacities
    update_installed_capacities(instance)
    
    violated_constraints = check_constraints(instance)
    return violated_constraints

def update_installed_capacities(instance):
    # Initialize genInstalledCap
    for (n, g) in instance.GeneratorsOfNode:
        for i in instance.PeriodActive:
            startPeriod = max(1, value(1 + i - (instance.genLifetime[g] / instance.LeapYearsInvestment)))
            instance.genInstalledCap[n, g, i].set_value(
                value(instance.genInitCap[n, g, i]) +
                sum(instance.genInvCap[n, g, j].value for j in instance.PeriodActive if startPeriod <= j <= i)
            )

    # Update transmission installed capacities
    for n1, n2 in instance.BidirectionalArc:
        for i in instance.PeriodActive:
            startPeriod = max(1, value(1 + i - (instance.transmissionLifetime[n1,n2] / instance.LeapYearsInvestment)))
            instance.transmissionInstalledCap[n1,n2,i].set_value(
                value(instance.transmissionInitCap[n1,n2,i] +
                sum(instance.transmisionInvCap[n1,n2,j].value for j in instance.PeriodActive if startPeriod <= j <= i))
            )

    # Update storage installed capacities
    for n, b in instance.StoragesOfNode:
        for i in instance.PeriodActive:
            startPeriod = max(1, value(1 + i - (instance.storageLifetime[b] / instance.LeapYearsInvestment)))
            instance.storENInstalledCap[n, b, i].set_value(
                value(instance.storENInitCap[n, b, i]) +
                sum(instance.storENInvCap[n, b, j].value for j in instance.PeriodActive if startPeriod <= j <= i)
            )
            instance.storPWInstalledCap[n, b, i].set_value(
                value(instance.storPWInitCap[n, b, i]) +
                sum(instance.storPWInvCap[n, b, j].value for j in instance.PeriodActive if startPeriod <= j <= i)
            )

def check_constraints(instance):
    violated_constraints = []
    for const in instance.component_objects(Constraint):
        for index in const:
            try:
                body_value = value(const[index].body())
                lower_value = value(const[index].lower) if const[index].lower is not None else None
                upper_value = value(const[index].upper) if const[index].upper is not None else None
                
                if lower_value is not None and body_value < lower_value - 1e-6:
                    violated_constraints.append((const.name, index, 'lower', body_value, lower_value))
                if upper_value is not None and body_value > upper_value + 1e-6:
                    violated_constraints.append((const.name, index, 'upper', body_value, upper_value))
            except ValueError as e:
                logging.info(f"Error evaluating constraint {const.name}[{index}]: {str(e)}")
    return violated_constraints


def resample_violated_constraints(instance, sample, violated_constraints):
    for const_name, index, bound_type, body_value, bound_value in violated_constraints:
        if const_name == 'investment_gen_cap':
            t, n, i = index
            related_samples = sample[(sample['Country'] == n) & 
                                     (sample['Period'] == i) & 
                                     (sample['Type'] == 'Generation')]
            current_sum = related_samples['Value'].sum()
            max_cap = value(instance.genMaxBuiltCap[n, t, i])
            
            if current_sum > max_cap:  # If sum is too high
                reduction_factor = max_cap / current_sum
                for idx in related_samples.index:
                    sample.at[idx, 'Value'] *= reduction_factor

        elif const_name == 'investment_trans_cap':
            n1, n2, i = index
            max_cap = value(instance.transmissionMaxBuiltCap[n1, n2, i])
            new_value = np.random.uniform(0, max_cap)
            sample.loc[(sample['Country'] == n1) & (sample['Energy_Type'] == n2) & 
                       (sample['Period'] == i) & (sample['Type'] == 'Transmission'), 'Value'] = new_value

        elif const_name in ['investment_storage_power_cap', 'investment_storage_energy_cap']:
            n, b, i = index
            if const_name == 'investment_storage_power_cap':
                max_cap = value(instance.storPWMaxBuiltCap[n, b, i])
                sample_type = 'Storage Power'
            else:
                max_cap = value(instance.storENMaxBuiltCap[n, b, i])
                sample_type = 'Storage Energy'
            new_value = np.random.uniform(0, max_cap)
            sample.loc[(sample['Country'] == n) & (sample['Energy_Type'] == b) & 
                       (sample['Period'] == i) & (sample['Type'] == sample_type), 'Value'] = new_value
    
    return sample


def sample_generation(instance, num_samples=1, max_iterations=1000):
    print("Sample Generating!")
    samples = []

    for sample_num in range(num_samples):
        try:
            sample = []
            # Generate initial sample
            for n, g in instance.GeneratorsOfNode:
                for i in instance.PeriodActive:
                    max_cap = max(value(instance.genMaxBuiltCap[n, t, i]) 
                                  for t in instance.Technology if (t, g) in instance.GeneratorsOfTechnology)
                    sample_value = np.random.uniform(0, max_cap) if np.random.rand() > 0.99 else 0
                    sample.append({'Country': n, 'Energy_Type': g, 'Period': i, 'Type': 'Generation', 'Value': sample_value})

            for n1, n2 in instance.BidirectionalArc:
                for i in instance.PeriodActive:
                    sample_value = np.random.uniform(0, value(instance.transmissionMaxBuiltCap[n1, n2, i])) 
                    sample.append({'Country': n1, 'Energy_Type': n2, 'Period': i, 'Type': 'Transmission', 'Value': sample_value})

            for n, b in instance.StoragesOfNode:
                for i in instance.PeriodActive:
                    en_sample_value = np.random.uniform(0, value(instance.storENMaxBuiltCap[n, b, i])) 
                    pw_sample_value = np.random.uniform(0, value(instance.storPWMaxBuiltCap[n, b, i]))
                    sample.append({'Country': n, 'Energy_Type': b, 'Period': i, 'Type': 'Storage Energy', 'Value': en_sample_value})
                    sample.append({'Country': n, 'Energy_Type': b, 'Period': i, 'Type': 'Storage Power', 'Value': pw_sample_value})

            sample_df = pd.DataFrame(sample)
            
            # Validate and resample if necessary
            for iteration in range(max_iterations):
                violated_constraints = validate_sample(instance, sample_df)
                if not violated_constraints:
                    break
                sample_df = resample_violated_constraints(instance, sample_df, violated_constraints)
                print(f"Sample {sample_num + 1}, Iteration {iteration + 1}: Resampled {len(violated_constraints)} violated constraints")
            
            if not violated_constraints:
                samples.append(sample_df)
                print(f"Sample {sample_num + 1} generated successfully after {iteration + 1} iterations")
            else:
                print(f"Failed to generate valid sample {sample_num + 1} after {max_iterations} iterations")
        
        except Exception as e:
            print(f"Error generating sample {sample_num + 1}: {str(e)}")

    return samples

def generate_samples_from_statistics(instance, statistics_df):
    sample = []
    
    for _, row in statistics_df.iterrows():
        mean = row['Mean']
        std = row['Std']
        # Generate a sample value from a normal distribution centered at the mean with the given std dev
        sampled_value = max(0,np.random.normal(mean, std))
        # Append the generated sample to the sample list
        sample.append({
            'Country': row['Country'],
            'Energy_Type': row['Energy_Type'],
            'Period': row['Period'],
            'Type': row['Type'],
            'Value': sampled_value
        })
    
    # Convert list of dicts to DataFrame
    sample_df = pd.DataFrame(sample)
    # Validate and apply sample if necessary
    if validate_sample(instance, sample_df):
        new_samples = [sample_df]
    
    return new_samples


# Pass to the dataset building
def new_samples(instance,num_samples):
    best_samples_directory = 'FSD'
    best_samples = load_multiple_best_samples(best_samples_directory)

    combined_samples = pd.concat(best_samples, ignore_index=True)
    statistics = calculate_statistics(combined_samples)
    print("Model setup complete. Starting sample generation...")
    valid_samples = generate_samples_from_statistics(instance, statistics)
    return valid_samples

if __name__ == "__main__":
    tab_file_path = 'Data handler/sampling'
    num_samples = 1000
    max_attempts = 10000
    model, data = sample_model(tab_file_path)
    instance = model.create_instance(data)

    best_samples_directory = 'FSD'
    best_samples = load_multiple_best_samples(best_samples_directory)

    combined_samples = pd.concat(best_samples, ignore_index=True)
    statistics = calculate_statistics(combined_samples)
    print("Model setup complete. Starting sample generation...")

    # valid_samples = sample_generation(instance, num_samples)
    for i in range(num_samples):
        valid_samples = generate_samples_from_statistics(instance, statistics)

        if valid_samples:
            output_dir = "FSDsamples"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"valid_samples_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.csv")
            pd.concat(valid_samples).to_csv(output_path, index=False)
            logging.info(f"Saved {len(valid_samples)} valid samples to {output_path}")
        else:
            logging.warning(f"No valid samples found after {max_attempts} attempts. Check constraints and sampling ranges.")

        for const in instance.component_objects(Constraint):
            for index in const:
                lower = value(const[index].lower) if const[index].lower is not None else "-inf"
                upper = value(const[index].upper) if const[index].upper is not None else "inf"
                logging.debug(f"Constraint {const.name}[{index}] range: [{lower}, {upper}]")