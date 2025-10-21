from empire_model import run_empire
import os 
import re
import argparse
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ph import PH
from mpisppy.opt.lshaped import LShapedMethod
import sys
import time
import pandas as pd
import fcntl
from scenario_generator import scenario_generator

import warnings
warnings.filterwarnings("ignore")


# Import fcntl for file locking on Unix-based systems
try:
    import fcntl
except ImportError:
    fcntl = None # Set to None on non-Unix systems like Windows


def gather_xbars_from_ph(ph_object):
    if ph_object.cylinder_rank != 0:
        return None
    
    random_sname = ph_object.local_scenario_names[0]
    scenario = ph_object.local_scenarios[random_sname]
    
    xbar_values = {}
    for node in scenario._mpisppy_node_list:
        for i, var_data in enumerate(node.nonant_vardata_list):
            var_name = var_data.name
            # .get((node.name, i)) 대신 [(node.name, i)]를 사용합니다.
            xbar_param = scenario._mpisppy_model.xbars[(node.name, i)]
            xbar_values[var_name] = xbar_param.value
            
    return xbar_values



def scenario_creator(scenario_name,seed,num_sce):
    match = re.fullmatch(r'scenario(\d+)', scenario_name)
    if not match:
        raise ValueError(f"Unrecognized scenario name: {scenario_name}")
    idx = int(match.group(1))
    print(f"scenario_name: {scenario_name}, idx: {idx}")
    _common_scenariopath = scenario_generator(seed,idx,num_sce)
    scenariopath = os.path.join(_common_scenariopath, scenario_name)
    model, data = run_empire(scenariopath)
    instance = model.create_instance(data)
    print(f"model instance created for {scenario_name}, idx: {idx}")
    sputils.attach_root_node(instance, instance.investcost, [instance.genInvCap, instance.transmisionInvCap,instance.storPWInvCap, instance.storENInvCap])
    # model._mpisppy_probability = 1.0 / 5, ######## if don't set assumign equal..
    return instance



def save_investment_solution(solution_dict, method_name, seed, num_sce,time_limit):
    if not solution_dict:
        print("Could not retrieve solution or solution is empty.")
        return

    print("Processing solution and saving to CSV...")
    
    # Regex for standard investments: Gen, Storage Power, Storage Energy
    pattern_standard = re.compile(
        r'^(?P<Type>\w+)\[(?P<Node>[^,]+),(?P<Energy_Type>[^,]+),(?P<Period>\d+)\]$'
    )
    # Regex for transmission investments
    pattern_transmission = re.compile(
        r'^(?P<Type>\w+)\[(?P<Node1>[^,]+),(?P<Node2>[^,]+),(?P<Period>\d+)\]$'
    )

    # Map internal variable names to more descriptive labels
    type_mapping = {
        "genInvCap": "Generation",
        "storPWInvCap": "Storage Power",
        "storENInvCap": "Storage Energy",
        "transmisionInvCap": "Transmission"
    }

    records = []
    for var_name, var_val in solution_dict.items():
        # Use a small tolerance to filter out negligible investment values
        # if var_val > 1e-6:
        m_std = pattern_standard.match(var_name)
        m_trans = pattern_transmission.match(var_name)

        if m_std:
            rec = m_std.groupdict()
            rec['Type'] = type_mapping.get(rec['Type'], rec['Type'])
            rec['Value'] = var_val
            records.append(rec)
        elif m_trans:
            rec = m_trans.groupdict()
            rec['Node'] = f"{rec.pop('Node1')}-{rec.pop('Node2')}"
            rec['Energy_Type'] = 'Electricity'
            rec['Type'] = type_mapping.get(rec['Type'], rec['Type'])
            rec['Value'] = var_val
            records.append(rec)
        else:
            print(f"Warning: Variable name did not match any pattern: {var_name}")

    if not records:
        print("No non-zero investment variables found in the solution.")
    else:
        df = pd.DataFrame(records, columns=['Node', 'Energy_Type', 'Period', 'Type', 'Value'])
        
        output_dir = 'sol_sets'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'full_{method_name}_solution_{num_sce}_{seed}_{time_limit}.csv')
        df.to_csv(output_path, index=False)
        print(f"\nSolution saved to {output_path}")


############### progressive hedging ##################

def Progressive_Hedging(options, scenario_names, seed, num_sce, time_limit):

    print("!!Solving with Progressive Hedging!!")
    start_time = time.time()
    ph_kwargs = {"seed": seed, "num_sce": num_sce}
    ph = PH(options, scenario_names, scenario_creator, scenario_creator_kwargs=ph_kwargs)
    conv, obj, bnd = ph.ph_main()
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"PH terminated. conv: {conv}, obj: {obj}, bnd: {bnd}")

    # All post-processing, logging, and saving happens on rank 0
    if ph.cylinder_rank == 0:
        print(f"\nPH run took: {elapsed:.2f} (sec)")
        best_inner_bound = bnd
        best_outer_bound = obj
        print(f"Best Inner (Lower) Bound: {best_inner_bound:.4f}")
        print(f"Best Outer (Upper) Bound: {best_outer_bound:.4f}")

        # Log runtime and bounds to a CSV file with file locking
        log_file = "runtime_full_ph_log.csv"
        need_header = not os.path.exists(log_file)
        
        try:
            with open(log_file, "a") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                if need_header:
                    lf.write("seed,num_of_sce,time_limit,lower_bound,upper_bound\n")
                lf.write(f"{seed},{num_sce},{time_limit},{best_inner_bound},{best_outer_bound}\n")
                fcntl.flock(lf, fcntl.LOCK_UN)
        except IOError as e:
            print(f"Could not write to log file {log_file}. Error: {e}")

    print("\nExtracting final consensus solution (xbars).")
    soln = gather_xbars_from_ph(ph)
    save_investment_solution(soln, "ph", seed, num_sce, time_limit)

    print("!!End: Progressive Hedging!!")



############### Benders Decomposition ##################

def BendersDecomposition(options, scenario_names, seed, num_sce):

    print("!!Solving with Benders Decomposition!!")
    start_time = time.time()
    bd_kwargs = {"seed": seed, "num_sce": num_sce}
    ls = LShapedMethod(options, scenario_names, scenario_creator, scenario_creator_kwargs=bd_kwargs)
    result = ls.lshaped_algorithm()
    end_time = time.time()
    elapsed = end_time - start_time

    # All post-processing, logging, and saving happens on rank 0
    if ls.cylinder_rank == 0:
        print(f"\nBenders run took: {elapsed:.2f} (sec)")
        
        # Extract results
        best_bound = result.get('best_bound', 'N/A')
        best_incumbent = result.get('best_incumbent', 'N/A')
        
        print(f"Best Lower Bound: {best_bound}")
        print(f"Best Upper Bound (Incumbent): {best_incumbent}")

        # Log runtime and bounds to a CSV file with file locking
        log_file = "runtime_bd_log.csv"
        need_header = not os.path.exists(log_file)
        try:
            with open(log_file, "a") as lf:
                if fcntl:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                if need_header:
                    lf.write("seed,num_of_sce,runtime,lower_bound,upper_bound\n")
                lf.write(f"{seed},{num_sce},{elapsed:.6f},{best_bound},{best_incumbent}\n")
                if fcntl:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except IOError as e:
            print(f"Could not write to log file {log_file}. Error: {e}")

        print("\nExtracting final first-stage solution.")
        variables = ls.gather_var_values_to_rank0()
        
        if variables:
            first_stage_vars = {}
            # First-stage variables are consistent across scenarios, so we only need one.
            scen_name_to_check = scenario_names[0]
            for (s_name, var_name), val in variables.items():
                if s_name == scen_name_to_check:
                    # Identify first-stage variables by their name prefixes
                    if var_name.startswith(('genInvCap', 'transmisionInvCap', 'storPWInvCap', 'storENInvCap')):
                        first_stage_vars[var_name] = val
            
            save_investment_solution(first_stage_vars, "bd", seed, num_sce)
        else:
            print("Could not retrieve solution from Benders object.")

    print("!!End: Benders Decomposition!!")
    


if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    peak_memory_mb = 0

    parser = argparse.ArgumentParser(description="EMPIRE model run")
    parser.add_argument("--seed",    "-s", type=int, default=1, help="random seed")
    parser.add_argument("--num-sce", "-n", type=int, default=5, help="number of scenarios")
    parser.add_argument("--method",  "-m", type=str, default="PH", choices=["PH", "BD"], 
                        help="Solution method: 'PH' for Progressive Hedging or 'BD' for Benders Decomposition.")
    parser.add_argument("--time",    "-t", type=int, default=60, help="time limit")
    args = parser.parse_args()

    # assign the global variables
    SEED    = args.seed
    num_sce = args.num_sce
    method = args.method
    time_limit = args.time

    scenario_names = [f"scenario{i}" for i in range(1, num_sce+1)]
    print(f"num_sce: {num_sce}")


    if method == "PH":
        convthresh = 1e-3
        print(f"convthresh : {convthresh}")
        
        options_PH = {
            "solver_name": "gurobi",
            "PHIterLimit": 1000,
            "defaultPHrho": 1,
            "time_limit": time_limit,   
            "convthresh": convthresh,
            "verbose": False,
            "display_progress": True,
            "display_timing": False,
            "iter0_solver_options": {"Method": 1, "threads": 4, "MIPGap": 1e-2},
            "iterk_solver_options": {"Method": 1, "threads": 4, "MIPGap": 1e-2},   
        }


        Progressive_Hedging(options_PH,scenario_names,SEED,num_sce,time_limit)
    
    elif method == "BD":
        options_BD = {
            "root_solver": "gurobi_persistent",
            "sp_solver": "gurobi_persistent",
            "sp_solver_options": {
                "Threads": 4,
                "MIPGap": 1e-2,
            },
            "root_solver_options": {
                "Threads": 4,
                "MIPGap": 1e-2,
            },
            "store_subproblems": True,
            "valid_eta_lb": {name: 0.0 for name in scenario_names},  
            "max_iter": 1000,
            "rel_gap": 0.01,
            "abs_gap": 0.0,
            "max_stalled_iters": 20,
            "verbose": True,
            "display_progress": True,
        }
        BendersDecomposition(options_BD, scenario_names,SEED,num_sce)
    
    else:
        print(f"Error: Unknown method '{method}'. Please choose 'PH' or 'BD'.")
        sys.exit(1)
