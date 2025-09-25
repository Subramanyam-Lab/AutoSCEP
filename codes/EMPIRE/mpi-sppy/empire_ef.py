from __future__ import division
from pyomo.environ import *
import os
import pandas as pd
from scenario_generator import scenario_generator
import os 
import time
import re
import argparse
from empire_model import run_empire
from mpi4py import MPI
import mpisppy.utils.sputils as sputils



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
    sputils.attach_root_node(instance, instance.investcost, [instance.genInvCap, instance.transmisionInvCap,instance.storPWInvCap, instance.storENInvCap])
    return instance



def EF(all_scenario_names,SEED,num_sce):
    from mpisppy.opt.ef import ExtensiveForm
    options = {"solver": "gurobi","solver_options": {"threads":32,"MIPGap": 0.01}}
    ef_kwargs = {"seed": SEED, "num_sce": num_sce}
    ef = ExtensiveForm(options, all_scenario_names, scenario_creator, scenario_creator_kwargs=ef_kwargs)
    start_time = time.time()
    results = ef.solve_extensive_form()
    end_time = time.time()

    elapsed = end_time - start_time
    objval = ef.get_objective_value()
    print(f"obj: {objval:.1f}")
    print(f"took: {elapsed} (sec)")


    import fcntl
    log_file = "full_ef_runtime_log.csv"
    need_header = not os.path.exists(log_file)
    with open(log_file, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if need_header:
            lf.write("seed,num_of_sce, runtime\n")
        lf.write(f"{SEED},{num_sce},{elapsed:.6f}\n")
        fcntl.flock(lf, fcntl.LOCK_UN)

    soln = ef.get_root_solution()

    for (var_name, var_val) in soln.items():
        print(var_name, var_val)


    pattern = re.compile(
        r'^(?P<Type>\w+)\['            
        r'(?P<Node>[^,]+),'            
        r'(?P<Energy_Type>[^,]+),'     
        r'(?P<Period>\d+)\]'           
        r'$'
    )

    type_mapping = {
        "genInvCap":        "Generation",
        "storPWInvCap":     "Storage Power",
        "storENInvCap":     "Storage Energy",
        "transmisionInvCap":"Transmission"
    }

    records = []
    for var_name, var_val in soln.items():
        m = pattern.match(var_name)
        if not m:
            continue
        rec = m.groupdict()
        rec['Type'] = type_mapping.get(rec['Type'], rec['Type'])
        rec['Value'] = var_val
        records.append(rec)

    df = pd.DataFrame(records, columns=['Node','Energy_Type','Period','Type','Value'])
    df.to_csv(f'sol_sets/full_ef_solution_{num_sce}_{SEED}.csv', index=False)




if __name__ == '__main__':    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    parser = argparse.ArgumentParser(description="EMPIRE model run")
    parser.add_argument("--seed",    "-s", type=int, default=1, help="random seed")
    parser.add_argument("--num-sce", "-n", type=int, default=5, help="number of scenarios")
    args = parser.parse_args()

    SEED    = args.seed
    num_sce = args.num_sce

    all_scenario_names = [f"scenario{i}" for i in range(1, num_sce+1)]
    print(f"num_sce: {num_sce}")
    
    
    EF(all_scenario_names, SEED, num_sce)

