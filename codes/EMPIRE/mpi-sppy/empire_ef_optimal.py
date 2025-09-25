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

warnings.filterwarnings("ignore")


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
    # model._mpisppy_probability = 1.0 / 5, ######## if don't set assumign equal..
    return instance


from mpisppy.opt.ef import ExtensiveForm
def ef(all_scenario_names, num_sce):
    options = {"solver": "gurobi"}
    ef_kwargs = {"seed": num_sce, "num_sce": num_sce}
    ef = ExtensiveForm(options, all_scenario_names, scenario_creator, scenario_creator_kwargs=ef_kwargs)
    start_time = time.time()
    results = ef.solve_extensive_form()
    end_time = time.time()


    elapsed = end_time - start_time
    objval = ef.get_objective_value()
    print(f"obj: {objval:.1f}")
    print(f"took: {elapsed} (sec)")
    
    soln = ef.get_root_solution()

    # print out all variables and values
    for (var_name, var_val) in soln.items():
        print(var_name, var_val)


    # define a pattern to parse the variable name
    pattern = re.compile(
        r'^(?P<Type>\w+)\['            
        r'(?P<Node>[^,]+),'            
        r'(?P<Energy_Type>[^,]+),'     
        r'(?P<Period>\d+)\]'           
        r'$'
    )

    # define a mapping from internal var names to desired Type labels
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
        # rec['Type'] 에 원래 들어 있던 키를 변환
        rec['Type'] = type_mapping.get(rec['Type'], rec['Type'])
        rec['Value'] = var_val
        records.append(rec)

    df = pd.DataFrame(records, columns=['Node','Energy_Type','Period','Type','Value'])
    df.to_csv(f'sol_sets/solution_optimal.csv', index=False)
    
    print("!!End: Extensive Form (optimal)!!")
    



if __name__ == '__main__':

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    num_sce = 100

    all_scenario_names = [f"scenario{i}" for i in range(1, num_sce+1)]
    print(f"num_sce: {num_sce}")
    
    ef(all_scenario_names,num_sce)

