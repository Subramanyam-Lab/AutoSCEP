import random
import math
import os
import glob

### We need to change ###

def euclidean_distance(p1, p2):
    return 10 * math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_problem_data(size):
    clients, facilities = size
    client_points = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(clients)]
    facility_points = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(facilities)]
    transportation_costs = [[euclidean_distance(c, f) for f in facility_points] for c in client_points]
    demands = [random.uniform(5, 35) for _ in range(clients)]
    capacities = [random.uniform(10, 160) for _ in range(facilities)]
    fixed_costs = [random.uniform(0, 90) + random.uniform(100, 110) * math.sqrt(s) for s in capacities]
    problem_data = {
        "clients": clients,
        "facilities": facilities,
        "transportation_costs": transportation_costs,
        "demands": demands,
        "capacities": capacities,
        "fixed_costs": fixed_costs,
    }
    return problem_data

def scale_problem_data(args):
    problem_data, ratio, i, problem_sizes = args
    scaled_problem_data = problem_data.copy()
    scaling_factor = ratio
    scaled_problem_data["capacities"] = [c * scaling_factor for c in problem_data["capacities"]]
    scaled_problem_data["fixed_costs"] = [random.uniform(0, 90) + random.uniform(100, 110) * math.sqrt(s) for s in scaled_problem_data["capacities"]]
    
    if ratio in [1.5, 2]:
        scaled_problem_data["fixed_costs"] = [f * 2 for f in scaled_problem_data["fixed_costs"]]
    
    return scaled_problem_data

def clear_directory(directory):
    # Check if the directory exists
    if os.path.exists(directory):
        # Get all .dat files in the directory
        files = glob.glob(os.path.join(directory, "*.dat"))
        # Remove all .dat files
        for file in files:
            os.remove(file)
    else:
        # If the directory does not exist, create it
        os.makedirs(directory)

def save_problem_to_dat(problem_data, size, scenario_index):
    clients, facilities = size
    directory = f"data/CPLP_{clients}_{facilities}/"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}scenario_{scenario_index}.dat"
    with open(filename, "w") as f:
        f.write(f"set P := {' '.join(['P' + str(i) for i in range(problem_data['facilities'])])} ;\n")
        f.write(f"set C := {' '.join(['C' + str(i) for i in range(problem_data['clients'])])} ;\n\n")
        
        f.write("param: f c :=\n")
        for i, (fixed_cost, capacity) in enumerate(zip(problem_data['fixed_costs'], problem_data['capacities'])):
            f.write(f"    P{i} {fixed_cost} {capacity}\n")
        f.write(";\n\n")
        
        f.write("param d :=\n")
        for i, demand in enumerate(problem_data['demands']):
            f.write(f"    C{i} {demand}\n")
        f.write(";\n\n")
        
        
        f.write("param t :")
        for j in range(facilities):  # Change i to j for clarity
            f.write(f" P{j}")  # Change C to P
        f.write(" :=\n")
        for i, costs in enumerate(problem_data['transportation_costs']):
            f.write(f"    C{i} {' '.join(map(str, costs))}\n")  # Change P to C
        f.write(";\n\n")



def generate_and_save_problem_data(size, scenario_index, scaling_ratios):
    clients, facilities = size
    directory = f"data/CPLP_{clients}_{facilities}/"

    if scenario_index == 0:
        clear_directory(directory)
    for ratio_index, ratio in enumerate(scaling_ratios):
        problem_data = generate_problem_data(size)  # Move this line inside the loop
        scaled_problem_data = scale_problem_data((problem_data, ratio, scenario_index, [size]))
        save_problem_to_dat(scaled_problem_data, size, scenario_index * len(scaling_ratios) + ratio_index)

if __name__ == '__main__':
    random.seed(42)
    
    num_sets = 20
    problem_sizes = [(10, 10), (25, 25), (50, 50)]
    scaling_ratios = [1.5, 2, 3, 5, 10]

    for size_index, size in enumerate(problem_sizes):
        for scenario_index in range(num_sets):
            generate_and_save_problem_data(size, scenario_index, scaling_ratios)
