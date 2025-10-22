# AutoSCEP

## Table of Contents
- [Abstract](#abstract)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Setup](#setup)
- [Usage](#usage)
  - [Important Note on Environment](#important-note-on-environment)
  - [Automated End-to-End Workflow](#automated-end-to-end-workflow)
  - [How to Run](#how-to-run)
- [Solution Validation & Baselines](#solution-validation--baselines)
  - [Solution Validation](#solution-validation)
  - [Baselines](#baselines)
- [Changelog](#changelog)

---

## Abstract
This repository accompanies the paper "Machine Learning-Enabled Large-Scale Capacity Expansion Planning under Uncertainty" and contains the official implementation of the models and experiments.

---

## Repository Structure
```
codes/
├── Data handler/
├── Datalog/
├── Experiments/
├── models/
├── scripts/
├── src/
├── requirements.txt
└── README.md
```

### Directory Descriptions

#### `Data handler/`
Contains dataset files for the EMPIRE model, including input data and scenario files.

#### `Datalog/`
Stores aggregated results used to a paper. 

#### `Experiments/`
Contains baseline models and solution validation scripts:
- `Data handler/` - Contains dataset files for the EMPIRE model
- `Datalog/` - Stores validatione results and runtime log. 
- `parameter_convergence/` - Parameter selection algorithm testing codes and result plots.
- `MLEMBEDSOLS_{adaptive, fixed}/` - AutoSCEP's solutions (adaptive) and fixed parameters solutions.
- `sol_sets/` - Baesline Solutions
- `scripts/` - Shell scripts
    - `main_ef.sh` - Extensive Form baseline
    - `main_bm.sh` - Benders Decomposition & Progressive Hedging baseline
    - `sol_valid_ML.sh` - Solution validation for ML surrogate method
    - `sol_valid.sh` - Solution validation for baseline 
- `src/` - relavant python codes


#### `models/`
Stores trained machine learning models and checkpoints.

#### `scripts/`
Shell scripts for automated workflow execution:
- `job.sh` - Main job submission script
- `sampling_script.sh` - Data sampling script
- `ml_train.sh` - Model training script
- `embedding.sh` - Embedding execution script
- `worker_script.sh` - Worker script for parallel processing
- `wrapper.sh` - Wrapper script for coordination

#### `src/`
Python source code and modules:
- `config_run.yaml` - Configuration file
- `data_preprocessing.py` - Data preprocessing module
- `label_generation_adaptive.py` - Adaptive label generation
- `label_generation_fixed.py` - Fixed label generation
- `ml_embedding.py` - ML model embedding
- `ml_train.py` - Model training
- `NEUREMPIRE.py` - Main EMPIRE model implementation
- `reader.py` - Data reader utilities
- `run.py` - Main execution script
- `sampling.py` - Sampling utilities
- `scenario_random.py` - Random scenario generation
- `second_stage_label.py` - Second stage labeling

---

## Installation

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Subramanyam-Lab/NeurMHSP.git
cd NeurMHSP
cd codes
```

2. Create a virtual environment:
```bash
conda create -n myenv python=3.11
conda activate myenv
conda install pip
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Important Note on Environment
All code is optimized for use on a **High-Performance Computing (HPC)** cluster. Due to the large size of the EMPIRE model and its dataset, significant computational resources are required. Therefore, running this code on a local laptop or desktop is **not recommended**.

---

### Automated End-to-End Workflow
The entire workflow, from initial data generation to solving the final problem, is **fully automated**. Submitting a single job script will automatically execute the following sequence:

1. **Sampling**: Generates the initial data samples.
2. **Labeling**: Processes and labels the generated samples.
3. **Preprocessing**: Cleans and prepares the data for model training.
4. **Model Training**: Trains the machine learning surrogate model.
5. **Embedding & Solving**: Embeds the trained model into the optimization problem and solves it.

To run this complete pipeline, you only need to specify the desired number of samples and submit the main job script.

---

### How to Run
1. Open `scripts/job.sh` and set the `TOTAL_FILES` variable to the number of samples you want.
2. Submit the job by running the following command in your terminal:
```bash
cd scripts
sbatch job.sh
```

---

## Solution Validation & Baselines
This section outlines how to validate the solution from the machine learning model and how to run the baseline optimization models for comparison.

### Solution Validation
To validate the feasibility and cost of the solution obtained from the ML-driven approach, follow these steps:

1. Navigate to the `Experiments` directory.
2. Submit the validation job script.
```bash
cd Experiments
sbatch sol_valid_ML.sh
```

### Baselines
The baseline models are implemented using the [``mpi-sppy``](https://mpi-sppy.readthedocs.io/en/latest/) library, which is also optimized for HPC environments. If you wish to use this library, please follow the instructions provided in the official documentation.

Once your environment is configured, you can run the baseline models as follows:

#### Extensive Form (EF)
Submit the job using this script:
```bash
sbatch main_ef.sh
```

#### Benders Decomposition (BD) & Progressive Hedging (PH)
Submit the job using this script:
```bash
sbatch main_bm.sh
```

**Note**: The BD and PH implementations use multiple nodes for parallel computing. Therefore, you **must ensure** that the number of nodes requested in the Slurm script (`--nodes`) matches the number of scenarios you are running.

---

## Changelog

### v1.0.0 (Initial Release)
- Initial code release
- Reproducible experiments
- Complete documentation