# NeurMHSP

## Abstract

This repository accompanies the paper "Machine Learning-Enabled Large-Scale Transmission Capacity Expansion Planning" and contains the official implementation of the models and experiments.

## Repository Structure

```
├── Data handler/           # Dataset files for EMPIRE model
├── models/                 # Trained Model
├── BM_Validation/           # Baselines and Solution Validation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

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
conda install pip
```

3. Install dependencies:
```bash
conda activate myenv
pip install -r requirements.txt
```


## Usage
All code is optimized for use on a High-Performance Computing (HPC) cluster. Due to the large size of the EMPIRE model and its dataset, significant computational resources are required. Therefore, running this code on a local laptop or desktop is not recommended.

The dataset generation process is fully automated. Before starting, you must specify the number of samples to generate for the training dataset. Please follow the procedures below:

1. Open `job.sh`, set the ``TOTAL_FILES`` variable to the number of samples you want.
2. Submit the job by running the following command in your terminal:

```bash
sbatch job.sh
```

### Solution Validation

For the validation, go to the `Experiments` directory

```bash
cd Experiments
```

Run validation script.

```bash
sbatch sol_valid_ML.sh
```


### Baselines 

The baseline models are implemented using the [``mpi-sppy``](https://mpi-sppy.readthedocs.io/en/latest/) library. which is also optimized for HPC environments. If you wish to use this library, please follow the instructions provided in the official documentation.

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

+ **Note**: The BD and PH implementations use multiple nodes for parallel computing. Therefore, you **must ensure** that the number of nodes requested in the Slurm script (`--nodes`) matches the number of scenarios you are running.




## Contact

- [Your Name] - [tzk5446@psu.edu]
- Project Link: [NeurMHSP](https://github.com/Subramanyam-Lab/NeurMHSP.git)

## Acknowledgments


## Changelog

### v1.0.0 (Initial Release)
- Initial code release
- Reproducible experiments
- Complete documentation