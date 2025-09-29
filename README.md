# NeurMHSP

## Abstract

This repository supports codes for a paper "Machine Learning-Enabled Large-Scale Transmission Capacity Expansion Planning"

## Repository Structure

```
├── Data handler/           # Dataset files for EMPIRE model
├── models/                 # Model architectures and implementations
├── BM_Validation/            # Baselines and Solution Validation
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
cd EMPIRE
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


All codes are optimized for using in High Performance Computing (HPC). Since the EMPIRE model and its dataset is extremely large and required huge computatioal resources. Do not recommned running this code on local laptop and desktop.


Dataset generation process are fully connected. Before you starting your job, you must choose the numbe of samples in dataset for training machine learning models. You must following the below procedures:

1. Open `job.sh` and change ``TOTAL_FILES`` to numbers what you want.
2. Run 

```bash
sbatch job.sh
```

### Solution Validation

For the validation, go to the `BM_Validation` directory

```bash
# Evaluation command
python evaluate.py \
    --model_path results/best_model.pth \
    --test_data data/test.csv
```

### Reproducing Paper Results

To reproduce the main results from the paper:

```bash
# Run all experiments from the paper
bash scripts/run_all_experiments.sh
```

Individual experiments:
```bash
# Experiment 1: [Description]
python experiment1.py --config experiments/exp1_config.yaml

# Experiment 2: [Description]
python experiment2.py --config experiments/exp2_config.yaml
```

## Code Structure

### Key Files

- `main.py`: Main entry point for experiments
- `models/model.py`: Implementation of the proposed model
- `train.py`: Training script
- `evaluate.py`: Evaluation script
- `utils/data_loader.py`: Data loading utilities
- `utils/metrics.py`: Evaluation metrics


## Contact

- [Your Name] - [tzk5446@psu.edu]
- Project Link: [NeurMHSP](https://github.com/Subramanyam-Lab/NeurMHSP.git)

## Acknowledgments


## Changelog

### v1.0.0 (Initial Release)
- Initial code release
- Reproducible experiments
- Complete documentation