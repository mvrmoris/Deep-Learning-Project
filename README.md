# FlowNAS

This repository contains the code for **FlowNAS**, a project developed for the Deep Learning course at Sapienza University of Rome.

The experiments were conducted on two Neural Architecture Search benchmarks: **NAS-Bench-201** and **NAS-Bench-301**. For a simpler and faster setup, we recommend starting with NAS-Bench-201, since NAS-Bench-301 requires additional dependencies, is more computationally demanding, and may take longer to configure.

To make the repository easier to navigate and the experiments easier to reproduce, this README is divided into the following sections:

* Setup and execution instructions for NAS-Bench-201
* Setup and execution instructions for NAS-Bench-301
* An overview of the project structure and main files


# NAS-Bench-201 Setup
After cloning the repository and creating a virtual environment, install the required dependencies with:

```bash
pip install -r requirements.txt
```

We recommend starting with the two notebooks:

* `notebooks/training.ipynb`
* `notebooks/latent_space.ipynb`


# NAS-Bench-301 setup

NAS-Bench-301 uses a pretrained surrogate model to estimate the
CIFAR-10 performance of architectures in the DARTS search space.

1. Install the NAS-Bench-301 package:

   ```bash
   pip install git+https://github.com/automl/nasbench301.git
    ````

2. Download the pretrained performance surrogate model:

   ```python
   import nasbench301 as nb

   nb.download_models(
       version="1.0",
       download_dir="datasets"
   )
   ```

3. The expected model directory is:

   ```text
   datasets/
   └── nb_models_1.0/
       └── xgb_v1.0/
   ```



# Project Overview

## `notebooks/`

The `notebooks/` directory contains the analyses and experiments developed throughout the project.

* `notebooks/training.ipynb` provides a step-by-step implementation of the FlowNAS training procedure.
* `notebooks/latent_space.ipynb` contains visualizations of the learned latent space and Flow directions.
* `notebooks/experiments.ipynb`  ad `notebooks/experiments.ipynb` contains the code used to run the main experiments and evaluations.

## `datasets/`

The `datasets/` directory contains the utilities required to load and process the NAS benchmarks used in the project.

* `datasets/dataset_loader_nas201.py` contains the dataset builder, architecture conversion utilities, and API loader for NAS-Bench-201.
* `datasets/dataset_loader_nas301.py` contains the corresponding utilities for NAS-Bench-301.
* `datasets/nas301/` contains 50,000 preprocessed NAS-Bench-301 architectures to speed up testing and experimentation.

## `models/`

The `models/` directory contains the neural network architectures used by FlowNAS.

* `models/flow.py` implements the Flow model and its loss function.
* `models/nas201_models.py` contains the VAE architecture and loss functions used for NAS-Bench-201.
* `models/nas301_models.py` contains the VAE architecture and loss functions used for NAS-Bench-301.

## `utils_functions/`

The `utils_functions/` directory contains reusable utilities shared across training, evaluation, and visualization scripts.

* `utils_functions/utils.py` includes general utilities such as seed initialization, architecture generation, population construction, and NAS-Bench-201 evaluation.
* `utils_functions/utilsnas301.py` contains utilities specific to NAS-Bench-301.
* `utils_functions/tests_utils.py` contains the functions used to compare the learned Flow direction with a random direction of equal norm.
* `utils_functions/plots_utils.py` contains plotting and visualization utilities.

## `ws_universale/`

The `ws_universale/` directory contains the weight-sharing implementation used to evaluate NAS-Bench-201 architectures through a shared supernet.

## Main directory `./`

The main directory contains the scripts and configuration files used to train and evaluate FlowNAS.

* `train.py` contains the main training pipeline, including VAE pretraining, Flow training, architecture generation, and population updates.
* `requirements.txt` lists the dependencies required to run the NAS-Bench-201 experiments.
