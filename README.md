# Analyzing Error Sources in Global Feature Effect Estimation

This repository contains the full source code and results for the paper "Analyzing Error Sources in Global Feature Effect Estimation" by Timo Heiß, Coco Bögel, Bernd Bischl, and Giuseppe Casalicchio, submitted to the XAI Conference 2026.

## Important Materials (Quick Links)

- hyperparameter configurations: [analyses/model_performance_analysis.ipynb](analyses/model_performance_analysis.ipynb)
- hyperparameter search spaces: [current_research_feature_effects/mappings.py](current_research_feature_effects/mappings.py#L75)
- model performances: [analyses/model_performance_analysis.ipynb](analyses/model_performance_analysis.ipynb)
- bias-variance-decomposition results: [analyses/bias_variance_analysis.ipynb](analyses/bias_variance_analysis.ipynb)
- variance decomposition results: [analyses/variance_decomposition_analysis.ipynb](analyses/variance_decomposition_analysis.ipynb)
- effect of sample size on estimation error results: [analyses/estimation_error_sample_size_analysis.ipynb](analyses/estimation_error_sample_size_analysis.ipynb)

## Repository Structure

- *analyses*: jupyter notebooks containing analyses and plots of the experiment results
    - *bias_variance_analysis.ipynb*: analyses corresponding to chapter §6.1 in the paper regarding MSE decomposition into bias and variance
    - *estimation_error_sample_size_analysis.ipynb*: analyses corresponding to chapter §6.3 on the effect of the sample size on the estimation error
    - *groundtruth_effects.ipynb*: visualizations of the the ground truth feature effects for each dataset
    - *model_performance_analysis.ipynb*: evaluation of model performances and overview of selected model hyperparameters
    - *ablation_study_variance_decomposition.ipynb*: analyses corresponding to chapter §6.2 in the paper on variance decomposition into model and estimation variance
-  *configs*: configurations of the experiments
    - *ablation_study_mc.ini*: configuration of the experiment on the effect of sample size on estimation error
    - *ablation_study_variance_decomposition.ini*: configuration of the experiments to estimate estimation variances (enabling variance decomposition)
    - *datasets.yaml*: dataset configurations for all experiments
    - *main_study.ini*: configuration of the experiments on MSE bias-variance-decomposition
    - *models_ablation_study.yaml*: model configurations for the "ablation study" on estimation variance estimation for variance decomposition
    - *models.yaml*: model configurations for the MSE bias-variance-decomposition experiments
- *current_research_feature_effects*: 'library' with implementations of functions and classes used in the experiment scripts, detailed documentation is provided in the code
- *experiments*: results of the experiments
    - *ablation_study_mc_error*: results for sample size impact on estimation error
    - *ablation_study_variance_decomposition_new*: results for estimation variances (used for variance decomposition)
    - *main_study_parallel*: results for MSE bias-variance-decomposition
- *scripts*: python scripts for the experiments
    - *ablation_study_mc.py*: script for experiment on the effect of sample size on estimation error
    - *ablation_study_variance_decomposition.py*: script for experiment to estimate estimation variances (enabling variance decomposition)
    - *main.py*: script for experiments on MSE bias-variance-decomposition

## Setup / Installation
Before executing experiments in this project, it is recommended to follow these steps:

1.) Clone or fork this repository

2.) Install pipx:
```
pip install --user pipx
```

3.) Install Poetry:
```
pipx install poetry
```

*3a.) Optionally (if you want the env-folder to be created in your project)*:
```
poetry config virtualenvs.in-project true
```

4.) Install this project:
```
poetry install
```

*4a.) If the specified Python version (3.11 for this project) is not found:*

If the required Python version is not installed on your device, install Python from the official [python.org](https://www.python.org/downloads) website.

Then run
```
poetry env use <path-to-your-python-version>
```
and install the project:
```
poetry install
```

## Reproduce the Experiments

The experiments can be reproduced after completing the setup/installation above by running the following commands:

### Run MSE Bias-Variance-Decomposition Experiment

Adapt the configs/main_study.ini with your paths, navigate to the root folder of this repository and run:

```
poetry run python scripts/main_study.py --config configs/main_study.ini
```

This will automatically run the main study based on the parameters provided in the `configs/main_study.ini`.
It will create a new subdirectory in `experiments` based on the simulation name you specified in the config,
copy the config-file to this directory, and store all simulation results (model results, tuning trials, and feature effects results in database-files).

Random seeds are used throughout the entire simulation to make all results reproducible. Data genereated in each simulation run are not stored, but can simply be recreated afterwards if needed by using the number of the simulation run as seed for the data generation.

### Run Experiment on Estimation Variance

Adapt the ablation_study_variance_decomposition.ini with your paths, navigate to the root folder of this repository and run:

```
poetry run python scripts/ablation_study_variance_decomposition.py --config configs/ablation_study_variance_decomposition.ini
```

Results can also be found in the `experiments` directory under the corresponding experiment.

### Run Experiment on Effect of Sample Size on Estimation Error

Adapt the ablation_study_mc.ini with your paths, navigate to the root folder of this repository and run:

```
poetry run python scripts/ablation_study_mc.py --config configs/ablation_study_mc.ini
```

Results can also be found in the `experiments` directory under the corresponding experiment.
