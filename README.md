# Time Can Invalidate Algorithmic Recourse

This is the source code for the paper "Time Can Invalidate Algorithmic Recourse". The implementation is based on the code of Dominguez-Olmedo et al. (2022), available [here](https://github.com/RicardoDominguez/AdversariallyRobustRecourse).

## Structure

The project repository is organized with the following directories:
- `analytics/`: it contains scripts to analyze the results and generate the figures of the paper.
- `data/`: it contains the data for the experiments and the suitable pre-trained models.
- `learned_scms/`: it contains the pre-trained approximate SCMs for the experiments with realistic data (see Section 4.2 of the paper).
- `results/`: it contains all the raw results from the experiments.
- `src/`: it contains the working code e.g., SCMs, implementations of TSAR, CAR, SAR and IMF, etc.
- `testing/`: it contains unit-tests checking some basic implementation details. 

## Install

We report here the instructions to install a `conda` environment with Python 3.10 to run the experiments. We also report the needed packages. In theory, it should not matter which environment manager is used. 

```bash
# Create a suitable environment with python 3.10
conda create --name temporal-recourse python=3.10
conda activate temporal-recourse

# Install the required libraries into the environment
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install numpy scipy matplotlib scikit-learn pandas seaborn tqdm cvxpy mpi4py
```

Once the environment is set up, we suggest to run the unittest to ensure everything is in order:

```bash 
python -m unittest testing/test_temporal_causal_models.py
```

## Reproduce the experiments

We prepared several bash scripts to run automatically all the experiments presented in the main paper.
Beware that it might take some time since we run exclusively on CPU. 
Before running any of these scripts, make sure to be in the parent directory and to have run:

```bash
cd temporal-recourse-submission
conda activate temporal-recourse
export PYTHONPATH=.
```

The following will run the experiments detailed in Section 4.1.

```bash
bash run_proposition_1.sh
bash run_task_1.sh
```

The following will instead run the experiments detailed in Section 4.2 and Appendix D, respectively. 

```bash
bash run_task_2.sh
bash run_task_2_ground_truth.sh
```

Lastly, we can run the following to regenerate the plot in Figure 2.

```bash
bash run_alpha.sh
```

### Re-train the generative models

We also provide some scripts to retrain the generative model used in Section 4.2 of the main paper. For more details, please have a look at Appendix B. In practice, this step is not needed since we provide the pre-trained models in `learned_scms/`.

```bash
bash run_train_scms_syn.sh
bash run_train_scms.sh
```

## Re-generate the plots

Given the raw data files in `results/`, we provide some scripts and bash commands to generate all the figures of the main paper.

```bash
# Generate Figure 2
python analytics/plot_alphas.py results/task_0_alpha/*.csv

# Generate Figure 3
python analytics/plot_test_1.py results/task_1_uncertainty/dnn_*.csv

# Generate Figure 4
python analytics/plot_trends_syn.py results/task_2_syn_data/*.csv

# Generate Figure 5
python analytics/plot_trends.py results/appendix/task_3_real_linear/*.csv
bash plots/analysis_actions.sh

# Generate plots in Appendix C
bash plots/cost_actions.sh
bash plots/sparsity_actions.sh

# Generate plots in Appendix D
python analytics/plot_trends.py results/task_3_real_data_ground_truth/*.csv
```
