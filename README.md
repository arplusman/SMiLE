# SMiLE: Provably Enforcing Global Relational Properties in Neural Networks

## Description
This repository allows for the reproduction of the findings reported in the manuscript __"SMiLE: Provably Enforcing Global Relational Properties in Neural Networks"__, submitted to __AAAI 2026 AI Alignment Track__ and currently under revision.

The paper extends the property-enforcement SMiLE framework, designed for trace properties, to also support relational ones, such as monotonicity, robustness and fairness.

## Installation
1. Python: `pip install -r requirements.txt`
2. Gurobi: follow the instructions on https://www.gurobi.com/features/academic-named-user-license/ to download Gurobi Optimizer and obtain a free academic license

## Project Structure
This repository is structured into three folders: 
    * `core` -- contains the implementation of the methodology (Methodology Section)
    * `benchmarks` -- contains the implementation of the experiments (Experimentation Section)
    * `synthetic_example` -- contains the implementation of the synthetic example (Figure 2)

1. `core` contains:
    * `data.py` -- implements the data preprocessing for the three benchmarks
    * `model.py` -- implements the SMiLE architecture
    * `optimization.py` -- implements optimization components, in particular Generator and Projector, both used in Algorithm 1

2. `benchmarks` consists of a folder for each benchmark: `monotonicity`, `roubstness` and `fairness`, containing:
    * `process.py` -- trains the models and produces the logs with the corresponding raw results
    * `process_train.py` (only for `roubstness`) -- trains the models and produces the logs with the corresponding raw results 
    * `process_attack.py` (only for `roubstness`) -- attacks the trained models via a 100-step PGD attack on 1,000 test samples
    * `postprocess.py` -- retrieves and aggregates the raw results into plots and/or tables
    * `data`: (only for `fairness`) contains the training/testing data used by `process.py`
    * `results`: stores the raw results produced by `process.py`
    * `aggregated_results`: stores the aggregated results produced by `postprocess.py`

3. `synthetic_example`:
    * `process.py` -- trains a globally robust SMiLE model on the synthetic task
    * `postprocess.py` -- retrieves and aggregates the raw results into plots and/or tables

## Usage
The results presented in the paper can be reproduced, for the syntetic example and for each benchmark, as follows. 

1. Generate raw results: 
    * `python benchmarks/{benchamrk}/process.py` (`monotonicity`, `fairness`)
    * `python benchmarks/{benchamrk}/process_train.py`, then `python benchmarks/{benchamrk}/process_attack.py` (`robustness`)
    * `python synthetic_example/process.py` (`synthetic_example`)
2. Generate aggregated results: 
    * `python benchmarks/{benchmark}/postprocess.py`
    * `python synthetic_example/postprocess.py` (`synthetic_example`)

__NOTE__: All aggregated results (plots/tables) reported in the paper can be direclty accessed from `benchmarks/{benchmark}/aggregated_results` and `synthetic_example/aggregated_results`, without running any process. However, the raw results are not available in `benchmarks/{benchmark}/results` or `synthetic_example/results`, as they exceed OpenReview size limit. The only way to obtain these results is by re-generating them through the pipeline described above.
