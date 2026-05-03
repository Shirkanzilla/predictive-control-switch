# Alternative Control-Switch Method for Guided Safe Exploration

This project was started as an alternative approach to the Control-Switch Method described in the paper:

> **Reinforcement Learning by Guided Safe Exploration**  
> Qisong Yang, Thiago D. Simão, Nils Jansen, Simon H. Tindemans, Matthijs T. J. Spaan \
> [Original Paper](https://arxiv.org/abs/2307.14316) 

Implementation of the framework is adapted from [sagui-container](https://github.com/MarkelZ/sagui-container), which provides a simple and straightforward installation method to the original framework 
and the extended robust one described in [Robust Transfer of Safety-Constrained Reinforcement Learning Agents](https://openreview.net/forum?id=rvXdGL4pCJ).

TODO: Add part of the abstract here maybe as a intro to the project

---

## Repository Structure

This repository consists of four main components:

### 1. Dataset Generation in /Predictive Control-Switch/dataset_creation
train_safe_agent.py is used to train a safe agent using, which can then be used to measure the expected cost of a sample.
In create_dataset.py, a dataset is created, preprocessed, and saved.

### 2. Neural network training in /Predictive Control-Switch/neural_network_training
In train_neural_network.ipynb, we train:
- **A classifier**: To predict whether the expected cost is zero.
- **A regression model**: To directly estimate the expected cost based on observations and actions, but only on non-zero samples.
These models are then saved as scikit-learn compatible predictors to allow for a wide range of predictor types.

### 3. Benchmarks in /Predictive Control-Switch/method_comparison
In comparison.ipynb, the vanilla Control-Switch and the newly proposed Predictive Control-Switch methods of the original thesis project are benchmarked.

With train_comparisons.py and train_neural_shielding.py the neural shielding method was benchmarked.

### 4. Utility used throughout in /Predictive Control-Switch/misc and the shared_files package
With record_video.py, this safe agent can be recorded and sanity checked.
The other files contain code to test various things used throughout the project.



---

## Installation

### Prerequisites

- Python 3.8
- safety gymnasium v.1.2.0 installed from [GitHub](https://github.com/PKU-Alignment/safety-gymnasium)
- installing the shared_files package
- installing all python packages listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Replicating the experiments

### From Scratch
- Train safe agents with predictive_control_switch/dataset_creation/train_safe_agent.py
- Create a dataset with predictive_control_switch/dataset_creation/create_dataset.py 
- Train and safe a classifier and regressor with  predictive_control_switch/neural_network_training/train_neural_network.ipynb 
- Train with neural shielding using predictive_control_switch/method_comparison/train_neural_shielding.py
- All other files were used for debugging and or creating plots

### From our results
- The neural shielding can be replicated by just passing the run dict of one of the finished experiments again. TODO: Add a path to a specific run dict.

