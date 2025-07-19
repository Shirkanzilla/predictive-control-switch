# Alternative Control-Switch Method for Guided Safe Exploration

This repository presents an alternative approach to the Control-Switch Method described in the paper:

> **Reinforcement Learning by Guided Safe Exploration**  
> Qisong Yang, Thiago D. Simão, Nils Jansen, Simon H. Tindemans, Matthijs T. J. Spaan \
> [Original Paper](https://arxiv.org/abs/2307.14316) 

Implementation of the framework is adapted from [sagui-container](https://github.com/MarkelZ/sagui-container), which provides a simple and straightforward installation method to the original framework 
and the extended robust one described in [Robust Transfer of Safety-Constrained Reinforcement Learning Agents](https://openreview.net/forum?id=rvXdGL4pCJ).

Our method introduces a predictive model that estimates the **expected cost** of an agent’s behavior based on its observations and actions. 
This enables a preventive and potentially more accurate safety mechanism compared to the original Control-Switch method, which is only reactive.

---

## Repository Structure

This repository consists of two main components:

### 1. Dataset Generation & Model Training in /model-training/

train_safe_agent.py is used to train a safe agent using PPOLag, which is used to measure the expected cost connected to a sample.

With record_video.py, this safe agent can be recorded and sanity checked.

In create_dataset.ipynb, a dataset is created, preprocessed, and saved. We do that for the "SafetyPointGoal1-v0" environment, but by changing the env, a dataset could be generated for a different task.

In train_neural_network.ipynb, we train:
- **A classifier**: To predict whether the expected cost is zero.
- **A regression model**: To directly estimate the expected cost based on observations and actions, but only on non-zero samples.
These models are then saved.

manual_nn_debugging.py offers a way to manually play a safety-gymnasium environment and do a sanity check on the performance of the models. 
The predicted cost is printed to the console for every timestep.


### 2. Modified Guided Safe Exploration (GSE) Framework

This component is based on a cloned and adapted version of the [sagui-container](https://github.com/MarkelZ/sagui-container) repository. 
It integrates our cost prediction models in place of the original Control-Switch method.

We do that for two 

---

## Installation

### Prerequisites

- Python 3.7
- mujoco 2.0+ in home/mujoco and in ./sagui-container/mujoco
- Python packages listed in `requirements.txt`
- For easier installation: Docker engine

```bash
pip install -r requirements.txt

