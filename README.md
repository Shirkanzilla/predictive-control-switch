# Predictive Control Switch

Research pipeline for training safe RL agents, generating datasets, training cost predictors, and benchmarking a neural shielding approach against baselines.

Environments we want to run the pipeline on:
- InvertedPendulum (customized)
- SafetyPointGoal1
- SafetyPointCircle2
- SafetyPointFormulaOne1
- Frozenlake (customized, for testing purposes)

## Setup

Dependencies and the required Python version are managed with [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/ai-fm/shielded-transfer-rl.git
cd shielded-transfer-rl
uv venv
uv sync
```

Activate the venv as usual (`source .venv/bin/activate`) before running any scripts.

## Repository Structure

- **`predictive_control_switch/`** - main pipeline scripts: training the safe agent, dataset creation, neural network training, and benchmarking.
- **`shared_files/src/helpers/`** - shared code used throughout the pipeline (custom environments, utilities, etc.).
- **`runs/`** - all trained agents throughout the pipeline and envs.
- **`relevant_runs/`** - some curated runs I wanted to highlight.

Most relevant code is in `.py` scripts. `.ipynb` notebooks were mainly used for the bachelor's thesis, with one exception: `predictive_control_switch/neural_network_training/train_neural_network.ipynb` is still the main entry point for training the predictors.

## Pipeline

### 1. Train the safe agent

```bash
python predictive_control_switch/dataset_creation/train_safe_agent.py \
    <env_id> <algorithm> <epochs> [device] [threads] \
    [curriculum_learning y/n] [curriculum_learning_epochs>]\
    [previous_model_path]
```

Curriculum learning has so far only been used for `InvertedPendulum`. The agent is first trained in the non-Monte-Carlo sampling environment, then the probability of random starting states is gradually increased. Just training with the random start did not work well.

With
```bash
python predictive_control_switch/misc/record_video.py \
    <save_dir>
```
videos of trained agents can be captured. Useful for sanity checks. 

With 
```bash
python evaluate_policy.py <env_id> <algorithm> <seed/subdir name> <epoch>
```
a policy can be evaluated by collection offline data and printing statistics about that.

### 2. Create a dataset

```bash
python predictive_control_switch/dataset_creation/create_dataset.py \
    <env_id> <amount> <safe_agent_dir> <pt_file_name> \
    [min_rand_steps] [max_rand_steps] [cost_window] [deterministic] [n_processes]
```

A random agent runs for a random number of steps (between `min_rand_steps` and `max_rand_steps`), after which the trained safe agent takes over for `cost_window` steps, acting deterministically if specified.

> Note: `n_processes` was an attempt to parallelize dataset creation, but in practice `n_processes=1` has been fastest (I don't know why).

### 3. Train cost predictors

Main file: `predictive_control_switch/neural_network_training/train_neural_network.ipynb`

Trains two models:
- a **classifier** distinguishing zero vs. non-zero cost,
- a **regression model** predicting cost in the non-zero case.

For `InvertedPendulum`, no zero-cost cases occured up until now, so only the regression model is used.

> Everything in this notebook beyond the core PyTorch training loop is experimental and not fully tested.

### 4. Benchmark

```bash
python train_all_variants_parallel.py <base_env_id> <safe_agent_dir_path> <pt-file name> <epochs>
python predictive_control_switch/benchmark/plot_results.py
```

Trains and compares:
- **PPOSaute**
- **PPOLag**
- **PPOSaute + Neural Shielding** (our approach)

Results in ./results.