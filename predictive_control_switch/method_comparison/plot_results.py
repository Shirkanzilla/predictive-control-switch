import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob

envs = {
    "NeuralShieldingInvertedPendulum": {"color": "blue", "label": "NeuralShieldingPPOLag"},
    "SauteInvertedPendulum": {"color": "green", "label": "PPOSaute"},
    "SafetyInvertedPendulum": {"color": "red", "label": "PPOLag"},
}

data = {env: {"rewards": [], "costs": [], "steps": []} for env in envs}

for env_name in envs:
    # Find progress.csv recursively
    pattern = f"./results/{env_name}/**/progress.csv"
    csv_files = glob(pattern, recursive=True)

    if not csv_files:
        print(f"Warning: No progress.csv files found for {env_name}. Skipping.")
        continue

    seed_data = {}
    for csv_file in csv_files:
        parts = csv_file.split(os.sep)
        seed_dir = [p for p in parts if p.startswith("seed")][0] if any(p.startswith("seed") for p in parts) else "seed_0"
        seed = int(seed_dir.split("-")[0].replace("seed", "")) if "-" in seed_dir else int(seed_dir.split("_")[1])

        df = pd.read_csv(csv_file)
        if seed not in seed_data:
            seed_data[seed] = {"rewards": [], "costs": [], "steps": []}
        rewards = df["Metrics/EpRet"].tolist()
        costs = df["Metrics/EpCost"].tolist()
        steps = df["TotalEnvSteps"].tolist()

        rewards = [0] + rewards
        costs = [0] + costs
        steps = [0] + steps

        seed_data[seed]["rewards"].append(rewards)
        seed_data[seed]["costs"].append(costs)
        seed_data[seed]["steps"].append(steps)

    for seed in sorted(seed_data.keys()):
        data[env_name]["rewards"].append(seed_data[seed]["rewards"][0])
        data[env_name]["costs"].append(seed_data[seed]["costs"][0])
        data[env_name]["steps"].append(seed_data[seed]["steps"][0])

def compute_stats(values_list):
    if not values_list:
        return None, None
    arr = np.array(values_list)
    return np.mean(arr, axis=0), np.std(arr, axis=0)

plt.figure(figsize=(12, 6))
for env_name, env_info in envs.items():
    if data[env_name]["rewards"]:
        rewards_mean, rewards_std = compute_stats(data[env_name]["rewards"])
        steps = data[env_name]["steps"][0]  # Assume all seeds have the same steps

        plt.plot(steps, rewards_mean, label=env_info["label"], color=env_info["color"], linewidth=2)
        plt.fill_between(steps, rewards_mean - rewards_std, rewards_mean + rewards_std, alpha=0.2, color=env_info["color"])

# create a reward plot
plt.xlabel("Training Step")
plt.ylabel("Episode Reward")
plt.title("Reward Comparison Across Environments (Mean ± Std)")
plt.legend()
plt.grid(True)
plt.savefig("./results/reward_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 6))
for env_name, env_info in envs.items():
    if data[env_name]["costs"]:
        costs_mean, costs_std = compute_stats(data[env_name]["costs"])
        steps = data[env_name]["steps"][0]

        plt.plot(steps, costs_mean, label=env_info["label"], color=env_info["color"], linewidth=2)
        plt.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std, alpha=0.2, color=env_info["color"])

# create a cost plot
plt.xlabel("Training Step")
plt.ylabel("Episode Cost")
plt.title("Cost Comparison Across Environments (Mean ± Std)")
plt.legend()
plt.grid(True)
plt.savefig("./results/cost_comparison.png", dpi=300, bbox_inches="tight")
plt.show()