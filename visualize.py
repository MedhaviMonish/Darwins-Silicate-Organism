import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("logs/metrics.jsonl", lines=True)
df[["avg_episode_reward", "running_reward", "policy_loss", "entropy"]].plot()
plt.show()
