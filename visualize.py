import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def find_latest_log_file(log_dir="logs", extension="metrics.jsonl"):
    log_dirs = sorted(
        Path(log_dir).glob(f"*/{extension}"),
        key=lambda f: f.parent.stat().st_mtime,
        reverse=True,
    )
    return log_dirs[0] if log_dirs else None


def load_metrics(path):
    return pd.read_json(path, lines=True)


def plot_metrics(df):
    # 1. Reward and Losses
    df[
        [
            "avg_episode_reward",
            "running_reward",
            "loss/total",
            "loss/policy",
            "loss/value",
            "entropy",
        ]
    ].plot(title="Reward & Loss Metrics")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Advantage stats
    df[["advantage/mean", "advantage/std"]].plot(title="Advantage Stats")
    plt.xlabel("Step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Log prob std
    df[["log_prob/std"]].plot(title="Log Probability Std Dev")
    plt.xlabel("Step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", type=str, default=None, help="Path to a metrics .jsonl file"
    )
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        path = find_latest_log_file()
        if path:
            print(f"[Info] Using latest log file: {path}")
        else:
            print("❌ No log files found.")
            return

    df = load_metrics(path)
    plot_metrics(df)


if __name__ == "__main__":
    main()
