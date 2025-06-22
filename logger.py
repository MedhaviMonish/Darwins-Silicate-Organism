import json
from pathlib import Path
from typing import Dict, Any


class MetricsLogger:
    def __init__(self, path: str = "logs/metrics.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self.episode_data = {}

    def log_scalar(self, key: str, value: float) -> None:
        self.episode_data[key] = float(value)

    def write(self) -> None:
        with self.path.open("a") as f:
            json.dump(self.episode_data, f)
            f.write("\n")
        self.reset()
