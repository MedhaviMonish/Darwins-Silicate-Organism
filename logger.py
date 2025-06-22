import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsLogger:
    def __init__(self, path: str = "logs", run_name: Optional[str] = None) -> None:
        run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(path) / run_name / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.reset()

    def reset(self) -> None:
        self.episode_data = {}

    def log_scalar(self, key: str, value: float) -> None:
        self.episode_data[key] = float(value)

    def write(self) -> None:
        record = {"step": self.step, **self.episode_data}
        with self.path.open("a") as f:
            json.dump(record, f)
            f.write("\n")
        self.step += 1
        self.reset()
