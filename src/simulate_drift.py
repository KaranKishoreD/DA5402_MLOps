import yaml
import pandas as pd
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJ_ROOT / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

processed_dir = PROJ_ROOT / config["data"]["processed_dir"]
production_dir = PROJ_ROOT / config["data"]["production_dir"]
data_version = config["training"]["data_version"]

test_path = processed_dir / f"{data_version}_test.csv"
df = pd.read_csv(test_path)
target_col = config["features"]["target"]
sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
for col in sensor_cols:
    df[col] = df[col] * 1.15

flip_indices = df.sample(frac = 0.2, random_state = 42).index
df.loc[flip_indices, target_col] = 1 - df.loc[flip_indices, target_col]

day2_path = production_dir / f"day2_data.csv"
df.to_csv(day2_path, index = False)

print(f"Day 2 data created.")

