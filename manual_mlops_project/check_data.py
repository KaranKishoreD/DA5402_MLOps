import pandas as pd
import yaml
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[0]
print("Hi:", PROJ_ROOT)
processed_dir = PROJ_ROOT / "data/processed"
config_path = PROJ_ROOT / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
version = config["data"]["current_version"]

train_path = processed_dir / f"{version}_train.csv"
test_path = processed_dir / f"{version}_test.csv"

if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        f"""Processed files for version {version} not found.
        Please run src/data_prep.py to generate the files."""
    )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Train df details: {train_df.describe()}\n")
print(f"Test df details: {test_df.describe()}")
print(f"{test_df.columns}")