import json
import yaml
import pandas as pd
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJ_ROOT / "models"
CONFIG_PATH = PROJ_ROOT / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

model_version = config["training"]["model_version"]
prediction_log_path = PROJ_ROOT / config["monitoring"]["prediction_log"]
day2_path = PROJ_ROOT / config["monitoring"]["day2_data"]
threshold = config["monitoring"]["error_threshold"]

metadata_path = MODELS_DIR / f"{model_version}_metadata.json"
with open(metadata_path, "r") as f:
    metadata = json.load(f)

train_acc = metadata["accuracy"]
train_err = 1 - train_acc

pred_log = pd.read_csv(prediction_log_path)
day2_data = pd.read_csv(day2_path)
target_col = config["features"]["target"]

n = min(len(pred_log), len(day2_data))
pred_log = pred_log.iloc[:n]
day2_df = day2_data.iloc[:n]

prod_acc = (
    pred_log["prediction"] == day2_df[target_col]
).mean()

prod_err = 1 - prod_acc
print("Training error: ", train_err)
print("Production error:", prod_err)

if prod_err > train_err + threshold:
    print("Dirft detected. Retrain model")
else:
    print("Model stable.")