import yaml
import json
import joblib
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJ_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJ_ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

data_version = config["training"]["data_version"]
model_version = config["training"]["model_version"]
processed_dir = PROJ_ROOT / config["data"]["processed_dir"]
target_col = config["features"]["target"]
model_dir = PROJ_ROOT / "models"
model_dir.mkdir(exist_ok = True)
model_path = model_dir / f"{model_version}_model.joblib"
train_path = processed_dir / f"{data_version}_train.csv"
test_path = processed_dir / f"{data_version}_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns = target_col)
y_train = train_df[target_col]
X_test = test_df.drop(columns = target_col)
y_test = test_df[target_col]

params = config["model_params"]
algorithm = params["algorithm"]

if algorithm == "RandomForest":
    model = RandomForestClassifier(
        n_estimators = params["n_estimators"],
        max_depth = params["max_depth"],
        random_state = params["random_state"]
    )
elif algorithm == "LogisticRegression":
    model = LogisticRegression(
        C = params["C"],
        max_iter = params["max_iter"],
        random_state = params["random_state"]
    )
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
if model_path.exists():
    raise FileExistsError(
        f"""Model file already exists for version {model_version}.
        Please update the model version in config.yaml"""
    )
joblib.dump(model, model_path)
try:
    status = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd = PROJ_ROOT
    ).decode("utf-8").strip()
except Exception as e:
    git_commit = "unknown"

if status:
    raise RuntimeError(f"Commit changes before training.")

metadata = {
    "model_version": model_version,
    "data_version": data_version,
    "algorithm": algorithm,
    "accuracy": accuracy,
    "git_commit": git_commit,
    "timestamp": datetime.now().isoformat(),
    "hyperparameters": params,
    "metrics": {
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict = True)
    }
}

metadata_path = model_dir / f"{model_version}_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent = 4)

registry_log = model_dir / "registry_log.txt"
with open(registry_log, "a") as f:
    f.write(
        f"""
Model Version: {model_version}
Data Version: {data_version}
Algorithm: {algorithm}
Accuracy: {accuracy:.4f}
Git Commit: {git_commit}
Timestamp: {metadata['timestamp']}
Hyperparameters: {params}
Metrics: {metadata['metrics']}\n
"""
    )

print(f"{model_version} trained with accuracy: {accuracy:.4f}")