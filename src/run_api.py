import yaml
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJ_ROOT / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

base_url = config["api"]["base_url"]
predict_endpoint = config["api"]["predict_endpoint"]
day2_path = PROJ_ROOT / config["monitoring"]["day2_data"]

url = base_url + predict_endpoint

df = pd.read_csv(day2_path)
target_col = config["features"]["target"]
feature_cols = [col for col in df.columns if col != target_col]

print(f"Running {len(df)} records through API.")
successes = 0
failures = 0

for _, row in tqdm(df.iterrows(), total = len(df)):
    payload = row[feature_cols].to_dict()
    try:
        response = requests.post(url, json = payload)
        if response.status_code == 200:
            successes += 1
        else:
            failures += 1
            print("Error: ", response.text)
    except Exception as e:
        failures += 1
        print("Request failed:", str(e))

print("\nFinished.")
print("Successful requests:", successes)
print("Failed requests:", failures)