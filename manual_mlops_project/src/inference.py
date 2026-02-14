import csv
import json
import yaml
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from fastapi import FastAPI, HTTPException

PROJ_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJ_ROOT / "config.yaml"
MODELS_DIR = PROJ_ROOT / "models"
DEPLOYMENT_LOG = PROJ_ROOT / "deployment_log.csv"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

model_version = config["training"]["model_version"]
model_path = MODELS_DIR / f"{model_version}_model.joblib"
metadata_path = MODELS_DIR / f"{model_version}_metadata.json"

if not model_path.exists():
    raise FileNotFoundError(
        f"""Model file for version {model_version} not found.
        Please run src/train.py to train the model."""
    )

if not metadata_path.exists():
    raise FileNotFoundError(
        f"""Metadata file for version {model_version} not found.
        Please run src/train.py to train the model."""
    )

log_exists = DEPLOYMENT_LOG.exists()
with open(DEPLOYMENT_LOG, "a") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "model_version", "status"])
    writer.writerow([datetime.now().isoformat(), model_version, "deployed"])

model = joblib.load(model_path)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

expected_features = metadata["feature_names"]

processed_dir = PROJ_ROOT / config["data"]["processed_dir"]
data_version = config["training"]["data_version"]
train_path = processed_dir / f"{data_version}_train.csv"

scaler_path = processed_dir / f"{data_version}_scaler.joblib"
scaler = joblib.load(scaler_path)
if not train_path.exists():
    raise FileNotFoundError(f"Training data {train_path} not found")

train_df = pd.read_csv(train_path)
X_train = train_df[expected_features]

feature_stats = {
    col: {
        "mean": float(X_train[col].mean()),
        "std": float(X_train[col].std()),
        "min": float(X_train[col].min()),
        "max": float(X_train[col].max())
    }
    for col in expected_features
}

app = FastAPI(title = "ML Inference API", version = "1.0")

class InferenceRequest(BaseModel):
    Air_temperature_K: float = Field(
        ...,
        alias="Air temperature [K]",
        ge = 0,
        description = "Ambient air temperature"
    )
    Process_Temperature_K: float = Field(
        ...,
        alias="Process temperature [K]",
        ge = 0,
        description = "Machine process temperature"
    )
    Rotational_speed_rpm: float = Field(
        ...,
        alias="Rotational speed [rpm]",
        ge = 0,
        description = "Spindnle rotation speed"
    )
    Torque_Nm: float = Field(
        ...,
        alias="Torque [Nm]",
        ge = 0,
        description="Applied mechanical torque"
    )
    Tool_wear_min: float= Field(
        ...,
        alias="Tool wear [min]",
        ge = 0,
        description =  "Tool wear duration"
    )
    TWF: int = Field(..., ge = 0, le = 1, description = "Tool wear failure")
    HDF: int = Field(..., ge = 0, le = 1, description = "Heat dissipation failure")
    PWF: int = Field(..., ge = 0, le = 1, description = "Power failure flag")
    OSF: int = Field(..., ge = 0, le = 1, description = "Overstrain failure flag")
    RNF: int = Field(..., ge = 0, le = 1, description = "Random failure flag")
    Type_L: int = Field(..., ge = 0, le = 1, description = "Machine type L")
    Type_M: int = Field(..., ge = 0, le = 1, description = "Machine type M")

    @model_validator(mode = "after")
    def validate_machine_type(self):
        if self.Type_L + self.Type_M > 1: # If both are zero, it implies type H
            raise ValueError("Only one machine type allowed")
        return self
# class InferenceRequest(BaseModel):
#    features: Dict[str, Any]

numeric_cols_path = processed_dir / f"{data_version}_numeric_cols.json"
with open(numeric_cols_path, "r") as f:
    numeric_cols = json.load(f)

@app.get("/")
def health_check():
    return {"status": "ok",
            "model_version": model_version,
            "expected_features": expected_features,
            "example_payload":feature_stats}

@app.get("/schema")
def schema():
    return {
        "model_version": model_version,
        "features": feature_stats,
        "example_payload": {
            "features": {
                f: round(stats["mean"], 3)
                for f, stats in feature_stats.items()
            }
        }
    }


@app.post("/predict")
def predict(request: InferenceRequest):
    input_dict = request.model_dump()
    
    input_dict = request.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])
    df = df[expected_features]

    df[numeric_cols] = scaler.transform(df[numeric_cols])

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return {
        "model_version": model_version,
        "prediction": int(prediction),
        "probability_class_0": float(proba[0]),
        "probability_class_1": float(proba[1])
    }
    """
    input_features = {
        "Air Temperature [K]": request.Air_temperature_K,
        "Process Temperature [K]": request.Process_Temperature_K,
        "Rotational speed [rpm]": request.Rotational_speed_rpm,
        "Torque [Nm]": request.Torque_Nm,
        "Tool wear [min]": request.Tool_wear_min,
        "TWF": request.TWF,
        "HDF": request.HDF,
        "PWF": request.PWF,
        "OSF": request.OSF,
        "RNF": request.RNF,
        "Type_L": request.Type_L,
        "Type_M": request.Type_M
    }
    missing = set(expected_features) - set(input_features.keys())
    extra = set(input_features.keys()) - set(expected_features)

    if missing or extra:
        raise HTTPException(
            status_code = 400,
            detail = {
                "message": "Invalid input features",
                "missing_features": list(missing),
                "extra_features": list(extra)
            }
        )
    df = pd.DataFrame([[input_features[f] for f in expected_features]], columns = expected_features)
    prediction = model.predict(df)[0]
    return {
        "model_version": model_version,
        "prediction": int(prediction)
    }
    """