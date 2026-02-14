import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    data = r.json()
    print(data)
    assert "model_version" in data
    print("Model version present")

def test_schema():
    r = requests.get(f"{BASE_URL}/schema")
    assert r.status_code == 200
    data = r.json()
    assert "features" in data # Check if any input is provided under features in the Swagger UI
    print("Schema structure is correct")

def test_prediction():
    payload = {
    "Air temperature [K]": 298.9,
    "Process temperature [K]": 309.1,
    "Rotational speed [rpm]": 2861,
    "Torque [Nm]": 4.6,
    "Tool wear [min]": 143,
    "TWF": 0,
    "HDF": 0,
    "PWF": 1,
    "OSF": 0,
    "RNF": 0,
    "Type_L": 1,
    "Type_M": 0
    }


    r = requests.post(f"{BASE_URL}/predict", json = payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    print("Prediction is present")

def test_schema_valid():
    payload = {
        "random_attribute": 123
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload) # pass a random attribute to check the returned result
    assert r.status_code in [400, 422] # Corresponds to 400 - Bda request and 422 - Unprocessable content
    print("Correct message for wrong input structure")

if __name__ == "__main__":
    test_health()
    test_schema()
    test_prediction()
    test_schema_valid()
    print("All smoke tests passed")