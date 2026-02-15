# MLOps Assignment 1
Karan Kishore, DA25D400

**MLOps Assignment 1** – Manual MLOps Project

## Contents

- Python scripts for the assignment
- `.gitignore` to exclude large or temporary files
- PDFs / Documentation or explanations related to the project

## Usage

1. Clone the repository:
git clone https://github.com/KaranKishoreD/DA5402_MLOps.git

2. Navigate to the project folder:
cd assignment-1-KaranKishoreD

3. Run the API server (inside the same container as step 4):
python -m uvicorn src.inference:app --reload

4. Use monitor.py to find data drifts from simulate_drift.py (runs in the same container as step 3)
