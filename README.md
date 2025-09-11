# AnomalyDetection

## ⚙️ Setup

1. Clone repo:
   ```bash
   git clone https://github.com/sj302003/AnomalyDetection.git
   cd AnomalyDetection

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate -- windows
   source venv/bin/activate --macOS/Linux

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install numpy pandas scikit-learn

4. Running the Baseline:
   ```bash
   python -m experiments.run_experiment --data "data/2025-04-01.csv" --label failure

5. Run with DEAS scoring:
   ```bash
   python -m experiments.run_experiment --data data/2025-04-01.csv --label failure --scoring deas




