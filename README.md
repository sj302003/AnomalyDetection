# AnomalyDetection

## ⚙️ Setup

1. Clone repo:
   ```bash
   git clone https://github.com/<your-username>/GO-ODIF.git
   cd GO-ODIF

2. Create and activate virtual environment:
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
   python -m experiments.run_experiment --data data/toy.csv --label label --trees 100 --psi 128 --d_out 64 --depth 1

5. Run with DEAS scoring:
   ```bash
   python -m experiments.run_experiment --data data/toy.csv --label label --scoring deas



