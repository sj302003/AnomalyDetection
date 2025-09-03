# experiments/run_experiment.py
import argparse
import os
from go_odif.datasets import load_csv
from go_odif.preprocessing import prepare_splits
from go_odif.cere import CERE
from go_odif.iforest import ODIForest
from go_odif.deas import deas_scores
from go_odif.metrics import pr_auc, roc_auc
from go_odif.utils import timer
import numpy as np
import csv

def main(args):
    X_train, y_train, X_val, y_val, X_test, y_test = load_csv(
        args.data, label_col=args.label if args.label else None, test_size=0.2, val_size=0.2, random_state=args.seed
    )

    prep, X_train_t, X_val_t, X_test_t = prepare_splits(X_train, X_val, X_test)

    cere = CERE(input_dim=X_train_t.shape[1], d_out=args.d_out, depth=args.depth, activation=args.activation, seed=args.seed)

    with timer("fit_ODIF"):
        forest = ODIForest(t=args.trees, psi=args.psi, seed=args.seed).fit(X_train_t, cere=cere)

    with timer("score_val"):
        val_scores = forest.score_samples(X_val_t, cere=cere)
    with timer("score_test"):
        test_scores = forest.score_samples(X_test_t, cere=cere)

    # Optionally compute DEAS scores for comparison
    compute_deas = args.scoring == "deas"
    if compute_deas:
        Z_val = cere.transform(X_val_t)
        Z_test = cere.transform(X_test_t)
        val_scores_deas = deas_scores(Z_val, forest.trees)
        test_scores_deas = deas_scores(Z_test, forest.trees)
    else:
        val_scores_deas = None
        test_scores_deas = None

    results = {
        "val_pr_auc": pr_auc(y_val, val_scores) if y_val is not None else None,
        "val_roc_auc": roc_auc(y_val, val_scores) if y_val is not None else None,
        "test_pr_auc": pr_auc(y_test, test_scores) if y_test is not None else None,
        "test_roc_auc": roc_auc(y_test, test_scores) if y_test is not None else None,
        "val_pr_auc_deas": pr_auc(y_val, val_scores_deas) if (y_val is not None and val_scores_deas is not None) else None,
        "test_pr_auc_deas": pr_auc(y_test, test_scores_deas) if (y_test is not None and test_scores_deas is not None) else None,
    }

    print("\n=== Results ===")
    print(f"VAL PR-AUC (classic): {results['val_pr_auc']}")
    print(f"VAL ROC-AUC (classic): {results['val_roc_auc']}")
    if compute_deas:
        print(f"VAL PR-AUC (DEAS): {results['val_pr_auc_deas']}")
    print(f"TEST PR-AUC (classic): {results['test_pr_auc']}")
    if compute_deas:
        print(f"TEST PR-AUC (DEAS): {results['test_pr_auc_deas']}")

    # Save to CSV report
    os.makedirs("reports", exist_ok=True)
    out_file = os.path.join("reports", "results.csv")
    header = ["data", "val_pr_auc", "val_roc_auc", "test_pr_auc", "test_roc_auc", "val_pr_auc_deas", "test_pr_auc_deas"]
    exists = os.path.exists(out_file)
    with open(out_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([os.path.basename(args.data), results["val_pr_auc"], results["val_roc_auc"], results["test_pr_auc"], results["test_roc_auc"], results["val_pr_auc_deas"], results["test_pr_auc_deas"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--psi", type=int, default=256)
    parser.add_argument("--d_out", type=int, default=128)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu","gelu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scoring", type=str, default="classic", choices=["classic","deas"])
    args = parser.parse_args()
    main(args)
