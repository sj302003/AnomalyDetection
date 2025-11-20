# experiments/run_experiment.py
import argparse
import os
import csv
import numpy as np

from go_odif.datasets import load_csv
from go_odif.preprocessing import prepare_splits
from go_odif.cere import CERE
from go_odif.iforest import ODIForest
from go_odif.deas import deas_scores
from go_odif.metrics import pr_auc, roc_auc
from go_odif.utils import timer
from go_odif.ga import GAOptimizer
from go_odif.feature_stability import run_feature_stability


def main(args):

    # -------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_csv(
        args.data,
        label_col=args.label if args.label else None,
        test_size=0.2,
        val_size=0.2,
        random_state=args.seed
    )

    # -------------------------------------------------------
    # 2. STANDARDIZE
    # -------------------------------------------------------
    prep, X_train_t, X_val_t, X_test_t = prepare_splits(
        X_train, X_val, X_test
    )

    print("\n[INFO] Data loaded and standardized.")
    print(f"Train shape = {X_train_t.shape}, Test shape = {X_test_t.shape}")

    # -------------------------------------------------------
    # STEP 3 — Feature Stability Mode
    # -------------------------------------------------------
    if args.stability_runs > 0:
        print("\n[MODE] Running FEATURE STABILITY ANALYSIS MODE ONLY.")
        run_feature_stability(
            X_train_t, y_train,
            X_val_t, y_val,
            runs=args.stability_runs,
            seed=args.seed
        )
        print("\n[INFO] Stability analysis completed. Exiting.")
        return

    # -------------------------------------------------------
    # 3. RUN GENETIC ALGORITHM (normal mode)
    # -------------------------------------------------------
    ga = GAOptimizer(
        X_train=X_train_t,
        y_train=y_train,
        X_val=X_val_t,
        y_val=y_val,
        feature_count=X_train_t.shape[1],
        pop_size=args.ga_pop,
        generations=args.ga_gens,
        mutation_rate=0.3,
        seed=args.seed
    )

    print("\n[INFO] Running Genetic Algorithm Optimization...")
    best = ga.run()

    print("\n=== GA Best Chromosome ===")
    for k, v in best.items():
        print(f"{k}: {v}")

    # -------------------------------------------------------
    # 4. USE BEST GA FEATURES
    # -------------------------------------------------------
    selected = np.where(best["feature_mask"] == 1)[0]
    X_train_sel = X_train_t[:, selected]
    X_val_sel = X_val_t[:, selected]
    X_test_sel = X_test_t[:, selected]

    # -------------------------------------------------------
    # 5. BUILD FINAL CERE
    # -------------------------------------------------------
    cere = CERE(
        input_dim=len(selected),
        d_out=best["d_out"],
        depth=best["depth"],
        activation=best["activation"],
        seed=args.seed
    )

    # -------------------------------------------------------
    # 6. TRAIN ODIForest
    # -------------------------------------------------------
    print("\n[INFO] Training final ODIForest...")
    with timer("final_fit"):
        forest = ODIForest(
            t=best["t"],
            psi=best["psi"],
            max_depth=best["max_depth"],
            seed=args.seed
        ).fit(X_train_sel, cere=cere)

    # -------------------------------------------------------
    # 7. SCORE
    # -------------------------------------------------------
    with timer("score_val"):
        val_scores = forest.score_samples(X_val_sel, cere=cere)
    with timer("score_test"):
        test_scores = forest.score_samples(X_test_sel, cere=cere)

    if args.scoring == "deas":
        Z_val = cere.transform(X_val_sel)
        Z_test = cere.transform(X_test_sel)
        val_scores_deas = deas_scores(Z_val, forest.trees)
        test_scores_deas = deas_scores(Z_test, forest.trees)
    else:
        val_scores_deas = None
        test_scores_deas = None

    # -------------------------------------------------------
    # 8. METRICS
    # -------------------------------------------------------
    results = {
        "val_pr_auc": pr_auc(y_val, val_scores),
        "val_roc_auc": roc_auc(y_val, val_scores),
        "test_pr_auc": pr_auc(y_test, test_scores),
        "test_roc_auc": roc_auc(y_test, test_scores),
        "val_pr_auc_deas": pr_auc(y_val, val_scores_deas) if val_scores_deas is not None else None,
        "test_pr_auc_deas": pr_auc(y_test, test_scores_deas) if test_scores_deas is not None else None,
    }

    print("\n=== FINAL RESULTS ===")
    print(results)

    # -------------------------------------------------------
    # 9. SAVE RESULTS
    # -------------------------------------------------------
    os.makedirs("reports", exist_ok=True)
    with open("reports/results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "data", "val_pr_auc", "val_roc_auc",
                "test_pr_auc", "test_roc_auc",
                "val_pr_auc_deas", "test_pr_auc_deas"
            ])
        writer.writerow([
            os.path.basename(args.data),
            results["val_pr_auc"],
            results["val_roc_auc"],
            results["test_pr_auc"],
            results["test_roc_auc"],
            results["val_pr_auc_deas"],
            results["test_pr_auc_deas"],
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)

    # GA params
    parser.add_argument("--ga_pop", type=int, default=10)
    parser.add_argument("--ga_gens", type=int, default=5)

    # Stability study
    parser.add_argument("--stability_runs", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scoring", type=str, default="classic", choices=["classic", "deas"])

    args = parser.parse_args()
    main(args)
