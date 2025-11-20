import numpy as np
from .ga import GAOptimizer
import csv
import os

def run_feature_stability(X_train, y_train, X_val, y_val, runs=20, seed=0):
    feature_count = X_train.shape[1]

    all_masks = []
    all_best = []

    print(f"\n[INFO] Running {runs} GA runs for Feature Stability Analysis...")

    for r in range(runs):
        print(f"\n==== GA RUN {r+1}/{runs} ====")

        ga = GAOptimizer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_count=feature_count,
            pop_size=10,
            generations=5,
            mutation_rate=0.3,
            seed=seed + r
        )

        best = ga.run()
        all_best.append(best)
        all_masks.append(best["feature_mask"])

    # Convert to matrix
    mask_matrix = np.vstack(all_masks)

    # Stability score = fraction of runs where feature mask=1
    stability_scores = mask_matrix.sum(axis=0) / runs

    # Prepare feature ranking
    ranking = [
        (i, stability_scores[i])
        for i in range(feature_count)
    ]
    ranking.sort(key=lambda x: x[1], reverse=True)

    print("\n===== FEATURE STABILITY RANKING =====")
    for feat_idx, score in ranking:
        print(f"Feature {feat_idx} → Stability = {score:.3f}")

    # Save to CSV
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "feature_stability.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "stability_score"])
        for feat_idx, score in ranking:
            writer.writerow([feat_idx, score])

    print(f"\n[INFO] Feature stability report saved at: {out_path}")

    return ranking
