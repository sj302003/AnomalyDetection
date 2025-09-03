# go_odif/deas.py
import numpy as np
from .iforest import ITreeNode

def _deas_single(z: np.ndarray, tree: ITreeNode):
    node = tree
    deviations = []
    length = 0
    while not node.is_leaf:
        feat = node.split_feature
        thr = node.threshold
        deviations.append(abs(z[feat] - thr))
        if z[feat] <= thr:
            node = node.left
        else:
            node = node.right
        length += 1
    mean_dev = float(np.mean(deviations)) if len(deviations) > 0 else 0.0
    return length * mean_dev

def deas_scores(X_z: np.ndarray, trees):
    # X_z = already-transformed features (n x d)
    n = X_z.shape[0]
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        z = X_z[i]
        vals = []
        for tree in trees:
            vals.append(_deas_single(z, tree))
        scores[i] = float(np.mean(vals))
    return scores
