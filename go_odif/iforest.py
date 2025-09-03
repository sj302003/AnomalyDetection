# go_odif/iforest.py
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import math

@dataclass
class ITreeNode:
    is_leaf: bool
    size: int
    split_feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["ITreeNode"] = None
    right: Optional["ITreeNode"] = None

def _c(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

def _build_itree(Z: np.ndarray, current_depth: int, max_depth: int, rng: np.random.RandomState) -> ITreeNode:
    n, d = Z.shape
    if current_depth >= max_depth or n <= 1 or np.allclose(Z, Z[0]):
        return ITreeNode(is_leaf=True, size=n)
    feat = rng.randint(0, d)
    col = Z[:, feat]
    cmin, cmax = col.min(), col.max()
    if cmin == cmax:
        return ITreeNode(is_leaf=True, size=n)
    thr = rng.uniform(cmin, cmax)
    left_mask = col <= thr
    right_mask = ~left_mask
    if not left_mask.any() or not right_mask.any():
        return ITreeNode(is_leaf=True, size=n)
    left = _build_itree(Z[left_mask], current_depth + 1, max_depth, rng)
    right = _build_itree(Z[right_mask], current_depth + 1, max_depth, rng)
    return ITreeNode(is_leaf=False, size=n, split_feature=feat, threshold=thr, left=left, right=right)

def _path_length_point(z: np.ndarray, node: ITreeNode, current_depth: int) -> float:
    if node.is_leaf:
        return current_depth + _c(node.size)
    if z[node.split_feature] <= node.threshold:
        return _path_length_point(z, node.left, current_depth + 1)
    else:
        return _path_length_point(z, node.right, current_depth + 1)

class ODIForest:
    def __init__(self, t: int = 100, psi: int = 256, max_depth: Optional[int] = None, seed: int = 0):
        self.t = t
        self.psi = psi
        self.max_depth = max_depth
        self.seed = seed
        self.trees: List[ITreeNode] = []
        self.sample_indices: List[np.ndarray] = []
        self._rng = np.random.RandomState(seed)
        self._c_psi = _c(psi)

    def fit(self, X: np.ndarray, cere) -> "ODIForest":
        n, d = X.shape
        if self.max_depth is None:
            self.max_depth = int(math.ceil(math.log2(max(2, self.psi))))
        self.trees.clear()
        self.sample_indices.clear()
        for i in range(self.t):
            rng_i = np.random.RandomState(self._rng.randint(1_000_000_000))
            if n <= self.psi:
                idx = np.arange(n)
            else:
                idx = rng_i.choice(n, size=self.psi, replace=False)
            self.sample_indices.append(idx)
            Z_sub = cere.transform(X[idx])  # transform only subsample (ODIF trick)
            tree = _build_itree(Z_sub, current_depth=0, max_depth=self.max_depth, rng=rng_i)
            self.trees.append(tree)
        return self

    def score_samples(self, X: np.ndarray, cere) -> np.ndarray:
        # For week1: assume same cere used for all trees (transform X once)
        Z = cere.transform(X)
        n = X.shape[0]
        scores = np.zeros(n, dtype=float)
        for i in range(n):
            z = Z[i]
            h_sum = 0.0
            for tree in self.trees:
                h_sum += _path_length_point(z, tree, 0)
            E_h = h_sum / max(1, self.t)
            s = 2 ** (-E_h / self._c_psi) if self._c_psi > 0 else 0.0
            scores[i] = s
        return scores
