# go_odif/preprocessing.py
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "Preprocessor":
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

def prepare_splits(X_train, X_val, X_test):
    prep = Preprocessor()
    X_train_t = prep.fit_transform(X_train)
    X_val_t = prep.transform(X_val)
    X_test_t = prep.transform(X_test)
    return prep, X_train_t, X_val_t, X_test_t
