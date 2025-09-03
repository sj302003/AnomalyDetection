# go_odif/datasets.py
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv(
    path: str,
    label_col: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    if label_col is not None and label_col in df.columns:
        y = df[label_col].values.astype(int)
        X = df.drop(columns=[label_col]).values.astype(float)
        strat = y
    else:
        y = None
        X = df.values.astype(float)
        strat = None

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat if strat is not None else None
    )
    # compute validation split from remaining
    if val_size <= 0 or val_size >= 1:
        X_train, X_val, y_train, y_val = X_tmp, np.empty((0, X_tmp.shape[1])), y_tmp, None
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_size, random_state=random_state, stratify=y_tmp if y_tmp is not None else None
        )
    return X_train, y_train, X_val, y_val, X_test, y_test































