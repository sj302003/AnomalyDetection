from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_csv(
    path: str,
    label_col: str = "failure",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load hard drive failure prediction dataset with proper scaling.
    
    Args:
        path: Path to the CSV file
        label_col: Name of the label column (default: 'failure')
        test_size: Proportion of dataset to include in test split
        val_size: Proportion of dataset to include in validation split
        random_state: Random state for reproducibility
    """
    # Load the dataset
    df = pd.read_csv(path)
    
    # Print column info for debugging
    print("Dataset columns:", df.columns.tolist())
    
    # Drop non-feature columns we don't need
    drop_cols = ['date', 'serial_number', 'model', 'datacenter', 
                 'cluster_id', 'vault_id', 'pod_id']
    
    # Handle label column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset")
    
    y = df[label_col].values.astype(int)
    df = df.drop(columns=[label_col] + drop_cols)
    
    # Convert categorical columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle missing values
    df = df.fillna(0)
    
    # Scale features to prevent overflow
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values.astype(float))
    
    # Split data with stratification
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, random_state=random_state,
        stratify=y_tmp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test