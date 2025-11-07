from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Important SMART attributes for failure prediction
SMART_ATTRIBUTES = {
    1: "Read Error Rate",              # Frequency of read errors during normal operation
    5: "Reallocated Sectors Count",    # Bad sectors that were remapped - direct indicator of disk problems
    10: "Spin Retry Count",            # Number of retry attempts to spin up the drive
    184: "End-to-End Error",           # Hardware ECC recovered error count
    187: "Reported Uncorrectable",     # Number of uncorrectable errors - indicates severe read/write issues
    188: "Command Timeout",            # Number of aborted operations due to HDD timeout
    193: "Load/Unload Cycle Count",    # Count of load/unload cycles into head landing zone
    194: "Temperature",                # Current temperature of the drive
    197: "Current Pending Sectors",    # Sectors waiting to be remapped - potential future failures
    198: "Offline Uncorrectable",      # Sectors that couldn't be remapped - severe disk damage
    199: "UDMA CRC Error Count"        # Cyclic Redundancy Check error rate during UDMA operations
}

def load_csv(
    path: str = "data/2025-04-02.csv",
    label_col: str = "failure",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load hard drive failure prediction dataset focusing on critical SMART attributes.
    
    The selected SMART attributes are:
    - SMART 5: Shows reallocated bad sectors
    - SMART 187: Shows uncorrectable errors
    - SMART 197: Shows pending bad sectors
    - SMART 198: Shows offline uncorrectable sectors
    
    These attributes are chosen because they directly indicate physical disk problems.
    """
    # Load the dataset
    df = pd.read_csv(path)
    print("Loading dataset:", path)
    
    # Select only relevant SMART attributes (raw values)
    selected_columns = [f"smart_{id}_raw" for id in SMART_ATTRIBUTES.keys()]
    selected_columns.append(label_col)
    
    # Keep only relevant columns
    df = df[selected_columns]
    
    # Print feature information
    print("\nSelected SMART attributes:")
    for smart_id, description in SMART_ATTRIBUTES.items():
        print(f"SMART {smart_id}: {description}")
    
    # Handle label column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset")
    
    # Split features and target
    y = df[label_col].values.astype(int)
    X = df.drop(columns=[label_col])
    
    # Print class distribution
    n_failures = y.sum()
    print(f"\nClass distribution:")
    print(f"Total samples: {len(y)}")
    print(f"Normal drives: {len(y) - n_failures}")
    print(f"Failed drives: {n_failures}")
    print(f"Failure rate: {n_failures/len(y)*100:.2f}%")
    
    # Handle missing values
    X = X.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.values.astype(float))
    
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
