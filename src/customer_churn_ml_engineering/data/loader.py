import pandas as pd
import numpy as np
from src.customer_churn_ml_engineering.config import DATA_PATH, TARGET_COL

def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
        
    # Normalize missing values / Convert blank strings to NaN
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # 🚨 CRITICAL: drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    
    # 🔑 CRITICAL: convert pd.NA → np.nan for sklearn
    df = df.replace({pd.NA: np.nan})
    return df
