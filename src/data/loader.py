import pandas as pd
from src.config import DATA_PATH, TARGET_COL

def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    # Convert blank strings to NaN
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df
