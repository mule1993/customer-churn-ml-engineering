import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV data from a given path.

    Parameters
    ----------
    path : str
        Path to CSV file (local, Drive mount, or cloud-synced)

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty")

    return df
