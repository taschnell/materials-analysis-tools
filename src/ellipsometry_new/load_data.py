from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_csv_data(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    if data.dtype.names is None:
        return np.asarray(data)

    return {name: np.asarray(data[name]) for name in data.dtype.names}


def load_elippso_data(csv_path=None):
    """Load CSV data from the repository data directory.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to a specific CSV file. If None, all CSV files in DATA_DIR are loaded.

    Returns
    -------
    dict
        A dictionary keyed by CSV stem name when multiple files are loaded.
        If a single file is requested, the raw array or column mapping is returned.
    """
    if csv_path is None:
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

        return {csv_file.stem: _load_csv_data(csv_file) for csv_file in csv_files}

    path = Path(csv_path)
    if not path.is_absolute():
        path = DATA_DIR / path

    return _load_csv_data(path)


if __name__ == "__main__":
    data = load_elippso_data()
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{key}: columns={list(value.keys())}")
            
        else:
            print(f"{key}: shape={value.shape}")
