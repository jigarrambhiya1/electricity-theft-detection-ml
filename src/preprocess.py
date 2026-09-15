"""Shared preprocessing for TDD2022 electricity-theft data.

Mirrors the logic from notebooks/01_base_models_soft_voting.ipynb so that
both the notebooks and the standalone scripts in src/ use one source of truth.

Dataset columns (actual CSV):
    '0' (row index, dropped), 10 hourly consumption features,
    'Class' (consumer type, 17 values), 'theft' (target:
    'Normal', 'Theft1' ... 'Theft6').

Scenarios (from Mohammad et al. 2023):
    P7C -> 7 classes, 11 features (10 consumption + Class)
    P7U -> 7 classes, 10 features (consumption only)
    P6C -> 6 classes (Theft6 removed), 11 features
    P6U -> 6 classes (Theft6 removed), 10 features

Usage:
    from src.preprocess import get_scenario
    X_train, X_test, y_train, y_test = get_scenario('P7C')
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Default dataset location relative to this file: <repo>/data/Dataset_actual.csv
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Dataset_actual.csv"

CONSUMPTION_FEATURES = [
    'Electricity:Facility [kW](Hourly)',
    'Fans:Electricity [kW](Hourly)',
    'Cooling:Electricity [kW](Hourly)',
    'Heating:Electricity [kW](Hourly)',
    'InteriorLights:Electricity [kW](Hourly)',
    'InteriorEquipment:Electricity [kW](Hourly)',
    'Gas:Facility [kW](Hourly)',
    'Heating:Gas [kW](Hourly)',
    'InteriorEquipment:Gas [kW](Hourly)',
    'Water Heater:WaterSystems:Gas [kW](Hourly)',
]

TARGET_COL = 'theft'
CONSUMER_TYPE_COL = 'Class'

# Set once via load_dataset(); get_scenario() loads lazily on first call.
_DF = None
_DATA_PATH = DEFAULT_DATA_PATH


def load_dataset(data_path=None):
    """Load and encode the raw CSV. Returns the encoded DataFrame."""
    global _DF, _DATA_PATH
    if data_path is not None:
        _DATA_PATH = Path(data_path)
    if _DF is not None:
        return _DF
    path = Path(_DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "See data/README.md for the download link, then place the file in data/."
        )
    df = pd.read_csv(path)
    print("Dataset loaded:", df.shape)
    if '0' in df.columns:
        df = df.drop(columns=['0'])
    print("Missing values:", df.isnull().sum().sum())
    df = df.dropna()
    print("Unique classes in theft column:", df[TARGET_COL].unique())
    df[CONSUMER_TYPE_COL] = df[CONSUMER_TYPE_COL].astype('category').cat.codes
    df[TARGET_COL] = df[TARGET_COL].astype('category').cat.codes
    print("Encoded theft classes:", sorted(df[TARGET_COL].unique()))
    _DF = df
    return _DF


def get_scenario(scenario_name, data_path=None, test_size=0.20, random_state=42):
    """Return stratified 80/20 train/test split for scenario P7C/P7U/P6C/P6U."""
    df = load_dataset(data_path)
    data = df.copy()
    if scenario_name in ('P6C', 'P6U'):
        theft6_code = data[TARGET_COL].max()
        data = data[data[TARGET_COL] != theft6_code]
        print(f"Removed theft6 (code={theft6_code}), rows left: {len(data)}")
    y = data[TARGET_COL].values
    if scenario_name in ('P7C', 'P6C'):
        X = data[CONSUMPTION_FEATURES + [CONSUMER_TYPE_COL]].values
    else:
        X = data[CONSUMPTION_FEATURES].values
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Scenario={scenario_name} | Features={X.shape[1]} | "
          f"Classes={len(np.unique(y))} | "
          f"Train={len(X_train)} | Test={len(X_test)}")
    return X_train, X_test, y_train, y_test
