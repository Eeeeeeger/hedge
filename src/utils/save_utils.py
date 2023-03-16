import pickle
import json
from pathlib import Path


def save_pickle(data, path: str or Path):
    with open(Path(path), 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str or Path):
    with open(Path(path), 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, path: str or Path):
    with open(Path(path), 'w') as f:
        json.dump(data, f)


def load_json(path: str or Path):
    with open(Path(path), 'r') as f:
        data = json.loads(f)
    return data
