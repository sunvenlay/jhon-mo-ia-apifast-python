"""Utilidades para cargar y usar el modelo de costos."""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

import joblib

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "cost_model.pkl"
_MODEL_CACHE: Optional["CostModel"] = None
_MODEL_LOCK = Lock()


class CostModel:
    """Wrapper simple alrededor del estimador de scikit-learn."""

    def __init__(self, estimator, features: Iterable[str]):
        self.estimator = estimator
        self.features = list(features)

    def predict(self, feature_values: Dict[str, float]) -> float:
        row: List[float] = [float(feature_values[name]) for name in self.features]
        prediction = self.estimator.predict([row])
        return float(prediction[0])


def load_model(path: Path = DEFAULT_MODEL_PATH) -> CostModel:
    artifact = joblib.load(path)
    estimator = artifact.get("model")
    features = artifact.get("features")
    if estimator is None or features is None:
        raise ValueError("El archivo del modelo no contiene los objetos requeridos")
    return CostModel(estimator=estimator, features=features)


def get_cost_model(path: Path = DEFAULT_MODEL_PATH) -> CostModel:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    with _MODEL_LOCK:
        if _MODEL_CACHE is None:
            if not path.exists():
                raise FileNotFoundError(path)
            _MODEL_CACHE = load_model(path)
    return _MODEL_CACHE
