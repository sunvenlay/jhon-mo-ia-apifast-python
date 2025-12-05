"""Script para generar datos sinteticos y entrenar el modelo de costos."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cost_model.pkl"
N_SAMPLES = 500
SEED = 42


def generar_dataset(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distancia_km = rng.uniform(50, 800, size=n_samples)
    tipo_vehiculo = rng.integers(1, 4, size=n_samples)

    # Rendimiento estimado (km por galon) segun tipo de vehiculo.
    rendimiento_por_tipo = {1: 7.5, 2: 5.5, 3: 3.8}
    mantenimiento_por_tipo = {1: 80.0, 2: 120.0, 3: 180.0}
    costo_peaje_base = 12.0
    precio_combustible = 5.2  # USD por galon aproximado.

    rendimientos = np.vectorize(rendimiento_por_tipo.get)(tipo_vehiculo)
    consumo_galones = distancia_km / rendimientos + rng.normal(0, 1.5, size=n_samples)
    consumo_galones = np.clip(consumo_galones, a_min=0.0, a_max=None)

    tramos_estimados = np.ceil(distancia_km / 120)
    peajes = tramos_estimados * costo_peaje_base + rng.normal(0, 1.0, size=n_samples)
    peajes = np.clip(peajes, a_min=0.0, a_max=None)

    mantenimiento = np.vectorize(mantenimiento_por_tipo.get)(tipo_vehiculo)
    ruido_operativo = rng.normal(0, 35.0, size=n_samples)
    costo_total_real = consumo_galones * precio_combustible + peajes + mantenimiento + ruido_operativo

    data = pd.DataFrame(
        {
            "distancia_km": distancia_km,
            "tipo_vehiculo": tipo_vehiculo,
            "consumo_galones": consumo_galones,
            "peajes": peajes,
            "costo_total_real": costo_total_real,
        }
    )
    return data


def entrenar_modelo(data: pd.DataFrame) -> LinearRegression:
    features = ["distancia_km", "tipo_vehiculo", "peajes", "consumo_galones"]
    X = data[features]
    y = data["costo_total_real"]

    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo


def guardar_modelo(modelo: LinearRegression, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": modelo, "features": ["distancia_km", "tipo_vehiculo", "peajes", "consumo_galones"]}
    joblib.dump(payload, path)


def main() -> None:
    data = generar_dataset()
    modelo = entrenar_modelo(data)
    guardar_modelo(modelo)
    print(f"Modelo entrenado y guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
