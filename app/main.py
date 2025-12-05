"""Punto de entrada de FastAPI para el microservicio de costos."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .ml_logic import CostModel, get_cost_model
from .models import InputAnomalia, InputPrediccion, OutputAnomalia, OutputPrediccion
from .utils import ANOMALY_THRESHOLD, evaluar_anomalia, estimar_consumo

logger = logging.getLogger("jhon-mo-ai")

app = FastAPI(title="Jhon Mo IA", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def cargar_modelo() -> None:
    try:
        app.state.cost_model = get_cost_model()
        logger.info("Modelo de costos cargado correctamente")
    except FileNotFoundError:
        app.state.cost_model = None
        logger.warning("Archivo cost_model.pkl no encontrado; ejecutar ml/train_model.py")
    except Exception as exc:  # pragma: no cover - solo logging defensivo
        app.state.cost_model = None
        logger.exception("Error inesperado al cargar el modelo: %s", exc)


def obtener_modelo() -> CostModel:
    modelo: Optional[CostModel] = getattr(app.state, "cost_model", None)
    if modelo is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no disponible. Ejecuta ml/train_model.py antes de iniciar la API.",
        )
    return modelo


@app.get("/health")
def healthcheck() -> dict[str, str]:
    modelo_cargado = "true" if getattr(app.state, "cost_model", None) else "false"
    return {"status": "ok", "model_ready": modelo_cargado}


@app.post("/predict_cost", response_model=OutputPrediccion)
def predict_cost(payload: InputPrediccion, modelo: CostModel = Depends(obtener_modelo)) -> OutputPrediccion:
    tipo_valor = int(payload.tipo_vehiculo)
    consumo_estimado = estimar_consumo(payload.distancia_km, tipo_valor)
    features = {
        "distancia_km": payload.distancia_km,
        "tipo_vehiculo": tipo_valor,
        "peajes": payload.peajes_estimados,
        "consumo_galones": consumo_estimado,
    }
    costo_estimado = modelo.predict(features)
    return OutputPrediccion(costo_estimado=round(costo_estimado, 2))


@app.post("/detect_anomaly", response_model=OutputAnomalia)
def detect_anomaly(payload: InputAnomalia) -> OutputAnomalia:
    es_anomalia, diferencia = evaluar_anomalia(payload.costo_real, payload.costo_estimado_ia)
    if es_anomalia:
        mensaje = f"ALERTA: Gasto excede {ANOMALY_THRESHOLD:.0f}% de la prediccion (desviacion {diferencia:.2f}%)"
    else:
        mensaje = f"Gasto dentro de lo normal (desviacion {diferencia:.2f}%)"
    return OutputAnomalia(es_anomalia=es_anomalia, mensaje=mensaje)
