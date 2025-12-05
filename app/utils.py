"""Funciones auxiliares para logica de negocio."""
from __future__ import annotations

from typing import Tuple

CONSUMO_POR_TIPO = {1: 7.5, 2: 5.5, 3: 3.8}
ANOMALY_THRESHOLD = 15.0  # Porcentaje permitido antes de marcar anomalia.


def estimar_consumo(distancia_km: float, tipo_vehiculo: int) -> float:
    rendimiento = CONSUMO_POR_TIPO.get(tipo_vehiculo, 6.0)
    return max(distancia_km / rendimiento, 0.0)


def calcular_diferencia_porcentual(costo_real: float, costo_estimado: float) -> float:
    if costo_estimado == 0:
        return 0.0
    return ((costo_real - costo_estimado) / costo_estimado) * 100


def evaluar_anomalia(costo_real: float, costo_estimado: float) -> Tuple[bool, float]:
    diferencia = calcular_diferencia_porcentual(costo_real, costo_estimado)
    es_anomalia = diferencia >= ANOMALY_THRESHOLD
    return es_anomalia, diferencia
