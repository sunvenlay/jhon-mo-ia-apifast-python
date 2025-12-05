"""Esquemas Pydantic utilizados por la API. No se toca ninguna base de datos."""
from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class TipoVehiculo(IntEnum):
    CAMION = 1
    BUS = 2
    TRAILER = 3


class InputPrediccion(BaseModel):
    distancia_km: float = Field(gt=0, description="Distancia planificada del viaje")
    tipo_vehiculo: TipoVehiculo = Field(description="1=Camion, 2=Bus, 3=Trailer")
    peajes_estimados: float = Field(ge=0, description="Monto esperado de peajes")


class OutputPrediccion(BaseModel):
    costo_estimado: float


class InputAnomalia(BaseModel):
    costo_real: float = Field(gt=0)
    costo_estimado_ia: float = Field(gt=0)
    distancia_km: float = Field(gt=0)


class OutputAnomalia(BaseModel):
    es_anomalia: bool
    mensaje: str
