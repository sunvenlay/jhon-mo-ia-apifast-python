# Microservicio de IA - Jhon Mo

Este proyecto implementa una API REST con **FastAPI** diseñada para servir como el "cerebro" de inteligencia artificial para la plataforma de transporte Jhon Mo. Su función principal es estimar costos operativos y detectar anomalías en los gastos reportados utilizando modelos de Machine Learning (**Scikit-learn**).

## 🚀 Características

- **Predicción de Costos**: Estima el costo total de un viaje basándose en la distancia, el tipo de vehículo y los peajes esperados.
- **Detección de Anomalías**: Compara los costos reales reportados contra las estimaciones de la IA para alertar sobre desviaciones sospechosas (umbral > 15%).
- **Arquitectura Ligera**: Funciona totalmente en memoria (stateless), recibiendo parámetros y devolviendo cálculos sin depender de una conexión directa a base de datos en esta capa.

## 📋 Requisitos

- **Python 3.13**
- Gestor de paquetes: `uv` (recomendado) o `pip`.

## 🛠️ Instalación y Configuración

### 1. Preparar el entorno

Si usas `uv` (como en el entorno actual):
```bash
uv sync
```

O usando `pip` tradicional:
```bash
python -m venv .venv
# Activar entorno:
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Entrenar el Modelo de IA

Antes de levantar la API, necesitas generar el archivo del modelo (`cost_model.pkl`). Este script genera datos sintéticos y entrena una Regresión Lineal.

```bash
# Ejecutar desde la raíz del proyecto
python ml/train_model.py
```
*Deberías ver un mensaje confirmando que el modelo se guardó en `ml/cost_model.pkl`.*

### 3. Iniciar el Servidor

Levanta el servicio utilizando Uvicorn:

```bash
uvicorn app.main:app --reload
```

El servicio estará disponible en: `http://127.0.0.1:8000`

## 📖 Documentación de la API

Una vez levantado el servidor, puedes acceder a la documentación interactiva automática en:
👉 **http://127.0.0.1:8000/docs**

### Endpoints Principales

#### 1. `POST /predict_cost`
Calcula el costo estimado de un viaje.

**Ejemplo de Request:**
```json
{
  "distancia_km": 300.5,
  "tipo_vehiculo": 1,
  "peajes_estimados": 50.0
}
```
*Nota: `tipo_vehiculo`: 1 (Camión), 2 (Bus), 3 (Trailer).*

**Respuesta:**
```json
{
  "costo_estimado": 1250.00
}
```

#### 2. `POST /detect_anomaly`
Evalúa si un gasto real se desvía demasiado de lo esperado.

**Ejemplo de Request:**
```json
{
  "costo_real": 1500.0,
  "costo_estimado_ia": 1250.0,
  "distancia_km": 300.5
}
```

**Respuesta:**
```json
{
  "es_anomalia": true,
  "mensaje": "ALERTA: Gasto excede 15% de la prediccion..."
}
```

#### 3. `GET /health`
Healthcheck para monitoreo. Devuelve si el servicio está activo y si el modelo ML se cargó correctamente.

## 📂 Estructura del Proyecto

```text
/
├── app/
│   ├── main.py       # Punto de entrada de FastAPI y definición de rutas
│   ├── models.py     # Esquemas de datos (Pydantic) para validación
│   ├── ml_logic.py   # Lógica para cargar y consultar el modelo .pkl
│   └── utils.py      # Funciones auxiliares (cálculos matemáticos)
├── ml/
│   ├── train_model.py # Script para generar dataset y entrenar modelo
│   └── cost_model.pkl # Archivo binario del modelo entrenado (ignorado en git)
└── README.md         # Esta documentación
```
