# F1 Race Engineer AI (FastAPI + Groq)

Microservicio en Python con FastAPI que recibe telemetría procesada (OpenF1 vía tu backend NestJS) y devuelve un análisis estratégico en JSON usando Groq (modelo `llama-3.1-8b-instant`).

## Estructura

- app/main.py — App FastAPI y endpoint `/analyze`.
- app/services/ai_service.py — Llamada a Groq + validación estricta de JSON.
- app/schemas/telemetry.py — Esquemas Pydantic de entrada y salida.
- requirements.txt — Dependencias.

## Variables de entorno

- `GROQ_API_KEY` — API key de Groq.

Puedes usar un archivo `.env` en la raíz del proyecto:

```
GROQ_API_KEY=tu_api_key
```

Ejemplo: mira [.env.example](.env.example).

## Instalar y ejecutar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Opción A: usar .env (recomendado)
cp .env.example .env  # edita el valor
# Opción B: exportar variable de entorno
# export GROQ_API_KEY=tu_api_key
uvicorn app.main:app --reload --port 8080
```

Healthcheck: http://localhost:8080/health

## Uso

POST `/analyze` con el JSON de telemetría. La respuesta devuelve únicamente análisis (sin sugerencias de gráficos):

```json
{
  "summary": "...",
  "key_findings": [ { "topic": "...", "description": "...", "severity": "low|med|high" } ],
  "strategic_report": {
    "race_narrative": "...",
    "next_race_projections": "..."
  }
}
```

El servicio identifica valores `null` como "anomalías de sensores" y los incorpora al análisis.

## Probar con Postman

1. Arranca el servidor:

```bash
uvicorn app.main:app --reload --port 8080
```

2. En Postman, crea una petición `POST` a `http://localhost:8080/analyze`.
- Header: `Content-Type: application/json`
- Body (raw, JSON):

```json
{
  "raceSummary": { "totalLaps": 58, "totalStops": 2, "compounds": {"C2": 30, "C3": 28} },
  "telemetry": [
    {
      "lapNumber": 1,
      "lapDuration": 92.315,
      "sectors": {"s1": 28.1, "s2": 31.0, "s3": 33.2},
      "tireCompound": "C3",
      "pitStop": 0,
      "weather": {"trackTemp": 41.2, "airTemp": 28.0, "rain": false}
    },
    {
      "lapNumber": 2,
      "lapDuration": null,
      "sectors": {"s1": null, "s2": 31.2, "s3": 33.0},
      "tireCompound": "C3",
      "pitStop": 0,
      "weather": {"trackTemp": 41.0, "airTemp": 28.1, "rain": false}
    }
  ]
}
```

3. Asegúrate de tener `GROQ_API_KEY` configurada (en `.env` o exportada). Si hay problemas, revisa el endpoint `/health`.
