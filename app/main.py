import os
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.schemas.telemetry import TelemetryInput, AIOutput
from app.services.ai_service import analyze_telemetry


load_dotenv()  # Load variables from .env if present

app = FastAPI(
    title="F1 Race Engineer AI",
    version="0.1.0",
    description=(
        "FastAPI microservice that consumes a processed telemetry JSON and returns a strategic, "
        "race-engineering analysis via Groq (llama-3.1-8b-instant).\n\n"
        "Input JSON supports: \n"
        "- raceSummary: { totalLaps?, totalStops?/totalPitStops?, compounds?/compoundsUsed? }\n"
        "- pitStops: [ { lapNumber, duration?, totalDuration? } ] (optional but preferred for accuracy)\n"
        "- telemetry: [ laps with lapNumber, lapDuration?, sectors{s1,s2,s3}? or sector1/2/3, tireCompound?, pitStop?, weather? ]\n\n"
        "Nulls are treated as sensor anomalies but do not block analysis."
    ),
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health():
    return {"status": "ok", "groq": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/analyze", response_model=AIOutput, tags=["analysis"])
async def analyze(
    payload: TelemetryInput = Body(
        ..., 
        examples={
            "minimal": {
                "summary": "Minimal valid payload with pitStops + telemetry",
                "value": {
                    "raceSummary": {"totalLaps": 58, "totalPitStops": 1, "compoundsUsed": ["MEDIUM", "HARD"]},
                    "pitStops": [{"lapNumber": 23, "duration": 21.7}],
                    "telemetry": [
                        {"lapNumber": 1, "sector2": 38.489, "sector3": 32.363, "tireCompound": "MEDIUM"},
                        {"lapNumber": 2, "lapDuration": 89.117, "sector1": 18.085, "sector2": 38.383, "sector3": 32.649, "tireCompound": "MEDIUM", "weather": {"trackTemperature": 31.4}},
                        {"lapNumber": 23, "lapDuration": 91.365, "sector1": 17.903, "sector2": 38.59, "sector3": 34.872, "tireCompound": "MEDIUM"},
                        {"lapNumber": 24, "lapDuration": 108.923, "sector1": 38.129, "sector2": 38.518, "sector3": 32.276, "tireCompound": "HARD"}
                    ]
                }
            }
        }
    )
):
    try:
        result = analyze_telemetry(payload)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except RuntimeError as re:
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")
