import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.schemas.telemetry import TelemetryInput
from app.services.ai_service import analyze_telemetry


load_dotenv()  # Load variables from .env if present

app = FastAPI(title="F1 Race Engineer AI", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "groq": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/analyze")
async def analyze(payload: TelemetryInput):
    try:
        result = analyze_telemetry(payload)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except RuntimeError as re:
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")
