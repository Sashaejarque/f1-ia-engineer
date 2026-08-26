# F1 Race Engineer AI (FastAPI + Groq)

A Python microservice that ingests processed F1 telemetry JSON (from your NestJS backend) and returns a strictly-JSON strategic race analysis using Groq (`openai/gpt-oss-20b`).

## Technologies
- FastAPI — HTTP server + OpenAPI/Swagger UI
- Uvicorn — ASGI server
- Pydantic — Input/output validation
- Groq — LLM inference
- python-dotenv — Environment variables via `.env`
- (Optional) Pandas — future data helpers

## Environment
- Set `GROQ_API_KEY` in `.env` (recommended) or export it.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit GROQ_API_KEY
uvicorn app.main:app --reload --port 8080
```

- Health: http://localhost:8080/health
- Swagger: http://localhost:8080/docs
- OpenAPI JSON: http://localhost:8080/openapi.json
- Redoc: http://localhost:8080/redoc

## Input JSON Format (No Errors)
The service is robust to missing fields and `null` values — they are treated as sensor anomalies rather than blocking analysis. To avoid validation issues, use the following shapes:

- `raceSummary` (object):
  - `totalLaps` (int, optional)
  - `totalStops` or `totalPitStops` (int, optional)
  - `compounds` (object of counts, optional) or `compoundsUsed` (array of strings)

- `pitStops` (array, optional but preferred):
  - Each item: `{ lapNumber: int, duration?: number, totalDuration?: number }`
  - This is treated as source of truth for pit-stop detection.

- `telemetry` (array of objects): each lap may include:
  - `lapNumber` (int, required)
  - `lapDuration` (number, optional)
  - Either `sectors` (object) with keys `s1`, `s2`, `s3`; or any of `sector1`, `sector2`, `sector3` (numbers or null). The service normalizes these into `sectors`.
  - `tireCompound` (string, optional)
  - `pitStop` (bool | int | object, optional)
  - `weather` (object or null, optional). Known keys include `trackTemperature` or `trackTemp`, plus any additional fields.

### Example Payload
```json
{
  "raceSummary": { "totalLaps": 58, "totalPitStops": 1, "compoundsUsed": ["MEDIUM", "HARD"] },
  "pitStops": [ { "lapNumber": 23, "duration": 21.7 } ],
  "telemetry": [
    { "lapNumber": 1, "sector2": 38.489, "sector3": 32.363, "tireCompound": "MEDIUM" },
    { "lapNumber": 2, "lapDuration": 89.117, "sector1": 18.085, "sector2": 38.383, "sector3": 32.649, "tireCompound": "MEDIUM", "weather": { "trackTemperature": 31.4 } },
    { "lapNumber": 23, "lapDuration": 91.365, "sector1": 17.903, "sector2": 38.59, "sector3": 34.872, "tireCompound": "MEDIUM" },
    { "lapNumber": 24, "lapDuration": 108.923, "sector1": 38.129, "sector2": 38.518, "sector3": 32.276, "tireCompound": "HARD" }
  ]
}
```

## Output JSON Schema
The service returns strict JSON:
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
- No charts are included for token savings.
- Missing inputs are acknowledged as sensor anomalies but do not block analysis.

## Testing
- Postman: import the collection at `postman/F1-IA-Engineer.postman_collection.json` and environment `postman/F1-IA-Engineer.local.postman_environment.json`.
- Curl example:
```bash
curl -X POST http://localhost:8080/analyze \
  -H 'Content-Type: application/json' \
  -d @payload.json
```

## Notes
- The prompt is optimized to be decisive with partial data, prioritize `pitStops`, detect stints, and provide actionable recommendations.
- Sector data is normalized to `sectors` to improve consistency analysis.
