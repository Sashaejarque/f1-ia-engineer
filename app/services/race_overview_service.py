import json
import os
from typing import Any, Dict, List, Optional

from groq import Groq
from pydantic import ValidationError

from app.schemas.telemetry import AIOutput
from app.schemas.race_overview import RaceOverviewInput


SYSTEM_PROMPT = (
    "Role: F1 race chronicler and analyst covering the FULL GRID of a race, not a single driver's strategy. "
    "You receive a compact digest: final classification, pit strategy per driver, and a weather summary. "
    "Be specific, use the available data (avoid 'cannot evaluate'). Cover the overall race story: who won, "
    "key battles implied by close gaps, strategy divergences (drivers who pitted more/less or used different "
    "compound sequences than the field), and weather impact if relevant. "
    "Some drivers may have position: null (DNF/did not classify) -- this is real race data, not a data error; "
    "mention notable retirements if present, do not treat them as anomalies to explain away. "
    "Apply engineering/strategic inference from the digest; avoid conjecture beyond what the data supports. "
    "Write all natural-language content (summary, topic, description, race_narrative, next_race_projections) in neutral Spanish (español neutro/rioplatense) -- the frontend UI is in Spanish, the analysis should match. "
    "Output ONLY valid JSON: {\"summary\":string, \"key_findings\":[{\"topic\":string, \"description\":string, \"severity\":\"low|med|high\"}], \"strategic_report\":{\"race_narrative\":string, \"next_race_projections\":string}}. "
    "JSON keys stay exactly as specified above (English), only the string VALUES are in Spanish. "
    "severity must be EXACTLY one of the three literal strings low, med, high (English, lowercase) -- never translate it and never write out 'medium', 'moderate', 'critical' or any other word."
)


def _driver_label(d) -> str:
    return f"{d.fullName} (#{d.driverNumber})"


def _build_race_prompt(data: RaceOverviewInput) -> str:
    strategy_by_driver: Dict[int, Any] = {s.driverNumber: s for s in data.strategies}

    # Clasificados primero (por posición), DNFs al final (orden estable por driverNumber)
    classified = sorted(
        [d for d in data.classification if d.position is not None],
        key=lambda d: d.position,
    )
    dnfs = sorted(
        [d for d in data.classification if d.position is None],
        key=lambda d: d.driverNumber,
    )
    ordered = classified + dnfs

    header_bits = [f"SessionKey: {data.sessionKey}"]
    if data.circuitShortName:
        header_bits.append(f"Circuit: {data.circuitShortName}")
    if data.year:
        header_bits.append(f"Year: {data.year}")
    header = ", ".join(header_bits)

    lines = ["Pos,Num,Piloto,Equipo,Pts,DNF,Paradas,Gomas,Gap"]
    for d in ordered:
        strat = strategy_by_driver.get(d.driverNumber)
        pit_stops = strat.pitStopCount if strat else "-"
        compounds = "-".join(strat.compoundSequence) if strat and strat.compoundSequence else "-"
        pos_val = str(d.position) if d.position is not None else "DNF"
        gap_val = d.gapToLeader if d.gapToLeader else "-"
        row = f"{pos_val},{d.driverNumber},{d.fullName},{d.teamName},{d.points},{d.dnf},{pit_stops},{compounds},{gap_val}"
        lines.append(row)
    table = "\n".join(lines)

    weather_line = "Weather: n/a"
    if data.weather:
        w = data.weather
        weather_line = (
            f"Weather: AirTemp {w.airTempStart}->{w.airTempEnd}, "
            f"TrackTemp {w.trackTempStart}->{w.trackTempEnd}, Rained: {w.rained}"
        )

    return f"{header}\n\n{table}\n\n{weather_line}"


def _validate_output(payload: Dict[str, Any]) -> AIOutput:
    try:
        if hasattr(AIOutput, "model_validate"):
            return AIOutput.model_validate(payload)  # pydantic v2
        return AIOutput.parse_obj(payload)  # pydantic v1
    except ValidationError as ve:
        raise ValueError(f"AI output validation failed: {ve}")


def analyze_race(data: RaceOverviewInput) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY no configurada en el entorno.")

    client = Groq(api_key=api_key)

    user_prompt = _build_race_prompt(data)

    # Mismo patrón de retry que analyze_telemetry en ai_service.py: reintentamos toda la
    # llamada (no solo el parseo) ante un JSON inválido o que no valida contra AIOutput,
    # el modelo no es 100% consistente en json_object mode.
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            last_error = RuntimeError(f"Error llamando a Groq: {e}")
            continue

        if hasattr(completion, "usage"):
            usage = completion.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            print(f"TOKENS_USAGE | attempt={attempt + 1} | prompt_tokens={prompt_tokens} | completion_tokens={completion_tokens} | total_tokens={total_tokens}")

        try:
            content = completion.choices[0].message.content  # type: ignore[index]
            parsed = json.loads(content)
            output = _validate_output(parsed)
            return output.to_dict()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Groq no devolvió una salida válida tras 3 intentos: {last_error}")
