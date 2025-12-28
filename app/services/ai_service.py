import json
import os
from typing import Any, Dict, List, Tuple
import math

from groq import Groq
from pydantic import ValidationError

from app.schemas.telemetry import AIOutput, TelemetryInput


SYSTEM_PROMPT = (
    "Eres el Jefe de Estrategia de un equipo de F1. "
    "Sé concluyente y específico aunque falte información parcial: usa lo disponible, evita frases como 'no se puede evaluar'. "
    "Produce insights técnicos sobre degradación térmica, ventanas de pit stop, consistencia de sectores e impacto del clima en el ritmo. "
    "Si faltan datos, aplica criterios de ingeniería (inferencias razonables, outliers, interpolación simple) y continúa el análisis. "
    "Si el campo 'pitStops' está presente, úsalo como fuente de verdad: indica las vueltas y duración de las paradas. "
    "Si hay discrepancia entre 'pitStops' y el resto de la telemetría, prioriza 'pitStops' y explica la discrepancia brevemente. "
    "Responde exclusivamente con un JSON válido que respete exactamente este esquema (sin sugerencias de gráficos): "
    "{\n"
    "  \"summary\": string,\n"
    "  \"key_findings\": [ { \"topic\": string, \"description\": string, \"severity\": \"low|med|high\" } ],\n"
    "  \"strategic_report\": {\n"
    "    \"race_narrative\": string,\n"
    "    \"next_race_projections\": string\n"
    "  }\n"
    "}\n"
    "No incluyas texto fuera del JSON ni comentarios. No incluyas claves de gráficos."
)


def _scan_anomalies(data: TelemetryInput) -> Tuple[int, List[str]]:
    count = 0
    examples: List[str] = []
    for lap in data.telemetry:
        ln = getattr(lap, "lapNumber", None)
        if getattr(lap, "lapDuration", None) is None:
            count += 1
            if len(examples) < 5:
                examples.append(f"lap {ln}: lapDuration=None")
        # sectors (dict form)
        for k, v in (lap.sectors or {}).items():
            if v is None:
                count += 1
                if len(examples) < 5:
                    examples.append(f"lap {ln}: sector {k}=None")
        # sectors (flat form sector1/2/3)
        for k in ("sector1", "sector2", "sector3"):
            if hasattr(lap, k):
                v = getattr(lap, k)
                if v is None:
                    count += 1
                    if len(examples) < 5:
                        examples.append(f"lap {ln}: {k}=None")
        # weather (any None)
        if isinstance(lap.weather, dict):
            for k, v in (lap.weather or {}).items():
                if v is None:
                    count += 1
                    if len(examples) < 5:
                        examples.append(f"lap {ln}: weather {k}=None")
        elif lap.weather is None:
            count += 1
            if len(examples) < 5:
                examples.append(f"lap {ln}: weather=None")
    return count, examples


def _normalize_sectors(data: TelemetryInput) -> None:
    # If sectors dict is empty, fill from sector1/sector2/sector3 when available
    for lap in data.telemetry:
        try:
            sectors_dict = getattr(lap, "sectors", None)
            needs_fill = not sectors_dict or len(sectors_dict) == 0
            if not needs_fill:
                continue

            s1 = getattr(lap, "sector1", None) if hasattr(lap, "sector1") else None
            s2 = getattr(lap, "sector2", None) if hasattr(lap, "sector2") else None
            s3 = getattr(lap, "sector3", None) if hasattr(lap, "sector3") else None

            new_sectors = {}
            # Preserve explicit None as anomaly so AI sees the gap
            if hasattr(lap, "sector1"):
                new_sectors["s1"] = s1
            if hasattr(lap, "sector2"):
                new_sectors["s2"] = s2
            if hasattr(lap, "sector3"):
                new_sectors["s3"] = s3

            if new_sectors:
                lap.sectors = new_sectors
        except Exception:
            # Best-effort normalization; ignore edge cases silently
            pass


def _derived_stats(data: TelemetryInput) -> Dict[str, Any]:
    pit_stop_laps: List[int] = []
    # 1) Top-level pitStops as source of truth
    if hasattr(data, "pitStops") and data.pitStops:
        for ps in data.pitStops:
            lap_no = getattr(ps, "lapNumber", None)
            if isinstance(lap_no, int):
                pit_stop_laps.append(lap_no)
    # 2) Fallback detection from each lap's pitStop field
    for lap in data.telemetry:
        ps = getattr(lap, "pitStop", None)
        if isinstance(ps, dict) and len(ps) > 0:
            pit_stop_laps.append(lap.lapNumber)
        elif isinstance(ps, (bool, int)) and ps:
            pit_stop_laps.append(lap.lapNumber)
    pit_stop_laps = sorted(list(dict.fromkeys(pit_stop_laps)))

    compounds_seq: List[str] = []
    for lap in data.telemetry:
        comp = getattr(lap, "tireCompound", None)
        if comp is not None:
            compounds_seq.append(str(comp))

    return {
        "pit_stop_count_detected": len(pit_stop_laps),
        "pit_stop_laps": pit_stop_laps,
        "compounds_sequence_present": len(compounds_seq) > 0,
    }


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    return (num / den) if den > 0 else 0.0


def _std(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(var)


def _extract_track_temp(lap: Any) -> Any:
    w = getattr(lap, "weather", None)
    if isinstance(w, dict):
        if "trackTemperature" in w:
            return w.get("trackTemperature")
        if "trackTemp" in w:
            return w.get("trackTemp")
    return None


def _stint_summaries(data: TelemetryInput, derived: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Build map lapNumber -> lap for ordering
    laps = sorted(list(data.telemetry), key=lambda l: getattr(l, "lapNumber", 0))
    stop_laps = set(derived.get("pit_stop_laps", []))

    stints: List[Dict[str, Any]] = []
    if not laps:
        return stints

    def close_stint(start_idx: int, end_idx_exclusive: int):
        if start_idx >= end_idx_exclusive:
            return
        sub = laps[start_idx:end_idx_exclusive]
        comp = next((getattr(l, "tireCompound", None) for l in sub if getattr(l, "tireCompound", None) is not None), None)
        start_lap = getattr(sub[0], "lapNumber", None)
        end_lap = getattr(sub[-1], "lapNumber", None)
        xs: List[float] = []
        ys: List[float] = []
        s1s: List[float] = []
        s2s: List[float] = []
        s3s: List[float] = []
        for l in sub:
            ln = getattr(l, "lapNumber", None)
            d = getattr(l, "lapDuration", None)
            if isinstance(ln, int) and isinstance(d, (int, float)):
                xs.append(float(ln))
                ys.append(float(d))
            sec = getattr(l, "sectors", {}) or {}
            if isinstance(sec.get("s1"), (int, float)):
                s1s.append(float(sec["s1"]))
            if isinstance(sec.get("s2"), (int, float)):
                s2s.append(float(sec["s2"]))
            if isinstance(sec.get("s3"), (int, float)):
                s3s.append(float(sec["s3"]))
        slope = _linear_slope(xs, ys) if ys else 0.0
        avg = (sum(ys) / len(ys)) if ys else None
        stints.append({
            "compound": comp,
            "start": start_lap,
            "end": end_lap,
            "laps": len(sub),
            "avgLap": avg,
            "slopeSecPerLap": slope,
            "stdSectors": {
                "s1": _std(s1s),
                "s2": _std(s2s),
                "s3": _std(s3s),
            }
        })

    i = 0
    while i < len(laps):
        start = i
        curr_comp = getattr(laps[i], "tireCompound", None)
        i += 1
        while i < len(laps):
            ln_prev = getattr(laps[i - 1], "lapNumber", None)
            ln = getattr(laps[i], "lapNumber", None)
            comp = getattr(laps[i], "tireCompound", None)
            # cut if pit stop at previous lap (next stint starts after stop)
            if isinstance(ln_prev, int) and ln_prev in stop_laps:
                break
            # cut on compound change
            if comp is not None and curr_comp is not None and comp != curr_comp:
                break
            i += 1
        close_stint(start, i)
    return stints



def _build_user_prompt(data: TelemetryInput) -> str:
    total_laps = len(data.telemetry)
    # Support both totalStops and totalPitStops
    total_stops = (
        getattr(data.raceSummary, "totalStops", None)
        if hasattr(data.raceSummary, "totalStops")
        else getattr(data.raceSummary, "totalPitStops", None)
    )
    # Support both compounds dict and compoundsUsed list
    compounds = getattr(data.raceSummary, "compounds", None)
    compounds_used = getattr(data.raceSummary, "compoundsUsed", None)

    anomalies_count, samples = _scan_anomalies(data)
    derived = _derived_stats(data)

    lines = [
        "Contexto de la telemetría:",
        f"- Vueltas recibidas: {total_laps}",
        f"- Paradas reportadas (resumen): {total_stops}",
        f"- Paradas provistas (pitStops): {derived['pit_stop_count_detected']} en vueltas {derived['pit_stop_laps']}",
        f"- Compuestos: {list(compounds.keys()) if isinstance(compounds, dict) else (compounds_used if isinstance(compounds_used, list) else compounds)}",
        f"- Anomalías de sensores detectadas (valores null): {anomalies_count}",
    ]
    if samples:
        lines.append(f"  Ejemplos: {', '.join(samples)}")

    # Stints summary
    stints = _stint_summaries(data, derived)
    if stints:
        pretty = []
        for s in stints:
            avg = f"{s['avgLap']:.3f}s" if isinstance(s.get('avgLap'), (int, float)) else "n/d"
            slope = s.get('slopeSecPerLap') or 0.0
            slope_txt = ("+" if slope >= 0 else "") + f"{slope:.3f}s/lap"
            ss = s.get('stdSectors', {}) or {}
            s1 = ss.get('s1', 0.0); s2 = ss.get('s2', 0.0); s3 = ss.get('s3', 0.0)
            pretty.append(f"{s.get('compound') or 'UNK'} L{s.get('start')}-{s.get('end')}: avg {avg}, slope {slope_txt}, std(s1/s2/s3)={s1:.3f}/{s2:.3f}/{s3:.3f}")
        lines.append("\nStints detectados:" )
        lines.extend([f"- {p}" for p in pretty])

    # Simple climate correlation
    temps: List[float] = []
    laps_: List[float] = []
    for lap in data.telemetry:
        t = _extract_track_temp(lap)
        d = getattr(lap, "lapDuration", None)
        if isinstance(t, (int, float)) and isinstance(d, (int, float)):
            temps.append(float(t)); laps_.append(float(d))
    if len(temps) >= 8:
        # Pearson r
        r = _linear_slope(temps, laps_) * ( (sum((t - sum(temps)/len(temps))**2 for t in temps) / len(temps)) ** 0.5 )
        # Above isn't exact Pearson, but slope*std_x approximates covariance/var_x*std_x = cov/std_x = r*std_y
        # Keep it simple: show monotonic trend via slope of lapDuration vs temp
        slope_temp = _linear_slope(temps, laps_)
        lines.append(f"\nClima-ritmo: slope(lapDuration vs trackTemp)={slope_temp:+.4f} s/°C (n={len(temps)})")

    lines.append(
        "\nInstrucciones de salida: Devuelve SOLO JSON válido con las claves: "
        "summary, key_findings, strategic_report (race_narrative, next_race_projections). "
        "No incluyas sugerencias de gráficos."
    )

    lines.append(
        "\nReglas estrictas para el análisis (sé específico y accionable):\n"
        "1) No digas que 'no se puede evaluar' por datos faltantes; infiere con lo disponible.\n"
        "2) Si existe el campo 'pitStops', úsalo como fuente de verdad (vueltas y duración).\n"
        "   Si también detectas paradas en el ritmo/compuesto, compara y explica.\n"
        "3) Identifica stints por compuesto y comenta degradación (tendencia de tiempos por vuelta/sector).\n"
        "4) Menciona clima solo si hay correlación visible con el ritmo; si falta en algunas vueltas, no bloquea el análisis.\n"
        "5) Resalta fallas (inconsistencia sectorial, gestión térmica, ejecución de pit) y da 3-5 acciones concretas para la próxima carrera.\n"
        "6) Usa lenguaje técnico pero claro, sin relleno."
    )

    return "\n".join(lines)


def _validate_output(payload: Dict[str, Any]) -> AIOutput:
    try:
        if hasattr(AIOutput, "model_validate"):
            return AIOutput.model_validate(payload)  # pydantic v2
        return AIOutput.parse_obj(payload)  # pydantic v1
    except ValidationError as ve:
        raise ValueError(f"AI output validation failed: {ve}")


def analyze_telemetry(data: TelemetryInput) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY no configurada en el entorno.")

    client = Groq(api_key=api_key)

    # Normalize sectors to a consistent dict form before building the prompt
    _normalize_sectors(data)
    user_prompt = _build_user_prompt(data)

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise RuntimeError(f"Error llamando a Groq: {e}")

    try:
        content = completion.choices[0].message.content  # type: ignore[index]
    except Exception:
        raise RuntimeError("Respuesta de Groq incompleta o sin contenido.")

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"No se pudo parsear el JSON de Groq: {e}")

    output = _validate_output(parsed)
    return output.to_dict()
