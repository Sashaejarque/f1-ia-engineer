import json
import os
from typing import Any, Dict, List, Tuple, Optional
import math

from groq import Groq
from pydantic import ValidationError

from app.schemas.telemetry import AIOutput, TelemetryInput


SYSTEM_PROMPT = (
    "Role: F1 Strategy Chief. Be specific, use available data (avoid 'cannot evaluate'). Provide technical insights on thermal degradation, pit windows, sector consistency, weather impact. "
    "If pitStops present, treat as truth (lap numbers & durations). "
    "Apply engineering inference for gaps; avoid conjecture. "
    "Output ONLY valid JSON: {\"summary\":string, \"key_findings\":[{\"topic\":string, \"description\":string, \"severity\":\"low|med|high\"}], \"strategic_report\":{\"race_narrative\":string, \"next_race_projections\":string}}. "
    "severity must be EXACTLY one of the three literal strings low, med, high -- never write out 'medium', 'moderate', 'critical' or any other word."
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


def _abbrev_compound(compound: Optional[str]) -> str:
    """Abbreviate tire compound to first capital letter."""
    if compound is None:
        return "-"
    upper = compound.upper()
    if upper in ("MEDIUM", "MED"):
        return "M"
    elif upper in ("HARD",):
        return "H"
    elif upper in ("SOFT",):
        return "S"
    elif upper.startswith("I"):
        return "I"  # Intermediate
    else:
        return upper[0] if upper else "-"


def _telemetry_to_csv(data: TelemetryInput) -> str:
    """
    Transform telemetry data into optimized CSV format for token efficiency.
    
    Columns: L (Lap), T (Time), S1, S2, S3, C (Compound), Tmp (Track Temp)
    
    Rules:
    - Times rounded to 2 decimals
    - Compounds abbreviated to first capital letter
    - Nulls represented as "-"
    - Track temp shown only on first lap or when delta > 0.5°C
    """
    if not data.telemetry:
        return "L,T,S1,S2,S3,C,Tmp"
    
    lines = ["L,T,S1,S2,S3,C,Tmp"]
    last_temp: Optional[float] = None
    
    for lap in data.telemetry:
        lap_num = getattr(lap, "lapNumber", None)
        lap_dur = getattr(lap, "lapDuration", None)
        sectors = getattr(lap, "sectors", None) or {}
        compound = getattr(lap, "tireCompound", None)
        
        # Format lap number
        l_val = str(lap_num) if lap_num is not None else "-"
        
        # Format lap duration (time)
        if isinstance(lap_dur, (int, float)):
            t_val = f"{lap_dur:.2f}"
        else:
            t_val = "-"
        
        # Format sectors
        s1_val = f"{sectors.get('s1'):.2f}" if isinstance(sectors.get("s1"), (int, float)) else "-"
        s2_val = f"{sectors.get('s2'):.2f}" if isinstance(sectors.get("s2"), (int, float)) else "-"
        s3_val = f"{sectors.get('s3'):.2f}" if isinstance(sectors.get("s3"), (int, float)) else "-"
        
        # Format compound
        c_val = _abbrev_compound(compound)
        
        # Smart temperature logic
        curr_temp = _extract_track_temp(lap)
        tmp_val = ""
        
        if curr_temp is not None and isinstance(curr_temp, (int, float)):
            curr_temp_f = float(curr_temp)
            # Show on first lap
            if lap_num == 1:
                tmp_val = str(curr_temp_f)
                last_temp = curr_temp_f
            # Show if change > 0.5°C
            elif last_temp is not None and abs(curr_temp_f - last_temp) > 0.5:
                tmp_val = str(curr_temp_f)
                last_temp = curr_temp_f
            # Otherwise, empty (or use quote to indicate "no change")
            else:
                tmp_val = '"'
                if last_temp is None:
                    last_temp = curr_temp_f
        else:
            tmp_val = '"'
        
        # Build CSV row
        row = f"{l_val},{t_val},{s1_val},{s2_val},{s3_val},{c_val},{tmp_val}"
        lines.append(row)
    
    return "\n".join(lines)


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
    total_stops = (
        getattr(data.raceSummary, "totalStops", None)
        if hasattr(data.raceSummary, "totalStops")
        else getattr(data.raceSummary, "totalPitStops", None)
    )
    compounds = getattr(data.raceSummary, "compounds", None)
    compounds_used = getattr(data.raceSummary, "compoundsUsed", None)

    anomalies_count, samples = _scan_anomalies(data)
    derived = _derived_stats(data)
    
    # Compact header with critical info
    pit_laps_str = str(derived['pit_stop_laps']) if derived['pit_stop_laps'] else "none"
    compounds_list = list(compounds.keys()) if isinstance(compounds, dict) else (compounds_used if isinstance(compounds_used, list) else compounds)
    
    header = f"Context: {total_laps} laps, Stops: {derived['pit_stop_count_detected']} (truth: {pit_laps_str}), Compounds: {compounds_list}, Anomalies: {anomalies_count}"

    # Generate CSV
    csv_data = _telemetry_to_csv(data)
    csv_header = "Data (CSV): Lap,Time,S1,S2,S3,C(M/H/S/I),Tmp(=same)"

    return f"{header}\n\n{csv_header}\n{csv_data}"


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

    # gpt-oss-20b en json_object mode casi siempre respeta el schema, pero no siempre --
    # es una salida no determinística de un LLM, no una API tipada. Reintentamos toda la
    # llamada (no solo el parseo) un par de veces ante un JSON inválido o que no valida
    # contra AIOutput, antes de darnos por vencidos.
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                # llama-3.1-8b-instant fue discontinuado por Groq -- gpt-oss-20b es el
                # reemplazo más cercano en tamaño/velocidad que sigue disponible. Es un
                # modelo de razonamiento (gasta tokens en un campo "reasoning" separado
                # antes del JSON final), por eso max_tokens subió de 1200 a 2000 -- con el
                # límite viejo el razonamiento se comía el presupuesto antes de llegar al
                # JSON real.
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                # 2000 se quedaba corto para carreras largas (más vueltas/paradas -> prompt
                # más largo -> menos presupuesto para razonamiento + JSON) -- Groq devolvía
                # json_validate_failed con failed_generation vacío, señal de truncamiento.
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
