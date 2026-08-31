from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class RaceSummary(BaseModel):
    totalLaps: Optional[int] = None
    totalStops: Optional[int] = None
    compounds: Optional[Dict[str, int]] = None

    class Config:
        extra = "allow"


class TelemetryLap(BaseModel):
    lapNumber: int
    lapDuration: Optional[float] = None
    sectors: Dict[str, Optional[float]] = Field(default_factory=dict)
    tireCompound: Optional[str] = None
    # Accepts bool/int/object/null per upstream data variations
    pitStop: Optional[Union[bool, int, Dict[str, Optional[Any]]]] = None
    # Accept dict or null
    weather: Optional[Dict[str, Optional[Any]]] = None

    class Config:
        extra = "allow"


class TelemetryInput(BaseModel):
    raceSummary: RaceSummary
    telemetry: List[TelemetryLap]
    class Config:
        extra = "allow"


class PitStop(BaseModel):
    lapNumber: int
    duration: Optional[float] = None
    totalDuration: Optional[float] = None

    class Config:
        extra = "allow"


class Precedent(BaseModel):
    """Resumen compacto de la carrera pasada más parecida (RAG por vector de features,
    calculado y elegido en f1-data-bc antes de llamar acá -- ver docs/RAG-PLAN.md)."""
    circuitShortName: Optional[str] = None
    similarity: Optional[float] = None
    pitStopCount: Optional[int] = None
    compoundSequence: List[str] = Field(default_factory=list)
    summary: Optional[str] = None

    class Config:
        extra = "allow"


class TelemetryInput(BaseModel):
    raceSummary: RaceSummary
    telemetry: List[TelemetryLap]
    pitStops: Optional[List[PitStop]] = None
    precedent: Optional[Precedent] = None
    class Config:
        extra = "allow"


# === AI Output Schema ===
Severity = Literal["low", "med", "high"]
ChartType = Literal["line", "bar", "radar"]


class Finding(BaseModel):
    topic: str
    description: str
    severity: Severity

    # El LLM no siempre respeta el enum abreviado del prompt (ej. devuelve "medium"
    # en vez de "med") -- normalizamos variantes conocidas antes de validar en vez
    # de confiar en que el modelo sea 100% consistente.
    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        aliases = {"medium": "med", "moderate": "med", "critical": "high", "severe": "high", "minor": "low"}
        normalized = value.strip().lower()
        return aliases.get(normalized, normalized)


class ChartSuggestion(BaseModel):
    type: ChartType
    title: str
    data_keys: List[str]
    reason: str


class StrategicReport(BaseModel):
    race_narrative: str
    next_race_projections: str
    # charts disabled for token savings; keep field but default empty
    charts: List[ChartSuggestion] = Field(default_factory=list)


class AIOutput(BaseModel):
    summary: str
    key_findings: List[Finding]
    strategic_report: StrategicReport

    def to_dict(self) -> Dict[str, Any]:
        # pydantic v1/v2 compatible export
        if hasattr(self, "model_dump"):
            return self.model_dump(exclude_none=True, exclude_defaults=True)
        # pydantic v1
        return self.dict(exclude_none=True, exclude_defaults=True)
