from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel


class DriverClassification(BaseModel):
    driverNumber: int
    fullName: str
    # Optional: un piloto que corrió carreras anteriores del año pero ya no está en el
    # roster vigente (ej. reemplazado a mitad de temporada) no tiene team_name en /drivers.
    teamName: Optional[str] = None
    position: Optional[int] = None  # None = no clasificó / DNF
    points: float = 0.0
    dnf: bool = False
    # OpenF1 devuelve gap_to_leader como número (segundos) para la mayoría, pero como
    # string ("+1 LAP", "+2 LAPS") para pilotos que quedaron vueltas atrás.
    gapToLeader: Optional[Union[str, float]] = None


class DriverStrategy(BaseModel):
    driverNumber: int
    pitStopCount: int
    compoundSequence: List[str] = []  # ej. ["MEDIUM", "HARD"]


class WeatherSummary(BaseModel):
    airTempStart: Optional[float] = None
    airTempEnd: Optional[float] = None
    trackTempStart: Optional[float] = None
    trackTempEnd: Optional[float] = None
    rained: Optional[bool] = None


class RaceOverviewInput(BaseModel):
    sessionKey: int
    circuitShortName: Optional[str] = None
    year: Optional[int] = None
    classification: List[DriverClassification]
    strategies: List[DriverStrategy] = []
    weather: Optional[WeatherSummary] = None

    class Config:
        extra = "allow"
