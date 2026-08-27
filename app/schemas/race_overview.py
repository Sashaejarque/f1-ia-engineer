from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DriverClassification(BaseModel):
    driverNumber: int
    fullName: str
    teamName: str
    position: Optional[int] = None  # None = no clasificó / DNF
    points: float = 0.0
    dnf: bool = False
    gapToLeader: Optional[str] = None


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
